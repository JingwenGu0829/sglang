# SPDX-License-Identifier: Apache-2.0
"""Owned, loop-local snapshots of explicitly declared denoising streams.

This module does not inspect requests, stages, schedulers or pipeline configs.
Integration sites supply resolved schedules and a gather operation per stream.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import torch

from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDitTrajectory,
)


def identity_gather(latents: torch.Tensor) -> torch.Tensor:
    return latents


def legacy_video_trajectory(
    streams: Mapping[str, RolloutDitTrajectory],
) -> RolloutDitTrajectory | None:
    """Keep legacy sparse timesteps aligned with retained video latents.

    Named streams always carry full clocks; the legacy projection historically
    returned only the timesteps at retained boundaries, alongside full sigmas.
    """
    video = streams.get("video")
    if video is None:
        return None
    return replace(
        video, timesteps=video.timesteps.index_select(0, video.latent_step_indices)
    )


@dataclass(frozen=True)
class RolloutStreamSpec:
    name: str
    local_shape: tuple[int, ...]  # [B, ...], including for packed single samples
    timesteps: torch.Tensor  # full [N+1] rollout clock, including final boundary
    sigmas: torch.Tensor  # full [N+1] resolved sigma grid
    gather: Callable[[torch.Tensor], torch.Tensor] = identity_gather
    model_timesteps: torch.Tensor | None = None  # optional explicit [N] model clock


class RolloutRecorder:
    """Capture all declared streams at the same pre-update boundary.

    Snapshots own their storage. finish() transfers completed CPU trajectories to
    the caller and clears temporary storage. abort() performs no collectives.
    """

    def __init__(
        self,
        streams: Sequence[RolloutStreamSpec],
        *,
        retain_steps: Sequence[int] | None = None,
    ) -> None:
        if not streams or len({s.name for s in streams}) != len(streams):
            raise ValueError("Rollout streams must be nonempty and uniquely named")
        self._specs: dict[str, RolloutStreamSpec] = {}
        self._num_steps = streams[0].timesteps.numel() - 1
        if self._num_steps < 1:
            raise ValueError("Rollout schedules require at least one transition")
        batch_size = streams[0].local_shape[0] if streams[0].local_shape else 0
        for spec in streams:
            if (
                len(spec.local_shape) < 2
                or spec.local_shape[0] != batch_size
                or batch_size < 1
            ):
                raise ValueError(
                    "Every stream requires an explicit matching batch axis"
                )
            for schedule in (spec.timesteps, spec.sigmas):
                if schedule.ndim != 1 or schedule.numel() != self._num_steps + 1:
                    raise ValueError("Streams require aligned full N+1 schedules")
                if not torch.isfinite(schedule).all():
                    raise ValueError("Rollout schedules must be finite")
            if spec.model_timesteps is not None and (
                spec.model_timesteps.ndim != 1
                or spec.model_timesteps.numel() != self._num_steps
            ):
                raise ValueError("Model timesteps must contain N entries")
            self._specs[spec.name] = RolloutStreamSpec(
                name=spec.name,
                local_shape=spec.local_shape,
                timesteps=spec.timesteps.detach().cpu().clone(),
                sigmas=spec.sigmas.detach().cpu().clone(),
                gather=spec.gather,
                model_timesteps=(
                    spec.model_timesteps.detach().cpu().clone()
                    if spec.model_timesteps is not None
                    else None
                ),
            )
        retained = set(
            range(self._num_steps + 1) if retain_steps is None else retain_steps
        )
        if any(type(i) is not int or not 0 <= i <= self._num_steps for i in retained):
            raise ValueError("Retained indices must be loop boundaries in [0, N]")
        self._retain = retained
        self._snapshots: dict[str, list[torch.Tensor]] = {
            name: [] for name in self._specs
        }
        self._indices: list[int] = []
        self._next_step = 0
        self._closed = False

    def capture_before(self, step: int, state: Mapping[str, torch.Tensor]) -> None:
        if self._closed or step != self._next_step or step >= self._num_steps:
            raise ValueError(
                "Rollout capture must follow consecutive active loop steps"
            )
        self._capture(step, state)
        self._next_step += 1

    def _capture(self, step: int, state: Mapping[str, torch.Tensor]) -> None:
        if state.keys() != self._specs.keys():
            raise ValueError("Joint state must contain exactly the declared streams")
        for name, spec in self._specs.items():
            if tuple(state[name].shape) != spec.local_shape:
                raise ValueError(
                    f"{name} state shape must be {spec.local_shape}, got {tuple(state[name].shape)}"
                )
        if step in self._retain:
            for name in self._specs:
                self._snapshots[name].append(state[name].detach().clone())
            self._indices.append(step)

    def finish(
        self, state: Mapping[str, torch.Tensor]
    ) -> dict[str, RolloutDitTrajectory]:
        if self._closed or self._next_step != self._num_steps:
            raise ValueError("Rollout can only finish once, after every transition")
        try:
            self._capture(self._num_steps, state)
            result = {}
            for name, spec in self._specs.items():
                if self._indices:
                    stacked = torch.stack(self._snapshots[name], dim=1)
                    gathered = spec.gather(stacked)
                    if gathered.shape[:2] != stacked.shape[:2]:
                        raise ValueError("Gather must preserve batch and boundary axes")
                    result[name] = RolloutDitTrajectory(
                        latents=gathered.cpu(),
                        timesteps=spec.timesteps,
                        sigmas=spec.sigmas,
                        latent_step_indices=torch.tensor(
                            self._indices, dtype=torch.int64
                        ),
                        model_timesteps=spec.model_timesteps,
                    )
            return result
        finally:
            self.abort()

    def abort(self) -> None:
        for snapshots in self._snapshots.values():
            snapshots.clear()
        self._indices.clear()
        self._closed = True
