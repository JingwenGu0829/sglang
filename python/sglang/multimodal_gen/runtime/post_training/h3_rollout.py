# SPDX-License-Identifier: Apache-2.0
"""H3 packed-row adapter: video policy with native deterministic audio updates."""

from collections.abc import Callable, Sequence

import torch

from sglang.multimodal_gen.runtime.post_training.flow_transition import (
    FlowMethod,
    flow_step,
)
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDebugTensors,
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.post_training.rollout_recorder import (
    RolloutRecorder,
    RolloutStreamSpec,
    legacy_video_trajectory,
)


class H3RolloutSession:
    """Request-local adapter; has no Req, stage, scheduler, or SP sharding state.

    H3's denoise rows are replicated outside its model-side sequence parallelism.
    Every rank therefore uses the same request generator and full packed shape.
    Only the video stream is sampled stochastically in this first integration.
    """

    def __init__(
        self,
        *,
        video_shape: tuple[int, int],
        video_sigmas: Sequence[float],
        generator: torch.Generator,
        method: FlowMethod,
        noise_level: float,
        legacy_score: bool,
        capture: bool,
        debug: bool,
        retain_steps: Sequence[int] | None,
        sde_steps: Sequence[int] | None,
    ) -> None:
        if method not in ("sde", "cps", "ode"):
            raise ValueError(f"Unsupported H3 rollout method: {method}")
        if method == "ode" and not legacy_score:
            raise ValueError("ODE rollout has no Gaussian density; use legacy scores")
        if len(video_sigmas) < 2:
            raise ValueError("H3 rollout requires at least one video transition")
        self._video_sigmas = tuple(video_sigmas)
        self._num_steps = len(video_sigmas) - 1
        self._sde_steps = None if sde_steps is None else frozenset(sde_steps)
        if self._sde_steps is not None and any(
            type(i) is not int or not 0 <= i < self._num_steps for i in self._sde_steps
        ):
            raise ValueError("H3 SDE indices must be valid loop steps")
        self._shapes = {"video": (1, *video_shape)}
        specs = []
        for name, schedule in (("video", video_sigmas),):
            sigmas = torch.tensor(schedule, dtype=torch.float32)
            if (
                not torch.isfinite(sigmas).all()
                or torch.any(sigmas[1:] > sigmas[:-1])
                or torch.any(sigmas < 0)
            ):
                raise ValueError(
                    "H3 sigmas must be finite, nonnegative and nonincreasing"
                )
            specs.append(
                RolloutStreamSpec(
                    name=name,
                    local_shape=self._shapes[name],
                    timesteps=sigmas * 1000.0,
                    sigmas=sigmas,
                    # Match the native loop's Python subtraction, then fp32 conversion.
                    model_timesteps=torch.tensor([1.0 - s for s in schedule[:-1]]),
                )
            )
        self._recorder = (
            RolloutRecorder(specs, retain_steps=retain_steps) if capture else None
        )
        self._generator = generator
        self._method = method
        self._noise_level = noise_level
        self._legacy_score = legacy_score
        self._debug = debug
        self._scores: list[torch.Tensor] = []
        self._noises: list[torch.Tensor] = []
        self._means: list[torch.Tensor] = []
        self._stds: list[torch.Tensor] = []
        self._predictions: list[torch.Tensor] = []
        self._next_step = 0
        self._closed = False

    def advance(
        self,
        step: int,
        video: torch.Tensor,
        video_velocity: torch.Tensor,
        audio: torch.Tensor,
        audio_velocity: torch.Tensor,
        update_video: Callable[[], None],
        update_audio: Callable[[], None],
    ) -> None:
        """Capture video before updating it; advance audio with its native callback."""
        if self._closed or step != self._next_step or step >= self._num_steps:
            raise ValueError("H3 rollout steps must be consecutive and active")
        state = {"video": video.unsqueeze(0)}
        for name, tensor in state.items():
            if tuple(tensor.shape) != self._shapes[name]:
                raise ValueError(f"H3 {name} shape changed during rollout")
        if self._recorder is not None:
            self._recorder.capture_before(step, state)
        method = self._method
        if self._sde_steps is not None and step not in self._sde_steps:
            method = "ode"
        sample = state["video"].float()
        noise = (
            torch.randn(
                sample.shape,
                generator=self._generator,
                device=sample.device,
                dtype=torch.float32,
            )
            if method != "ode"
            else None
        )
        result = flow_step(
            sample=sample,
            model_output=(-video_velocity).float().unsqueeze(0),
            current_sigma=sample.new_tensor(self._video_sigmas[step]),
            next_sigma=sample.new_tensor(self._video_sigmas[step + 1]),
            sigma_max=self._video_sigmas[1],
            method=method,
            noise_level=self._noise_level,
            legacy_score=self._legacy_score,
            variance_noise=noise,
        )
        self._scores.append((result.score_sum / result.element_count).detach())
        if self._debug:
            self._noises.append(
                noise.clone() if noise is not None else torch.zeros_like(sample)
            )
            self._means.append(result.mean.detach().clone())
            self._stds.append(result.noise_std.detach().expand(1, 1).clone())
            self._predictions.append(result.model_output.detach().clone())
        video.copy_(result.sample.squeeze(0))
        update_audio()
        self._next_step += 1

    def finish(self, video: torch.Tensor) -> RolloutTrajectoryData:
        if self._closed or self._next_step != self._num_steps:
            raise ValueError("H3 rollout must finish once after all transitions")
        try:
            streams = (
                self._recorder.finish({"video": video.unsqueeze(0)})
                if self._recorder is not None
                else {}
            )
            debug = None
            if self._debug:
                debug = RolloutDebugTensors(
                    rollout_variance_noises=torch.stack(self._noises, dim=1).cpu(),
                    rollout_prev_sample_means=torch.stack(self._means, dim=1).cpu(),
                    rollout_noise_std_devs=torch.stack(self._stds, dim=1).cpu(),
                    rollout_model_outputs=torch.stack(self._predictions, dim=1).cpu(),
                )
            return RolloutTrajectoryData(
                rollout_log_probs=torch.stack(self._scores, dim=1).cpu(),
                rollout_debug_tensors=debug,
                dit_trajectory=legacy_video_trajectory(streams),
                stream_trajectories=streams,
            )
        finally:
            self.abort()

    def abort(self) -> None:
        if self._recorder is not None:
            self._recorder.abort()
        for values in (
            self._scores,
            self._noises,
            self._means,
            self._stds,
            self._predictions,
        ):
            values.clear()
        self._closed = True
