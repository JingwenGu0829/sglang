# SPDX-License-Identifier: Apache-2.0
"""RL-specific dataclasses used by post-training and rollout paths."""

from dataclasses import dataclass, fields, replace
from typing import Any

import torch


@dataclass
class RolloutDebugTensors:
    """Container for rollout debug tensors collected during denoising."""

    rollout_variance_noises: torch.Tensor | None = None
    rollout_prev_sample_means: torch.Tensor | None = None
    rollout_noise_std_devs: torch.Tensor | None = None
    rollout_model_outputs: torch.Tensor | None = None


@dataclass
class RolloutDenoisingEnv:
    image_kwargs: dict[str, Any] | None = None
    pos_cond_kwargs: dict[str, Any] | None = None
    neg_cond_kwargs: dict[str, Any] | None = None
    guidance: torch.Tensor | None = None


@dataclass
class RolloutDitTrajectory:
    # [B, T+1, ...]: per-step noisy latents x_{t_0..t_{T-1}} followed by the
    # final denoised latent x_{t_T} (last scheduler.step output).
    latents: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None  # [T+1], includes the terminal timestep
    # [T+1] scheduler.sigmas snapshot (post-shift, includes terminal 0).
    sigmas: torch.Tensor | None = None
    # [K] original step index of each kept latent (0..T); None means the full
    # 0..T trajectory. Set whenever rollout_return_step_indices filtered it.
    latent_step_indices: torch.Tensor | None = None


@dataclass
class RolloutStreamData:
    """One stream's transition statistics and noisy latent trajectory."""

    rollout_log_probs: torch.Tensor | None = None
    rollout_debug_tensors: RolloutDebugTensors | None = None
    dit_trajectory: RolloutDitTrajectory | None = None


@dataclass
class RolloutTrajectoryData(RolloutStreamData):
    denoising_env: RolloutDenoisingEnv | None = None
    audio: RolloutStreamData | None = None


def _cat_per_output(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Concatenate complete per-output tensor fields along the batch dimension."""
    if not tensors or any(tensor is None for tensor in tensors):
        return None
    return torch.cat(tensors, dim=0)


def _concat_stream_data(
    streams: list[RolloutStreamData | None],
) -> RolloutStreamData | None:
    if not streams or any(stream is None for stream in streams):
        return None
    first = streams[0]
    debug = None
    if all(stream.rollout_debug_tensors is not None for stream in streams):
        debug = RolloutDebugTensors(
            **{
                field.name: _cat_per_output(
                    [
                        getattr(stream.rollout_debug_tensors, field.name)
                        for stream in streams
                    ]
                )
                for field in fields(RolloutDebugTensors)
            }
        )
    trajectory = None
    if all(stream.dit_trajectory is not None for stream in streams):
        trajectory = RolloutDitTrajectory(
            latents=_cat_per_output(
                [stream.dit_trajectory.latents for stream in streams]
            ),
            timesteps=first.dit_trajectory.timesteps,
            sigmas=first.dit_trajectory.sigmas,
            latent_step_indices=first.dit_trajectory.latent_step_indices,
        )
    return replace(
        first,
        rollout_log_probs=_cat_per_output(
            [stream.rollout_log_probs for stream in streams]
        ),
        rollout_debug_tensors=debug,
        dit_trajectory=trajectory,
    )


def concat_rollout_trajectory_data(
    per_output: list[RolloutTrajectoryData | None],
) -> RolloutTrajectoryData | None:
    """Combine aligned per-output trajectories along each stream's batch axis."""
    if not per_output or any(data is None for data in per_output):
        return None
    if len(per_output) == 1:
        return per_output[0]
    visual = _concat_stream_data(per_output)
    return replace(
        visual, audio=_concat_stream_data([data.audio for data in per_output])
    )


def _slice_output_dim(
    tensor: torch.Tensor | None, output_index: int
) -> torch.Tensor | None:
    if tensor is None or tensor.dim() < 1 or output_index >= tensor.shape[0]:
        return tensor
    return tensor[output_index : output_index + 1]


def _select_stream_data(
    stream: RolloutStreamData | None,
    output_index: int,
) -> RolloutStreamData | None:
    if stream is None:
        return None
    debug = stream.rollout_debug_tensors
    if debug is not None:
        debug = RolloutDebugTensors(
            **{
                field.name: _slice_output_dim(getattr(debug, field.name), output_index)
                for field in fields(RolloutDebugTensors)
            }
        )
    trajectory = stream.dit_trajectory
    if trajectory is not None:
        trajectory = RolloutDitTrajectory(
            latents=_slice_output_dim(trajectory.latents, output_index),
            timesteps=trajectory.timesteps,
            sigmas=trajectory.sigmas,
            latent_step_indices=trajectory.latent_step_indices,
        )
    return replace(
        stream,
        rollout_log_probs=_slice_output_dim(stream.rollout_log_probs, output_index),
        rollout_debug_tensors=debug,
        dit_trajectory=trajectory,
    )


def select_output_rollout_trajectory(
    rollout_trajectory_data: RolloutTrajectoryData | None,
    output_index: int | None,
) -> RolloutTrajectoryData | None:
    """Keep one output row, preserving the batch dimension of both streams."""
    if rollout_trajectory_data is None or output_index is None:
        return rollout_trajectory_data
    visual = _select_stream_data(rollout_trajectory_data, output_index)
    return replace(
        visual, audio=_select_stream_data(rollout_trajectory_data.audio, output_index)
    )
