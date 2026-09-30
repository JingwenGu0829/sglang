"""Mixin for rollout-related denoising hooks.

Moved out of DenoisingStage to keep the core stage lean.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import torch

from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutCollectionState,
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)
from sglang.multimodal_gen.runtime.post_training.sp_utils import (
    gather_stacked_latents_for_sp,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _kwargs_to_cpu(d: Any) -> Any:
    if isinstance(d, torch.Tensor):
        return d.detach().cpu().clone()
    if isinstance(d, dict):
        return {k: _kwargs_to_cpu(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_kwargs_to_cpu(v) for v in d]
    if isinstance(d, tuple):
        return tuple(_kwargs_to_cpu(v) for v in d)
    return d


if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


class RolloutDenoisingMixin:
    @contextmanager
    def rollout_lifecycle(self, batch: Req):
        """Scope request-owned collection state; cleanup never performs collectives."""
        try:
            yield
        finally:
            if batch.rollout:
                batch._rollout_denoising_env_state = None
                batch._rollout_session_data = None

    def step_latents(
        self,
        batch: Req,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        step_index: int,
        apply: Callable[[], torch.Tensor | None],
    ) -> torch.Tensor:
        """Record ``x_t`` if this is a rollout, then run the caller's step.

        ``apply`` is the real update: ``lambda: scheduler.step(...)[0]`` or an
        in-place Euler that returns ``None``. ``log π`` still comes from
        ``SchedulerRLMixin`` inside ``scheduler.step``.
        """
        if batch.rollout:
            batch._rollout_loop_step_index = step_index
        self._maybe_append_dit_trajectory_step(
            batch=batch,
            latents=latents,
            timestep_value=timestep,
            step_index=step_index,
        )
        updated = apply()
        return latents if updated is None else updated

    def _maybe_prepare_rollout(self, batch: Req, *, latents_shape: tuple | None = None):
        """Prepare denoising loop for rollout."""
        scheduler = batch.scheduler
        if not isinstance(scheduler, SchedulerRLMixin):
            if batch.rollout:
                raise ValueError(
                    f"Scheduler {type(scheduler)} does not support rollout"
                )
            return

        scheduler.release_rollout_resources(batch)
        if batch.rollout:
            scheduler.prepare_rollout(
                batch=batch,
                pipeline_config=self.server_args.pipeline_config,
                latents_shape=latents_shape,
            )

    def _maybe_collect_rollout_log_probs(self, batch: Req):
        scheduler = batch.scheduler
        if not isinstance(scheduler, SchedulerRLMixin):
            if batch.rollout:
                raise ValueError(
                    f"Scheduler {type(scheduler)} does not support rollout"
                )
            return

        if batch.rollout:
            if batch.rollout_trajectory_data is None:
                batch.rollout_trajectory_data = RolloutTrajectoryData()
            batch.rollout_trajectory_data.rollout_log_probs = (
                scheduler.collect_rollout_log_probs(batch)
            )
            if batch.rollout_debug_mode:
                batch.rollout_trajectory_data.rollout_debug_tensors = (
                    scheduler.collect_rollout_debug_tensors(batch)
                )
            scheduler.release_rollout_resources(batch)

    def _postprocess_rollout_outputs(
        self,
        batch: Req,
        latents: torch.Tensor,
        num_inference_steps: int,
        final_timestep: torch.Tensor,
        server_args: ServerArgs,
    ) -> None:
        """Finalize rollout-only outputs.

        Must be called before ``_post_denoising_loop`` so that ``latents`` (the
        last ``scheduler.step`` output) is still SP-sharded and can be gathered
        uniformly with the per-step trajectory latents.
        """
        self._maybe_collect_rollout_log_probs(batch)
        # Append final denoised latent as the (T+1)-th entry (step_index=T),
        # routed through the same filter so rollout_return_step_indices can
        # include/exclude it.
        self._maybe_append_dit_trajectory_step(
            batch=batch,
            latents=latents,
            timestep_value=final_timestep,
            step_index=num_inference_steps,
        )
        self._maybe_finalize_denoising_env_collection(
            batch=batch,
            pipeline_config=server_args.pipeline_config,
        )

    def _maybe_init_denoising_env_collection(
        self,
        batch,
        pipeline_config,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any],
        guidance: torch.Tensor | None,
        sample_layout: Literal["batched", "single_packed"] = "batched",
    ) -> None:
        collect_env = batch.rollout_return_denoising_env
        collect_traj = batch.rollout_return_dit_trajectory
        if not (collect_env or collect_traj):
            batch._rollout_denoising_env_state = None
            return

        if collect_env:
            env = RolloutDenoisingEnv(
                image_kwargs=_kwargs_to_cpu(image_kwargs),
                pos_cond_kwargs=_kwargs_to_cpu(
                    pipeline_config.gather_denoising_env_static_for_sp(
                        batch, pos_cond_kwargs
                    )
                ),
                neg_cond_kwargs=(
                    _kwargs_to_cpu(
                        pipeline_config.gather_denoising_env_static_for_sp(
                            batch, neg_cond_kwargs
                        )
                    )
                    if neg_cond_kwargs
                    else None
                ),
                guidance=_kwargs_to_cpu(guidance),
                sample_layout=sample_layout,
            )
        else:
            env = None

        batch._rollout_denoising_env_state = RolloutCollectionState(
            sigmas=batch.scheduler.sigmas.detach().cpu().clone(), env=env
        )

    def _maybe_append_dit_trajectory_step(
        self,
        batch,
        latents: torch.Tensor,
        timestep_value: torch.Tensor,
        step_index: int,
    ) -> None:
        if not batch.rollout or not batch.rollout_return_dit_trajectory:
            return
        state = batch._rollout_denoising_env_state
        if state is None:
            return

        return_step_indices = batch.rollout_return_step_indices
        if return_step_indices is not None and step_index not in return_step_indices:
            return

        if latents.ndim < 2:
            raise ValueError("Rollout latents require an explicit [B, ...] shape")
        if state.step_latents and state.step_latents[0].shape != latents.shape:
            raise ValueError("Rollout latent shape changed within a denoising loop")
        state.step_latents.append(latents.detach().clone())
        state.step_timesteps.append(timestep_value.detach().cpu().clone())

    def _maybe_finalize_denoising_env_collection(self, batch, pipeline_config) -> None:
        state = batch._rollout_denoising_env_state
        if state is None:
            return

        env = state.env
        step_latents = state.step_latents
        step_timesteps = state.step_timesteps

        if batch.rollout_trajectory_data is None:
            batch.rollout_trajectory_data = RolloutTrajectoryData()

        if step_latents and batch.rollout_return_dit_trajectory:
            step_latents_tensor = torch.stack(step_latents, dim=1)
            step_latents_tensor = gather_stacked_latents_for_sp(
                pipeline_config=pipeline_config,
                batch=batch,
                stacked_latents=step_latents_tensor,
            )
            batch.rollout_trajectory_data.dit_trajectory = RolloutDitTrajectory(
                latents=step_latents_tensor.cpu(),
                timesteps=torch.stack(step_timesteps, dim=0).cpu(),
                sigmas=state.sigmas,
            )

        if env is not None and batch.rollout_return_denoising_env:
            batch.rollout_trajectory_data.denoising_env = env

        batch._rollout_denoising_env_state = None
