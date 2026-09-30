"""Compatibility boundary for legacy rollout schedulers and conditioning.

Trajectory ownership lives in RolloutRecorder. This mixin only bridges existing
request/scheduler APIs while their transition interface is migrated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch

from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDenoisingEnv,
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


def _kwargs_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _kwargs_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_kwargs_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_kwargs_to_cpu(v) for v in value)
    return value


class RolloutDenoisingMixin:
    def _maybe_prepare_rollout(self, batch: Req):
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

    def _snapshot_rollout_environment(
        self,
        batch: Req,
        *,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        neg_cond_kwargs: dict | None,
        guidance: torch.Tensor | None,
        sample_layout: Literal["batched", "single_packed"] = "batched",
    ) -> RolloutDenoisingEnv | None:
        if not batch.rollout or not batch.rollout_return_denoising_env:
            return None
        config = self.server_args.pipeline_config
        return RolloutDenoisingEnv(
            image_kwargs=_kwargs_to_cpu(image_kwargs),
            pos_cond_kwargs=_kwargs_to_cpu(
                config.gather_denoising_env_static_for_sp(batch, pos_cond_kwargs)
            ),
            neg_cond_kwargs=(
                _kwargs_to_cpu(
                    config.gather_denoising_env_static_for_sp(batch, neg_cond_kwargs)
                )
                if neg_cond_kwargs
                else None
            ),
            guidance=_kwargs_to_cpu(guidance),
            sample_layout=sample_layout,
        )
