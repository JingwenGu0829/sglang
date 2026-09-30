# SPDX-License-Identifier: Apache-2.0
"""Loop-level rollout hooks: snapshot ``c`` at start, pack outputs at end.

Per-step ``x_t`` is recorded by ``RolloutDenoisingMixin.step_latents``.
``log π`` is recorded inside ``SchedulerRLMixin.flow_sde_sampling``
(generic loops reach it via ``scheduler.step``; H3 calls it after
mapping ``−v``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch

from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    from sglang.multimodal_gen.runtime.post_training.rollout_denoising_mixin import (
        RolloutDenoisingMixin,
    )


class DenoiseLoopObserver:
    """No-op observer. Custom loops call ``init_env`` / ``finalize`` only."""

    def init_env(
        self,
        stage: RolloutDenoisingMixin,
        batch: Req,
        pipeline_config: PipelineConfig,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any] | None,
        guidance: torch.Tensor | None,
        sample_layout: Literal["batched", "single_packed"] = "batched",
    ) -> None:
        return None

    def finalize(
        self,
        stage: RolloutDenoisingMixin,
        batch: Req,
        latents: torch.Tensor,
        num_inference_steps: int,
        final_timestep: torch.Tensor,
        server_args: ServerArgs,
    ) -> None:
        return None


class NullDenoiseLoopObserver(DenoiseLoopObserver):
    pass


class RolloutDenoiseLoopObserver(DenoiseLoopObserver):
    """Snapshot DiT cond at loop start; pack trajectory / env / log-probs at end."""

    def init_env(
        self,
        stage: RolloutDenoisingMixin,
        batch: Req,
        pipeline_config: PipelineConfig,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any] | None,
        guidance: torch.Tensor | None,
        sample_layout: Literal["batched", "single_packed"] = "batched",
    ) -> None:
        scheduler = batch.scheduler
        if not isinstance(
            scheduler, SchedulerRLMixin
        ) or not scheduler.already_prepared_rollout(batch):
            stage._maybe_prepare_rollout(batch)
        stage._maybe_init_denoising_env_collection(
            batch=batch,
            pipeline_config=pipeline_config,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            neg_cond_kwargs=neg_cond_kwargs,
            guidance=guidance,
            sample_layout=sample_layout,
        )

    def finalize(
        self,
        stage: RolloutDenoisingMixin,
        batch: Req,
        latents: torch.Tensor,
        num_inference_steps: int,
        final_timestep: torch.Tensor,
        server_args: ServerArgs,
    ) -> None:
        stage._postprocess_rollout_outputs(
            batch=batch,
            latents=latents,
            num_inference_steps=num_inference_steps,
            final_timestep=final_timestep,
            server_args=server_args,
        )


_NULL_OBSERVER = NullDenoiseLoopObserver()
_ROLLOUT_OBSERVER = RolloutDenoiseLoopObserver()


def get_denoise_loop_observer(batch: Req) -> DenoiseLoopObserver:
    # Observers carry no state; selection must reflect this execution's mode.
    return _ROLLOUT_OBSERVER if batch.rollout else _NULL_OBSERVER
