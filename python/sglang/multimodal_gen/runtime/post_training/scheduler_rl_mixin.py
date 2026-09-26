# SPDX-License-Identifier: Apache-2.0
"""Flow-matching rollout step utilities for log-prob computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.post_training.flow_transition import flow_step
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutSessionData,
)
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_debug_mixin import (
    SchedulerRLDebugMixin,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


class SchedulerRLMixin(SchedulerRLDebugMixin):
    @staticmethod
    def _get_rollout_session_data(batch) -> RolloutSessionData:
        """Return the RolloutSessionData attached to *batch*, or raise if not prepared."""
        rollout_session_data = getattr(batch, "_rollout_session_data", None)
        if rollout_session_data is None:
            raise RuntimeError("prepare_rollout() not called before rollout")
        return rollout_session_data

    def release_rollout_resources(self, batch) -> None:
        """Release rollout-owned resources. Call when denoising ends or before a new rollout."""
        batch._rollout_session_data = None

    def prepare_rollout(self, batch: Req, pipeline_config: Any = None) -> None:
        """Enable rollout and set SDE/CPS params. Call once before the denoising loop."""
        if get_sp_world_size() > 1 and pipeline_config is None:
            raise RuntimeError(
                "SP rollout requires pipeline_config to be passed to prepare_rollout()."
            )
        batch._rollout_session_data = RolloutSessionData(
            pipeline_config=pipeline_config,
            sigma_max=self.sigmas[min(1, len(self.sigmas) - 1)].item(),
            latents_shape=(
                tuple(batch.latents.shape) if batch.latents is not None else None
            ),
        )

    def already_prepared_rollout(self, batch) -> bool:
        return getattr(batch, "_rollout_session_data", None) is not None

    def _get_or_create_rollout_noise_buffer(
        self,
        rollout_session_data: RolloutSessionData,
        full_shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get or create the reusable noise buffer (local or full shape) for rollout."""
        buffer = rollout_session_data.noise_buffer
        if (
            buffer is None
            or buffer.shape != full_shape
            or buffer.dtype != dtype
            or buffer.device != device
        ):
            buffer = torch.empty(full_shape, device=device, dtype=dtype)
            rollout_session_data.noise_buffer = buffer
        return buffer

    def _rollout_variance_noise(
        self,
        batch,
        model_output: torch.FloatTensor,
        generator: Union[torch.Generator, list[torch.Generator]],
    ) -> torch.FloatTensor:
        """Generate variance noise for rollout. If generator is a list, use generator[i] for the i-th batch item."""
        assert generator is not None, "Generator must be provided"

        rollout_session_data = self._get_rollout_session_data(batch)
        device = model_output.device
        dtype = model_output.dtype
        local_shape = tuple(model_output.shape)

        B = local_shape[0]
        if isinstance(generator, torch.Generator):
            assert B == 1, "Generator must be a list if batch size is not 1"
            generator = [generator]
        else:
            assert len(generator) == B, (
                "Generator list must have the same length as batch size"
            )

        buffer = self._get_or_create_rollout_noise_buffer(
            rollout_session_data, rollout_session_data.latents_shape, device, dtype
        )
        for i in range(B):
            torch.randn(
                (1, *rollout_session_data.latents_shape[1:]),
                out=buffer[i : i + 1],
                generator=generator[i],
            )

        sharded_noise, _ = rollout_session_data.pipeline_config.shard_latents_for_sp(
            batch=batch, latents=buffer
        )
        if tuple(sharded_noise.shape) != local_shape:
            raise ValueError(
                "Rollout SP noise shape mismatch after shard. "
                f"Expected local_shape={local_shape}, got {tuple(sharded_noise.shape)}."
            )
        return sharded_noise

    def flow_sde_sampling(
        self,
        batch,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        current_sigma: torch.FloatTensor,
        next_sigma: torch.FloatTensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Flow rollout step for log-prob / sampling (see FlowGRPO-style references).

        ``rollout_sde_type`` (from batch SamplingParams):

        1. ``"sde"``: Standard stochastic differential equation transition (Gaussian).
        2. ``"cps"``: Coupled Particle Sampling.
        3. ``"ode"``: Deterministic ODE step (no diffusion noise).
        """
        rollout_session_data = self._get_rollout_session_data(batch)
        sde_type = batch.rollout_sde_type
        noise_level = float(batch.rollout_noise_level)
        log_prob_no_const = batch.rollout_log_prob_no_const
        debug_mode = bool(getattr(batch, "rollout_debug_mode", False))

        if not log_prob_no_const and sde_type != "ode":
            assert noise_level > 0, (
                "True log-probability computation requires a non-zero noise level."
            )

        # step_index comes from the denoising-loop counter stashed by
        # DenoisingStage — scheduler._step_index would differ when
        # _begin_index != 0 (e.g. partial denoising).
        sde_step_indices = getattr(batch, "rollout_sde_step_indices", None)
        loop_step_index = getattr(batch, "_rollout_loop_step_index", None)
        if (
            sde_type != "ode"
            and sde_step_indices is not None
            and loop_step_index is not None
            and loop_step_index not in sde_step_indices
        ):
            effective_sde_type = "ode"
        else:
            effective_sde_type = sde_type

        if effective_sde_type == "ode":
            if sde_type == "ode" and not log_prob_no_const:
                raise ValueError("ODE transitions do not have a Gaussian density")
            variance_noise = None
            full_variance_noise = None
        else:
            variance_noise = self._rollout_variance_noise(
                batch, model_output.float(), generator
            )
            full_variance_noise = rollout_session_data.noise_buffer
        result = flow_step(
            sample=sample,
            model_output=model_output,
            current_sigma=current_sigma,
            next_sigma=next_sigma,
            sigma_max=rollout_session_data.sigma_max,
            method=effective_sde_type,
            noise_level=noise_level,
            legacy_score=log_prob_no_const,
            variance_noise=variance_noise,
            score_noise=full_variance_noise,
            score_shape=rollout_session_data.latents_shape,
        )
        if debug_mode:
            self.append_local_rollout_debug_tensors(
                batch,
                variance_noise=(
                    variance_noise
                    if variance_noise is not None
                    else torch.zeros_like(model_output)
                ),
                prev_sample_mean=result.mean,
                noise_std_dev=result.noise_std,
                model_output=result.model_output,
            )
        self.append_local_rollout_log_probs(
            batch, result.score_sum, result.element_count
        )
        return result.sample

    def append_local_rollout_log_probs(
        self, batch, log_prob_sum: torch.Tensor, log_prob_count: torch.Tensor
    ) -> None:
        rollout_session_data = self._get_rollout_session_data(batch)
        rollout_session_data.local_log_prob_sum.append(log_prob_sum)
        rollout_session_data.local_log_prob_count.append(log_prob_count)

    def consume_local_rollout_log_probs(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rollout_session_data = self._get_rollout_session_data(batch)
        # [B, T]: batch dim 0, denoising step dim 1
        values_sum = torch.stack(rollout_session_data.local_log_prob_sum, dim=1)
        values_count = torch.stack(rollout_session_data.local_log_prob_count, dim=1)
        rollout_session_data.local_log_prob_sum = []
        rollout_session_data.local_log_prob_count = []
        return values_sum, values_count

    def collect_rollout_log_probs(self, batch: Req) -> torch.Tensor | None:
        """Per-step sums are already computed on the full pre-shard noise
        buffer inside flow_sde_sampling, so every SP rank holds identical
        values here and no all-reduce is needed."""

        trajectory_log_prob_sum, trajectory_log_prob_count = (
            self.consume_local_rollout_log_probs(batch)
        )
        rollout_log_probs_tensor = trajectory_log_prob_sum / trajectory_log_prob_count
        return rollout_log_probs_tensor.cpu()
