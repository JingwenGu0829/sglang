# SPDX-License-Identifier: Apache-2.0
"""Exercise the real generic preparation order without loading a transformer."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import denoising


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rollout_noise_uses_full_shape_after_generic_sp_preparation(batch_size):
    """Moving session setup below sharding halves the noise shape a second time."""
    config = PipelineConfig.__new__(PipelineConfig)
    config.expand_conditioning_to_sample_batch = lambda batch: batch
    config.get_pos_prompt_embeds = lambda batch: torch.zeros(1, 2, 4)
    config.cfg_policy = SimpleNamespace(build=lambda *args: None)
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(3, device="cpu")
    batch = SimpleNamespace(
        scheduler=scheduler,
        timesteps=scheduler.timesteps,
        num_inference_steps=3,
        rollout=True,
        rollout_return_denoising_env=False,
        rollout_return_dit_trajectory=False,
        latents=torch.zeros(batch_size, 4, 8, 2, 2),
        image_latent=None,
        generator=[torch.Generator().manual_seed(7 + i) for i in range(batch_size)],
        eta=0.0,
        image_embeds=[],
        do_classifier_free_guidance=False,
        enable_sequence_shard=False,
        clip_embedding_pos=None,
        prompt_attention_mask=None,
        prompt_embeds_mask=None,
        is_warmup=False,
    )
    stage = denoising.DenoisingStage.__new__(denoising.DenoisingStage)
    stage.transformer = lambda **kwargs: None
    stage.pipeline = None
    stage.server_args = SimpleNamespace(pipeline_config=config, disable_autocast=True)
    stage._dual_transformer_execution_mode = lambda: None
    stage._maybe_enable_cache_dit_and_torch_compile = lambda *args: None
    stage.attn_backend = SimpleNamespace(get_enum=lambda: None)
    stage._sp_world_size = lambda: 2
    stage.get_or_build_guidance = lambda *args: None
    stage._get_transformer_attr = lambda *args: None
    stage._extra_func_kwarg_names_cache = {}
    with (
        patch.object(denoising, "load_transformer_if_needed", return_value=False),
        patch.object(denoising, "resolve_precision", return_value=torch.float32),
        patch.object(denoising, "should_apply_wan_ti2v", return_value=False),
        patch(
            "sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin.get_sp_world_size",
            return_value=2,
        ),
        patch(
            "sglang.multimodal_gen.configs.pipeline_configs.base.get_sp_world_size",
            return_value=2,
        ),
        patch(
            "sglang.multimodal_gen.configs.pipeline_configs.base.get_sp_parallel_rank",
            return_value=0,
        ),
    ):
        ctx = stage._prepare_denoising_loop(batch, stage.server_args)
        prepared = batch._rollout_session_data
        denoising.get_denoise_loop_observer(batch).init_env(
            stage, batch, config, {}, {}, None, None
        )
        assert batch._rollout_session_data is prepared
        noise = scheduler._rollout_variance_noise(
            batch, torch.zeros_like(ctx.latents), batch.generator
        )
    assert ctx.latents.shape == (batch_size, 4, 4, 2, 2)
    assert noise.shape == ctx.latents.shape
    assert batch._rollout_session_data.noise_buffer.shape == (batch_size, 4, 8, 2, 2)
    expected = torch.cat(
        [
            torch.randn((1, 4, 8, 2, 2), generator=torch.Generator().manual_seed(7 + i))
            for i in range(batch_size)
        ]
    )
    torch.testing.assert_close(
        batch._rollout_session_data.noise_buffer, expected, rtol=0, atol=0
    )
    torch.testing.assert_close(noise, expected[:, :, :4], rtol=0, atol=0)
