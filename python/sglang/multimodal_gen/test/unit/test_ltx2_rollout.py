# SPDX-License-Identifier: Apache-2.0
"""Exercise video capture around LTX's real Euler video/audio step."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.runtime.post_training.rollout_recorder import RolloutRecorder


@pytest.mark.parametrize("method", ["ode", "sde", "cps"])
def test_ltx_video_capture_preserves_sampling_and_native_audio(method, monkeypatch):
    for module in ("scheduler_rl_mixin", "sp_utils"):
        monkeypatch.setattr(
            f"sglang.multimodal_gen.runtime.post_training.{module}.get_sp_world_size",
            lambda: 1,
        )

    def run(capture):
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(3, device="cpu")
        audio_scheduler = FlowMatchEulerDiscreteScheduler()
        audio_scheduler.set_timesteps(3, device="cpu")
        config = SimpleNamespace(
            shard_latents_for_sp=lambda batch, latents: (latents, False)
        )
        batch = SimpleNamespace(
            rollout=True,
            rollout_sde_type=method,
            rollout_noise_level=0.4,
            rollout_log_prob_no_const=True,
            rollout_debug_mode=False,
            generator=torch.Generator().manual_seed(7),
            latents=torch.zeros(1, 4, 3),
            raw_latent_shape=(1, 4, 3),
            prompt_embeds=[torch.ones(1, 2, 4)],
            audio_prompt_embeds=[torch.ones(1, 2, 4)],
            do_classifier_free_guidance=False,
            did_sp_shard_audio_latents=False,
            eta=0.0,
            extra={},
        )
        ctx = SimpleNamespace(
            scheduler=scheduler,
            audio_scheduler=audio_scheduler,
            timesteps=scheduler.timesteps,
            latents=batch.latents,
            audio_latents=torch.ones(1, 2, 5),
            stage="one_stage",
            is_ltx23_variant=False,
            denoise_mask=None,
            clean_latent=None,
            reserved_frames_mask=None,
            z=None,
        )
        stage = LTX2DenoisingStage.__new__(LTX2DenoisingStage)
        stage.sampler_name = "euler"
        stage._extra_func_kwarg_names_cache = {}
        stage._prepare_ltx2_model_inputs = lambda *args: SimpleNamespace(
            latent_model_input=ctx.latents
        )
        stage._build_ltx2_base_model_kwargs = lambda *args: {}
        stage._build_ltx2_model_kwargs = lambda *args, **kwargs: {}
        stage._get_ltx_prompt_attention_mask = lambda *args, **kwargs: None
        stage._ltx2_model_forward_context = lambda *args: nullcontext()
        # Coupled prediction: video noise changes subsequent audio inputs.
        stage._ltx2_call_current_model = lambda *args: (
            torch.ones_like(ctx.latents) * ctx.audio_latents.mean(),
            torch.ones_like(ctx.audio_latents) * (1 + ctx.latents.mean()),
        )
        stage.post_forward_for_ti2v_task = lambda batch, args, mask, latents, z: latents
        args = SimpleNamespace(
            enable_cfg_parallel=False,
            pipeline_class_name="LTX2Pipeline",
            pipeline_config=config,
        )
        scheduler.prepare_rollout(batch, config)
        recorder = (
            RolloutRecorder(stage._rollout_stream_specs(ctx, batch, args))
            if capture
            else None
        )
        expected_video = []
        for i, t in enumerate(ctx.timesteps):
            state = stage._rollout_state(ctx)
            expected_video.append(ctx.latents.clone())
            if recorder is not None:
                recorder.capture_before(i, state)
            # Audio follows native Euler using the joint state before either update.
            dt = audio_scheduler.sigmas[i + 1] - audio_scheduler.sigmas[i]
            audio_next = ctx.audio_latents + dt * (1 + ctx.latents.mean())
            batch._rollout_loop_step_index = i
            stage._run_denoising_step(
                ctx, SimpleNamespace(step_index=i, t_device=t), batch, args
            )
            torch.testing.assert_close(ctx.audio_latents, audio_next, rtol=0, atol=0)
        streams = recorder.finish(stage._rollout_state(ctx)) if recorder else {}
        if capture:
            expected_video.append(ctx.latents.clone())
            torch.testing.assert_close(
                streams["video"].latents,
                torch.stack(expected_video, dim=1),
                rtol=0,
                atol=0,
            )
        return (
            ctx.latents,
            ctx.audio_latents,
            scheduler.collect_rollout_log_probs(batch),
            streams,
        )

    captured = run(True)
    uncaptured = run(False)
    for actual, expected in zip(captured[:3], uncaptured[:3]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert captured[3]["video"].latents.shape == (1, 4, 4, 3)
    assert set(captured[3]) == {"video"}
