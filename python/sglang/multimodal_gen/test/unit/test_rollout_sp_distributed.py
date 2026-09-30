# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=2 -m pytest -q <this file>."""

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.runtime.post_training.flow_transition import flow_step
from sglang.multimodal_gen.runtime.post_training.rollout_recorder import RolloutRecorder

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="requires two CUDA ranks launched by torchrun",
)


def test_padded_joint_capture_and_scores_match_unsharded_reference():
    """Gather different stream lengths without exposing padding or double-sharding RNG."""
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        backend="nccl",
        device_id=device,
        timeout=45,
    )
    initialize_model_parallel(
        sequence_parallel_degree=2, ulysses_degree=2, backend="nccl"
    )
    try:
        cfg = LTX2PipelineConfig.__new__(LTX2PipelineConfig)
        cfg.patch_size_t = cfg.patch_size = 1
        cfg.vae_config = SimpleNamespace(
            arch_config=SimpleNamespace(spatial_compression_ratio=1)
        )
        video = (
            torch.arange(2 * 5 * 3, device=device, dtype=torch.float32).reshape(2, 5, 3)
            / 30
        )
        audio = (
            torch.arange(2 * 7 * 2, device=device, dtype=torch.float32).reshape(2, 7, 2)
            / 30
        )
        reference = video.clone()
        batch = SimpleNamespace(
            latents=video,
            raw_latent_shape=video.shape,
            height=1,
            width=1,
            rollout_sde_type="sde",
            rollout_noise_level=0.4,
            rollout_log_prob_no_const=True,
            rollout_debug_mode=False,
            generator=[torch.Generator(device).manual_seed(70 + i) for i in range(2)],
        )
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(3, device=device)
        scheduler.prepare_rollout(batch, cfg)
        video_local, batch.did_sp_shard_latents = cfg.shard_latents_for_sp(batch, video)
        audio_local, batch.did_sp_shard_audio_latents = cfg.shard_audio_latents_for_sp(
            batch, audio
        )
        ctx = SimpleNamespace(
            latents=video_local,
            audio_latents=audio_local,
            scheduler=scheduler,
            audio_scheduler=scheduler,
            timesteps=scheduler.timesteps,
        )
        stage = LTX2DenoisingStage.__new__(LTX2DenoisingStage)
        recorder = RolloutRecorder(
            stage._rollout_stream_specs(
                ctx, batch, SimpleNamespace(pipeline_config=cfg)
            ),
            retain_steps=[0, 2, 3],
        )
        reference_generators = [
            torch.Generator(device).manual_seed(70 + i) for i in range(2)
        ]
        video_states, audio_states, scores = [], [], []
        for i in range(3):
            video_states.append(reference.clone())
            audio_states.append(audio.clone())
            recorder.capture_before(i, stage._rollout_state(ctx))
            batch._rollout_loop_step_index = i
            s0, s1 = scheduler.sigmas[i : i + 2]
            ctx.latents = scheduler.flow_sde_sampling(
                batch,
                torch.ones_like(ctx.latents),
                ctx.latents,
                s0,
                s1,
                batch.generator,
            )
            ctx.audio_latents.add_(s1 - s0)
            audio.add_(s1 - s0)
            noise = torch.cat(
                [
                    torch.randn((1, 5, 3), device=device, generator=g)
                    for g in reference_generators
                ]
            )
            step = flow_step(
                sample=reference,
                model_output=torch.ones_like(reference),
                current_sigma=s0,
                next_sigma=s1,
                sigma_max=float(scheduler.sigmas[1]),
                method="sde",
                noise_level=0.4,
                legacy_score=True,
                variance_noise=noise,
            )
            reference = step.sample
            scores.append(step.score_sum / step.element_count)
        video_states.append(reference)
        audio_states.append(audio)
        result = recorder.finish(stage._rollout_state(ctx))
        for name, states in [("video", video_states), ("audio", audio_states)]:
            expected = torch.stack([states[i] for i in [0, 2, 3]], dim=1).cpu()
            torch.testing.assert_close(result[name].latents, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            scheduler.collect_rollout_log_probs(batch),
            torch.stack(scores, dim=1).cpu(),
            rtol=0,
            atol=0,
        )
        print(
            f"rank={rank}: padded video/audio gather, B=2 global RNG and scores match unsharded reference exactly"
        )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
