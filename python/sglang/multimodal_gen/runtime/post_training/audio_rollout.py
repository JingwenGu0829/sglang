# SPDX-License-Identifier: Apache-2.0
"""Audio stream wiring for a loop with an independent Euler scheduler."""

import hashlib

import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.post_training.rollout_hooks import (
    AUDIO_STREAM,
    build_flow_rollout_controller,
    prepare_rollout_if_enabled,
)


def audio_rollout_generators(generators):
    """Derive per-sample audio RNGs without advancing the visual RNGs.

    The domain and seed are identical across SP/CFG ranks. Python's salted
    hash() and rank IDs must not enter the seed derivation.
    """
    if generators is None:
        raise ValueError("Audio rollout requires request generators")

    def derive(generator):
        digest = hashlib.sha256(
            f"sglang/rollout/audio/{generator.initial_seed()}".encode()
        ).digest()
        seed = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
        return torch.Generator(device=generator.device).manual_seed(seed)

    if isinstance(generators, torch.Generator):
        return derive(generators)
    return [derive(generator) for generator in generators]


def validate_audio_rollout_request(batch, server_args) -> None:
    """Validate active audio rollouts using the pipeline selected for serving."""
    if not (batch.rollout and batch.rollout_audio):
        return
    if batch.rollout_return_denoising_env:
        raise ValueError(
            "Audio rollout does not support rollout_return_denoising_env: "
            "the replay environment has no audio conditioning"
        )
    from sglang.multimodal_gen.registry import get_served_pipeline_class

    pipeline_cls = get_served_pipeline_class(server_args)
    if pipeline_cls is None:
        raise ValueError("Could not resolve the pipeline for audio rollout")
    pipeline_cls.validate_audio_rollout_request(batch)


def _replicated_audio(batch, tensor):
    # The audio controller draws full-shape noise and records whole latents per rank.
    if batch.did_sp_shard_audio_latents:
        raise RuntimeError(
            "Audio rollout requires audio latents replicated across SP ranks"
        )
    return tensor


def prepare_audio_rollout_if_enabled(*, batch, pipeline_config, scheduler, latents):
    """Register the replicated audio stream of single-stage LTX generation."""
    if not (batch.rollout and batch.rollout_audio):
        return
    if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        raise ValueError("Audio rollout requires a FlowMatch Euler scheduler")
    if not isinstance(latents, torch.Tensor) or latents.ndim != 3:
        raise ValueError("Audio rollout requires packed audio latents [B, S, D]")
    generators = audio_rollout_generators(batch.generator)

    def build_controller(*, batch, pipeline_config):
        return build_flow_rollout_controller(
            batch=batch,
            sigmas=scheduler.sigmas,
            latents=latents,
            generator=generators,
            shard_for_sp=_replicated_audio,
            gather_for_sp=_replicated_audio,
        )

    prepare_rollout_if_enabled(
        batch=batch,
        pipeline_config=pipeline_config,
        stream=AUDIO_STREAM,
        build_controller=build_controller,
    )
