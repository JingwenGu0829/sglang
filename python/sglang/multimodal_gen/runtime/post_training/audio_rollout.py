# SPDX-License-Identifier: Apache-2.0
"""Audio stream wiring for a loop with an independent Euler scheduler."""

import functools
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
from sglang.multimodal_gen.runtime.post_training.sp_utils import (
    gather_stacked_audio_latents_for_sp,
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


def _shard_audio_noise(pipeline_config, batch, latents):
    if not batch.did_sp_shard_audio_latents:
        return latents
    return pipeline_config.shard_audio_latents_for_sp(batch, latents)[0]


def prepare_audio_rollout_if_enabled(*, batch, pipeline_config, scheduler, latents):
    """Register before audio SP sharding to retain the unpadded full shape."""
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
            shard_for_sp=functools.partial(_shard_audio_noise, pipeline_config),
            gather_for_sp=functools.partial(
                gather_stacked_audio_latents_for_sp, pipeline_config
            ),
        )

    prepare_rollout_if_enabled(
        batch=batch,
        pipeline_config=pipeline_config,
        stream=AUDIO_STREAM,
        build_controller=build_controller,
    )
