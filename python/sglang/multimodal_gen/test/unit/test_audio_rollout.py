# SPDX-License-Identifier: Apache-2.0
"""Audio rollout integration through real Euler steps and the HTTP serializer."""

import copy
import gc
import types
import unittest
import weakref
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    RolloutRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import (
    _build_response,
    _build_sampling_kwargs,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_deserialize,
    msgpack_encode_spliced,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.post_training.audio_rollout import (
    audio_rollout_generators,
    prepare_audio_rollout_if_enabled,
)
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    concat_rollout_trajectory_data,
    select_output_rollout_trajectory,
)
from sglang.multimodal_gen.runtime.post_training.rollout_hooks import (
    AUDIO_STREAM,
    RolloutFinalState,
    finalize_rollout_if_enabled,
    init_rollout_env_if_enabled,
    prepare_rollout_if_enabled,
    rollout_step_kwargs,
)
from sglang.multimodal_gen.runtime.post_training.sp_utils import (
    gather_stacked_audio_latents_for_sp,
)


def _batch(*, audio=True):
    request = RolloutRequest(
        prompt="test",
        rollout=True,
        rollout_audio=audio,
        rollout_noise_level=0.5,
        rollout_debug_mode=True,
        rollout_return_dit_trajectory=True,
        rollout_return_step_indices=[0, 2, 3],
    )
    batch = Req(sampling_params=SamplingParams(**_build_sampling_kwargs(request)))
    batch.scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    batch.scheduler.set_timesteps(sigmas=[0.8, 0.5, 0.2])
    batch.latents = torch.zeros(2, 4, 2)
    batch.audio_latents = torch.ones(2, 5, 3)
    batch.generator = [torch.Generator().manual_seed(10 + i) for i in range(2)]
    return batch


_CONFIG = types.SimpleNamespace(
    shard_latents_for_sp=lambda batch, latents: (latents, False),
    gather_denoising_env_static_for_sp=lambda batch, kwargs: kwargs,
)


def _run(batch):
    audio_scheduler = copy.deepcopy(batch.scheduler)
    prepare_rollout_if_enabled(batch=batch, pipeline_config=_CONFIG)
    prepare_audio_rollout_if_enabled(
        batch=batch,
        pipeline_config=_CONFIG,
        scheduler=audio_scheduler,
        latents=batch.audio_latents,
    )
    init_rollout_env_if_enabled(
        batch=batch,
        image_kwargs={},
        pos_cond_kwargs={},
        neg_cond_kwargs={},
        guidance=None,
    )
    for timestep in batch.scheduler.timesteps:
        # Both predictions use the joint state before either scheduler advances.
        video_prediction = batch.latents * 0.1 + batch.audio_latents.mean(
            (1, 2), keepdim=True
        )
        audio_prediction = batch.audio_latents * 0.1 + batch.latents.mean(
            (1, 2), keepdim=True
        )
        batch.latents = batch.scheduler.step(
            video_prediction,
            timestep,
            batch.latents,
            **rollout_step_kwargs(batch),
            return_dict=False,
        )[0]
        batch.audio_latents = audio_scheduler.step(
            audio_prediction,
            timestep,
            batch.audio_latents,
            **rollout_step_kwargs(batch, AUDIO_STREAM),
            return_dict=False,
        )[0]
    finalize_rollout_if_enabled(
        batch=batch,
        latents=batch.latents,
        num_inference_steps=3,
        final_timestep=torch.tensor(0.0),
        pipeline_config=_CONFIG,
        additional_streams={
            AUDIO_STREAM: RolloutFinalState(batch.audio_latents, 3, torch.tensor(0.0))
        }
        if batch.rollout_audio
        else None,
    )
    return batch.rollout_trajectory_data


class TestAudioRollout(unittest.TestCase):
    def setUp(self):
        sp_size = patch(
            "sglang.multimodal_gen.runtime.post_training.sp_utils.get_sp_world_size",
            return_value=1,
        )
        sp_size.start()
        self.addCleanup(sp_size.stop)

    def test_audio_rng_is_repeatable_without_advancing_visual_rng(self):
        generators = [torch.Generator().manual_seed(i) for i in (10, 11)]
        states = [g.get_state().clone() for g in generators]
        first = audio_rollout_generators(generators)
        second = audio_rollout_generators(generators)
        for g, state, a, b in zip(generators, states, first, second):
            self.assertTrue(torch.equal(g.get_state(), state))
            self.assertNotEqual(g.initial_seed(), a.initial_seed())
            self.assertTrue(
                torch.equal(torch.randn(20, generator=a), torch.randn(20, generator=b))
            )
        self.assertNotEqual(first[0].initial_seed(), first[1].initial_seed())

        disabled, enabled = _batch(audio=False), _batch(audio=True)
        self.assertIsNone(_run(disabled).audio)
        _run(enabled)
        for a, b in zip(disabled.generator, enabled.generator):
            self.assertTrue(torch.equal(a.get_state(), b.get_state()))

    def test_audio_controller_does_not_retain_request(self):
        batch = _batch()
        prepare_audio_rollout_if_enabled(
            batch=batch,
            pipeline_config=_CONFIG,
            scheduler=copy.deepcopy(batch.scheduler),
            latents=batch.audio_latents,
        )
        controller = batch.rollout_controllers[AUDIO_STREAM]
        reference = weakref.ref(batch)
        gc.disable()
        try:
            del batch
            self.assertIsNone(reference())
            self.assertEqual(controller.latents_shape, (2, 5, 3))
        finally:
            gc.enable()

    def test_batched_merge_select_and_msgpack_preserve_audio_alignment(self):
        import msgpack

        batch = _batch()
        batch.rollout_sde_step_indices = [1]
        data = _run(batch)
        for stream, final_latents in (
            (data, batch.latents),
            (data.audio, batch.audio_latents),
        ):
            self.assertTrue(torch.isfinite(stream.rollout_log_probs).all())
            self.assertTrue(
                torch.equal(stream.rollout_log_probs[:, [0, 2]], torch.zeros(2, 2))
            )
            torch.testing.assert_close(
                stream.dit_trajectory.latents[:, -1], final_latents
            )
        self.assertEqual(batch.rollout_controllers, {})
        # Keep a batch dimension when selecting engine output rows, then
        # remove it exactly once while constructing each HTTP response.
        rows = [select_output_rollout_trajectory(data, i) for i in range(2)]
        merged = concat_rollout_trajectory_data(rows)
        torch.testing.assert_close(
            merged.audio.rollout_log_probs, data.audio.rollout_log_probs
        )
        self.assertEqual(
            merged.audio.dit_trajectory.latent_step_indices.tolist(), [0, 2, 3]
        )
        self.assertEqual(merged.dit_trajectory.latent_step_indices.tolist(), [0, 2, 3])
        result = OutputBatch(
            output=torch.zeros(2, 1),
            audio=torch.arange(24).view(2, 1, 12).float(),
            audio_sample_rate=24000,
            rollout_trajectory_data=merged,
        )
        responses = _build_response("r", "p", 0, True, result)
        payload = msgpack.unpackb(
            b"".join(msgpack_encode_spliced([r.model_dump() for r in responses])),
            raw=False,
        )
        for i, response in enumerate(payload):
            audio = response["audio_rollout"]
            torch.testing.assert_close(
                _maybe_deserialize(audio["rollout_log_probs"]),
                data.audio.rollout_log_probs[i],
            )
            torch.testing.assert_close(
                _maybe_deserialize(audio["generated_output"]), result.audio[i]
            )
            self.assertEqual(audio["sample_rate"], 24000)
            self.assertEqual(
                _maybe_deserialize(
                    audio["dit_trajectory"]["latent_step_indices"]
                ).tolist(),
                [0, 2, 3],
            )
            torch.testing.assert_close(
                _maybe_deserialize(audio["dit_trajectory"]["latents"]),
                data.audio.dit_trajectory.latents[i],
            )
            torch.testing.assert_close(
                _maybe_deserialize(audio["dit_trajectory"]["timesteps"]),
                data.audio.dit_trajectory.timesteps,
            )
            torch.testing.assert_close(
                _maybe_deserialize(audio["dit_trajectory"]["sigmas"]),
                data.audio.dit_trajectory.sigmas,
            )
            torch.testing.assert_close(
                _maybe_deserialize(
                    audio["rollout_debug_tensors"]["rollout_model_outputs"]
                ),
                data.audio.rollout_debug_tensors.rollout_model_outputs[i],
            )

    def test_audio_sp_gather_uses_audio_flag_and_restores_batch_and_steps(self):
        batch = types.SimpleNamespace(
            did_sp_shard_latents=False, did_sp_shard_audio_latents=True
        )
        stacked = torch.arange(2 * 3 * 2).view(2, 3, 2, 1)
        config = types.SimpleNamespace(
            gather_audio_latents_for_sp=lambda x, batch: torch.cat([x, x + 100], dim=1)[
                :, :3
            ]
        )
        with patch(
            "sglang.multimodal_gen.runtime.post_training.sp_utils.get_sp_world_size",
            return_value=2,
        ):
            gathered = gather_stacked_audio_latents_for_sp(config, batch, stacked)
            torch.testing.assert_close(
                gathered, torch.cat([stacked, stacked[:, :, :1] + 100], dim=2)
            )
            batch.did_sp_shard_latents = True
            batch.did_sp_shard_audio_latents = False
            self.assertIs(
                gather_stacked_audio_latents_for_sp(config, batch, stacked), stacked
            )

    def test_audio_noise_matches_unsharded_reference_with_two_padded_shards(self):
        from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
            LTX2PipelineConfig,
        )

        def step_audio(batch, config, rank=None):
            scheduler = copy.deepcopy(batch.scheduler)
            prepare_audio_rollout_if_enabled(
                batch=batch,
                pipeline_config=config,
                scheduler=scheduler,
                latents=batch.audio_latents,
            )
            if rank is not None:
                batch.audio_latents, batch.did_sp_shard_audio_latents = (
                    config.shard_audio_latents_for_sp(batch, batch.audio_latents)
                )
            result = scheduler.step(
                torch.zeros_like(batch.audio_latents),
                scheduler.timesteps[0],
                batch.audio_latents,
                **rollout_step_kwargs(batch, AUDIO_STREAM),
                return_dict=False,
            )[0]
            return result, batch.rollout_controllers[
                AUDIO_STREAM
            ].collect_rollout_log_probs()

        expected, expected_log_probs = step_audio(_batch(), _CONFIG)
        shards = []
        config = object.__new__(LTX2PipelineConfig)
        with patch(
            "sglang.multimodal_gen.configs.pipeline_configs.ltx_2.get_sp_world_size",
            return_value=2,
        ):
            for rank in range(2):
                batch = _batch()
                with patch(
                    "sglang.multimodal_gen.configs.pipeline_configs.ltx_2.get_sp_parallel_rank",
                    return_value=rank,
                ):
                    result, log_probs = step_audio(batch, config, rank)
                    shards.append(result)
                    torch.testing.assert_close(
                        log_probs, expected_log_probs, rtol=0, atol=0
                    )
                    self.assertTrue(batch.sp_audio_has_padding)
                    if rank == 1:
                        self.assertTrue(torch.equal(result[:, -1], torch.zeros(2, 3)))
        torch.testing.assert_close(
            torch.cat(shards, dim=1)[:, :5], expected, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
