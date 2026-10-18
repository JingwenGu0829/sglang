# SPDX-License-Identifier: Apache-2.0
"""Audio rollout integration through real Euler steps and the HTTP serializer."""

import asyncio
import copy
import gc
import types
import unittest
import weakref
from contextlib import nullcontext
from unittest.mock import AsyncMock, Mock, patch

import torch
from fastapi import HTTPException

from sglang.multimodal_gen.configs.sample.ltx_2 import (
    LTX2SamplingParams,
    LTX23SamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.post_training import rollout_api
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
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    LTX2Pipeline,
    LTX2TwoStagePipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.request_utils import (
    expand_request_outputs,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.runtime.post_training.audio_rollout import (
    audio_rollout_generators,
    prepare_audio_rollout_if_enabled,
)
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutTrajectoryData,
    concat_rollout_trajectory_data,
    select_output_rollout_trajectory,
)
from sglang.multimodal_gen.runtime.post_training.rollout_hooks import (
    AUDIO_STREAM,
    finalize_rollout_if_enabled,
    init_rollout_env_if_enabled,
    prepare_rollout_if_enabled,
    rollout_step_kwargs,
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
    """Use the real LTX Euler/CFG branch and shared finalizer with a tiny joint model."""
    stage = object.__new__(LTX2DenoisingStage)
    stage.sampler_name = "euler"
    stage.server_args = types.SimpleNamespace(
        pipeline_config=_CONFIG, enable_cfg_parallel=False
    )
    ctx = types.SimpleNamespace(
        scheduler=batch.scheduler,
        audio_scheduler=copy.deepcopy(batch.scheduler),
        timesteps=batch.scheduler.timesteps,
        latents=batch.latents,
        audio_latents=batch.audio_latents,
        stage="stage1",
        is_ltx23_variant=False,
        use_ltx23_legacy_one_stage=False,
        image_kwargs={},
        pos_cond_kwargs={},
        neg_cond_kwargs={},
        guidance=None,
        denoise_mask=None,
        clean_latent=None,
        reserved_frames_mask=None,
        z=None,
    )
    batch.do_classifier_free_guidance = False
    batch.prompt_embeds = batch.audio_prompt_embeds = [torch.zeros(2, 1, 1)]

    def prepare(batch, server_args):
        prepare_rollout_if_enabled(batch=batch, pipeline_config=_CONFIG)
        prepare_audio_rollout_if_enabled(
            batch=batch,
            pipeline_config=_CONFIG,
            scheduler=ctx.audio_scheduler,
            latents=ctx.audio_latents,
        )
        return ctx

    def model(ctx, step, kwargs):
        return (
            ctx.latents * 0.1 + ctx.audio_latents.mean((1, 2), keepdim=True),
            ctx.audio_latents * 0.1 + ctx.latents.mean((1, 2), keepdim=True),
        )

    def loop(ctx, batch, server_args):
        for i, timestep in enumerate(ctx.timesteps):
            stage._run_denoising_step(
                ctx,
                types.SimpleNamespace(step_index=i, t_device=timestep),
                batch,
                server_args,
            )
        return ctx.timesteps.cpu()

    def finish(ctx, batch, server_args):
        batch.latents, batch.audio_latents = ctx.latents, ctx.audio_latents

    stage._offload_for_torch_compile_warmup = lambda batch: nullcontext()
    stage._prepare_denoising_loop = prepare
    stage._run_denoising_loop = loop
    stage._finalize_denoising_loop = finish
    stage._get_ltx2_stage1_guider_params = lambda *args: None
    stage._prepare_ltx2_model_inputs = lambda *args: types.SimpleNamespace(
        latent_model_input=ctx.latents
    )
    stage._build_ltx2_base_model_kwargs = lambda *args: {}
    stage._build_ltx2_model_kwargs = lambda *args, **kwargs: kwargs
    stage._get_ltx_prompt_attention_mask = lambda *args, **kwargs: None
    stage._ltx2_model_forward_context = lambda *args: nullcontext()
    stage._ltx2_call_current_model = model
    stage.post_forward_for_ti2v_task = lambda batch, args, mask, latents, z: latents
    stage.forward(batch, stage.server_args)
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cpu_generators_with_cuda_latents(self):
        for mode in ("sde", "cps", "ode"):
            with self.subTest(mode=mode):
                cpu, gpu = _batch(), _batch()
                cpu.rollout_sde_type = gpu.rollout_sde_type = mode
                cpu.rollout_log_prob_no_const = gpu.rollout_log_prob_no_const = True
                gpu.latents = gpu.latents.cuda()
                gpu.audio_latents = gpu.audio_latents.cuda()
                expected, actual = _run(cpu), _run(gpu)
                for left, right in ((expected, actual), (expected.audio, actual.audio)):
                    torch.testing.assert_close(
                        left.rollout_log_probs, right.rollout_log_probs
                    )
                    torch.testing.assert_close(
                        left.dit_trajectory.latents, right.dit_trajectory.latents
                    )
                    torch.testing.assert_close(
                        left.rollout_debug_tensors.rollout_variance_noises,
                        right.rollout_debug_tensors.rollout_variance_noises,
                        rtol=0,
                        atol=0,
                    )
                for a, b in zip(cpu.generator, gpu.generator):
                    self.assertTrue(torch.equal(a.get_state(), b.get_state()))

    def test_expanded_outputs_own_their_rollout_state(self):
        for audio in (False, True):
            with self.subTest(audio=audio):
                parent = _batch(audio=audio)
                parent.num_outputs_per_prompt = 2
                children = expand_request_outputs(parent)
                self.assertIsNot(
                    children[0].rollout_controllers, children[1].rollout_controllers
                )
                self.assertIsNot(
                    children[0].rollout_controllers, parent.rollout_controllers
                )
                for child in children:
                    child.generator = [
                        torch.Generator().manual_seed(10 + i) for i in range(2)
                    ]
                    child.scheduler = copy.deepcopy(parent.scheduler)
                    _run(child)
                    self.assertEqual(child.rollout_controllers, {})
                    self.assertEqual(parent.rollout_controllers, {})
                    self.assertEqual(
                        child.rollout_trajectory_data.audio is not None, audio
                    )

    def test_denoising_stages_release_state_on_failure_and_retry(self):
        for cls, loop_name in (
            (DenoisingStage, "_denoise"),
            (Cosmos3DenoisingStage, "_denoise_once"),
        ):
            with self.subTest(stage=cls.__name__):
                batch = _batch(audio=False)
                stage = object.__new__(cls)
                stage._offload_for_torch_compile_warmup = lambda batch: nullcontext()
                stage.use_declared_component = lambda **kwargs: nullcontext()
                stage.transformer = Mock()
                fail = True

                def loop(batch, args):
                    prepare_rollout_if_enabled(batch=batch, pipeline_config=_CONFIG)
                    init_rollout_env_if_enabled(
                        batch=batch,
                        image_kwargs={},
                        pos_cond_kwargs={},
                        neg_cond_kwargs={},
                        guidance=None,
                    )
                    if fail:
                        raise RuntimeError("model failed")
                    batch.latents = batch.scheduler.step(
                        torch.zeros_like(batch.latents),
                        batch.scheduler.timesteps[0],
                        batch.latents,
                        **rollout_step_kwargs(batch),
                        return_dict=False,
                    )[0]
                    finalize_rollout_if_enabled(
                        batch=batch,
                        latents=batch.latents,
                        num_inference_steps=1,
                        final_timestep=torch.tensor(0.0),
                        pipeline_config=_CONFIG,
                    )
                    return batch

                setattr(stage, loop_name, loop)
                with self.assertRaisesRegex(RuntimeError, "model failed"):
                    stage.forward(batch, types.SimpleNamespace())
                self.assertEqual(batch.rollout_controllers, {})
                self.assertIsNone(batch._rollout_denoising_env_state)
                self.assertIsNone(batch.rollout_trajectory_data)
                fail = False
                stage.forward(batch, types.SimpleNamespace())
                self.assertEqual(batch.rollout_controllers, {})
                self.assertIsNone(batch._rollout_denoising_env_state)
                self.assertIsNotNone(batch.rollout_trajectory_data)

    def test_request_boundaries_validate_resolved_pipeline_before_execution(self):
        registry = "sglang.multimodal_gen.registry"
        cases = [
            (None, LTX2Pipeline, LTX2SamplingParams, True, {}, True),
            ("LTX2Pipeline", LTX2Pipeline, LTX23SamplingParams, True, {}, True),
            (None, LTX2TwoStagePipeline, LTX2SamplingParams, True, {}, False),
            (
                "LTX2TwoStagePipeline",
                LTX2TwoStagePipeline,
                LTX23SamplingParams,
                True,
                {},
                False,
            ),
            (None, ComposedPipelineBase, SamplingParams, True, {}, False),
            (
                None,
                LTX2Pipeline,
                LTX2SamplingParams,
                True,
                {"generate_audio": False},
                False,
            ),
            (
                None,
                LTX2Pipeline,
                LTX2SamplingParams,
                True,
                {"image_path": "input.png"},
                False,
            ),
            (None, LTX2Pipeline, LTX23SamplingParams, False, {}, True),
            (None, ComposedPipelineBase, SamplingParams, False, {}, True),
        ]
        for name, cls, params_cls, rollout, extra, valid in cases:
            with self.subTest(
                pipeline=cls.__name__, override=name, rollout=rollout, extra=extra
            ):
                args = types.SimpleNamespace(
                    pipeline_class_name=name,
                    model_path="test",
                    model_id=None,
                    backend="sglang",
                    attention_backend_config={},
                    enable_trace=False,
                )
                request = RolloutRequest(
                    prompt="test",
                    seed=42,
                    rollout=rollout,
                    rollout_audio=True,
                    extra_sampling_params=extra,
                )
                result = OutputBatch(
                    output=torch.zeros(1, 1),
                    rollout_trajectory_data=RolloutTrajectoryData(),
                )
                forward = AsyncMock(return_value=result)
                with (
                    patch(
                        f"{registry}.get_model_info",
                        return_value=types.SimpleNamespace(pipeline_cls=cls),
                    ),
                    patch(f"{registry}.get_pipeline_class", return_value=cls),
                    patch.object(
                        rollout_api, "get_global_server_args", return_value=args
                    ),
                    patch.object(
                        rollout_api,
                        "build_sampling_params",
                        side_effect=lambda rid, **kw: params_cls(**kw),
                    ),
                    patch.object(
                        rollout_api.async_scheduler_client, "forward", forward
                    ),
                ):
                    if valid:
                        self.assertEqual(
                            asyncio.run(
                                rollout_api.rollout_generate(request)
                            ).status_code,
                            200,
                        )
                        forward.assert_awaited_once()
                    else:
                        with self.assertRaises(HTTPException) as error:
                            asyncio.run(rollout_api.rollout_generate(request))
                        self.assertEqual(error.exception.status_code, 400)
                        forward.assert_not_awaited()
                    # The raw engine Req boundary uses the same policy before
                    # seed allocation, image loading, or any text encoder work.
                    raw = Req(
                        sampling_params=params_cls(**_build_sampling_kwargs(request))
                    )
                    raw.sampling_params.apply_request_extra(raw)
                    stage = object.__new__(InputValidationStage)
                    stage._generate_seeds = Mock(side_effect=RuntimeError("validated"))
                    with self.assertRaises(RuntimeError if valid else ValueError):
                        stage.forward(raw, args)
                    self.assertEqual(stage._generate_seeds.call_count, int(valid))


if __name__ == "__main__":
    unittest.main()
