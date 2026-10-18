# SPDX-License-Identifier: Apache-2.0
"""Regressions specific to audio rollout wiring and request ownership."""

import asyncio
import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from fastapi import HTTPException

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.sample.ltx_2 import LTX23SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.post_training import rollout_api
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    RolloutRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_deserialize,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    LTX2Pipeline,
    LTX2TwoStagePipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.request_utils import (
    expand_request_outputs,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DenoisingStage,
)
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
    finalize_rollout_if_enabled,
    prepare_rollout_if_enabled,
    rollout_scope,
    rollout_step_kwargs,
)

_CONFIG = SimpleNamespace(shard_latents_for_sp=lambda batch, latents: (latents, False))


@pytest.fixture
def batch(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.post_training.sp_utils.get_sp_world_size",
        lambda: 1,
    )
    req = Req(
        sampling_params=SamplingParams(
            prompt="test",
            seed=10,
            rollout=True,
            rollout_audio=True,
            rollout_return_dit_trajectory=True,
            rollout_return_step_indices=[0, 2],
        )
    )
    req.scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    req.scheduler.set_timesteps(sigmas=[0.8, 0.4])
    req.latents = torch.zeros(2, 4, 2)
    req.audio_latents = torch.ones(2, 5, 3)
    req.generator = [torch.Generator().manual_seed(10 + i) for i in range(2)]
    return req


def test_audio_rng_is_repeatable_and_independent(batch):
    states = [g.get_state() for g in batch.generator]
    first, second = [audio_rollout_generators(batch.generator) for _ in range(2)]
    assert first[0].initial_seed() != first[1].initial_seed()
    for visual, state, a, b in zip(batch.generator, states, first, second):
        assert visual.initial_seed() != a.initial_seed()
        torch.testing.assert_close(visual.get_state(), state, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.randn(8, generator=a), torch.randn(8, generator=b), rtol=0, atol=0
        )


def test_audio_steps_finalize_and_survive_output_slicing(batch):
    audio_scheduler = copy.deepcopy(batch.scheduler)
    with rollout_scope(batch):
        prepare_rollout_if_enabled(batch=batch, pipeline_config=_CONFIG)
        visual_controller = batch.rollout_controllers["visual"]
        prepare_audio_rollout_if_enabled(
            batch=batch,
            pipeline_config=_CONFIG,
            scheduler=audio_scheduler,
            latents=batch.audio_latents,
        )
        assert batch.rollout_controllers["visual"] is visual_controller
        for timestep in batch.scheduler.timesteps:
            batch.latents = batch.scheduler.step(
                torch.zeros_like(batch.latents),
                timestep,
                batch.latents,
                **rollout_step_kwargs(batch),
                return_dict=False,
            )[0]
            states = [g.get_state() for g in batch.generator]
            batch.audio_latents = audio_scheduler.step(
                torch.zeros_like(batch.audio_latents),
                timestep,
                batch.audio_latents,
                **rollout_step_kwargs(batch, AUDIO_STREAM),
                return_dict=False,
            )[0]
            for generator, state in zip(batch.generator, states):
                torch.testing.assert_close(generator.get_state(), state, rtol=0, atol=0)
        finalize_rollout_if_enabled(
            batch=batch,
            latents=batch.latents,
            num_inference_steps=2,
            final_timestep=torch.tensor(0.0),
            pipeline_config=_CONFIG,
            additional_latents={AUDIO_STREAM: batch.audio_latents},
        )
    data = batch.rollout_trajectory_data
    assert batch.rollout_controllers == {}
    for stream, latents in ((data, batch.latents), (data.audio, batch.audio_latents)):
        assert stream.dit_trajectory.latent_step_indices.tolist() == [0, 2]
        assert stream.rollout_log_probs.shape == (2, 2)
        assert torch.isfinite(stream.rollout_log_probs).all()
        torch.testing.assert_close(stream.dit_trajectory.latents[:, -1], latents)
    rows = [select_output_rollout_trajectory(data, i) for i in range(2)]
    result = OutputBatch(
        output=torch.zeros(2, 1),
        audio=torch.arange(16).view(2, 8).float(),
        audio_sample_rate=24000,
        rollout_trajectory_data=concat_rollout_trajectory_data(rows),
    )
    responses = rollout_api._build_response("r", "p", 10, True, result)
    for i, response in enumerate(responses):
        audio = _maybe_deserialize(response.audio_rollout)
        torch.testing.assert_close(
            audio["rollout_log_probs"], data.audio.rollout_log_probs[i]
        )
        torch.testing.assert_close(
            audio["dit_trajectory"]["latents"], data.audio.dit_trajectory.latents[i]
        )
        torch.testing.assert_close(
            audio["dit_trajectory"]["timesteps"], data.dit_trajectory.timesteps
        )
        assert audio["dit_trajectory"]["latent_step_indices"].tolist() == [0, 2]
        torch.testing.assert_close(audio["generated_output"], result.audio[i])
        assert audio["sample_rate"] == 24000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_generators_fill_cuda_noise_for_each_sample(batch):
    prepare_rollout_if_enabled(batch=batch, pipeline_config=_CONFIG)
    controller = batch.rollout_controllers["visual"]
    expected = torch.cat(
        [
            torch.randn(1, 4, 2, generator=torch.Generator().manual_seed(10 + i))
            for i in range(2)
        ]
    )
    actual = controller._rollout_variance_noise(batch, batch.latents.cuda())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_expanded_outputs_do_not_share_controllers(batch):
    batch.num_outputs_per_prompt = 2
    children = expand_request_outputs(batch)
    for child in children:
        prepare_rollout_if_enabled(batch=child, pipeline_config=_CONFIG)
    assert children[0].rollout_controllers is not children[1].rollout_controllers
    assert batch.rollout_controllers == {}


@pytest.mark.parametrize(
    "stage_cls,loop_name",
    [(DenoisingStage, "_denoise"), (Cosmos3DenoisingStage, "_denoise_once")],
)
def test_stage_cleanup_allows_retry(batch, stage_cls, loop_name):
    stage = object.__new__(stage_cls)
    stage._offload_for_torch_compile_warmup = lambda batch: nullcontext()
    stage.use_declared_component = lambda **kwargs: nullcontext()
    stage.transformer = Mock()
    fail = True

    def loop(batch, args):
        prepare_rollout_if_enabled(batch=batch, pipeline_config=_CONFIG)
        batch._rollout_denoising_env_state = {"env": None}
        if fail:
            raise RuntimeError("model failed")
        return batch

    setattr(stage, loop_name, loop)
    with pytest.raises(RuntimeError, match="model failed"):
        stage.forward(batch, None)
    assert batch.rollout_controllers == {}
    assert batch._rollout_denoising_env_state is None
    fail = False
    assert stage.forward(batch, None) is batch
    assert batch.rollout_controllers == {}


@pytest.mark.parametrize(
    "pipeline,override,rollout,valid",
    [
        (LTX2Pipeline, None, True, True),
        (LTX2Pipeline, "LTX2Pipeline", True, True),
        (LTX2TwoStagePipeline, None, True, False),
        (LTX2Pipeline, None, False, True),
    ],
)
def test_http_validates_the_resolved_pipeline(
    monkeypatch, pipeline, override, rollout, valid
):
    args = SimpleNamespace(
        pipeline_class_name=override,
        model_path="test",
        model_id=None,
        backend="sglang",
        attention_backend_config={},
        enable_trace=False,
    )
    monkeypatch.setattr(
        registry,
        "get_model_info",
        lambda *a, **kw: SimpleNamespace(pipeline_cls=pipeline),
    )
    monkeypatch.setattr(registry, "get_pipeline_class", lambda name: pipeline)
    monkeypatch.setattr(rollout_api, "get_global_server_args", lambda: args)
    monkeypatch.setattr(
        rollout_api,
        "build_sampling_params",
        lambda rid, **kw: LTX23SamplingParams(**kw),
    )
    forward = AsyncMock(return_value=OutputBatch(output=torch.zeros(1, 1)))
    monkeypatch.setattr(rollout_api.async_scheduler_client, "forward", forward)
    monkeypatch.setattr(rollout_api, "_build_response", lambda *a, **kw: [])
    request = RolloutRequest(
        prompt="test", seed=42, rollout=rollout, rollout_audio=True
    )
    if valid:
        assert asyncio.run(rollout_api.rollout_generate(request)).status_code == 200
        forward.assert_awaited_once()
    else:
        with pytest.raises(HTTPException) as error:
            asyncio.run(rollout_api.rollout_generate(request))
        assert error.value.status_code == 400
        forward.assert_not_awaited()
