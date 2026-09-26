# SPDX-License-Identifier: Apache-2.0
"""Behavioral contracts for owned joint-state capture and wire transport."""

import pytest
import torch

from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import (
    _build_response,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    bytes_to_tensor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDenoisingEnv,
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.post_training.rollout_recorder import (
    RolloutRecorder,
    RolloutStreamSpec,
    legacy_video_trajectory,
)


def spec(name="video", shape=(1, 4, 3), **kwargs):
    return RolloutStreamSpec(
        name=name,
        local_shape=shape,
        timesteps=torch.tensor([1000.0, 700.0, 300.0, 0.0]),
        sigmas=torch.tensor([1.0, 0.7, 0.3, 0.0]),
        **kwargs,
    )


@pytest.mark.parametrize("retained", [None, [0, 2], [0, 1, 2], [3], []])
def test_owned_joint_states_and_sparse_schedule_provenance(retained):
    specs = [spec(), spec("audio", (1, 2, 5))]
    recorder = RolloutRecorder(specs, retain_steps=retained)
    state = {"video": torch.zeros(1, 4, 3), "audio": torch.zeros(1, 2, 5)}
    for step in range(3):
        recorder.capture_before(step, state)
        for value in state.values():
            value.add_(1)
    result = recorder.finish(state)
    if retained == []:
        assert result == {}
        return
    expected_indices = list(range(4)) if retained is None else retained
    specs[0].sigmas.zero_()
    for value in state.values():
        value.zero_()
    for name, trajectory in result.items():
        assert trajectory.latent_step_indices.tolist() == expected_indices
        assert trajectory.latents.shape[:2] == (1, len(expected_indices))
        assert trajectory.timesteps.tolist() == [1000, 700, 300, 0]
        assert trajectory.sigmas[0] == 1
        for position, index in enumerate(expected_indices):
            assert (trajectory.latents[:, position] == index).all()


def test_lifecycle_and_stream_shape_errors():
    recorder = RolloutRecorder([spec()])
    state = {"video": torch.zeros(1, 4, 3)}
    with pytest.raises(ValueError, match="consecutive"):
        recorder.capture_before(1, state)
    with pytest.raises(ValueError, match="exactly"):
        recorder.capture_before(0, {"audio": state["video"]})
    with pytest.raises(ValueError, match="shape"):
        recorder.capture_before(0, {"video": torch.zeros(4, 3)})
    recorder.capture_before(0, state)
    with pytest.raises(ValueError, match="every transition"):
        recorder.finish(state)
    recorder.abort()
    with pytest.raises(ValueError, match="active"):
        recorder.capture_before(1, state)


def test_gather_is_stream_specific_and_abort_never_gathers():
    calls = []

    def gather(value):
        calls.append(tuple(value.shape))
        return torch.cat((value, value + 10), dim=2)

    recorder = RolloutRecorder([spec(gather=gather), spec("audio", (1, 2, 5))])
    state = {"video": torch.zeros(1, 4, 3), "audio": torch.ones(1, 2, 5)}
    for step in range(3):
        recorder.capture_before(step, state)
    result = recorder.finish(state)
    assert calls == [(1, 4, 4, 3)]
    assert result["video"].latents.shape == (1, 4, 8, 3)
    assert result["audio"].latents.shape == (1, 4, 2, 5)
    other = RolloutRecorder([spec(gather=gather)])
    other.capture_before(0, {"video": state["video"]})
    other.abort()
    assert len(calls) == 1


def test_wire_transport_slices_streams_and_preserves_packed_conditioning():
    recorder = RolloutRecorder([spec(), spec("audio", (1, 2, 5))], retain_steps=[0, 3])
    state = {"video": torch.zeros(1, 4, 3), "audio": torch.ones(1, 2, 5)}
    for step in range(3):
        recorder.capture_before(step, state)
    streams = recorder.finish(state)
    # A one-token tensor must not lose its token axis just because B == 1.
    env = RolloutDenoisingEnv(
        pos_cond_kwargs={"h3_token_tags": torch.ones(1)}, sample_layout="single_packed"
    )
    output = OutputBatch(
        output=torch.zeros(1, 3, 1, 4, 4),
        rollout_trajectory_data=RolloutTrajectoryData(
            rollout_log_probs=torch.zeros(1, 3),
            denoising_env=env,
            dit_trajectory=legacy_video_trajectory(streams),
            stream_trajectories=streams,
        ),
    )
    response = _build_response("test", "prompt", 1, True, output)[0]
    for name, expected in streams.items():
        wire = response.stream_trajectories[name]
        actual = bytes_to_tensor(wire["latents"]["data"])
        torch.testing.assert_close(actual, expected.latents[0])
        assert bytes_to_tensor(wire["latent_step_indices"]["data"]).tolist() == [0, 3]
        assert bytes_to_tensor(wire["timesteps"]["data"]).shape == (4,)
    assert response.denoising_env["pos_cond_kwargs"]["h3_token_tags"]["shape"] == [1]
    assert bytes_to_tensor(response.dit_trajectory["timesteps"]["data"]).tolist() == [
        1000,
        0,
    ]


@pytest.mark.parametrize("indices", [[-1], [4], [True]])
def test_invalid_retained_indices_fail_before_execution(indices):
    with pytest.raises(ValueError, match="indices"):
        RolloutRecorder([spec()], retain_steps=indices)
