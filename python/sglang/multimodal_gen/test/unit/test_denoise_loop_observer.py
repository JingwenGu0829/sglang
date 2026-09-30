# SPDX-License-Identifier: Apache-2.0
"""Observer is start/end only; per-step collection is ``step_latents``."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.post_training.denoise_loop_observer import (
    NullDenoiseLoopObserver,
    RolloutDenoiseLoopObserver,
    get_denoise_loop_observer,
)
from sglang.multimodal_gen.runtime.post_training.rollout_denoising_mixin import (
    RolloutDenoisingMixin,
)


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    for module in ("scheduler_rl_mixin", "sp_utils"):
        monkeypatch.setattr(
            f"sglang.multimodal_gen.runtime.post_training.{module}.get_sp_world_size",
            lambda: 1,
        )


class _RecordingStage(RolloutDenoisingMixin):
    def __init__(self):
        self.server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                gather_denoising_env_static_for_sp=lambda batch, cond: cond,
                shard_latents_for_sp=lambda batch, latents: (latents, False),
            )
        )


def _batch(*, rollout: bool, return_traj: bool = True):
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(3, device="cpu")
    return SimpleNamespace(
        rollout=rollout,
        rollout_return_dit_trajectory=return_traj,
        rollout_return_denoising_env=False,
        rollout_return_step_indices=None,
        rollout_trajectory_data=None,
        scheduler=scheduler,
        latents=torch.zeros(1, 8),
        rollout_sde_type="ode",
        rollout_noise_level=0.5,
        rollout_log_prob_no_const=True,
        rollout_debug_mode=False,
        _rollout_denoising_env_state=None,
    )


def test_observer_factory_and_serving_is_noop():
    serving = _batch(rollout=False)
    assert isinstance(get_denoise_loop_observer(serving), NullDenoiseLoopObserver)
    assert get_denoise_loop_observer(serving) is get_denoise_loop_observer(serving)

    rollout = _batch(rollout=True)
    assert isinstance(get_denoise_loop_observer(rollout), RolloutDenoiseLoopObserver)

    stage = _RecordingStage()
    observer = get_denoise_loop_observer(serving)
    stage.step_latents(
        serving, torch.zeros(1, 4), torch.tensor(1.0), 0, apply=lambda: None
    )
    observer.finalize(
        stage,
        serving,
        latents=torch.zeros(1, 4),
        num_inference_steps=1,
        final_timestep=torch.zeros(()),
        server_args=SimpleNamespace(pipeline_config=SimpleNamespace()),
    )
    assert serving.rollout_trajectory_data is None
    assert not hasattr(serving, "_rollout_loop_step_index")
    serving.rollout = True
    assert isinstance(get_denoise_loop_observer(serving), RolloutDenoiseLoopObserver)


def test_step_latents_collects_trajectory():
    batch = _batch(rollout=True)
    stage = _RecordingStage()
    observer = get_denoise_loop_observer(batch)
    server_args = stage.server_args

    observer.init_env(
        stage,
        batch,
        pipeline_config=server_args.pipeline_config,
        image_kwargs={},
        pos_cond_kwargs={"encoder_hidden_states": torch.zeros(1, 2, 4)},
        neg_cond_kwargs=None,
        guidance=None,
    )
    latents = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    for step in range(3):
        latents = stage.step_latents(
            batch,
            latents,
            torch.tensor(1.0 - 0.25 * step),
            step,
            apply=lambda current=latents: batch.scheduler.step(
                -torch.ones_like(current),
                batch.scheduler.timesteps[step],
                current,
                batch=batch,
                return_dict=False,
            )[0],
        )

    with patch(
        "sglang.multimodal_gen.runtime.post_training.sp_utils.get_sp_world_size",
        return_value=1,
    ):
        observer.finalize(
            stage,
            batch,
            latents=latents,
            num_inference_steps=3,
            final_timestep=torch.zeros(()),
            server_args=server_args,
        )

    traj = batch.rollout_trajectory_data.dit_trajectory
    assert traj.latents.shape[1] == 4
    assert traj.timesteps.shape == (4,)
    assert batch.rollout_trajectory_data.rollout_log_probs.shape == (1, 3)


@pytest.mark.parametrize("retained", [None, [0, 2, 3]])
def test_inplace_snapshots_and_environment_are_owned(retained):
    stage, batch = _RecordingStage(), _batch(rollout=True)
    batch.rollout_return_denoising_env = True
    batch.rollout_return_step_indices = retained
    condition = torch.ones(1, 4)
    stage._maybe_init_denoising_env_collection(
        batch, stage.server_args.pipeline_config, {}, {"tokens": condition}, None, None
    )
    state = torch.zeros(1, 4, 3)
    original_sigmas = batch.scheduler.sigmas.clone()
    for i in range(3):
        stage.step_latents(
            batch, state, torch.tensor(float(i)), i, apply=lambda: state.add_(1)
        )
    condition.zero_()
    batch.scheduler.sigmas.zero_()
    stage._maybe_append_dit_trajectory_step(batch, state, torch.tensor(3.0), 3)
    stage._maybe_finalize_denoising_env_collection(
        batch, stage.server_args.pipeline_config
    )
    result = batch.rollout_trajectory_data
    indices = list(range(4)) if retained is None else retained
    assert result.dit_trajectory.latents.shape == (1, len(indices), 4, 3)
    assert result.dit_trajectory.latents[0, :, 0, 0].tolist() == indices
    torch.testing.assert_close(result.dit_trajectory.sigmas, original_sigmas)
    assert result.denoising_env.pos_cond_kwargs["tokens"].sum() == 4
    assert batch._rollout_denoising_env_state is None


def test_failure_releases_state_without_finalizing():
    stage, batch = _RecordingStage(), _batch(rollout=True)
    with pytest.raises(RuntimeError, match="model failed"):
        with stage.rollout_lifecycle(batch):
            get_denoise_loop_observer(batch).init_env(
                stage, batch, stage.server_args.pipeline_config, {}, {}, None, None
            )
            raise RuntimeError("model failed")
    assert batch._rollout_session_data is None
    assert batch._rollout_denoising_env_state is None
    assert batch.rollout_trajectory_data is None


def test_ltx_trajectory_excludes_sp_padding(monkeypatch):
    from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
        LTX2DenoisingStage,
    )

    batch = _batch(rollout=True)
    batch.did_sp_shard_latents = True
    batch.raw_latent_shape = (1, 3, 4)
    config = LTX2PipelineConfig.__new__(LTX2PipelineConfig)
    # Rank 0 has two tokens; rank 1 has the third token and one padding token.
    config._gather_sp_tensor = lambda x, dim: torch.cat(
        [
            x,
            torch.cat(
                [torch.full_like(x[:, :1], 7), torch.zeros_like(x[:, :1])], dim=1
            ),
        ],
        dim=dim,
    )
    for module in (
        "runtime.post_training.sp_utils",
        "configs.pipeline_configs.ltx_2",
    ):
        monkeypatch.setattr(
            f"sglang.multimodal_gen.{module}.get_sp_world_size", lambda: 2
        )
    stage = LTX2DenoisingStage.__new__(LTX2DenoisingStage)
    stage._maybe_init_denoising_env_collection(batch, config, {}, {}, None, None)
    for step in range(2):
        stage._maybe_append_dit_trajectory_step(
            batch, torch.full((1, 2, 4), float(step)), torch.tensor(step), step
        )
    stage._maybe_finalize_denoising_env_collection(batch, config)
    trajectory = batch.rollout_trajectory_data.dit_trajectory.latents
    assert trajectory.shape == (1, 2, 3, 4)
    assert trajectory[0, :, 0, 0].tolist() == [0, 1]
    assert trajectory[0, :, 2, 0].tolist() == [7, 7]
