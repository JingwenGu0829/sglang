# SPDX-License-Identifier: Apache-2.0
"""Numerical contract for request-static H3 denoise metadata."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_minimax_h3_euler_ancestral import (
    _minimax_h3_euler_eta0_step,
    _minimax_h3_rf_v_to_x0,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    _build_local_embedding_layout,
    _minimax_h3_update_target_rows_,
    minimax_h3_denoise_loop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
    _build_cube_attn_metadata,
    _precompute_refined_prompt_embeds,
)


def _branch(
    mode: str, token_tags: torch.Tensor | None = None
) -> MiniMaxH3DenoiseBranch:
    common = dict(text_len=3, latent_t=2, latent_h=4, latent_w=4, audio_t=3)
    if mode == "t2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=False,
        )
    elif mode == "fl2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=True,
            keyframe_frame_indices=[0, -1],
            frame_count=5,
        )
    else:
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            **common,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 2},
            ],
        )
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(3, 5120),
        token_tags=packed["token_tags"] if token_tags is None else token_tags,
        device=torch.device("cpu"),
    )


def test_precomputed_timestep_plan_matches_full_unique_reference():
    """Preplanning must preserve fp32 collisions and every packed row class."""

    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        assert branch.static_kwargs["skip_mask_out_condition"]
        assert "token_tags" not in branch.static_kwargs
        assert not bool((branch.static_kwargs["block_token_tags"] < 0).any())
        torch.testing.assert_close(
            branch.static_kwargs["img_pos_for_infer_output_info"]["position_ids"],
            branch.img_pos_dev[branch.update_mask_dev],
            rtol=0,
            atol=0,
        )
        video_steps = [0.75, 0.1]
        audio_steps = [0.625, 0.2]
        plan = branch.prepare_timestep_plan(
            video_timesteps=video_steps,
            audio_timesteps=audio_steps,
            imgvid_cond_noise_aug=0.6,
            audio_ref_cond_noise_aug=0.4,
        )

        assert branch.static_kwargs["packed_seq_params"]["cu_seqlens_q_host"] == tuple(
            int(value)
            for value in branch.static_kwargs["packed_seq_params"][
                "cu_seqlens_q"
            ].tolist()
        )
        assert branch.static_kwargs["refiner_packed_seq_params"][
            "cu_seqlens_q_host"
        ] == (0, 3, 3)

        for step, (video_t, audio_t) in enumerate(
            zip(video_steps, audio_steps, strict=True)
        ):
            reference = torch.full((branch.seq_len,), video_t, dtype=torch.float32)
            reference[branch.img_cond_seq_idx] = max(video_t, 0.6)
            reference[branch.audio_target_seq_idx] = audio_t
            reference[branch.audio_ref_seq_idx] = max(audio_t, 0.4)
            expected = torch.unique(reference, sorted=True, return_inverse=True)
            torch.testing.assert_close(plan[step][0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(plan[step][1], expected[1], rtol=0, atol=0)
            torch.testing.assert_close(
                plan[step][2],
                branch.static_kwargs["block_token_tags"]
                + expected[1] * MINIMAX_H3_ADALN_MODALITY_NUM,
                rtol=0,
                atol=0,
            )

        repeated_plan = branch.prepare_timestep_plan(
            video_timesteps=[0.0, 0.1, 0.2],
            audio_timesteps=[0.0, 0.2, 0.4],
            imgvid_cond_noise_aug=0.999,
            audio_ref_cond_noise_aug=1.0,
        )
        assert repeated_plan[1][1] is repeated_plan[2][1]
        assert repeated_plan[1][2] is repeated_plan[2][2]


def test_inplace_target_update_matches_scheduler_math():
    generator = torch.Generator().manual_seed(7)
    for sigma_curr, sigma_next in ((1.0, 0.7), (0.2, 0.0), (0.0, 0.0)):
        state = torch.randn(11, 32, generator=generator)
        velocity = torch.randn(11, 32, generator=generator)
        timestep = torch.tensor(1.0 - sigma_curr)
        ratio = torch.tensor(0.0 if sigma_curr == 0.0 else sigma_next / sigma_curr)
        denoised = _minimax_h3_rf_v_to_x0(state, velocity, timestep)
        expected = _minimax_h3_euler_eta0_step(
            state,
            denoised,
            sigma_curr=sigma_curr,
            sigma_next=sigma_next,
            sigma_ratio=ratio,
        )

        actual = state.clone()
        velocity_scratch = velocity.clone()
        _minimax_h3_update_target_rows_(
            actual,
            velocity_scratch,
            sigma_t=1.0 - timestep,
            sigma_curr=sigma_curr,
            sigma_ratio=ratio,
            one_minus_sigma_ratio=1.0 - ratio,
            denoised_scratch=torch.empty_like(actual),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_local_text_layout_is_a_contiguous_prefix_per_ulysses_rank():
    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        text_len = int(branch.static_kwargs["prompt_embeds"].shape[0])
        for world_size in (1, 2, 4, 8):
            for rank in range(world_size):
                layout = _build_local_embedding_layout(
                    seq_len=branch.seq_len,
                    text_pos=torch.arange(text_len),
                    img_pos=branch.img_pos,
                    audio_pos=branch.audio_pos,
                    world_size=world_size,
                    rank=rank,
                    device=torch.device("cpu"),
                )
                start = int(layout["text_source_start"])
                stop = int(layout["text_source_stop"])
                row_start = rank * (branch.seq_len // world_size)
                expected = torch.nonzero(
                    (torch.arange(text_len) >= row_start)
                    & (
                        torch.arange(text_len)
                        < row_start + branch.seq_len // world_size
                    )
                ).view(-1)
                assert expected.tolist() == list(range(start, stop))


def test_rank_local_token_tags_match_reference_slice():
    for mode in ("t2va", "fl2va", "ref2va"):
        seq_len = _branch(mode).seq_len
        token_tags = torch.arange(seq_len, dtype=torch.long) - seq_len // 2
        for world_size in (1, 2, 4, 8):
            for rank in range(world_size):
                with patch(
                    "sglang.multimodal_gen.runtime.pipelines_core.stages."
                    "model_specific_stages.minimax_h3.denoise_loop.get_ulysses_ctx",
                    return_value=(world_size, rank),
                ):
                    branch = _branch(mode, token_tags=token_tags)
                local_rows = branch.seq_len // world_size
                expected = token_tags[
                    rank * local_rows : (rank + 1) * local_rows
                ].clamp(min=0)
                torch.testing.assert_close(
                    branch.static_kwargs["block_token_tags"], expected, rtol=0, atol=0
                )


def test_h3_sampling_params_accept_rollout():
    from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams

    params = MiniMaxH3SamplingParams(rollout=True, rollout_return_dit_trajectory=True)
    assert params.rollout is True
    assert params.rollout_return_dit_trajectory is True


def test_h3_negated_v_ode_matches_native_euler():
    from sglang.multimodal_gen.runtime.post_training.flow_transition import flow_step

    state = torch.randn(7, 16, generator=torch.Generator().manual_seed(11))
    velocity = torch.randn_like(state)
    expected = state.clone()
    _minimax_h3_update_target_rows_(
        expected,
        velocity.clone(),
        sigma_t=state.new_tensor(0.7),
        sigma_curr=0.7,
        sigma_ratio=state.new_tensor(0.2 / 0.7),
        one_minus_sigma_ratio=state.new_tensor(1 - 0.2 / 0.7),
        denoised_scratch=torch.empty_like(state),
    )
    actual = flow_step(
        sample=state.unsqueeze(0),
        model_output=-velocity.unsqueeze(0),
        current_sigma=state.new_tensor(0.7),
        next_sigma=state.new_tensor(0.2),
        sigma_max=0.7,
        method="ode",
        noise_level=0,
        legacy_score=True,
        variance_noise=None,
    ).sample.squeeze(0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["ode", "sde", "cps"])
def test_h3_video_rollout_captures_owned_states_without_changing_audio(method):
    from sglang.multimodal_gen.runtime.post_training.h3_rollout import H3RolloutSession

    branch = _branch("t2va")
    video = torch.zeros(branch.img_pos.numel(), 96)
    audio = torch.zeros(branch.audio_pos.numel(), 32)
    video_sigmas, audio_sigmas = [1.0, 0.7, 0.3, 0.0], [1.0, 0.8, 0.2, 0.0]
    outputs = []
    for capture in (False, True):
        session = H3RolloutSession(
            video_shape=tuple(video.shape),
            video_sigmas=video_sigmas,
            generator=torch.Generator().manual_seed(17),
            method=method,
            noise_level=0.5,
            legacy_score=True,
            capture=capture,
            debug=True,
            retain_steps=[0, 2, 3],
            sde_steps=None,
        )
        seen_video, seen_audio = [], []

        def advance(step, v, velocity, a, av, update_v, update_a):
            seen_video.append(v.clone())
            seen_audio.append(a.clone())
            session.advance(step, v, velocity, a, av, update_v, update_a)

        final_v, final_a = minimax_h3_denoise_loop(
            model=SimpleNamespace(prepare_adaln_plans=lambda _: None),
            model_forward=lambda *args: (
                torch.ones_like(video),
                torch.ones_like(audio),
            ),
            positive=branch,
            initial_video_rows=video,
            initial_audio_rows=audio,
            keyframe_cond_rows=None,
            sigmas_video=video_sigmas,
            sigmas_audio=audio_sigmas,
            device=torch.device("cpu"),
            apply_step=advance,
        )
        result = session.finish(final_v)
        outputs.append((final_v.clone(), final_a.clone(), result.rollout_log_probs))
        assert result.rollout_log_probs.shape == (1, 3)
        if capture:
            assert set(result.stream_trajectories) == {"video"}
            torch.testing.assert_close(
                result.dit_trajectory.latents,
                result.stream_trajectories["video"].latents,
            )
            assert result.dit_trajectory.timesteps.tolist() == [1000, 300, 0]
            for name, before, final, sigmas in (
                ("video", seen_video, final_v, video_sigmas),
            ):
                trajectory = result.stream_trajectories[name]
                assert trajectory.latent_step_indices.tolist() == [0, 2, 3]
                expected = torch.stack([before[0], before[2], final], dim=0)
                torch.testing.assert_close(trajectory.latents[0], expected)
                torch.testing.assert_close(trajectory.sigmas, torch.tensor(sigmas))
                torch.testing.assert_close(
                    trajectory.model_timesteps,
                    torch.tensor([1 - s for s in sigmas[:-1]]),
                )
                assert trajectory.timesteps.shape == (4,)
            assert not torch.equal(
                result.dit_trajectory.latents[:, 0],
                result.dit_trajectory.latents[:, -1],
            )
            with pytest.raises(ValueError, match="finish once"):
                session.finish(final_v)
        else:
            assert result.stream_trajectories == {}
    for left, right in zip(*outputs):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    # Audio uses exactly the original deterministic native update.
    _, native_audio = minimax_h3_denoise_loop(
        model=SimpleNamespace(prepare_adaln_plans=lambda _: None),
        model_forward=lambda *args: (torch.ones_like(video), torch.ones_like(audio)),
        positive=branch,
        initial_video_rows=video,
        initial_audio_rows=audio,
        keyframe_cond_rows=None,
        sigmas_video=video_sigmas,
        sigmas_audio=audio_sigmas,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(outputs[0][1], native_audio, rtol=0, atol=0)


def test_cube_metadata_builder_uses_packed_layout_and_validates_step_count():
    packed = minimax_h3_packed_sequence(
        text_len=3,
        latent_t=2,
        latent_h=8,
        latent_w=8,
        audio_t=3,
        include_keyframe_cond=False,
    )
    server_args = SimpleNamespace(
        attention_backend="cube_sparse_attn",
        component_attention_backends={},
        attention_backend_config={
            "local_cube_size": [4, 4, 4],
            "topk_ratio_list": [1.0, 0.5],
        },
    )

    metadata = _build_cube_attn_metadata(
        server_args,
        packed=packed,
        num_steps=2,
        device=torch.device("cpu"),
    )
    assert metadata.topk_ratio_list == [1.0, 0.5]
    assert metadata.precomputed.layout.cube_token_size == 64

    with pytest.raises(ValueError, match="denoise steps"):
        _build_cube_attn_metadata(
            server_args,
            packed=packed,
            num_steps=3,
            device=torch.device("cpu"),
        )


def test_cube_metadata_follows_transformer_backend_override():
    packed = minimax_h3_packed_sequence(
        text_len=3,
        latent_t=2,
        latent_h=8,
        latent_w=8,
        audio_t=3,
        include_keyframe_cond=False,
    )
    server_args = SimpleNamespace(
        attention_backend="fa",
        component_attention_backends={"transformer": "cube_sparse_attn"},
        attention_backend_config={
            "local_cube_size": [4, 4, 4],
            "topk_ratio_list": [0.5],
        },
    )
    assert (
        _build_cube_attn_metadata(
            server_args,
            packed=packed,
            num_steps=1,
            device=torch.device("cpu"),
        )
        is not None
    )

    server_args.attention_backend = "cube_sparse_attn"
    server_args.component_attention_backends["transformer"] = "fa"
    assert (
        _build_cube_attn_metadata(
            server_args,
            packed=packed,
            num_steps=1,
            device=torch.device("cpu"),
        )
        is None
    )


def test_cube_metadata_is_updated_per_step():
    branch = _branch("t2va")
    metadata = SimpleNamespace(current_timestep=-1, topk_ratio_list=[1.0, 0.25])
    seen = []

    def model_forward(_model, _kwargs, step):
        seen.append((step, metadata.current_timestep))
        return (
            torch.zeros(int(branch.update_mask.sum()), 96),
            torch.zeros(branch.audio_pos.numel(), 32),
        )

    minimax_h3_denoise_loop(
        model=SimpleNamespace(prepare_adaln_plans=lambda _: None),
        model_forward=model_forward,
        positive=branch,
        initial_video_rows=torch.zeros(branch.img_pos.numel(), 96),
        initial_audio_rows=torch.zeros(branch.audio_pos.numel(), 32),
        keyframe_cond_rows=None,
        sigmas_video=[1.0, 0.5, 0.0],
        sigmas_audio=[1.0, 0.5, 0.0],
        device=torch.device("cpu"),
        attn_metadata=metadata,
    )

    assert seen == [(0, 0), (1, 1)]


def test_native_dit_forward_publishes_cube_metadata_in_forward_context():
    metadata = SimpleNamespace(current_timestep=0, topk_ratio_list=[0.5])
    batch = SimpleNamespace()

    def model(**_kwargs):
        context = get_forward_context()
        assert context.current_timestep == 0
        assert context.attn_metadata is metadata
        assert context.forward_batch is batch
        return torch.zeros(1, 96), torch.zeros(1, 32)

    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)
    with patch.object(
        MiniMaxH3DenoisingStage,
        "_maybe_get_bcg_runner",
        return_value=None,
    ):
        video, audio = stage._forward_dit(
            model,
            {},
            0,
            batch=batch,
            attn_metadata=metadata,
        )
    assert video.shape == (1, 96)
    assert audio.shape == (1, 32)


def test_grouped_outputs_share_prompt_refinement():
    class Refiner:
        calls = 0

        def refine_prompt_embeds(self, prompt_embeds, refiner_cu, *, device):
            del refiner_cu
            self.calls += 1
            return torch.ones(
                prompt_embeds.shape[0], 5376, dtype=prompt_embeds.dtype, device=device
            )

    model = Refiner()
    conditioning = {}
    first, second = _branch("t2va"), _branch("t2va")

    for branch in (first, second):
        assert _precompute_refined_prompt_embeds(
            model,
            branch,
            device=torch.device("cpu"),
            shared_conditioning=conditioning,
        )

    assert model.calls == 1
    assert first.static_kwargs["prompt_embeds"] is second.static_kwargs["prompt_embeds"]
