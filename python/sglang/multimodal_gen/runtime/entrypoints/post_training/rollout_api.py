"""Rollout HTTP API (``POST /rollout/generate``)."""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from typing import Any

import torch
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    RolloutRequest,
    RolloutResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.request_timing import (
    STAGES_HEADER,
    TIMING_HEADER,
    RequestStamps,
    stages_header,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_serialize,
    _quantize_video_uint8,
    msgpack_encode_spliced,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutStreamData,
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/rollout", tags=["rollout"])


def _extract_single_sample_tensor(
    obj: Any, sample_idx: int, batch_size: int, *, current_key: str | None = None
) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.dim() >= 1 and obj.shape[0] == batch_size:
            return obj[sample_idx].contiguous()
        return obj
    if isinstance(obj, dict):
        return {
            k: _extract_single_sample_tensor(v, sample_idx, batch_size, current_key=k)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        if current_key == "img_shapes" and len(obj) == batch_size:
            return [obj[sample_idx]]
        return [
            _extract_single_sample_tensor(
                v, sample_idx, batch_size, current_key=current_key
            )
            for v in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            _extract_single_sample_tensor(
                v, sample_idx, batch_size, current_key=current_key
            )
            for v in obj
        )
    return obj


def _slice_stream_for_sample(
    stream: RolloutStreamData | None, sample_idx: int, batch_size: int
) -> RolloutStreamData | None:
    if stream is None:
        return None
    debug = stream.rollout_debug_tensors
    if debug is not None:
        debug = replace(
            debug,
            **{
                field.name: _extract_single_sample_tensor(
                    getattr(debug, field.name), sample_idx, batch_size
                )
                for field in fields(debug)
            },
        )
    trajectory = stream.dit_trajectory
    if trajectory is not None:
        trajectory = replace(
            trajectory,
            latents=_extract_single_sample_tensor(
                trajectory.latents, sample_idx, batch_size
            ),
        )
    return replace(
        stream,
        rollout_log_probs=_extract_single_sample_tensor(
            stream.rollout_log_probs, sample_idx, batch_size
        ),
        rollout_debug_tensors=debug,
        dit_trajectory=trajectory,
    )


def _slice_rollout_trajectory_for_sample(
    rtd: RolloutTrajectoryData | None,
    sample_idx: int,
    batch_size: int,
) -> RolloutTrajectoryData | None:
    if rtd is None:
        return None
    env = rtd.denoising_env
    if env is not None:
        env = replace(
            env,
            **{
                field.name: _extract_single_sample_tensor(
                    getattr(env, field.name), sample_idx, batch_size
                )
                if field.name == "guidance" or getattr(env, field.name)
                else None
                for field in fields(env)
            },
        )
    return replace(
        _slice_stream_for_sample(rtd, sample_idx, batch_size),
        denoising_env=env,
        audio=_slice_stream_for_sample(rtd.audio, sample_idx, batch_size),
    )


def _serialize_stream(
    stream: RolloutStreamData | None,
    *,
    serialized_dit_timesteps: dict | None = None,
    serialized_dit_sigmas: dict | None = None,
) -> dict:
    debug = stream.rollout_debug_tensors if stream else None
    trajectory = stream.dit_trajectory if stream else None
    return {
        "rollout_log_probs": _maybe_serialize(stream.rollout_log_probs)
        if stream
        else None,
        "rollout_debug_tensors": {
            field.name: _maybe_serialize(getattr(debug, field.name))
            for field in fields(debug)
        }
        if debug is not None
        else None,
        "dit_trajectory": {
            "latents": _maybe_serialize(trajectory.latents),
            "timesteps": serialized_dit_timesteps,
            "sigmas": serialized_dit_sigmas,
            "latent_step_indices": _maybe_serialize(trajectory.latent_step_indices),
        }
        if trajectory is not None
        else None,
    }


def _serialize_rollout_trajectory(
    rtd: RolloutTrajectoryData | None,
    *,
    serialized_dit_timesteps: dict | None = None,
    serialized_dit_sigmas: dict | None = None,
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    """Return order: rollout_log_probs, rollout_debug_tensors, denoising_env, dit_trajectory."""
    stream = _serialize_stream(
        rtd,
        serialized_dit_timesteps=serialized_dit_timesteps,
        serialized_dit_sigmas=serialized_dit_sigmas,
    )
    env = rtd.denoising_env if rtd else None
    serialized_env = (
        {
            field.name: _maybe_serialize(getattr(env, field.name))
            if field.name == "guidance" or getattr(env, field.name)
            else None
            for field in fields(env)
        }
        if env is not None
        else None
    )
    return (
        stream["rollout_log_probs"],
        stream["rollout_debug_tensors"],
        serialized_env,
        stream["dit_trajectory"],
    )


def _build_response(
    request_id: str,
    prompt: str,
    seed: int,
    rollout: bool,
    result: OutputBatch,
    video_dtype: str | None = None,
) -> list[RolloutResponse]:
    """
    rollout: bool - set to False when evaluating the model
    """
    batch_size = result.output.shape[0]
    inference_time_s = (
        result.metrics.total_duration_s
        if result.metrics and result.metrics.total_duration_s > 0
        else None
    )
    peak_memory_mb = result.peak_memory_mb if result.peak_memory_mb > 0 else None
    rollout_trajectory_data = result.rollout_trajectory_data
    if rollout:
        assert rollout_trajectory_data is not None, (
            "rollout_trajectory_data must be present when rollout=True"
        )

    serialized_dit_timesteps = None
    serialized_dit_sigmas = None
    if rollout and rollout_trajectory_data and rollout_trajectory_data.dit_trajectory:
        serialized_dit_timesteps = _maybe_serialize(
            rollout_trajectory_data.dit_trajectory.timesteps
        )
        serialized_dit_sigmas = _maybe_serialize(
            rollout_trajectory_data.dit_trajectory.sigmas
        )

    audio = (
        rollout_trajectory_data.audio if rollout and rollout_trajectory_data else None
    )
    audio_timesteps = (
        _maybe_serialize(audio.dit_trajectory.timesteps)
        if audio and audio.dit_trajectory
        else None
    )
    audio_sigmas = (
        _maybe_serialize(audio.dit_trajectory.sigmas)
        if audio and audio.dit_trajectory
        else None
    )

    responses: list[RolloutResponse] = []
    for sample_idx in range(batch_size):
        out_i = result.output[sample_idx]
        if isinstance(out_i, torch.Tensor):
            if video_dtype == "uint8":
                out_i = _quantize_video_uint8(out_i)
            out_i = out_i.contiguous()
        serialized_generated_output = _maybe_serialize(out_i)
        if not rollout:
            responses.append(
                RolloutResponse(
                    request_id=request_id,
                    prompt=prompt,
                    seed=seed,
                    generated_output=serialized_generated_output,
                    inference_time_s=inference_time_s,
                    peak_memory_mb=peak_memory_mb,
                )
            )
            continue
        per_sample_trajectory = _slice_rollout_trajectory_for_sample(
            result.rollout_trajectory_data, sample_idx, batch_size
        )
        (
            serialized_log_probs,
            serialized_debug_tensors,
            serialized_denoising_env,
            serialized_dit_trajectory,
        ) = _serialize_rollout_trajectory(
            per_sample_trajectory,
            serialized_dit_timesteps=serialized_dit_timesteps,
            serialized_dit_sigmas=serialized_dit_sigmas,
        )
        audio_rollout = None
        if per_sample_trajectory.audio is not None:
            audio_rollout = {
                **_serialize_stream(
                    per_sample_trajectory.audio,
                    serialized_dit_timesteps=audio_timesteps,
                    serialized_dit_sigmas=audio_sigmas,
                ),
                "generated_output": _maybe_serialize(result.audio[sample_idx])
                if result.audio is not None
                else None,
                "sample_rate": result.audio_sample_rate,
            }
        responses.append(
            RolloutResponse(
                request_id=request_id,
                prompt=prompt,
                seed=seed,
                generated_output=serialized_generated_output,
                rollout_log_probs=serialized_log_probs,
                rollout_debug_tensors=serialized_debug_tensors,
                denoising_env=serialized_denoising_env,
                dit_trajectory=serialized_dit_trajectory,
                audio_rollout=audio_rollout,
                inference_time_s=inference_time_s,
                peak_memory_mb=peak_memory_mb,
            )
        )
    return responses


def _build_sampling_kwargs(request: RolloutRequest) -> dict:
    sampling_kwargs: dict = dict(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
        generator_device=request.generator_device,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        num_outputs_per_prompt=request.num_outputs_per_prompt,
        guidance_scale=request.guidance_scale,
        true_cfg_scale=request.true_cfg_scale,
        num_frames=request.num_frames,
        fps=request.fps,
        image_path=request.image_path,
        rollout=request.rollout,
        rollout_audio=request.rollout_audio,
        rollout_sde_type=request.rollout_sde_type,
        rollout_noise_level=request.rollout_noise_level,
        rollout_log_prob_no_const=request.rollout_log_prob_no_const,
        rollout_debug_mode=request.rollout_debug_mode,
        rollout_return_denoising_env=request.rollout_return_denoising_env,
        rollout_return_dit_trajectory=request.rollout_return_dit_trajectory,
        rollout_sde_step_indices=request.rollout_sde_step_indices,
        rollout_return_step_indices=request.rollout_return_step_indices,
        suppress_logs=request.suppress_logs,
        save_output=False,
        return_trajectory_latents=False,
        return_trajectory_decoded=False,
    )
    if request.extra_sampling_params:
        sampling_kwargs.update(request.extra_sampling_params)
        sampling_kwargs["rollout"] = request.rollout
    return {k: v for k, v in sampling_kwargs.items() if v is not None}


@router.post(
    "/generate",
    response_class=Response,
    responses={
        200: {
            "model": list[RolloutResponse],
            "content": {"application/msgpack": {}},
        }
    },
)
async def rollout_generate(request: RolloutRequest):
    stamps = RequestStamps()
    stamps.mark("srv_recv")
    request_id = generate_request_id()
    stamps.request_id = request_id
    server_args = get_global_server_args()
    sampling_kwargs = _build_sampling_kwargs(request)
    try:
        sampling_params = build_sampling_params(request_id, **sampling_kwargs)
        pipeline_request = prepare_request(
            server_args=server_args, sampling_params=sampling_params
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid sampling params: {exc}"
        ) from exc
    try:
        with stamps.span("forward"):
            output_batch: OutputBatch = await async_scheduler_client.forward(
                pipeline_request
            )
    except Exception as exc:
        logger.error("Rollout generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {exc}"
        ) from exc
    if output_batch.error:
        raise HTTPException(status_code=500, detail=output_batch.error)

    def _serialize_response() -> list[bytes]:
        with stamps.span("build"):
            rollout_responses = _build_response(
                request_id,
                request.prompt,
                request.seed,
                request.rollout,
                output_batch,
                video_dtype=request.rollout_video_dtype,
            )
        with stamps.span("dump"):
            payload = [r.model_dump() for r in rollout_responses]
        with stamps.span("msgpack"):
            return msgpack_encode_spliced(payload)

    # Off the event loop: building the response copies hundreds of MB
    parts = await asyncio.to_thread(_serialize_response)

    # Timing rides a header: the last marks postdate the encoded body. The
    # explicit content-length keeps uvicorn on identity framing, not chunked.
    headers = {
        TIMING_HEADER: stamps.to_header(),
        "content-length": str(sum(len(p) for p in parts)),
    }
    stages = stages_header(output_batch.metrics)
    if stages:
        headers[STAGES_HEADER] = stages
    return StreamingResponse(
        iter(parts),
        media_type="application/msgpack",
        headers=headers,
    )
