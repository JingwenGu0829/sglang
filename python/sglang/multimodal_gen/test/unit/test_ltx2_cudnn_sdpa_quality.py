# SPDX-License-Identifier: Apache-2.0
"""LTX-2 runs cuDNN SDPA only for requests above quality="exact".

cuDNN can select a different attention kernel in another process for the same
inputs, so an exact request must not depend on it.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.models.dits import ltx_2
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)


def _attention(use_local_attention: bool) -> ltx_2.LTX2Attention:
    with (
        mock.patch.object(ltx_2, "get_tp_world_size", return_value=1),
        mock.patch.object(ltx_2, "ColumnParallelLinear", return_value=nn.Identity()),
        mock.patch.object(ltx_2, "RowParallelLinear", return_value=nn.Identity()),
        mock.patch.object(layer, "get_ring_parallel_world_size", return_value=1),
    ):
        return ltx_2.LTX2Attention(
            query_dim=64,
            heads=2,
            dim_head=32,
            use_local_attention=use_local_attention,
        )


def _cudnn_flags(attention: ltx_2.LTX2Attention) -> set[bool]:
    layer = attention.attn
    return {
        layer.allow_cudnn_sdp,
        layer.attn_impl.allow_cudnn_sdp,
        layer._attn_impl_ctor_kwargs["allow_cudnn_sdp"],
    }


def _stage(transformer: nn.Module) -> DenoisingStage:
    stage = object.__new__(DenoisingStage)
    stage.transformer = transformer
    stage.transformer_2 = None
    stage._mounted_quality = "exact"
    return stage


def _request(quality: str) -> SimpleNamespace:
    return SimpleNamespace(sampling_params=SimpleNamespace(quality=quality))


@pytest.mark.parametrize("use_local_attention", [False, True])
def test_request_quality_decides_whether_ltx2_runs_cudnn_sdpa(use_local_attention):
    attention = _attention(use_local_attention)
    stage = _stage(nn.ModuleList([attention]))

    # A stage starts at "exact", so a first exact request toggles nothing.
    for quality, expected in (
        ("exact", False),
        ("lossless", True),
        ("exact", False),
        ("high", True),
    ):
        stage._maybe_toggle_quality_fusions(_request(quality))
        assert _cudnn_flags(attention) == {expected}, quality


def test_mount_reports_whether_the_model_has_ltx2_attention():
    assert ltx_2.mount_ltx2_cudnn_sdpa(nn.ModuleList([_attention(False)]))
    assert not ltx_2.mount_ltx2_cudnn_sdpa(nn.Linear(2, 2))
