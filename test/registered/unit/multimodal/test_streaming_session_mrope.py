"""CPU regressions for M-RoPE metadata across streaming-session appends."""

import unittest
from array import array
from types import SimpleNamespace

import torch
from sglang.srt.layers.rotary_embedding.mrope_rope_index import get_rope_index
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session, SessionController
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB = 1 << 20
BOS = 1
VISION_START = 100
IMAGE = 101
VIDEO = 102
GRID = [[1, 4, 4]]


def _mrope(input_ids, grids):
    positions, delta = get_rope_index(
        spatial_merge_size=2,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_start_token_id=VISION_START,
        model_type="qwen2_5_vl",
        tokens_per_second=2,
        input_ids=torch.tensor([input_ids]),
        image_grid_thw=torch.tensor(grids),
        video_grid_thw=None,
        second_per_grid_ts=None,
        use_audio_in_video=False,
        audio_seqlens=None,
        audio_token_id=None,
        audio_start_token_id=None,
        position_id_per_seconds=None,
    )
    return positions.squeeze(1), delta


def _item(offset):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=123,
        pad_value=1_000_123,
        offsets=[offset],
        model_specific_data={"image_grid_thw": torch.tensor(GRID)},
    )


def _recv(rid, input_ids, *, max_new_tokens=2, mm_inputs=None):
    return SimpleNamespace(
        rid=rid,
        input_ids=array("q", input_ids),
        mm_inputs=mm_inputs,
        session_params=SimpleNamespace(
            id="s", rid=None, offset=None, replace=False, drop_previous_output=False
        ),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        lora_id=None,
        custom_logit_processor=None,
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        return_sampling_mask=False,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        priority=None,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


class TestStreamingSessionMrope(CustomTestCase):
    def setUp(self):
        self.session = Session(capacity_of_str_len=0, session_id="s", streaming=True)
        self.tokenizer = SimpleNamespace(bos_token_id=BOS)

    def _first_turn(self, input_ids, output_ids):
        req = self.session.create_req(
            _recv("r1", input_ids, max_new_tokens=len(output_ids)),
            tokenizer=self.tokenizer,
            vocab_size=VOCAB,
        )
        positions, delta = _mrope(input_ids, GRID)
        req.multimodal_inputs = MultimodalInputs(
            mm_items=[_item((3, 6))],
            mrope_positions=positions,
            mrope_position_delta=delta,
        )
        req.output_ids.extend(output_ids)
        req._refresh_fill_ids()
        self.session.finish_req(req)
        return req

    def test_multimodal_append_matches_single_full_prompt(self):
        first = [BOS, 5, VISION_START, IMAGE, IMAGE, IMAGE, IMAGE, 6]
        output = [7, 8]
        second_with_bos = [BOS, 9, VISION_START, IMAGE, IMAGE, IMAGE, IMAGE, 10, 11]
        second = second_with_bos[1:]
        self._first_turn(first, output)

        second_positions, second_delta = _mrope(second_with_bos, GRID)
        processor_output = MultimodalProcessorOutput(
            input_ids=second_with_bos,
            padded_input_ids=list(second_with_bos),
            mm_items=[_item((3, 6))],
            mrope_positions=second_positions,
            mrope_position_delta=second_delta,
        )
        recv = _recv("r2", second_with_bos, mm_inputs=processor_output)
        req = self.session.create_req(recv, tokenizer=self.tokenizer, vocab_size=VOCAB)

        # Reproduce the lightweight scheduler-side conversion/merge.
        new_mm = MultimodalInputs(
            mm_items=processor_output.mm_items,
            padded_input_ids=processor_output.padded_input_ids,
            mrope_positions=processor_output.mrope_positions,
            mrope_position_delta=processor_output.mrope_position_delta,
        )
        prefix_len = SessionController.adjust_mm_offsets(recv, req, new_mm)
        req.extend_image_inputs(new_mm, sequence_prefix_len=prefix_len)

        full = first + output + second
        expected_positions, expected_delta = _mrope(full, GRID + GRID)
        self.assertEqual(list(req.origin_input_ids), full)
        self.assertEqual(processor_output.padded_input_ids, second)
        self.assertEqual(new_mm.mm_items[0].offsets, [(prefix_len + 2, prefix_len + 5)])
        torch.testing.assert_close(req.multimodal_inputs.mrope_positions, expected_positions)
        torch.testing.assert_close(req.multimodal_inputs.mrope_position_delta, expected_delta)

    def test_text_append_fills_output_and_prompt_positions(self):
        first = [BOS, 5, VISION_START, IMAGE, IMAGE, IMAGE, IMAGE, 6]
        output = [7, 8]
        suffix_with_bos = [BOS, 9, 10, 11]
        self._first_turn(first, output)

        req = self.session.create_req(
            _recv("r2", suffix_with_bos),
            tokenizer=self.tokenizer,
            vocab_size=VOCAB,
        )
        expected_positions, expected_delta = _mrope(first + output + suffix_with_bos[1:], GRID)
        torch.testing.assert_close(req.multimodal_inputs.mrope_positions, expected_positions)
        torch.testing.assert_close(req.multimodal_inputs.mrope_position_delta, expected_delta)

    def test_first_multimodal_chunk_after_text_prefix_matches_full_prompt(self):
        first = [BOS, 30, 31]
        output = [32]
        initial = self.session.create_req(
            _recv("r1", first, max_new_tokens=1),
            tokenizer=self.tokenizer,
            vocab_size=VOCAB,
        )
        initial.output_ids.extend(output)
        initial._refresh_fill_ids()
        self.session.finish_req(initial)

        second_with_bos = [BOS, 9, VISION_START, IMAGE, IMAGE, IMAGE, IMAGE, 10]
        second_positions, second_delta = _mrope(second_with_bos, GRID)
        processor_output = MultimodalProcessorOutput(
            input_ids=second_with_bos,
            padded_input_ids=list(second_with_bos),
            mm_items=[_item((3, 6))],
            mrope_positions=second_positions,
            mrope_position_delta=second_delta,
        )
        recv = _recv("r2", second_with_bos, mm_inputs=processor_output)
        req = self.session.create_req(recv, tokenizer=self.tokenizer, vocab_size=VOCAB)
        new_mm = MultimodalInputs(
            mm_items=processor_output.mm_items,
            mrope_positions=processor_output.mrope_positions,
            mrope_position_delta=processor_output.mrope_position_delta,
        )
        prefix_len = SessionController.adjust_mm_offsets(recv, req, new_mm)
        req.extend_image_inputs(new_mm, sequence_prefix_len=prefix_len)

        expected_positions, expected_delta = _mrope(first + output + second_with_bos[1:], GRID)
        torch.testing.assert_close(req.multimodal_inputs.mrope_positions, expected_positions)
        torch.testing.assert_close(req.multimodal_inputs.mrope_position_delta, expected_delta)

    def test_missing_chunk_mrope_is_computed_before_session_merge(self):
        first = [BOS, 5, VISION_START, IMAGE, IMAGE, IMAGE, IMAGE, 6]
        output = [7, 8]
        second_with_bos = [BOS, 9, VISION_START, IMAGE, IMAGE, IMAGE, IMAGE, 10]
        self._first_turn(first, output)

        processor_output = MultimodalProcessorOutput(
            input_ids=second_with_bos,
            padded_input_ids=list(second_with_bos),
            mm_items=[_item((3, 6))],
        )
        recv = _recv("r2", second_with_bos, mm_inputs=processor_output)
        req = self.session.create_req(recv, tokenizer=self.tokenizer, vocab_size=VOCAB)
        new_mm = MultimodalInputs(mm_items=processor_output.mm_items)

        processor = SimpleNamespace(
            compute_mrope_positions=lambda input_ids, _items: _mrope(
                list(input_ids), GRID
            )
        )
        scheduler = SimpleNamespace(_mm_processor=processor)
        Scheduler._maybe_compute_mrope_positions_for_inputs(
            scheduler, recv.input_ids, new_mm
        )
        prefix_len = SessionController.adjust_mm_offsets(recv, req, new_mm)
        req.extend_image_inputs(new_mm, sequence_prefix_len=prefix_len)

        expected_positions, expected_delta = _mrope(
            first + output + second_with_bos[1:], GRID + GRID
        )
        self.assertEqual(req.multimodal_inputs.mrope_positions.shape[1], len(req.origin_input_ids))
        torch.testing.assert_close(req.multimodal_inputs.mrope_positions, expected_positions)
        torch.testing.assert_close(req.multimodal_inputs.mrope_position_delta, expected_delta)

    def test_aborted_multimodal_append_does_not_mutate_rollback_point(self):
        first = [BOS, 5, VISION_START, IMAGE, IMAGE, IMAGE, IMAGE, 6]
        self._first_turn(first, [])
        committed_mm = next(iter(self.session.req_nodes.values())).req.multimodal_inputs
        committed_mm.mm_items[0].feature = torch.ones(1)

        second = MultimodalInputs(
            mm_items=[_item((10, 13))],
            mrope_positions=torch.arange(8).expand(3, -1),
            mrope_position_delta=torch.zeros((1, 1), dtype=torch.long),
        )
        aborted = self.session.create_req(
            _recv("r2", [20] * 8, mm_inputs=SimpleNamespace(mm_items=[])),
            tokenizer=None,
            vocab_size=VOCAB,
        )
        aborted.extend_image_inputs(second, sequence_prefix_len=len(first))
        self.assertIsNot(aborted.multimodal_inputs, committed_mm)
        self.assertIsNot(aborted.multimodal_inputs.mm_items[0], committed_mm.mm_items[0])
        self.assertEqual(len(aborted.multimodal_inputs.mm_items), 2)
        aborted.multimodal_inputs.release_features()
        self.assertIsNotNone(committed_mm.mm_items[0].feature)
        self.session.abort_req()

        continued = self.session.create_req(_recv("r3", [30]), tokenizer=None, vocab_size=VOCAB)
        self.assertEqual(len(continued.multimodal_inputs.mm_items), 1)
        self.assertEqual(continued.multimodal_inputs.mrope_positions.shape[1], 9)


if __name__ == "__main__":
    unittest.main()
