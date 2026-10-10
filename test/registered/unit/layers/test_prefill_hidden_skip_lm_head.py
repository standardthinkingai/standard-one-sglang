"""LM-head projection is unnecessary for last-hidden-only prefill requests."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestPrefillLastHiddenSkipsLMHead(unittest.TestCase):
    def setUp(self):
        self.hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        self.input_ids = torch.arange(5)
        self.mm_input_embeds = torch.ones(1, 4)
        self.projected_logits = torch.full((2, 7), 3.0)
        self.processor = LogitsProcessor.__new__(LogitsProcessor)
        torch.nn.Module.__init__(self.processor)
        self.processor._get_logits = Mock(return_value=self.projected_logits)

    def _forward_batch(
        self,
        *,
        forward_mode=ForwardMode.EXTEND,
        is_prefill_only=True,
        return_logprob=False,
        capture_hidden_mode=CaptureHiddenMode.LAST,
        has_grammar=False,
    ):
        return ForwardBatch(
            forward_mode=forward_mode,
            batch_size=2,
            input_ids=self.input_ids,
            req_pool_indices=torch.arange(2),
            seq_lens=torch.tensor([2, 3]),
            out_cache_loc=torch.arange(5),
            seq_lens_sum=5,
            extend_seq_lens=torch.tensor([2, 3]),
            extend_seq_lens_cpu=[2, 3],
            extend_logprob_start_lens_cpu=[2, 3],
            return_logprob=return_logprob,
            is_prefill_only=is_prefill_only,
            capture_hidden_mode=capture_hidden_mode,
            top_logprobs_nums=[0, 0],
            token_ids_logprobs=[None, None],
            mm_input_embeds=self.mm_input_embeds,
            sampling_info=SimpleNamespace(
                grammars=[Mock()] if has_grammar else None
            ),
        )

    def _run(self, batch):
        # LayerNorm SP is independent of the LM-head decision under test.
        with patch(
            "sglang.srt.layers.logits_processor.layernorm_sp.maybe_exit_gather",
            side_effect=lambda **kw: (
                kw["hidden_states"],
                kw["hidden_states_before_norm"],
            ),
        ):
            return self.processor.forward(
                self.input_ids, self.hidden_states, Mock(), batch
            )

    def test_last_hidden_only_prefill_returns_each_last_row_without_projection(self):
        output = self._run(self._forward_batch())

        self.assertIsNone(output.next_token_logits)
        torch.testing.assert_close(output.hidden_states, self.hidden_states[[1, 4]])
        self.assertIs(output.mm_input_embeds, self.mm_input_embeds)
        self.processor._get_logits.assert_not_called()

    def test_logprob_request_still_projects_when_no_input_logprobs_are_needed(self):
        batch = self._forward_batch(return_logprob=True)
        metadata = LogitsMetadata.from_forward_batch(batch)
        self.assertFalse(metadata.extend_return_logprob)
        self.assertTrue(metadata.return_logprob)

        output = self._run(batch)

        self.assertIs(output.next_token_logits, self.projected_logits)
        self.processor._get_logits.assert_called_once()

    def test_non_matching_batches_keep_the_lm_head(self):
        for name, overrides in (
            ("generating", {"is_prefill_only": False}),
            (
                "full_hidden_capture",
                {"capture_hidden_mode": CaptureHiddenMode.FULL},
            ),
            (
                "no_hidden_capture",
                {"capture_hidden_mode": CaptureHiddenMode.NULL},
            ),
            ("mixed_forward", {"forward_mode": ForwardMode.MIXED}),
            ("grammar", {"has_grammar": True}),
        ):
            with self.subTest(name=name):
                self.processor._get_logits.reset_mock()
                output = self._run(self._forward_batch(**overrides))

                self.assertIs(output.next_token_logits, self.projected_logits)
                self.processor._get_logits.assert_called_once()


if __name__ == "__main__":
    unittest.main()
