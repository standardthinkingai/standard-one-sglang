"""Logits-processor routing for the fixed-label decision readout."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.layers.decision_readout import DecisionReadout
from sglang.srt.layers.decision_schema_head import _segment_sum
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDecisionLogitsProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = LogitsProcessor.__new__(LogitsProcessor)
        torch.nn.Module.__init__(self.processor)
        self.processor._decision_native_fp8 = False
        self.processor._decision_fp8_head_validated = False
        self.processor.decision_readout = DecisionReadout(
            weights=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
            bias=np.zeros(2, dtype=np.float32),
            mode="learned",
            temperature=1.0,
            model_dtype=torch.bfloat16,
            token_ids=[3, 1],
        )
        self.processor.logit_scale = None
        self.processor.final_logit_softcapping = None
        self.vocab_logits = torch.tensor(
            [[0.5, 2.0, -1.0, 1.0, 0.0], [1.5, -2.0, 0.25, 3.0, 1.0]]
        )
        self.processor._get_logits = Mock(return_value=self.vocab_logits)
        self.processor._get_hidden_states_to_store = Mock()
        self.hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    def _run(self, *, mode=ForwardMode.EXTEND, is_prefill_only=True):
        metadata = LogitsMetadata(
            forward_mode=mode,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            extend_seq_lens=torch.tensor([2, 3]),
            is_prefill_only=is_prefill_only,
        )
        with patch(
            "sglang.srt.layers.logits_processor.layernorm_sp.maybe_exit_gather",
            side_effect=lambda **kw: (
                kw["hidden_states"],
                kw["hidden_states_before_norm"],
            ),
        ):
            return self.processor.forward(
                torch.arange(5), self.hidden, Mock(), metadata
            )

    def test_prefill_projects_only_last_hidden_rows(self):
        output = self._run()

        self.assertIsNone(output.next_token_logits)
        self.assertIsNone(output.hidden_states)
        # Readout logits, then the labels' full-vocabulary logprobs (stock
        # token_ids_logprob semantics) for options.return_logprobs.
        torch.testing.assert_close(
            output.decision_logits,
            torch.cat(
                [
                    self.hidden[[1, 4], :2],
                    torch.log_softmax(self.vocab_logits, -1)[:, [3, 1]],
                ],
                dim=1,
            ),
        )
        (rows, *_), kwargs = self.processor._get_logits.call_args
        torch.testing.assert_close(rows, self.hidden[[1, 4]])
        self.assertFalse(kwargs["use_logits_buffer"])
        self.processor._get_hidden_states_to_store.assert_not_called()

    def test_decode_and_generating_requests_fail_closed(self):
        for mode, prefill_only in (
            (ForwardMode.DECODE, True),
            (ForwardMode.EXTEND, False),
        ):
            with self.subTest(mode=mode, prefill_only=prefill_only):
                with self.assertRaisesRegex(ValueError, "prefill-only EXTEND"):
                    self._run(mode=mode, is_prefill_only=prefill_only)
        self.processor._get_logits.assert_not_called()


if __name__ == "__main__":
    unittest.main()


# --- schema head pooling: two passes over one request must be bit-identical ---


def _case(seed: int, n_nodes: int = 7, width: int = 5):
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(1, 40, (n_nodes,), generator=g)
    segment = torch.repeat_interleave(torch.arange(n_nodes), counts)
    values = torch.randn(int(counts.sum()), width, generator=g)
    return values, segment


def test_segment_sum_matches_index_add_per_node():
    values, segment = _case(1)
    expected = torch.zeros(7, 5).index_add(0, segment, values)
    torch.testing.assert_close(_segment_sum(values, segment, 7), expected)


def test_segment_sum_handles_one_dimensional_values_and_empty_nodes():
    values, segment = _case(2, n_nodes=4)
    # Node 5 and 6 own no tokens: their rows stay zero, as with index_add.
    expected = torch.zeros(7).index_add(0, segment, values[:, 0])
    torch.testing.assert_close(_segment_sum(values[:, 0], segment, 7), expected)


def test_segment_sum_long_segment_keeps_float32_digits():
    # One 20k-token state-like node plus small nodes; a float32 scan would drift.
    big = torch.full((20000, 3), 0.1) + torch.linspace(0, 1, 20000)[:, None] * 1e-3
    small = torch.randn(30, 3, generator=torch.Generator().manual_seed(3))
    values = torch.cat([big, small])
    segment = torch.cat([torch.zeros(20000, dtype=torch.long), torch.ones(30, dtype=torch.long)])
    expected = torch.stack([big.double().sum(0), small.double().sum(0)]).float()
    torch.testing.assert_close(_segment_sum(values, segment, 2), expected, rtol=1e-6, atol=1e-6)
