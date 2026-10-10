"""Artifact and arithmetic tests without importing the SGLang package."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/layers/decision_readout.py"
SPEC = importlib.util.spec_from_file_location("decision_readout_under_test", SOURCE)
readout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readout)

REVISION = "a" * 40
MODEL_PATH = "mistralai/Ministral-3-3B-Instruct-2512"


def _metadata(**updates):
    metadata = {
        "format": "jev-gpu-readout-v1",
        "mode": "lm_rows",
        "model_id": MODEL_PATH,
        "engine_model_path": MODEL_PATH,
        "revision": REVISION,
        "architecture": "Mistral3ForConditionalGeneration",
        "hidden_size": 4,
        "labels": ["yes", "no"],
        "token_ids": [1, 2],
        "temperature": 2.0,
    }
    metadata.update(updates)
    return metadata


class TestDecisionReadout(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / "readout.npz"
        self.weights = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.bias = np.array([0.5, -0.5], dtype=np.float32)

    def _write(self, metadata=None, weights=None, bias=None):
        np.savez(
            self.path,
            metadata=np.bytes_(json.dumps(metadata or _metadata())),
            weights=self.weights if weights is None else weights,
            bias=self.bias if bias is None else bias,
        )

    def _load(self, *, model_dtype=torch.bfloat16, projection_width=0):
        return readout.DecisionReadout.from_npz(
            self.path,
            engine_model_path=MODEL_PATH,
            revision=REVISION,
            hidden_size=4,
            vocab_size=8,
            model_dtype=model_dtype,
            projection_width=projection_width,
        )

    def test_lm_rows_loads_selected_rows_and_applies_temperature(self):
        self._write()
        head = self._load()

        self.assertEqual(head.weights.dtype, torch.bfloat16)
        self.assertEqual(head.bias.dtype, torch.bfloat16)
        self.assertEqual(tuple(head(torch.tensor([[2.0, 4.0, 0, 0]])).shape), (1, 2))
        torch.testing.assert_close(
            head(torch.tensor([[2.0, 4.0, 0, 0]])).float(),
            torch.tensor([[1.25, 1.75]]),
            atol=0.01,
            rtol=0,
        )

    def test_learned_head_uses_fp32_hidden_and_weights(self):
        self._write(_metadata(mode="learned"))
        head = self._load()

        self.assertEqual(head.weights.dtype, torch.float32)
        output = head(torch.tensor([[2.0, 4.0, 0, 0]], dtype=torch.bfloat16))
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, torch.tensor([[1.25, 1.75]]))

    def test_only_zero_bias_lm_rows_uses_bias_free_projection(self):
        # This validates routing, not the GPU rounding regression. Real H200
        # hidden-state captures exercise that separately.
        hidden = torch.tensor([[2.0, 4.0, 0, 0]], dtype=torch.bfloat16)
        linear = torch.nn.functional.linear
        for mode in ("lm_rows", "learned"):
            for bias in (np.zeros(2, dtype=np.float32), self.bias):
                with self.subTest(mode=mode, bias=bias.tolist()):
                    self._write(_metadata(mode=mode), bias=bias)
                    head = self._load()
                    with patch.object(readout.F, "linear", wraps=linear) as call:
                        output = head(hidden)
                    passed_bias = call.call_args.args[2]
                    if mode == "lm_rows" and not np.any(bias):
                        self.assertIsNone(passed_bias)
                    else:
                        self.assertIs(passed_bias, head.bias)
                    expected = linear(
                        hidden.to(head.weights.dtype), head.weights, head.bias
                    ).float() / head.temperature
                    torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_nonzero_artifact_bias_is_not_discarded_after_dtype_rounding(self):
        # Even a value that rounds to BF16 zero keeps the artifact's existing
        # biased projection route; the zero-bias check uses the original data.
        self._write(bias=np.array([1e-42, 0], dtype=np.float32))
        head = self._load()
        with patch.object(readout.F, "linear", wraps=torch.nn.functional.linear) as call:
            head(torch.tensor([[2.0, 4.0, 0, 0]], dtype=torch.bfloat16))
        self.assertIs(call.call_args.args[2], head.bias)

    def test_projection_width_preserves_original_buffers_and_reuses_padding(self):
        # CPU checks geometry and routing only, not GPU rounding equivalence.
        self._write(bias=np.zeros(2, dtype=np.float32))
        head = self._load(projection_width=8)
        self.assertEqual(head.projection_width, 8)
        self.assertEqual(tuple(head.weights.shape), (2, 4))
        self.assertEqual(tuple(head.bias.shape), (2,))
        self.assertEqual(tuple(head.projection_weights.shape), (8, 4))
        self.assertIs(dict(head.named_buffers())["projection_weights"], head.projection_weights)
        torch.testing.assert_close(head.weights.float(), torch.tensor(self.weights))
        torch.testing.assert_close(head.projection_weights[:2], head.weights)
        self.assertEqual(torch.count_nonzero(head.projection_weights[2:]).item(), 0)

        padded_buffer = head.projection_weights
        padded_pointer = padded_buffer.data_ptr()
        hidden = torch.tensor([[2.0, 4.0, 0, 0], [6.0, 8.0, 0, 0]])
        for _ in range(2):
            with patch.object(readout.F, "linear", wraps=torch.nn.functional.linear) as call:
                output = head(hidden)
            self.assertIs(call.call_args.args[1], padded_buffer)
            self.assertIsNone(call.call_args.args[2])
            self.assertEqual(tuple(output.shape), (2, 2))
            self.assertEqual(output.dtype, torch.float32)
            self.assertEqual(head.projection_weights.data_ptr(), padded_pointer)
            torch.testing.assert_close(output, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_projection_width_slices_before_scaling_softcap_and_temperature(self):
        self._write(bias=np.zeros(2, dtype=np.float32))
        head = self._load(projection_width=8)
        output = head(
            torch.tensor([[2.0, 4.0, 0, 0]]), logit_scale=1.5, softcap=2.0
        )
        expected = 2.0 * torch.tanh(torch.tensor([[2.0, 4.0]]) * 1.5 / 2.0) / 2.0
        self.assertEqual(tuple(output.shape), (1, 2))
        torch.testing.assert_close(output, expected)

    def test_projection_width_rejects_invalid_values_and_accepts_bounds(self):
        self._write(bias=np.zeros(2, dtype=np.float32))
        for width in (-1, True, False, 1.0, None, 1, 65537):
            with self.subTest(width=width), self.assertRaises(ValueError):
                self._load(projection_width=width)
        for width in (2, 65536):
            with self.subTest(width=width):
                head = self._load(projection_width=width)
                self.assertEqual(tuple(head.projection_weights.shape), (width, 4))
        head = self._load()
        self.assertIsNone(head.projection_weights)
        self.assertEqual(set(head.state_dict()), {"weights", "bias"})

    def test_projection_width_rejects_unsupported_heads_without_changing_defaults(self):
        for mode, bias, dtype in (
            ("learned", np.zeros(2, dtype=np.float32), torch.bfloat16),
            ("lm_rows", self.bias, torch.bfloat16),
            ("lm_rows", np.array([1e-42, 0], dtype=np.float32), torch.bfloat16),
            ("lm_rows", np.zeros(2, dtype=np.float32), torch.float16),
        ):
            with self.subTest(mode=mode, bias=bias.tolist(), dtype=dtype):
                self._write(_metadata(mode=mode), bias=bias)
                with self.assertRaises(ValueError):
                    self._load(model_dtype=dtype, projection_width=8)
                head = self._load(model_dtype=dtype)
                self.assertIsNone(head.projection_weights)
                with patch.object(readout.F, "linear", wraps=torch.nn.functional.linear) as call:
                    head(torch.tensor([[2.0, 4.0, 0, 0]]))
                self.assertIs(call.call_args.args[1], head.weights)

    def test_rejects_mismatched_model_or_revision(self):
        for name, updates in (
            ("path", {"engine_model_path": "other/model"}),
            ("revision", {"revision": "b" * 40}),
            ("architecture", {"architecture": "OtherForCausalLM"}),
            ("hidden_size", {"hidden_size": 8}),
        ):
            with self.subTest(name=name):
                self._write(_metadata(**updates))
                with self.assertRaises(ValueError):
                    self._load()

    def test_rejects_duplicate_tokens_and_nonfinite_weights(self):
        self._write(_metadata(token_ids=[1, 1]))
        with self.assertRaises(ValueError):
            self._load()

        bad_weights = self.weights.copy()
        bad_weights[0, 0] = np.nan
        self._write(weights=bad_weights)
        with self.assertRaises(ValueError):
            self._load()

    def _v2(self, **updates):
        values = {
            "format": "jev-gpu-readout-v2",
            "label_scheme": "upper",
            "tokenizer_id": MODEL_PATH,
            "tokenizer_revision": REVISION,
            "labels": ["A", "B"],
        }
        values.update(updates)
        return _metadata(**values)

    def test_v2_artifact_records_its_label_scheme(self):
        for scheme, labels in (("upper", ["A", "B"]), ("served", ["A", "B"])):
            with self.subTest(scheme=scheme):
                self._write(self._v2(label_scheme=scheme, labels=labels))
                self.assertEqual(self._load().label_scheme, scheme)
        # A v1 artifact predates schemes and always carries the served order.
        self._write()
        self.assertEqual(self._load().label_scheme, "served")

    def _load_expecting(self, scheme):
        return readout.DecisionReadout.from_npz(
            self.path,
            engine_model_path=MODEL_PATH,
            revision=REVISION,
            hidden_size=4,
            vocab_size=8,
            model_dtype=torch.bfloat16,
            expected_label_scheme=scheme,
        )

    def test_configured_scheme_must_match_the_artifact(self):
        for metadata, expected, accepted in (
            (self._v2(), "upper", True),
            (self._v2(label_scheme="served"), "served", True),
            (_metadata(), "served", True),
            (_metadata(), "upper", False),
            (self._v2(label_scheme="served"), "upper", False),
            (self._v2(), "served", False),
        ):
            with self.subTest(metadata=metadata["format"], expected=expected):
                self._write(metadata)
                if accepted:
                    self.assertEqual(
                        self._load_expecting(expected).label_scheme, expected
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "scheme"):
                        self._load_expecting(expected)

    def test_v2_rejects_missing_or_inconsistent_label_metadata(self):
        missing = self._v2()
        del missing["label_scheme"]
        for name, metadata in (
            ("missing", missing),
            ("unknown scheme", self._v2(label_scheme="lower")),
            ("lowercase upper label", self._v2(labels=["A", "a"])),
            ("reordered upper labels", self._v2(labels=["B", "A"])),
            ("longer before shorter", self._v2(labels=["AA", "Z"])),
            ("tokenizer revision", self._v2(tokenizer_revision="b" * 40)),
            ("tokenizer id", self._v2(tokenizer_id="other/tokenizer")),
            ("scheme in v1", _metadata(label_scheme="upper")),
        ):
            with self.subTest(name=name):
                self._write(metadata)
                with self.assertRaises(ValueError):
                    self._load()

    def test_optional_source_sha256_and_softcap_order(self):
        self._write(_metadata(source_sha256="f" * 64, temperature=2.0))
        head = self._load()
        logits = head(
            torch.tensor([[2.0, 4.0, 0, 0]]),
            logit_scale=1.5,
            softcap=2.0,
        )
        raw = torch.tensor([[2.5, 3.5]])
        expected = 2.0 * torch.tanh(raw * 1.5 / 2.0) / 2.0
        torch.testing.assert_close(logits, expected, atol=0.01, rtol=0)


if __name__ == "__main__":
    unittest.main()
