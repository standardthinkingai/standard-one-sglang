"""Small, fixed GPU readout for prefill-only decision serving.

The artifact is deliberately independent of the model checkpoint loader: it
contains only the selected LM-head rows or a learned linear classifier.  It is
loaded once while SGLang constructs the model on its target device.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# v2 adds the answer-label scheme and the tokenizer the labels were chosen
# with. A v1 artifact predates label schemes and always has the served order.
_FORMAT = "jev-gpu-readout-v2"
_LEGACY_FORMAT = "jev-gpu-readout-v1"
_LABEL_SCHEMES = ("served", "upper")
_LABEL_KEYS = frozenset({"label_scheme", "tokenizer_id", "tokenizer_revision"})
_UPPER_LABEL_RE = re.compile(r"[A-Z]{1,3}\Z")
_ARCHITECTURE = "Mistral3ForConditionalGeneration"
_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_REQUIRED_METADATA_KEYS = frozenset(
    {
        "format",
        "mode",
        "model_id",
        "engine_model_path",
        "revision",
        "architecture",
        "hidden_size",
        "labels",
        "token_ids",
        "temperature",
    }
)


def _decode_metadata(value: np.ndarray) -> dict[str, Any]:
    # np.bytes_(json_bytes) is the canonical exporter representation.  A byte
    # vector is accepted as well so an artifact can be written without NumPy's
    # fixed-width string type.
    if value.ndim == 0 and value.dtype.kind == "S":
        raw = bytes(value.item())
    elif value.ndim == 1 and value.dtype == np.uint8:
        raw = value.tobytes()
    else:
        raise ValueError("decision readout metadata must contain JSON bytes")
    try:
        metadata = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("decision readout metadata is not valid UTF-8 JSON") from exc
    if not isinstance(metadata, dict) or metadata.get("format") not in (
        _FORMAT,
        _LEGACY_FORMAT,
    ):
        raise ValueError(
            f"decision readout format must be {_FORMAT} (or {_LEGACY_FORMAT})"
        )
    required = _REQUIRED_METADATA_KEYS | (
        _LABEL_KEYS if metadata["format"] == _FORMAT else frozenset()
    )
    if not required.issubset(metadata) or set(metadata) - required - {"source_sha256"}:
        raise ValueError(
            f"decision readout metadata keys do not match the {metadata['format']} format"
        )
    return metadata


def readout_label_scheme(metadata: dict[str, Any]) -> str:
    """The answer-label scheme of an artifact (v1 artifacts are 'served')."""
    if metadata.get("format") == _LEGACY_FORMAT:
        return "served"
    scheme = metadata.get("label_scheme")
    if scheme not in _LABEL_SCHEMES:
        raise ValueError(
            f"decision readout label_scheme must be one of {', '.join(_LABEL_SCHEMES)}"
        )
    return scheme


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    engine_model_path: str,
    revision: str | None,
    hidden_size: int,
    vocab_size: int,
    expected_label_scheme: str | None = None,
) -> tuple[str, int, float, str]:
    if metadata["format"] not in (_FORMAT, _LEGACY_FORMAT):
        raise ValueError(f"decision readout format must be {_FORMAT}")
    mode = metadata["mode"]
    if mode not in ("lm_rows", "learned"):
        raise ValueError("decision readout mode must be lm_rows or learned")
    if metadata["architecture"] != _ARCHITECTURE:
        raise ValueError("decision readout architecture must be Mistral3")
    if "source_sha256" in metadata and (
        not isinstance(metadata["source_sha256"], str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", metadata["source_sha256"]) is None
    ):
        raise ValueError("decision readout source_sha256 must be a 64-digit hex digest")
    if not isinstance(metadata["model_id"], str) or not metadata["model_id"]:
        raise ValueError("decision readout model_id must be a nonempty string")
    if metadata["engine_model_path"] != engine_model_path:
        raise ValueError(
            "decision readout engine_model_path differs from the server model"
        )
    artifact_revision = metadata["revision"]
    if (
        not isinstance(artifact_revision, str)
        or _REVISION_RE.fullmatch(artifact_revision) is None
        or artifact_revision != revision
    ):
        raise ValueError(
            "decision readout revision must match the pinned server commit"
        )
    artifact_hidden_size = metadata["hidden_size"]
    if type(artifact_hidden_size) is not int or artifact_hidden_size != hidden_size:
        raise ValueError("decision readout hidden_size differs from the model")
    labels = metadata["labels"]
    token_ids = metadata["token_ids"]
    if not isinstance(labels, list) or not 2 <= len(labels) <= 255:
        raise ValueError("decision readout requires 2 to 255 labels")
    if not all(isinstance(label, str) and label for label in labels):
        raise ValueError("decision readout labels must be nonempty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("decision readout labels must be unique")
    if not isinstance(token_ids, list) or len(token_ids) != len(labels):
        raise ValueError("decision readout token_ids must match the label count")
    if not all(
        type(token_id) is int and 0 <= token_id < vocab_size
        for token_id in token_ids
    ):
        raise ValueError("decision readout token_ids must be valid vocabulary IDs")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("decision readout token_ids must be unique")
    label_scheme = readout_label_scheme(metadata)
    if metadata["format"] == _FORMAT:
        tokenizers = (metadata["model_id"], metadata["engine_model_path"])
        if (
            metadata["tokenizer_id"] not in tokenizers
            or metadata["tokenizer_revision"] != artifact_revision
        ):
            raise ValueError("decision readout tokenizer must be the pinned model's")
        # Uppercase only, by length then alphabetically; skipped candidates are
        # allowed (not single tokens), reordering or lowercase is not.
        ranks = [(len(label), label) for label in labels]
        if label_scheme == "upper" and (
            any(_UPPER_LABEL_RE.match(label) is None for label in labels)
            or any(before >= after for before, after in itertools.pairwise(ranks))
        ):
            raise ValueError("decision readout labels do not follow the upper scheme")
    if expected_label_scheme is not None and label_scheme != expected_label_scheme:
        raise ValueError(
            f"decision readout labels use the {label_scheme!r} scheme but the "
            f"server is configured for {expected_label_scheme!r} "
            "(--decision-label-scheme / JEV_LABEL_SCHEME); export a readout with "
            "the configured scheme or configure the artifact's scheme"
        )
    temperature = metadata["temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("decision readout temperature must be finite and positive")
    return mode, len(labels), float(temperature), label_scheme


class DecisionReadout(nn.Module):
    """Project one last hidden state per request onto a fixed set of labels."""

    def __init__(
        self,
        *,
        weights: np.ndarray,
        bias: np.ndarray,
        mode: str,
        temperature: float,
        model_dtype: torch.dtype,
        projection_width: int = 0,
        token_ids: list[int] | None = None,
        label_scheme: str | None = None,
    ) -> None:
        super().__init__()
        self.label_scheme = label_scheme
        if model_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                "decision readout requires an unquantized BF16 or FP16 model"
            )
        if type(projection_width) is not int or not 0 <= projection_width <= 65536:
            raise ValueError("decision readout projection_width must be an integer in [0, 65536]")
        zero_bias_lm_rows = mode == "lm_rows" and not np.any(bias)
        if projection_width:
            if not zero_bias_lm_rows or model_dtype != torch.bfloat16:
                raise ValueError(
                    "decision readout projection_width requires zero-bias BF16 LM rows"
                )
            if weights.ndim != 2 or projection_width < weights.shape[0]:
                raise ValueError(
                    "decision readout projection_width must cover all selected rows"
                )
        dtype = model_dtype if mode == "lm_rows" else torch.float32
        # Model construction runs inside SGLang's target-device context.  An
        # explicit device avoids torch.from_numpy silently retaining CPU data.
        device = torch.empty(0).device
        if device.type not in ("cuda", "cpu"):
            raise ValueError(f"decision readout does not support device {device}")
        self.register_buffer(
            "weights", torch.tensor(weights, dtype=dtype, device=device)
        )
        self.register_buffer("bias", torch.tensor(bias, dtype=dtype, device=device))
        # Vocabulary IDs of the labels, for their full-vocabulary logprobs.
        self.register_buffer(
            "token_ids",
            None
            if token_ids is None
            else torch.tensor(token_ids, dtype=torch.long, device=device),
            persistent=False,
        )
        projection_weights = None
        if projection_width:
            # Allocate and populate once during construction. The original K
            # rows remain available in weights; serving only selects a view of
            # the padded projection's first K output columns.
            projection_weights = torch.zeros(
                (projection_width, self.weights.shape[1]), dtype=dtype, device=device
            )
            projection_weights[: self.weights.shape[0]].copy_(self.weights)
        self.register_buffer("projection_weights", projection_weights)
        self.projection_width = projection_width
        self.mode = mode
        self.temperature = temperature
        # A zero bias selects addmm rather than matmul in F.linear. On H200,
        # that narrow BF16 addmm differed from the full LM head for batched
        # inputs. Select the bias-free path once, without a device sync during
        # serving; nonzero artifacts and learned heads retain their arithmetic.
        self._zero_bias_lm_rows = zero_bias_lm_rows

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        engine_model_path: str,
        revision: str | None,
        hidden_size: int,
        vocab_size: int,
        model_dtype: torch.dtype,
        projection_width: int = 0,
        expected_label_scheme: str | None = None,
    ) -> DecisionReadout:
        try:
            with np.load(path, allow_pickle=False) as artifact:
                if set(artifact.files) != {"metadata", "weights", "bias"}:
                    raise ValueError(
                        "decision readout NPZ must contain metadata, weights, bias"
                    )
                metadata = _decode_metadata(artifact["metadata"])
                mode, count, temperature, label_scheme = _validate_metadata(
                    metadata,
                    engine_model_path=engine_model_path,
                    revision=revision,
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    expected_label_scheme=expected_label_scheme,
                )
                weights = artifact["weights"]
                bias = artifact["bias"]
        except (OSError, EOFError) as exc:
            raise ValueError(f"cannot open decision readout NPZ: {path}") from exc
        if weights.dtype != np.float32 or weights.shape != (count, hidden_size):
            raise ValueError("decision readout weights must be float32[K, hidden_size]")
        if bias.dtype != np.float32 or bias.shape != (count,):
            raise ValueError("decision readout bias must be float32[K]")
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ValueError("decision readout weights and bias must be finite")
        return cls(
            weights=weights,
            bias=bias,
            mode=mode,
            temperature=temperature,
            model_dtype=model_dtype,
            projection_width=projection_width,
            token_ids=metadata["token_ids"],
            label_scheme=label_scheme,
        )

    def forward(
        self,
        last_hidden_states: torch.Tensor,
        *,
        logit_scale: float | None = None,
        softcap: float | None = None,
    ) -> torch.Tensor:
        if (
            last_hidden_states.ndim != 2
            or last_hidden_states.shape[1] != self.weights.shape[1]
        ):
            raise ValueError("decision readout needs one last hidden state per request")
        if last_hidden_states.device != self.weights.device:
            raise RuntimeError(
                "decision readout weights and hidden states must share a device"
            )
        if self.projection_weights is None:
            selected_logits = F.linear(
                last_hidden_states.to(dtype=self.weights.dtype),
                self.weights,
                None if self._zero_bias_lm_rows else self.bias,
            ).float()
        else:
            selected_logits = F.linear(
                last_hidden_states.to(dtype=self.weights.dtype),
                self.projection_weights,
                None,
            )[:, : self.weights.shape[0]].float()
        if self.mode == "lm_rows":
            if logit_scale is not None:
                selected_logits.mul_(logit_scale)
            if softcap:
                selected_logits = softcap * torch.tanh(selected_logits / softcap)
        if self.temperature != 1.0:
            selected_logits = selected_logits / self.temperature
        return selected_logits
