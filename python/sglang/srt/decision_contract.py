"""Pure validation for the dedicated, zero-output GPU decision worker.

This module deliberately imports no SGLang runtime objects. The server
resolution hook and tokenizer manager call it before GPU initialization and
before request normalization respectively.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from typing import Any

DECISION_WEIGHT_UPDATE_MESSAGE = (
    "Live model weight updates are unsupported with --decision-readout-path; "
    "restart the engine with a matching model and readout artifact."
)

# This experimental path converts these immutable BF16 checkpoints at load time.
# A prequantized checkpoint does not establish the provenance of the saved LM rows.
DECISION_FP8_CHECKPOINTS = {
    "mistralai/Ministral-3-3B-Instruct-2512-BF16": (
        "b6d637bef2393152b3da2b2fde72eecdee30557e"
    ),
    "mistralai/Ministral-3-8B-Instruct-2512-BF16": (
        "f6fae9795746f63c9be8344932f01275f3c63734"
    ),
}


# Answer labels beyond Z: "upper" (AA, AB, ...; the default) or the original
# "served" order (a-z, 0-9, then AA, ...). Kept in step with jev_adapter.labels.
DECISION_LABEL_SCHEMES = ("served", "upper")
DEFAULT_DECISION_LABEL_SCHEME = "upper"
LABEL_SCHEME_ENV = "JEV_LABEL_SCHEME"


def resolve_decision_label_scheme(
    value: str | None, environ: Mapping[str, str] | None = None
) -> str:
    """--decision-label-scheme, else JEV_LABEL_SCHEME, else upper."""
    if not value:
        environ = os.environ if environ is None else environ
        value = environ.get(LABEL_SCHEME_ENV) or DEFAULT_DECISION_LABEL_SCHEME
    if value not in DECISION_LABEL_SCHEMES:
        raise ValueError(
            "decision label scheme must be one of "
            f"{', '.join(DECISION_LABEL_SCHEMES)}; got {value!r}"
        )
    return value


def decision_projection_width(value: str | None) -> int:
    """Parse the worker's init-only experimental projection setting."""
    if value is None:
        return 0
    if value not in ("0", "8192"):
        raise ValueError("SGLANG_DECISION_PROJECTION_WIDTH must be exactly 0 or 8192")
    return int(value)


# The only image form a decision worker loads. SGLang's loader would otherwise
# fetch URLs and read local paths named by the caller.
_DECISION_IMAGE_DATA_URL = re.compile(
    r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}"
)


# --decision-readout-path names either a jev-gpu-readout-v1 NPZ or a directory
# holding a joint schema head (layers/decision_schema_head.py).
SCHEMA_HEAD_CONFIG_NAME = "schema_head_config.json"
_SCHEMA_QTYPE_OPTIONS = {"choice": (2, 255), "noul": (2, 2), "score": (2, 10)}
_SCHEMA_MAX_QUESTIONS = 256


def is_schema_head_artifact(path: Any) -> bool:
    return isinstance(path, str) and os.path.isfile(
        os.path.join(path, SCHEMA_HEAD_CONFIG_NAME)
    )


def _schema_span(span: Any, n_tokens: int) -> tuple:
    if (
        not isinstance(span, (list, tuple))
        or len(span) != 2
        or any(type(value) is not int for value in span)
        or not 0 <= span[0] < span[1] <= n_tokens
    ):
        raise ValueError("decision schema spans must be nonempty [start, end) token ranges")
    return tuple(span)


# Synthetic prompt for the server's own warmup and health probes on a schema
# worker: ordinary text tokens, no image marker, one true/false question.
SCHEMA_PROBE_INPUT_IDS = [1000, 1001, 1002, 1003, 1004, 1005]


def decision_schema_probe() -> dict:
    """Plan for SCHEMA_PROBE_INPUT_IDS: question, two options, then the state."""
    return {
        "template": "schema-v1",
        "n_tokens": len(SCHEMA_PROBE_INPUT_IDS),
        "qtype": ["noul"],
        "p_span": [None],
        "q_span": [[0, 1]],
        "o_span": [[[1, 2], [2, 3]]],
        "state_spans": [[3, len(SCHEMA_PROBE_INPUT_IDS)]],
        "image_span": None,
        "n_values": 2,
    }


def validate_decision_schema_plan(plan: Any) -> None:
    """Structure of one request's schema plan, before any tokenization or images."""
    if not isinstance(plan, dict):
        raise ValueError("decision_schema must be an object")  # noqa: TRY004
    n_tokens = plan.get("n_tokens")
    if type(n_tokens) is not int or n_tokens < 2:
        raise ValueError("decision_schema.n_tokens must be a positive token count")
    qtypes = plan.get("qtype")
    if not isinstance(qtypes, list) or not 1 <= len(qtypes) <= _SCHEMA_MAX_QUESTIONS:
        raise ValueError("decision_schema needs 1 to 256 questions")
    for name in ("q_span", "p_span", "o_span"):
        if not isinstance(plan.get(name), list) or len(plan[name]) != len(qtypes):
            raise ValueError(f"decision_schema.{name} needs one entry per question")
    values = 0
    for qtype, q_span, p_span, o_spans in zip(
        qtypes, plan["q_span"], plan["p_span"], plan["o_span"]
    ):
        if qtype not in _SCHEMA_QTYPE_OPTIONS:
            raise ValueError("decision_schema question types are choice, noul or score")
        _schema_span(q_span, n_tokens)
        if p_span is not None:
            _schema_span(p_span, n_tokens)
        low, high = _SCHEMA_QTYPE_OPTIONS[qtype]
        if not isinstance(o_spans, list) or not low <= len(o_spans) <= high:
            raise ValueError(f"decision_schema {qtype} question has a bad option count")
        for span in o_spans:
            _schema_span(span, n_tokens)
        values += len(o_spans)
    if plan.get("n_values") != values:
        raise ValueError("decision_schema.n_values must equal the total option count")
    state_spans = plan.get("state_spans")
    if not isinstance(state_spans, list) or not state_spans:
        raise ValueError("decision_schema needs the state token spans")
    for span in state_spans:
        _schema_span(span, n_tokens)
    image_span = plan.get("image_span")
    if image_span is not None and (
        _schema_span(image_span, n_tokens) != tuple(state_spans[0])
    ):
        raise ValueError("decision_schema image span must be the first state span")


def validate_decision_schema_tokens(plan: dict, *, input_ids: Any, mm_inputs: Any) -> None:
    """The plan's coordinates against the prompt as the processor expanded it."""
    if not isinstance(input_ids, list) or len(input_ids) != plan["n_tokens"]:
        raise ValueError("decision_schema.n_tokens differs from the processed prompt")
    items = _get(mm_inputs, "mm_items") or []
    if not items:
        if plan.get("image_span") is not None:
            raise ValueError("decision_schema has an image span but no images")
        return
    # Each image is rows of [IMG] separated by [IMG_BREAK] and closed by
    # [IMG_END]; the block runs from its first row to the closing marker.
    first = _get(items[0], "offsets")[0][0]
    last = _get(items[-1], "offsets")[-1][1] + 2
    if plan.get("image_span") is None or tuple(plan["image_span"]) != (first, last):
        raise ValueError("decision_schema image span differs from the processed images")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return (
        obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    )


def validate_decision_server(cfg: Any) -> None:
    """Reject resolved server settings that could enter another inference path."""
    path = _get(cfg, "decision_readout_path")
    if path is None:
        for name in (
            "decision_deadline_seconds",
            "decision_max_queued_evaluations",
            "decision_label_scheme",
        ):
            if _get(cfg, name) is not None:
                raise ValueError(f"{name} requires --decision-readout-path")
        return
    scheme = _get(cfg, "decision_label_scheme")
    if scheme is not None and scheme not in DECISION_LABEL_SCHEMES:
        raise ValueError(
            "decision_label_scheme must be one of " + ", ".join(DECISION_LABEL_SCHEMES)
        )
    if not isinstance(path, str) or not path.strip():
        raise ValueError("decision_readout_path must be a nonempty artifact path")
    for name in ("tp_size", "pp_size", "dp_size"):
        value = _get(cfg, name)
        if type(value) is not int or value != 1:
            raise ValueError(f"decision readout requires {name}=1")
    for name in (
        "is_embedding",
        "skip_tokenizer_init",
        "enable_return_hidden_states",
        "enable_custom_logit_processor",
        "enable_mis",
    ):
        if _get(cfg, name) is not False:
            raise ValueError(f"decision readout requires {name}=False")
    # The untouched SGLang default is None; its runtime meaning is disabled.
    if _get(cfg, "enable_lora") not in (None, False):
        raise ValueError("decision readout requires enable_lora=False")
    quantization = _get(cfg, "quantization")
    if quantization not in (None, "fp8"):
        raise ValueError("decision readout supports quantization=None or native fp8")
    if quantization == "fp8":
        if _get(cfg, "dtype") != "bfloat16":
            raise ValueError("native FP8 decision readout requires dtype=bfloat16")
        if _get(cfg, "kv_cache_dtype") != "auto":
            raise ValueError("native FP8 decision readout requires kv_cache_dtype=auto")
        if _get(cfg, "prefill_only_disable_kv_cache", False):
            raise ValueError(
                "native FP8 decision readout does not support KV-free prefill"
            )
        if _get(cfg, "enable_fp32_lm_head", False) is not False:
            raise ValueError("native FP8 decision readout requires a BF16 LM head")
        model = _get(cfg, "model_path")
        if (
            model not in DECISION_FP8_CHECKPOINTS
            or _get(cfg, "revision") != (DECISION_FP8_CHECKPOINTS[model])
        ):
            raise ValueError(
                "native FP8 decision readout requires a pinned BF16 checkpoint"
            )
        for name in (
            "quantization_param_path",
            "modelopt_quant",
            "modelopt_checkpoint_restore_path",
            "modelopt_checkpoint_save_path",
            "modelopt_export_path",
        ):
            if _get(cfg, name) is not None:
                raise ValueError(f"native FP8 decision readout requires {name}=None")
        if _get(cfg, "quantize_and_serve", False) is not False:
            raise ValueError(
                "native FP8 decision readout requires quantize_and_serve=False"
            )
    for name in ("speculative_algorithm", "return_hidden_states_mode"):
        if _get(cfg, name) is not None:
            raise ValueError(f"decision readout requires {name}=None")
    if _get(cfg, "disaggregation_mode") != "null":
        raise ValueError("decision readout requires disaggregation_mode=null")
    for name in ("encoder_only", "language_only", "enable_prefix_mm_cache"):
        if _get(cfg, name, False) is not False:
            raise ValueError(f"decision readout requires {name}=False")
    chunk_size = _get(cfg, "chunked_prefill_size")
    if type(chunk_size) is not int or chunk_size > 0:
        raise ValueError("decision readout requires chunked_prefill_size<=0")
    if _get(cfg, "disable_radix_cache") is not True:
        raise ValueError("decision readout requires disable_radix_cache=True")
    # A truncated prompt would be decided, and billed, as if it were whole.
    if _get(cfg, "allow_auto_truncate", False) is not False:
        raise ValueError("decision readout requires allow_auto_truncate=False")
    deadline = _get(cfg, "decision_deadline_seconds")
    if deadline is not None and (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
        or deadline <= 0
    ):
        raise ValueError("decision_deadline_seconds must be finite and positive")
    queued = _get(cfg, "decision_max_queued_evaluations")
    if queued is not None and (type(queued) is not int or queued < 0):
        raise ValueError("decision_max_queued_evaluations must be a nonnegative integer")
    # The decision service keeps up to max_running_requests evaluations in the
    # scheduler; a smaller scheduler queue refuses some midway through a request.
    queued_requests = _get(cfg, "max_queued_requests")
    running = _get(cfg, "max_running_requests")
    if (
        type(queued_requests) is int
        and type(running) is int
        and queued_requests <= running
    ):
        raise ValueError(
            "decision readout bounds its queue with --decision-max-queued-evaluations; "
            "leave --max-queued-requests unset or above --max-running-requests"
        )
    decode = _get(_get(cfg, "cuda_graph_config"), "decode")
    if _get(decode, "backend") != "disabled":
        raise ValueError(
            "decision readout requires a disabled resolved decode CUDA graph "
            "(--disable-decode-cuda-graph)"
        )
    if _get(cfg, "prefill_only_disable_kv_cache", False):
        # The optional no-paged-KV implementation is presently the eager FA3
        # path for BF16 Ministral on Hopper. Keep capture and other warmup
        # workloads out until they have been checked with a no-op KV pool.
        if chunk_size != -1:
            raise ValueError(
                "decision KV-free prefill requires chunked_prefill_size=-1"
            )
        if _get(cfg, "dtype") != "bfloat16":
            raise ValueError("decision KV-free prefill requires dtype=bfloat16")
        if _get(cfg, "kv_cache_dtype") != "auto":
            raise ValueError("decision KV-free prefill requires kv_cache_dtype=auto")
        if _get(cfg, "enable_unified_memory") is not False:
            raise ValueError(
                "decision KV-free prefill requires enable_unified_memory=False"
            )
        prefill = _get(_get(cfg, "cuda_graph_config"), "prefill")
        if _get(prefill, "backend") != "disabled":
            raise ValueError(
                "decision KV-free prefill requires a disabled resolved prefill CUDA graph "
                "(--disable-prefill-cuda-graph)"
            )
        if _get(cfg, "warmups") is not None:
            raise ValueError("decision KV-free prefill does not support --warmups")


def validate_decision_model_quantization(
    *, explicit_quantization: str | None, model_config: Any, model_dtype: str
) -> bool:
    """Validate the resolved full config, including multimodal text metadata.

    Returns whether a first-forward native FP8 head check is required. The
    standalone HF quant config can be empty for an unquantized source; the actual
    Fp8Config is checked separately after weights have been loaded.
    """
    if explicit_quantization not in (None, "fp8"):
        raise ValueError("decision readout supports quantization=None or native fp8")
    if _get(model_config, "quantization") != explicit_quantization:
        raise ValueError(
            "decision readout rejects autodetected or mismatched quantization"
        )
    hf_config = _get(model_config, "hf_config")
    for config in (hf_config, _get(hf_config, "text_config")):
        for name in ("quantization_config", "compression_config"):
            if _get(config, name) is not None:
                raise ValueError(
                    "decision readout requires checkpoint quantization metadata absent"
                )
    if _get(model_config, "hf_quant_config", {}) != {}:
        raise ValueError(
            "decision readout requires standalone checkpoint quantization metadata absent"
        )
    if explicit_quantization is None:
        return False
    if model_dtype != "torch.bfloat16" or str(_get(model_config, "dtype")) != (
        "torch.bfloat16"
    ):
        raise ValueError(
            "native FP8 decision readout requires BF16 activations and LM rows"
        )
    if _get(hf_config, "architectures") != ["Mistral3ForConditionalGeneration"]:
        raise ValueError(
            "native FP8 decision readout requires Mistral3ForConditionalGeneration"
        )
    model = _get(model_config, "model_path")
    if (
        model not in DECISION_FP8_CHECKPOINTS
        or _get(model_config, "revision") != (DECISION_FP8_CHECKPOINTS[model])
    ):
        raise ValueError(
            "native FP8 decision readout requires a pinned BF16 checkpoint"
        )
    return True


def validate_decision_fp8_readout(mode: str) -> None:
    if mode != "lm_rows":
        raise ValueError("native FP8 decision readout initially supports only lm_rows")


def validate_decision_fp8_head(
    *,
    quant_config: Any,
    is_native_fp8_config: bool,
    is_unquantized_embedding_method: bool,
    use_marlin: bool,
    head_dtype: str,
    hidden_dtype: str,
    readout_dtype: str,
) -> None:
    """Check actual loaded head metadata once, without reading GPU tensor values."""
    if is_native_fp8_config is not True:
        raise ValueError("native FP8 decision readout requires the exact Fp8Config")
    if use_marlin is not False:
        raise ValueError(
            "native FP8 W8A8 decision readout does not support weight-only Marlin"
        )
    for name in (
        "is_checkpoint_fp8_serialized",
        "use_mxfp8",
        "is_fp4_experts",
        "dequant_fp4_to_fp8",
        "is_dsv4_fp4_experts",
    ):
        if _get(quant_config, name) is not False:
            raise ValueError(f"native FP8 decision readout requires {name}=False")
    if _get(quant_config, "activation_scheme") != "dynamic":
        raise ValueError(
            "native FP8 decision readout requires dynamic activation quantization"
        )
    for name in ("weight_block_size", "kv_cache_quant_algo", "scale_fmt"):
        if _get(quant_config, name) is not None:
            raise ValueError(f"native FP8 decision readout requires {name}=None")
    if _get(quant_config, "ignored_layers") != []:
        raise ValueError(
            "native FP8 decision readout does not support ignored FP8 layers"
        )
    if is_unquantized_embedding_method is not True:
        raise ValueError("native FP8 decision readout requires an unquantized LM head")
    if any(
        dtype != "torch.bfloat16" for dtype in (head_dtype, hidden_dtype, readout_dtype)
    ):
        raise ValueError(
            "native FP8 decision readout requires BF16 head, hidden states, and rows"
        )


def _batch_shape(obj: Any) -> tuple[int, bool]:
    text = _get(obj, "text")
    ids = _get(obj, "input_ids")
    if (text is None) == (ids is None):
        raise ValueError("decision request needs exactly one of text or input_ids")
    if text is not None:
        if isinstance(text, str) and text:
            return 1, False
        if (
            isinstance(text, list)
            and text
            and all(isinstance(item, str) and item for item in text)
        ):
            return len(text), True
        raise ValueError("decision request text must be nonempty text or a text batch")
    if (
        isinstance(ids, list)
        and ids
        and all(type(item) is int and item >= 0 for item in ids)
    ):
        return 1, False
    if (
        isinstance(ids, list)
        and ids
        and all(
            isinstance(item, list)
            and item
            and all(type(token) is int and token >= 0 for token in item)
            for item in ids
        )
    ):
        return len(ids), True
    raise ValueError("decision request input_ids must be nonempty token IDs or a batch")


def _off_for_all(value: Any, *, count: int, batched: bool) -> bool:
    if isinstance(value, list):
        return (
            batched
            and len(value) == count
            and all(item is None or item is False for item in value)
        )
    return value is None or value is False


def _image_count(value: Any) -> int:
    """Accept encoded images, never caller-supplied processor tensors/embeddings."""
    items = value if isinstance(value, list) else [value]
    if not items or any(
        not isinstance(item, (str, bytes)) or not item for item in items
    ):
        raise ValueError(
            "decision image_data must contain nonempty image strings or bytes"
        )
    if any(
        isinstance(item, str) and _DECISION_IMAGE_DATA_URL.fullmatch(item) is None
        for item in items
    ):
        raise ValueError(
            "decision image_data strings must be base64 PNG, JPEG or WebP data URLs"
        )
    return len(items)


def _validate_decision_images(value: Any, *, count: int, batched: bool) -> None:
    if value is None:
        return
    if not batched:
        _image_count(value)
        return
    # SGLang distinguishes flat and nested batches using their first element.
    # Accept homogeneous explicit batches, avoiding ambiguous broadcast/mixed
    # shapes that could silently attach an image to a different prompt.
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("decision image_data batch must match the input batch")
    nested = isinstance(value[0], list)
    if any(isinstance(item, list) != nested for item in value):
        raise ValueError("decision image_data batch must use a consistent shape")
    for item in value:
        _image_count(item)


def validate_decision_image_context(
    *, model_config: Any, processor: Any, disagg: Any
) -> None:
    """Fail before decoding images if this worker cannot execute its vision tower.

    KV-free prefill is not a reason to refuse: the vision encoder has its own
    attention and the image embeddings reach the language model as ordinary
    token embeddings, so the raw-K/V prefill treats them like text.
    """
    if any(
        _get(disagg, name, False)
        for name in ("encoder_only", "language_only", "language_model_only")
    ):
        raise ValueError("decision images require the local vision encoder")
    if processor is None or _get(model_config, "is_multimodal") is not True:
        raise ValueError("decision images require a multimodal processor")
    if _get(_get(model_config, "hf_config"), "architectures") != [
        "Mistral3ForConditionalGeneration"
    ]:
        raise ValueError(
            "decision images currently require Mistral3ForConditionalGeneration"
        )


def decision_image_boundary(input_ids: Any, tokenizer: Any) -> tuple[int, list]:
    """The native [/INST] control token and the assistant-prefix tokens after it.

    Identifies the control token without re-encoding its display text. Only
    ordinary (non-special) tokens may follow it: an options.assistant_prefix.
    """
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or any(type(token) is not int for token in input_ids)
    ):
        raise ValueError(
            "decision images require tokenized input at the assistant boundary"
        )
    special = set(tokenizer.all_special_ids)
    index = next(
        (i for i in range(len(input_ids) - 1, -1, -1) if input_ids[i] in special),
        None,
    )
    if (
        index is None
        or tokenizer.decode([input_ids[index]], skip_special_tokens=False)
        != "[/INST]"
    ):
        raise ValueError(
            "decision images must end at the atomic [/INST] boundary, "
            "optionally followed by assistant-prefix text"
        )
    return input_ids[index], input_ids[index + 1 :]


def validate_decision_image_output(
    *,
    mm_inputs: Any,
    input_ids: Any,
    boundary_id: int,
    image_count: int,
    prefix_ids: Any = (),
) -> None:
    """Reject dropped images, bypassed encoders, and processor boundary drift.

    Checks metadata only: no GPU tensor reads or synchronization on this path.
    Pixtral returns one IMAGE item per source image (possibly many patch rows).
    """
    prefix_ids = list(prefix_ids)
    boundary = len(input_ids) - len(prefix_ids) - 1 if isinstance(input_ids, list) else -1
    if (
        boundary < 0
        or input_ids[boundary] != boundary_id
        or input_ids[boundary + 1 :] != prefix_ids
    ):
        raise ValueError("decision image processor changed the [/INST] boundary")
    items = _get(mm_inputs, "mm_items")
    if not isinstance(items, list) or len(items) != image_count:
        raise ValueError("decision image processor did not return every image")
    for item in items:
        if (
            _get(_get(item, "modality"), "name") != "IMAGE"
            or _get(item, "feature") is None
            or _get(item, "precomputed_embeddings") is not None
        ):
            raise ValueError(
                "decision images require image features for the local encoder"
            )
        offsets = _get(item, "offsets")
        if not isinstance(offsets, list) or not offsets:
            raise ValueError(
                "decision image processor returned no image token positions"
            )
        for span in offsets:
            if (
                not isinstance(span, (tuple, list))
                or len(span) != 2
                or any(type(index) is not int for index in span)
                or not 0 <= span[0] <= span[1] < boundary
            ):
                raise ValueError(
                    "decision image token positions overlap or exceed the boundary"
                )


def validate_decision_request(obj: Any, *, schema: bool = False) -> None:
    """Validate raw GenerateReqInput-like data before normalization changes it.

    A schema-head worker needs one decision_schema plan per input; a readout
    worker refuses the field.
    """
    count, batched = _batch_shape(obj)
    _validate_decision_images(_get(obj, "image_data"), count=count, batched=batched)
    plans = _get(obj, "decision_schema")
    if not schema:
        if plans is not None:
            raise ValueError("decision request does not support decision_schema")
    elif batched:
        if not isinstance(plans, list) or len(plans) != count:
            raise ValueError("decision_schema batch must match the input batch")
        for plan in plans:
            validate_decision_schema_plan(plan)
    else:
        validate_decision_schema_plan(plans)
    sampling = _get(obj, "sampling_params")
    if isinstance(sampling, dict):
        entries = [sampling] * count
    elif batched and isinstance(sampling, list) and len(sampling) == count:
        entries = sampling
    else:
        raise ValueError("decision request needs sampling_params for each input")
    for index, params in enumerate(entries):
        if not isinstance(params, dict):
            raise ValueError(f"decision sampling_params[{index}] must be a dict")  # noqa: TRY004
        if (
            type(params.get("max_new_tokens")) is not int
            or params["max_new_tokens"] != 0
        ):
            raise ValueError("decision request requires max_new_tokens exactly 0")
        if type(params.get("n", 1)) is not int or params.get("n", 1) != 1:
            raise ValueError("decision request requires n=1")
        beam = params.get("beam_width")
        if beam is not None and (type(beam) is not int or beam != 1):
            raise ValueError("decision request does not support beam search")
        if params.get("min_new_tokens", 0) not in (0, None):
            raise ValueError("decision request requires min_new_tokens=0")
        for key in (
            "json_schema",
            "regex",
            "ebnf",
            "structural_tag",
            "stop_regex",
            "logit_bias",
            "custom_params",
        ):
            if params.get(key) is not None:
                raise ValueError(f"decision request does not support {key}")

    if _get(obj, "stream", False) is not False:
        raise ValueError("decision request requires stream=False")
    for name in ("return_logprob", "return_hidden_states", "return_sampling_mask"):
        if not _off_for_all(_get(obj, name), count=count, batched=batched):
            raise ValueError(f"decision request requires {name}=False")
    for name in (
        "input_embeds",
        "audio_data",
        "video_data",
        "modalities",
        "mm_hashes",
        "mm_content_hashes",
        "video_config",
        "images_config",
        "positional_embed_overrides",
        "session_params",
        "lora_path",
        "lora_id",
        "multi_item_delimiter_indices",
        "token_ids_logprob",
        "custom_logit_processor",
    ):
        value = _get(obj, name)
        if name == "custom_logit_processor" and batched and value == [None] * count:
            continue
        if value is not None:
            raise ValueError(f"decision request does not support {name}")
    for name in (
        "use_audio_in_video",
        "return_text_in_logprobs",
        "return_flat_raw_top_logprobs",
        "return_flat_raw_top_logprobs_b64",
        "return_routed_experts",
        "return_indexer_topk",
        "return_entropy",
        "return_prompt_token_ids",
        "return_bytes",
    ):
        if _get(obj, name, False) is not False:
            raise ValueError(f"decision request requires {name}=False")
