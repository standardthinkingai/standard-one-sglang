"""Pure decision-mode contract tests that need no GPU or SGLang imports."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/decision_contract.py"
SPEC = importlib.util.spec_from_file_location("decision_contract_under_test", SOURCE)
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def server(**changes):
    values = {
        "decision_readout_path": "/models/decision.npz",
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "is_embedding": False,
        "enable_lora": None,
        "skip_tokenizer_init": False,
        "enable_return_hidden_states": False,
        "enable_custom_logit_processor": False,
        "enable_mis": False,
        "quantization": None,
        "speculative_algorithm": None,
        "return_hidden_states_mode": None,
        "disaggregation_mode": "null",
        "chunked_prefill_size": -1,
        "disable_radix_cache": True,
        "cuda_graph_config": SimpleNamespace(
            decode=SimpleNamespace(backend="disabled")
        ),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def request(**changes):
    values = {
        "text": "Choose the best option.",
        "input_ids": None,
        "sampling_params": {"max_new_tokens": 0, "n": 1},
        "stream": False,
        "return_logprob": False,
        "return_hidden_states": False,
        "return_sampling_mask": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_server_noop_without_artifact_and_valid_config_is_unchanged():
    contract.validate_decision_server(SimpleNamespace())
    cfg = server()
    before = copy.deepcopy(vars(cfg))
    contract.validate_decision_server(cfg)
    assert vars(cfg) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decision_readout_path", ""),
        ("tp_size", 2),
        ("pp_size", 2),
        ("dp_size", 2),
        ("is_embedding", True),
        ("enable_lora", True),
        ("quantization", "fp8"),
        ("speculative_algorithm", "EAGLE"),
        ("disaggregation_mode", "prefill"),
        ("encoder_only", True),
        ("language_only", True),
        ("enable_prefix_mm_cache", True),
        ("chunked_prefill_size", 512),
        ("disable_radix_cache", False),
        ("skip_tokenizer_init", True),
        ("return_hidden_states_mode", "last"),
        ("enable_return_hidden_states", True),
        ("enable_custom_logit_processor", True),
        ("enable_mis", True),
        (
            "cuda_graph_config",
            SimpleNamespace(decode=SimpleNamespace(backend="full")),
        ),
    ],
)
def test_server_rejects_incompatible_resolved_settings(field, value):
    with pytest.raises(ValueError):
        contract.validate_decision_server(server(**{field: value}))


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"allow_auto_truncate": True}, "allow_auto_truncate"),
        ({"decision_deadline_seconds": 0}, "decision_deadline_seconds"),
        ({"decision_deadline_seconds": -5.0}, "decision_deadline_seconds"),
        ({"decision_deadline_seconds": float("nan")}, "decision_deadline_seconds"),
        ({"decision_deadline_seconds": float("inf")}, "decision_deadline_seconds"),
        ({"decision_deadline_seconds": True}, "decision_deadline_seconds"),
        ({"decision_max_queued_evaluations": -1}, "decision_max_queued_evaluations"),
        ({"decision_max_queued_evaluations": 2.0}, "decision_max_queued_evaluations"),
        (
            {"max_running_requests": 32, "max_queued_requests": 32},
            "--max-queued-requests",
        ),
    ],
)
def test_server_rejects_unsafe_decision_limits(changes, expected):
    with pytest.raises(ValueError, match=expected):
        contract.validate_decision_server(server(**changes))


def test_server_accepts_decision_limits():
    contract.validate_decision_server(
        server(
            decision_deadline_seconds=110.0,
            decision_max_queued_evaluations=0,
            max_running_requests=32,
            max_queued_requests=33,
            allow_auto_truncate=False,
        )
    )
    contract.validate_decision_server(server(decision_deadline_seconds=30))


@pytest.mark.parametrize(
    "name",
    [
        "decision_deadline_seconds",
        "decision_max_queued_evaluations",
        "decision_label_scheme",
    ],
)
def test_decision_limits_require_decision_mode(name):
    with pytest.raises(ValueError, match="--decision-readout-path"):
        contract.validate_decision_server(
            SimpleNamespace(decision_readout_path=None, **{name: 1})
        )


def test_server_reads_resolved_decode_phase_not_only_legacy_flag():
    cfg = server(disable_decode_cuda_graph=True)
    cfg.cuda_graph_config.decode.backend = "full"
    with pytest.raises(ValueError, match="resolved decode CUDA graph"):
        contract.validate_decision_server(cfg)


def kv_free_server(**changes):
    cfg = server(
        prefill_only_disable_kv_cache=True,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        enable_unified_memory=False,
        warmups=None,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend="disabled"),
            prefill=SimpleNamespace(backend="disabled"),
        ),
    )
    for name, value in changes.items():
        setattr(cfg, name, value)
    return cfg


def test_kv_free_decision_server_accepts_narrow_bf16_eager_contract():
    cfg = kv_free_server()
    before = copy.deepcopy(vars(cfg))
    contract.validate_decision_server(cfg)
    assert vars(cfg) == before


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"chunked_prefill_size": 0}, "chunked_prefill_size=-1"),
        ({"dtype": "float16"}, "dtype=bfloat16"),
        ({"kv_cache_dtype": "fp8_e4m3"}, "kv_cache_dtype=auto"),
        ({"enable_unified_memory": True}, "enable_unified_memory=False"),
        ({"warmups": "prefill_shapes"}, "--warmups"),
        (
            {
                "cuda_graph_config": SimpleNamespace(
                    decode=SimpleNamespace(backend="disabled"),
                    prefill=SimpleNamespace(backend="breakable"),
                )
            },
            "resolved prefill CUDA graph",
        ),
    ],
)
def test_kv_free_decision_server_rejects_unvalidated_paths(changes, expected):
    with pytest.raises(ValueError, match=expected):
        contract.validate_decision_server(kv_free_server(**changes))


def test_single_request_is_accepted_without_mutation():
    obj = request(input_ids=[1, 2, 3], text=None)
    before = copy.deepcopy(vars(obj))
    contract.validate_decision_request(obj)
    assert vars(obj) == before


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"sampling_params": {}}, "max_new_tokens"),
        ({"sampling_params": {"max_new_tokens": True}}, "max_new_tokens"),
        ({"sampling_params": {"max_new_tokens": 1}}, "max_new_tokens"),
        ({"sampling_params": {"max_new_tokens": 0, "n": 2}}, "n=1"),
        ({"sampling_params": {"max_new_tokens": 0, "beam_width": 2}}, "beam"),
        ({"sampling_params": {"max_new_tokens": 0, "regex": "a"}}, "regex"),
        (
            {"sampling_params": {"max_new_tokens": 0, "json_schema": "{}"}},
            "json_schema",
        ),
        (
            {"sampling_params": {"max_new_tokens": 0, "custom_params": {}}},
            "custom_params",
        ),
        ({"return_logprob": True}, "return_logprob"),
        ({"return_hidden_states": "last"}, "return_hidden_states"),
        ({"return_sampling_mask": True}, "return_sampling_mask"),
        ({"stream": True}, "stream"),
        ({"image_data": {"precomputed_embeddings": [0.1]}}, "image_data"),
        ({"audio_data": "audio.wav"}, "audio_data"),
        ({"video_data": "video.mp4"}, "video_data"),
        ({"input_embeds": [[0.1]]}, "input_embeds"),
        ({"custom_logit_processor": "processor"}, "custom_logit_processor"),
        ({"multi_item_delimiter_indices": [1]}, "multi_item_delimiter_indices"),
        ({"token_ids_logprob": [1]}, "token_ids_logprob"),
    ],
)
def test_request_rejects_other_inference_paths(changes, expected):
    with pytest.raises(ValueError, match=expected):
        contract.validate_decision_request(request(**changes))


def test_batch_parameters_are_checked_for_every_item():
    good = request(
        text=["first", "second"],
        sampling_params=[{"max_new_tokens": 0}, {"max_new_tokens": 0}],
        return_logprob=[False, False],
        return_hidden_states=[False, False],
        return_sampling_mask=[False, False],
        custom_logit_processor=[None, None],
    )
    contract.validate_decision_request(good)
    bad = copy.deepcopy(good)
    bad.sampling_params[1]["max_new_tokens"] = 1
    with pytest.raises(ValueError, match="max_new_tokens"):
        contract.validate_decision_request(bad)
    bad = copy.deepcopy(good)
    bad.return_hidden_states[1] = "last"
    with pytest.raises(ValueError, match="return_hidden_states"):
        contract.validate_decision_request(bad)
    bad = copy.deepcopy(good)
    bad.sampling_params = bad.sampling_params[:1]
    with pytest.raises(ValueError, match="sampling_params"):
        contract.validate_decision_request(bad)


def test_batch_broadcast_and_input_id_batch_are_supported():
    contract.validate_decision_request(
        request(text=["a", "b"], sampling_params={"max_new_tokens": 0})
    )
    contract.validate_decision_request(
        request(
            text=None, input_ids=[[1, 2], [3]], sampling_params={"max_new_tokens": 0}
        )
    )


def test_malformed_single_and_batch_shapes_are_rejected():
    with pytest.raises(ValueError, match="sampling_params"):
        contract.validate_decision_request(
            request(sampling_params=[{"max_new_tokens": 0}])
        )
    with pytest.raises(ValueError, match="text"):
        contract.validate_decision_request(request(text=[]))
    with pytest.raises(ValueError, match="return_logprob"):
        contract.validate_decision_request(
            request(text=["a", "b"], return_logprob=[False])
        )


def schema_plan(**changes):
    # Two questions over a 40-token prompt: a choice with three options and a noul.
    plan = {
        "template": "schema-v1",
        "n_tokens": 40,
        "qtype": ["choice", "noul"],
        "p_span": [[2, 4], [4, 6]],
        "q_span": [[12, 15], [24, 27]],
        "o_span": [[[16, 18], [18, 20], [20, 23]], [[28, 30], [30, 33]]],
        "state_spans": [[7, 11]],
        "image_span": None,
        "n_values": 5,
    }
    plan.update(changes)
    return plan


def test_schema_head_artifact_is_a_directory_with_its_config(tmp_path):
    assert not contract.is_schema_head_artifact(str(tmp_path))
    (tmp_path / contract.SCHEMA_HEAD_CONFIG_NAME).write_text("{}")
    assert contract.is_schema_head_artifact(str(tmp_path))
    assert not contract.is_schema_head_artifact(str(tmp_path / "decision.npz"))


def test_schema_plan_is_required_by_a_schema_worker_and_refused_otherwise():
    with_plan = request(text=None, input_ids=[1, 2, 3], decision_schema=schema_plan())
    contract.validate_decision_request(with_plan, schema=True)
    with pytest.raises(ValueError, match="decision_schema"):
        contract.validate_decision_request(with_plan)
    with pytest.raises(ValueError, match="decision_schema must be an object"):
        contract.validate_decision_request(request(text=None, input_ids=[1, 2, 3]), schema=True)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"n_values": 4}, "n_values"),
        ({"qtype": ["choice", "maybe"]}, "question types"),
        ({"o_span": [[[16, 18]], [[28, 30], [30, 33]]]}, "option count"),
        ({"o_span": [[[16, 18], [18, 20], [20, 23]], [[28, 30], [30, 33], [33, 35]]]}, "option count"),
        ({"q_span": [[12, 12], [24, 27]]}, "nonempty"),
        ({"q_span": [[12, 41], [24, 27]]}, "nonempty"),
        ({"p_span": [[2, 4]]}, "one entry per question"),
        ({"state_spans": []}, "state token spans"),
        ({"image_span": [0, 7]}, "first state span"),
    ],
)
def test_schema_plan_structure_is_checked_before_tokenization(changes, expected):
    with pytest.raises(ValueError, match=expected):
        contract.validate_decision_schema_plan(schema_plan(**changes))


def test_schema_plan_must_match_the_processed_prompt_and_image_block():
    item = SimpleNamespace(offsets=[(3, 5), (7, 9)])
    plan = schema_plan(image_span=[3, 11], state_spans=[[3, 11], [12, 14]])
    contract.validate_decision_schema_plan(plan)
    contract.validate_decision_schema_tokens(
        plan, input_ids=[0] * 40, mm_inputs=SimpleNamespace(mm_items=[item])
    )
    with pytest.raises(ValueError, match="n_tokens"):
        contract.validate_decision_schema_tokens(plan, input_ids=[0] * 39, mm_inputs=None)
    with pytest.raises(ValueError, match="image span differs"):
        contract.validate_decision_schema_tokens(
            schema_plan(image_span=[3, 10], state_spans=[[3, 10]]),
            input_ids=[0] * 40,
            mm_inputs=SimpleNamespace(mm_items=[item]),
        )
    with pytest.raises(ValueError, match="no images"):
        contract.validate_decision_schema_tokens(plan, input_ids=[0] * 40, mm_inputs=None)


BOUNDARY_TOKENIZER = SimpleNamespace(
    all_special_ids=[1, 3, 4, 10],
    decode=lambda ids, **kwargs: {3: "[/INST]", 4: "[INST]"}.get(ids[0], "text"),
)


def test_image_boundary_allows_an_assistant_prefix_of_ordinary_tokens():
    assert contract.decision_image_boundary([1, 4, 10, 7, 3], BOUNDARY_TOKENIZER) == (3, [])
    assert contract.decision_image_boundary(
        [1, 4, 10, 7, 3, 500, 501], BOUNDARY_TOKENIZER
    ) == (3, [500, 501])
    for ids in ([1, 4, 10, 7, 3, 500, 4], [1, 4, 10, 7, 500]):
        with pytest.raises(ValueError, match="boundary"):
            contract.decision_image_boundary(ids, BOUNDARY_TOKENIZER)


def test_image_output_keeps_the_boundary_and_prefix_after_the_images():
    image = SimpleNamespace(
        modality=SimpleNamespace(name="IMAGE"),
        feature=[0.1],
        precomputed_embeddings=None,
        offsets=[(2, 4)],
    )
    mm_inputs = SimpleNamespace(mm_items=[image])
    expanded = [1, 4, 10, 10, 10, 7, 3, 500, 501]
    contract.validate_decision_image_output(
        mm_inputs=mm_inputs, input_ids=expanded, boundary_id=3, image_count=1,
        prefix_ids=[500, 501],
    )
    for ids, prefix in (
        (expanded, [501]),
        ([1, 4, 10, 10, 10, 7, 3, 500, 502], [500, 501]),
    ):
        with pytest.raises(ValueError, match="boundary"):
            contract.validate_decision_image_output(
                mm_inputs=mm_inputs, input_ids=ids, boundary_id=3, image_count=1,
                prefix_ids=prefix,
            )
    late = SimpleNamespace(**{**vars(image), "offsets": [(7, 7)]})
    with pytest.raises(ValueError, match="exceed the boundary"):
        contract.validate_decision_image_output(
            mm_inputs=SimpleNamespace(mm_items=[late]), input_ids=expanded,
            boundary_id=3, image_count=1, prefix_ids=[500, 501],
        )


@pytest.mark.parametrize("scheme", ["served", "upper", None])
def test_server_accepts_known_label_schemes(scheme):
    contract.validate_decision_server(server(decision_label_scheme=scheme))


def test_server_rejects_an_unknown_label_scheme():
    with pytest.raises(ValueError, match="decision_label_scheme"):
        contract.validate_decision_server(server(decision_label_scheme="lower"))


def test_label_scheme_resolution_order():
    resolve = contract.resolve_decision_label_scheme
    assert resolve(None, {}) == "upper"
    assert resolve("", {"JEV_LABEL_SCHEME": ""}) == "upper"
    assert resolve(None, {"JEV_LABEL_SCHEME": "served"}) == "served"
    assert resolve("upper", {"JEV_LABEL_SCHEME": "served"}) == "upper"
    with pytest.raises(ValueError, match="label scheme"):
        resolve(None, {"JEV_LABEL_SCHEME": "lower"})
