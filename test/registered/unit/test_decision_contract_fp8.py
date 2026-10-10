"""GPU-free tests for the isolated online FP8 and init-only projection contracts."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/decision_contract.py"
SPEC = importlib.util.spec_from_file_location("fp8_decision_contract", SOURCE)
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
        "quantization": "fp8",
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "model_path": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        "revision": "b6d637bef2393152b3da2b2fde72eecdee30557e",
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


def model(**changes):
    values = {
        "quantization": "fp8",
        "model_path": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        "revision": "b6d637bef2393152b3da2b2fde72eecdee30557e",
        "dtype": "torch.bfloat16",
        "hf_config": SimpleNamespace(
            architectures=["Mistral3ForConditionalGeneration"],
            text_config=SimpleNamespace(),
        ),
        "hf_quant_config": {},
    }
    values.update(changes)
    return SimpleNamespace(**values)


def validate_model(cfg, explicit="fp8", dtype="torch.bfloat16"):
    return contract.validate_decision_model_quantization(
        explicit_quantization=explicit, model_config=cfg, model_dtype=dtype
    )


def head(**changes):
    values = {
        "is_checkpoint_fp8_serialized": False,
        "use_mxfp8": False,
        "is_fp4_experts": False,
        "dequant_fp4_to_fp8": False,
        "is_dsv4_fp4_experts": False,
        "activation_scheme": "dynamic",
        "weight_block_size": None,
        "kv_cache_quant_algo": None,
        "scale_fmt": None,
        "ignored_layers": [],
    }
    values.update(changes)
    return SimpleNamespace(**values)


def validate_head(cfg, **changes):
    values = {
        "quant_config": cfg,
        "is_native_fp8_config": True,
        "is_unquantized_embedding_method": True,
        "use_marlin": False,
        "head_dtype": "torch.bfloat16",
        "hidden_dtype": "torch.bfloat16",
        "readout_dtype": "torch.bfloat16",
    }
    values.update(changes)
    contract.validate_decision_fp8_head(**values)


@pytest.mark.parametrize(
    "model_path,revision", contract.DECISION_FP8_CHECKPOINTS.items()
)
def test_online_fp8_pinned_bf16_checkpoints_accept_without_mutation(
    model_path, revision
):
    cfg = server(model_path=model_path, revision=revision)
    before = copy.deepcopy(vars(cfg))
    contract.validate_decision_server(cfg)
    assert vars(cfg) == before
    assert validate_model(model(model_path=model_path, revision=revision)) is True
    contract.validate_decision_fp8_readout("lm_rows")
    validate_head(head())


def test_unquantized_worker_defaults_remain_supported():
    cfg = server(quantization=None)
    del cfg.dtype, cfg.kv_cache_dtype, cfg.model_path, cfg.revision
    contract.validate_decision_server(cfg)
    assert validate_model(model(quantization=None), explicit=None) is False
    contract.validate_decision_server(SimpleNamespace(quantization="awq"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantization", "mxfp8"),
        ("quantization", "modelopt_fp8"),
        ("quantization", "compressed-tensors"),
        ("dtype", "float16"),
        ("dtype", "auto"),
        ("kv_cache_dtype", "fp8_e4m3"),
        ("prefill_only_disable_kv_cache", True),
        ("enable_fp32_lm_head", True),
        ("model_path", "mistralai/Ministral-3-3B-Instruct-2512"),
        ("revision", "main"),
        ("quantization_param_path", "/tmp/scales.json"),
        ("modelopt_quant", "fp8"),
        ("modelopt_checkpoint_restore_path", "/tmp/checkpoint"),
        ("modelopt_checkpoint_save_path", "/tmp/checkpoint"),
        ("modelopt_export_path", "/tmp/checkpoint"),
        ("quantize_and_serve", True),
        ("tp_size", 2),
        ("disable_radix_cache", False),
        ("chunked_prefill_size", 512),
        ("enable_lora", True),
        ("speculative_algorithm", "EAGLE"),
    ],
)
def test_server_rejects_precision_or_execution_path_changes(field, value):
    with pytest.raises(ValueError):
        contract.validate_decision_server(server(**{field: value}))


@pytest.mark.parametrize("scope", ["root", "text"])
@pytest.mark.parametrize("key", ["quantization_config", "compression_config"])
@pytest.mark.parametrize("metadata", [{}, {"quant_method": "fp8"}])
def test_present_checkpoint_metadata_rejected_even_when_empty(scope, key, metadata):
    cfg = model()
    target = cfg.hf_config if scope == "root" else cfg.hf_config.text_config
    setattr(target, key, metadata)
    with pytest.raises(ValueError, match="checkpoint quantization metadata absent"):
        validate_model(cfg)


@pytest.mark.parametrize(
    "explicit,resolved", [(None, "fp8"), ("fp8", None), ("fp8", "mxfp8")]
)
def test_model_rejects_autodetected_or_mismatched_quantization(explicit, resolved):
    with pytest.raises(ValueError, match="autodetected or mismatched"):
        validate_model(model(quantization=resolved), explicit=explicit)


def test_model_rejects_standalone_quant_metadata_and_wrong_architecture_or_dtype():
    with pytest.raises(ValueError, match="standalone"):
        validate_model(model(hf_quant_config={"quant_method": "fp8"}))
    with pytest.raises(ValueError, match="Mistral3ForConditionalGeneration"):
        validate_model(model(hf_config={"architectures": ["OtherModel"]}))
    with pytest.raises(ValueError, match="BF16"):
        validate_model(model(dtype="torch.float16"))
    with pytest.raises(ValueError, match="BF16"):
        validate_model(model(), dtype="torch.float16")
    with pytest.raises(ValueError, match="pinned"):
        validate_model(model(revision="main"))


def test_fp8_learned_readout_is_out_of_scope():
    with pytest.raises(ValueError, match="only lm_rows"):
        contract.validate_decision_fp8_readout("learned")


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_checkpoint_fp8_serialized", True),
        ("use_mxfp8", True),
        ("is_fp4_experts", True),
        ("dequant_fp4_to_fp8", True),
        ("is_dsv4_fp4_experts", True),
        ("activation_scheme", "static"),
        ("weight_block_size", [128, 128]),
        ("kv_cache_quant_algo", "FP8"),
        ("scale_fmt", "ue8m0"),
        ("ignored_layers", ["model.layers.0.self_attn"]),
    ],
)
def test_actual_head_rejects_serialized_or_alternate_fp8_configs(field, value):
    with pytest.raises(ValueError):
        validate_head(head(**{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_native_fp8_config", False),
        ("is_unquantized_embedding_method", False),
        ("use_marlin", True),
        ("head_dtype", "torch.float8_e4m3fn"),
        ("hidden_dtype", "torch.float16"),
        ("readout_dtype", "torch.float32"),
    ],
)
def test_actual_loaded_head_and_output_precision_are_enforced(field, value):
    with pytest.raises(ValueError):
        validate_head(head(), **{field: value})


def test_head_validation_never_reads_tensor_values():
    class UnreadableTensor:
        dtype = "torch.bfloat16"

        def item(self):
            raise AssertionError("host synchronization")

        def __bool__(self):
            raise AssertionError("tensor value inspection")

    tensor = UnreadableTensor()
    validate_head(
        head(),
        head_dtype=tensor.dtype,
        hidden_dtype=tensor.dtype,
        readout_dtype=tensor.dtype,
    )


@pytest.mark.parametrize("value,expected", [(None, 0), ("0", 0), ("8192", 8192)])
def test_projection_startup_choices(value, expected):
    assert contract.decision_projection_width(value) == expected


@pytest.mark.parametrize(
    "value", ["", " 0", "0 ", "00", "8192\n", "4096", "-1", "True", 8192]
)
def test_projection_choice_has_no_implicit_coercion(value):
    with pytest.raises(ValueError, match="exactly 0 or 8192"):
        contract.decision_projection_width(value)
