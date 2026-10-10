"""CPU tests for decision-image admission and the actual tokenizer-manager path."""

import ast
import asyncio
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
SPEC = importlib.util.spec_from_file_location(
    "decision_images_contract", SRT / "decision_contract.py"
)
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


PNG_A = "data:image/png;base64,aW1hZ2UtYQ=="
PNG_B = "data:image/png;base64,aW1hZ2UtYg=="


def request(**changes):
    values = {
        "text": None,
        "input_ids": [1, 10, 3],
        "image_data": ["data:image/png;base64,abc"],
        "audio_data": None,
        "video_data": None,
        "input_embeds": None,
        "sampling_params": {"max_new_tokens": 0},
        "stream": False,
        "mm_hashes": None,
        "mm_content_hashes": None,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def output(**changes):
    item = SimpleNamespace(
        modality=SimpleNamespace(name="IMAGE"),
        feature=object(),
        precomputed_embeddings=None,
        offsets=[(1, 2)],
    )
    values = {"mm_items": [item], "input_ids": [1, 10, 10, 3], "token_type_ids": None}
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("image_data", [PNG_A, b"png", [PNG_A, PNG_B]])
def test_encoded_images_accepted_without_changing_zero_output_contract(image_data):
    obj = request(image_data=image_data)
    before = copy.deepcopy(vars(obj))
    contract.validate_decision_request(obj)
    assert vars(obj) == before
    obj.sampling_params = {"max_new_tokens": 1}
    with pytest.raises(ValueError, match="max_new_tokens"):
        contract.validate_decision_request(obj)


@pytest.mark.parametrize(
    "image_data",
    [[], "", b"", [None], [["nested"]], {"feature": [0.2]}, ["a", {"feature": 1}]],
)
def test_malformed_or_precomputed_image_payloads_rejected(image_data):
    with pytest.raises(ValueError, match="image_data"):
        contract.validate_decision_request(request(image_data=image_data))


@pytest.mark.parametrize("images", [[PNG_A, PNG_B], [[PNG_A], [PNG_B, PNG_A]]])
def test_explicit_homogeneous_image_batches(images):
    contract.validate_decision_request(
        request(input_ids=[[1, 10, 3], [1, 10, 3]], image_data=images)
    )


@pytest.mark.parametrize(
    "images", ["broadcast", ["short"], [["a"], "b"], [None, ["a"]]]
)
def test_ambiguous_image_batches_rejected(images):
    with pytest.raises(ValueError, match="image_data"):
        contract.validate_decision_request(
            request(input_ids=[[1, 10, 3], [1, 10, 3]], image_data=images)
        )


@pytest.mark.parametrize(
    "image",
    [
        "https://example.test/a.png",
        "http://169.254.169.254/latest/meta-data",
        "file:///etc/hosts",
        "/etc/hosts.png",
        "image.png",
        "aGVsbG8=",
        "data:image/gif;base64,R0lGODlh",
        "data:image/png;base64,not base64",
        "data:image/png,raw",
    ],
)
def test_decision_images_are_never_fetched_or_read_from_disk(image):
    # SGLang's loader fetches URLs and reads local paths; a decision worker
    # accepts only inline base64 PNG, JPEG or WebP (or raw bytes in-process).
    with pytest.raises(ValueError, match="data URLs"):
        contract.validate_decision_request(request(image_data=[image]))
    with pytest.raises(ValueError, match="data URLs"):
        contract.validate_decision_request(
            request(input_ids=[[1, 10, 3], [1, 10, 3]], image_data=[[PNG_A], [image]])
        )


def _subject(monkeypatch, *, decision=True, images=True):
    source = SRT / "managers/tokenizer_manager.py"
    tree = ast.parse(source.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TokenizerManager"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_tokenize_one_request"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            ast.ClassDef(
                name="Subject", bases=[], keywords=[], body=[method], decorator_list=[]
            ),
        ],
        type_ignores=[],
    )
    disagg = SimpleNamespace(
        language_model_only=False, language_only=False, encoder_only=False
    )
    schedule = SimpleNamespace(prefill_only_disable_kv_cache=False)

    class GenerateReqInput(SimpleNamespace):
        decision_schema = None

        def contains_mm_input(self):
            return bool(self.image_data)

    namespace = {
        "GenerateReqInput": GenerateReqInput,
        "EmbeddingReqInput": type("EmbeddingReqInput", (), {}),
        "get_model": lambda: SimpleNamespace(
            decision_readout_path="readout.npz" if decision else None
        ),
        "get_disagg": lambda: disagg,
        "get_schedule": lambda: schedule,
        "get_memory": lambda: SimpleNamespace(disable_radix_cache=True),
        "envs": SimpleNamespace(
            SGLANG_MM_PRECOMPUTE_HASH=SimpleNamespace(get=lambda: False)
        ),
    }
    monkeypatch.setitem(sys.modules, "sglang.srt.decision_contract", contract)
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)  # noqa: S102
    manager = namespace["Subject"]()
    manager.model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(architectures=["Mistral3ForConditionalGeneration"]),
    )
    manager.tokenizer = SimpleNamespace(
        all_special_ids=[1, 3, 10],
        decode=lambda ids, **kwargs: "[/INST]" if ids == [3] else "not a boundary",
    )
    manager.max_req_input_len = 8192
    calls = []
    result = output()

    async def process(**kwargs):
        calls.append(kwargs)
        return result

    async def tokenize(text, is_cross_encoder):
        return [1, 10, 3], None

    manager.mm_processor = SimpleNamespace(
        prefer_tokenized_input=False, process_mm_data_async=process
    )
    manager._tokenize_texts = tokenize
    manager._validate_mm_limits = lambda obj: None
    manager._normalize_mm_content_hashes = lambda obj: None
    manager._validate_one_request = lambda obj, ids: None
    manager._create_tokenized_object = lambda obj, text, ids, embeds, mm, types: (
        ids,
        mm,
    )
    obj = GenerateReqInput(
        **vars(request(image_data=["image.png"] if images else None))
    )
    return manager, obj, calls, result, disagg, schedule


def test_manager_processes_pixels_and_expanded_ids_before_scheduler(monkeypatch):
    manager, obj, calls, result, _, _ = _subject(monkeypatch)
    ids, mm = asyncio.run(manager._tokenize_one_request(obj))
    assert len(calls) == 1
    assert calls[0]["input_text"] == [1, 10, 3]  # force IDs despite wrapper preference
    assert calls[0]["image_data"] == ["image.png"]
    assert ids == [1, 10, 10, 3]
    assert mm is result and mm.mm_items[0].feature is not None
    assert obj.sampling_params == {"max_new_tokens": 0}


def test_raw_text_uses_ids_for_decision_processor(monkeypatch):
    manager, obj, calls, _, _, _ = _subject(monkeypatch)
    obj.text, obj.input_ids = "native rendered prompt", None
    asyncio.run(manager._tokenize_one_request(obj))
    assert calls[0]["input_text"] == [1, 10, 3]


@pytest.mark.parametrize(
    "field", ["language_model_only", "language_only", "encoder_only"]
)
def test_missing_or_external_vision_encoder_rejected_before_processor(
    monkeypatch, field
):
    manager, obj, calls, _, disagg, _ = _subject(monkeypatch)
    setattr(disagg, field, True)
    with pytest.raises(ValueError, match="local vision encoder"):
        asyncio.run(manager._tokenize_one_request(obj))
    assert not calls


@pytest.mark.parametrize(
    "change,error",
    [
        ("processor", "multimodal processor"),
        ("model", "Mistral3"),
    ],
)
def test_unsupported_model_paths_rejected_before_processor(monkeypatch, change, error):
    manager, obj, calls, _, _, _ = _subject(monkeypatch)
    if change == "processor":
        manager.mm_processor = None
    else:
        manager.model_config.hf_config.architectures = ["AnotherVisionModel"]
    with pytest.raises(ValueError, match=error):
        asyncio.run(manager._tokenize_one_request(obj))
    assert not calls


def test_kv_free_prefill_processes_images_like_text(monkeypatch):
    # The raw-K/V prefill sees image embeddings as ordinary tokens; the vision
    # encoder never touches the paged KV pool. The request takes the same path.
    manager, obj, calls, result, _, schedule = _subject(monkeypatch)
    schedule.prefill_only_disable_kv_cache = True
    ids, mm = asyncio.run(manager._tokenize_one_request(obj))
    assert len(calls) == 1 and calls[0]["image_data"] == ["image.png"]
    assert ids == [1, 10, 10, 3] and mm is result


def test_invalid_input_boundary_rejected_before_loading_image(monkeypatch):
    manager, obj, calls, _, _, _ = _subject(monkeypatch)
    obj.input_ids[-1] = 7
    with pytest.raises(ValueError, match="atomic"):
        asyncio.run(manager._tokenize_one_request(obj))
    assert not calls


@pytest.mark.parametrize(
    "change,error",
    [
        ("boundary", "boundary"),
        ("missing", "every image"),
        ("empty_features", "local encoder"),
        ("precomputed", "local encoder"),
        ("audio", "local encoder"),
        ("empty_offsets", "no image token positions"),
        ("tail_offset", "boundary"),
        ("negative_offset", "boundary"),
    ],
)
def test_processor_cannot_drop_images_or_corrupt_readout_position(
    monkeypatch, change, error
):
    manager, obj, calls, result, _, _ = _subject(monkeypatch)
    item = result.mm_items[0]
    if change == "boundary":
        result.input_ids[-1] = 7
    elif change == "missing":
        result.mm_items = []
    elif change == "empty_features":
        item.feature = None
    elif change == "precomputed":
        item.precomputed_embeddings = object()
    elif change == "audio":
        item.modality.name = "AUDIO"
    elif change == "empty_offsets":
        item.offsets = []
    elif change == "tail_offset":
        item.offsets = [(1, 3)]
    else:
        item.offsets = [(-1, 1)]
    with pytest.raises(ValueError, match=error):
        asyncio.run(manager._tokenize_one_request(obj))
    assert len(calls) == 1


def test_auto_truncation_cannot_change_image_prompt_silently(monkeypatch):
    manager, obj, _, _, _, _ = _subject(monkeypatch)
    manager._validate_one_request = lambda obj, ids: ids.pop()
    with pytest.raises(ValueError, match="automatic prompt truncation"):
        asyncio.run(manager._tokenize_one_request(obj))


def test_text_only_decision_worker_keeps_language_model_only_path(monkeypatch):
    manager, obj, calls, _, disagg, _ = _subject(monkeypatch, images=False)
    manager.mm_processor = None
    disagg.language_model_only = True
    obj.input_ids = [1, 7, 3]
    assert asyncio.run(manager._tokenize_one_request(obj)) == ([1, 7, 3], None)
    assert not calls


def test_normal_generation_does_not_apply_decision_image_restrictions(monkeypatch):
    manager, obj, calls, result, _, _ = _subject(monkeypatch, decision=False)
    obj.text, obj.input_ids = "normal prompt", None
    result.input_ids = [1, 10, 10, 7]
    assert asyncio.run(manager._tokenize_one_request(obj))[0][-1] == 7
    assert calls[0]["input_text"] == "normal prompt"
