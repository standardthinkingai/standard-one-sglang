"""Exercise real Pixtral and Base processor methods without importing GPU kernels.

Only image decoding and the HF processor are stubbed. Dispatch,
native-ID reconstruction, offset calculation and Pixtral image splitting use the
checked-in methods. GPU transport is an observed boundary, never invoked here.
"""

import ast
import asyncio
import copy
import importlib.util
import math
import sys
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"
BASE_SOURCE = ROOT / "multimodal/processors/base_processor.py"
PIXTRAL_SOURCE = ROOT / "multimodal/processors/pixtral.py"
UTILS_SOURCE = ROOT / "managers/mm_utils.py"
SPEC = importlib.util.spec_from_file_location(
    "decision_pixtral_helper", ROOT / "multimodal/decision_images.py"
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
CACHE_SPEC = importlib.util.spec_from_file_location(
    "decision_preprocess_cache", ROOT / "multimodal/cache/preprocess_cache.py"
)
preprocess_cache = importlib.util.module_from_spec(CACHE_SPEC)
# Dataclasses resolve string annotations through the module's sys.modules entry.
sys.modules[CACHE_SPEC.name] = preprocess_cache
CACHE_SPEC.loader.exec_module(preprocess_cache)


class Modality(Enum):
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()


class InputFormat(Enum):
    NORMAL = auto()
    PROCESSOR_OUTPUT = auto()
    PRECOMPUTED_EMBEDDING = auto()


class Item:
    def __init__(self, *, modality):
        self.modality = modality
        self.offsets = None
        self.feature = None
        self.precomputed_embeddings = None
        self.format = InputFormat.NORMAL
        self.model_specific_data = {}
        self.hash = self.pad_value = None
        self.hash_calls = 0

    def set_pad_value(self):
        if self.pad_value is not None:
            return
        self.hash_calls += 1
        self.hash = hash(repr(self.feature))
        self.pad_value = self.hash % 1000

    def is_image(self):
        return self.modality == Modality.IMAGE

    def is_video(self):
        return self.modality == Modality.VIDEO

    def set(self, key, value):
        if key == "feature":
            self.feature = value
        else:
            self.model_specific_data[key] = value


class Tokens:
    def __init__(self, *, image_token, image_token_id):
        self.image_token, self.image_token_id = image_token, image_token_id

    def build(self, processor):
        return self

    def get_token_id_by_modality(self, modality):
        assert modality == Modality.IMAGE
        return self.image_token_id


class Tokenizer:
    def add_special_tokens(self, config):
        pass

    def convert_tokens_to_ids(self, token):
        return {"[IMG]": 10, "[IMG_BREAK]": 12, "[IMG_END]": 13}[token]

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([1, 701, 4]))


def _class_node(source, name):
    return next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _compile(namespace, body, source):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *body,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)  # noqa: S102


def processor(monkeypatch, *, decision=True, cache_bytes=0, transport_mode="cpu"):
    chosen = {
        "get_mm_items_offset",
        "get_mm_item_offsets",
        "collect_mm_items_from_processor_output",
        "_process_and_collect_mm_items",
        "process_and_combine_mm_data",
        "process_and_combine_mm_data_async",
        "_finalize_mm_items",
        "keep_mm_features_on_device",
    }
    base = _class_node(BASE_SOURCE, "BaseMultimodalProcessor")
    base.body = [
        method for method in base.body if getattr(method, "name", None) in chosen
    ]
    base.bases = []
    namespace = {
        "copy": copy,
        "math": math,
        "torch": torch,
        "np": np,
        "_PATCH_ALIGNED_KEYS": frozenset(("patch_pixel_values", "patch_newline_mask")),
        "Modality": Modality,
        "MultimodalInputFormat": InputFormat,
        "MultimodalDataItem": Item,
        "MultimodalProcessorOutput": SimpleNamespace,
        "MultimodalSpecialTokens": Tokens,
        "PreTrainedTokenizerBase": type("PreTrainedTokenizerBase", (), {}),
        "PixtralVisionModel": SimpleNamespace(DEFAULT_IMAGE_TOKEN_ID=10),
        "PixtralForConditionalGeneration": object,
        "rebuild_decision_image_ids": helper.rebuild_decision_image_ids,
        "_get_pixtral_hf_num_image_tokens": lambda shape, patch: (
            math.ceil(shape[0] / patch[0]),
            math.ceil(shape[1] / patch[1]),
        ),
        "envs": SimpleNamespace(
            SGLANG_MM_AVOID_RETOKENIZE=SimpleNamespace(get=lambda: True)
        ),
        "logger": SimpleNamespace(warning=lambda *args: None),
    }
    _compile(namespace, [base], BASE_SOURCE)
    base_class = namespace["BaseMultimodalProcessor"]

    def init(self, hf_config, server_args, hf_processor, *args, **kwargs):
        self.server_args = server_args
        self._processor = hf_processor
        self._tokenizer = hf_processor.tokenizer
        self.mm_processor_executor = None
        self.mm_feature_transport = transport_mode
        self.mm_preprocess_cache = preprocess_cache.MultimodalPreprocessCache(
            max_size_bytes=cache_bytes
        )
        self.ATTR_NAME_TO_MODALITY = {
            "pixel_values": Modality.IMAGE,
            "image_sizes": Modality.IMAGE,
        }
        self.FEATURE_NAMES = {"pixel_values"}
        self.preserve_processor_input_ids = False

    base_class.__init__ = init
    _compile(
        namespace, [_class_node(PIXTRAL_SOURCE, "PixtralProcessor")], PIXTRAL_SOURCE
    )
    split_functions = {
        "get_new_expanded_mm_items",
        "_get_length",
        "_is_rank2_grid",
        "_try_simple_split",
        "_compute_patch_slices",
        "_split_model_data_for_item",
        "_slice_value",
        "_slice_model_data",
    }
    _compile(
        namespace,
        [
            node
            for node in ast.parse(UTILS_SOURCE.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in split_functions
        ],
        UTILS_SOURCE,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.mm_utils",
        SimpleNamespace(
            get_new_expanded_mm_items=namespace["get_new_expanded_mm_items"],
            mark_shm_feature_reused=lambda tensor: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.multimodal.cache.preprocess_cache", preprocess_cache
    )
    hf_processor = SimpleNamespace(
        tokenizer=Tokenizer(),
        image_token="[IMG]",
        image_break_token="[IMG_BREAK]",
        image_end_token="[IMG_END]",
    )
    config = SimpleNamespace(
        image_token_index=10,
        vision_config=SimpleNamespace(image_size=1024, patch_size=16),
    )
    subject = namespace["PixtralProcessor"](
        config,
        SimpleNamespace(decision_readout_path="head.npz" if decision else None),
        hf_processor,
    )
    subject._resolve_processor = lambda override: (hf_processor, hf_processor.tokenizer)
    events = []
    state = SimpleNamespace(
        processed_ids=[1, 10, 10, 12, 10, 10, 13, 10, 10, 10, 13],
        offsets=None,
        sizes=[(32, 32), (48, 16)],
    )

    async def load(**kwargs):
        events.append(("load", kwargs))
        images = [
            SimpleNamespace(size=size)
            for size in state.sizes[: len(kwargs["image_data"] or [])]
        ]
        return SimpleNamespace(
            input_text=kwargs["prompt"],
            input_ids=None,
            images=images,
            organize_results=lambda: [(Modality.IMAGE, image) for image in images],
        )

    def process_mm_data(**kwargs):
        events.append(("hf", kwargs))
        count = len(kwargs["images"])
        result = {
            "input_ids": torch.tensor(state.processed_ids),
            "pixel_values": [f"pixels-{index}" for index in range(count)],
            "image_sizes": [image.size for image in kwargs["images"]],
        }
        if state.offsets is not None:
            result["offsets"] = state.offsets
        return result

    def transport(items):
        events.append(("transport", items))
        return items

    subject.load_mm_data = load
    subject.process_mm_data = process_mm_data
    subject._precompute_hashes_before_cpu_transfer = lambda items: None
    subject._prepare_mm_items_for_transport = transport
    return subject, state, events


def run(subject, ids, images):
    return asyncio.run(subject.process_mm_data_async(images, ids, SimpleNamespace()))


def test_native_ids_rebuilt_before_row_offsets_and_transport(monkeypatch):
    subject, _, events = processor(monkeypatch)
    original = [1, 3, 901, 10, 902, 10, 4]
    result = run(subject, original, ["first.png", "second.png"])
    assert original == [1, 3, 901, 10, 902, 10, 4]
    assert result.input_ids == [
        1,
        3,
        901,
        10,
        10,
        12,
        10,
        10,
        13,
        902,
        10,
        10,
        10,
        13,
        4,
    ]
    assert [item.offsets for item in result.mm_items] == [[(3, 4), (6, 7)], [(10, 12)]]
    assert [item.feature for item in result.mm_items] == [["pixels-0"], ["pixels-1"]]
    assert [item.model_specific_data["image_sizes"] for item in result.mm_items] == [
        [(32, 32)],
        [(48, 16)],
    ]
    assert [event[0] for event in events] == ["load", "hf", "transport"]
    assert events[0][1]["prompt"] == "[IMG][IMG]"
    assert events[1][1]["input_text"] == "[IMG][IMG]"
    assert "decision_input_ids" not in events[1][1]  # consumed before HF call


@pytest.mark.parametrize(
    "original", ["rendered chat", [1, 10, 4], [1, True, 10, 10, 4]]
)
def test_bad_native_prompt_rejected_before_image_loading(monkeypatch, original):
    subject, _, events = processor(monkeypatch)
    with pytest.raises(ValueError, match="native placeholder"):
        run(subject, original, ["a", "b"])
    assert not events


@pytest.mark.parametrize("processed", [[1, 10, 10, 12], [1, 10, 13], [1, 10, 13, 12]])
def test_bad_image_blocks_rejected_before_transport(monkeypatch, processed):
    subject, state, events = processor(monkeypatch)
    state.processed_ids = processed
    with pytest.raises(ValueError):
        run(subject, [1, 3, 10, 10, 4], ["a", "b"])
    assert not any(event[0] == "transport" for event in events)


def test_text_only_processor_path_is_unchanged(monkeypatch):
    subject, _, events = processor(monkeypatch)
    result = run(subject, "text prompt", [])
    assert result.input_ids == [1, 701, 4]
    assert result.mm_items == []
    assert events[0][1]["prompt"] == "text prompt"
    assert [event[0] for event in events] == ["load"]


def test_normal_generation_keeps_existing_processor_ids(monkeypatch):
    subject, state, events = processor(monkeypatch, decision=False)
    result = run(subject, "normal [IMG][IMG] prompt", ["a", "b"])
    assert result.input_ids == state.processed_ids
    assert events[0][1]["prompt"] == "normal [IMG][IMG] prompt"
    assert [item.offsets for item in result.mm_items] == [[(1, 2), (4, 5)], [(7, 9)]]


def test_precomputed_patch_offsets_rejected_before_transport(monkeypatch):
    subject, state, events = processor(monkeypatch)
    state.offsets = [(1, 2), (4, 5), (7, 9)]
    with pytest.raises(ValueError, match="offsets"):
        run(subject, [1, 3, 10, 10, 4], ["a", "b"])
    assert not any(event[0] == "transport" for event in events)


def test_single_row_images_keep_alignment_after_generic_item_split(monkeypatch):
    subject, state, events = processor(monkeypatch)
    state.sizes = [(32, 16), (48, 16)]
    state.processed_ids = [1, 10, 10, 13, 10, 10, 10, 13]
    result = run(subject, [1, 3, 901, 10, 902, 10, 4], ["a", "b"])
    assert result.input_ids == [1, 3, 901, 10, 10, 13, 902, 10, 10, 10, 13, 4]
    assert [item.offsets for item in result.mm_items] == [[(3, 4)], [(7, 9)]]
    assert [item.feature for item in result.mm_items] == [["pixels-0"], ["pixels-1"]]
    assert [event[0] for event in events] == ["load", "hf", "transport"]


SHARED_CACHE = 1 << 20
QUESTION_IDS = ([1, 3, 901, 10, 902, 10, 4], [1, 3, 905, 906, 907, 10, 908, 10, 909, 4])


def _summary(result):
    return (
        result.input_ids,
        [item.offsets for item in result.mm_items],
        [item.feature for item in result.mm_items],
        [item.model_specific_data["image_sizes"] for item in result.mm_items],
    )


def _hf_calls(events):
    return sum(event[0] == "hf" for event in events)


def test_shared_images_match_per_evaluation_processing(monkeypatch):
    per_evaluation, _, _ = processor(monkeypatch)
    shared, _, events = processor(monkeypatch, cache_bytes=SHARED_CACHE)
    images = ["first.png", "second.png"]
    results = [run(shared, list(ids), images) for ids in QUESTION_IDS]
    for ids, result in zip(QUESTION_IDS, results):
        assert _summary(result) == _summary(run(per_evaluation, list(ids), images))
    assert _hf_calls(events) == 1
    first, second = (result.mm_items for result in results)
    assert all(a is not b and a.feature is b.feature for a, b in zip(first, second))
    assert [item.hash for item in first] == [item.hash for item in second]
    cached_items, block_ids = next(iter(shared.mm_preprocess_cache._entries.values())).value
    assert [item.hash_calls for item in cached_items] == [1, 1]
    # Offsets of the cached items index the bare patch blocks only.
    assert block_ids == [10, 10, 12, 10, 10, 13, 10, 10, 10, 13]
    assert [item.offsets for item in cached_items] == [[(0, 1), (3, 4)], [(6, 8)]]


def test_concurrent_evaluations_join_one_preprocessing(monkeypatch):
    shared, _, events = processor(monkeypatch, cache_bytes=SHARED_CACHE)

    async def evaluations():
        return await asyncio.gather(
            *(
                shared.process_mm_data_async(["a", "b"], list(ids), SimpleNamespace())
                for ids in QUESTION_IDS * 3
            )
        )

    results = asyncio.run(evaluations())
    assert _hf_calls(events) == 1
    assert [result.input_ids for result in results[:2]] == [
        result.input_ids for result in results[2:4]
    ]


def test_shared_path_rejects_bad_blocks_before_transport_and_retries(monkeypatch):
    shared, state, events = processor(monkeypatch, cache_bytes=SHARED_CACHE)
    state.processed_ids = [1, 10, 10, 12]
    for _ in range(2):
        with pytest.raises(ValueError):
            run(shared, [1, 3, 10, 10, 4], ["a", "b"])
    assert not any(event[0] == "transport" for event in events)
    assert _hf_calls(events) == 2
    assert len(shared.mm_preprocess_cache) == 0


@pytest.mark.parametrize(
    ("cache_bytes", "transport_mode"), [(0, "cpu"), (SHARED_CACHE, "cuda_ipc")]
)
def test_images_processed_per_evaluation_without_shareable_cache(
    monkeypatch, cache_bytes, transport_mode
):
    subject, _, events = processor(
        monkeypatch, cache_bytes=cache_bytes, transport_mode=transport_mode
    )
    for ids in QUESTION_IDS:
        run(subject, list(ids), ["a", "b"])
    assert _hf_calls(events) == 2
    assert len(subject.mm_preprocess_cache) == 0


@pytest.mark.parametrize(("decision", "megabytes"), [(True, 1024), (False, 0)])
def test_decision_mode_enables_the_preprocess_cache_by_default(
    monkeypatch, decision, megabytes
):
    subject, _, _ = processor(monkeypatch, decision=decision)
    assert subject.auto_mm_preprocess_cache_size_mb == megabytes


def _data_url(width, height, image_format="PNG"):
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, image_format)
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[image_format]
    return f"data:image/{mime};base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize(
    ("size", "tokens"),
    [
        # Harness geometry: 1024-pixel longest edge, 16-pixel effective patch.
        # Each patch row ends in [IMG_BREAK] or [IMG_END].
        ((16, 16), 1 * 2),
        ((32, 32), 2 * 3),
        ((48, 16), 1 * 4),
        ((17, 1), 1 * 3),
        ((1000, 3), 1 * 64),
        ((4096, 1024), 16 * 65),
        ((1024, 4096), 64 * 17),
    ],
)
def test_decision_image_tokens_follow_the_processor_resize(monkeypatch, size, tokens):
    pytest.importorskip("PIL")
    subject, _, _ = processor(monkeypatch)
    url = _data_url(*size)
    assert subject.decision_image_tokens(url) == tokens
    # Its row count is the one the multi-image split already relies on.
    rows = subject._get_image_nrows([SimpleNamespace(size=size)])[0]
    assert tokens % rows == 0 and tokens // rows >= 2


@pytest.mark.parametrize("image_format", ["JPEG", "WEBP"])
def test_decision_image_tokens_read_jpeg_and_webp_headers(monkeypatch, image_format):
    features = pytest.importorskip("PIL.features")
    if image_format == "WEBP" and not features.check("webp"):
        pytest.skip("Pillow was built without WebP")
    subject, _, _ = processor(monkeypatch)
    assert subject.decision_image_tokens(_data_url(32, 32, image_format)) == 6


def test_decision_image_tokens_accept_raw_bytes(monkeypatch):
    import base64

    pytest.importorskip("PIL")
    subject, _, _ = processor(monkeypatch)
    raw = base64.b64decode(_data_url(48, 16).split(",", 1)[1])
    assert subject.decision_image_tokens(raw) == 4


@pytest.mark.parametrize(
    "image",
    [
        "https://example.test/a.png",
        "/etc/hosts.png",
        "data:image/gif;base64,R0lGODlh",
        "data:image/png;base64,QUJD",
        "data:image/png;base64,QUJ",
        b"",
        b"not an image",
    ],
)
def test_decision_image_tokens_refuse_anything_but_a_readable_image(monkeypatch, image):
    pytest.importorskip("PIL")
    subject, _, _ = processor(monkeypatch)
    with pytest.raises(ValueError):
        subject.decision_image_tokens(image)


def test_llava_wrapper_exposes_the_decision_image_counter():
    """The Mistral3 wrapper hid the Pixtral counter, so image pre-checks never ran."""
    source = ROOT / "multimodal/processors/llava.py"
    node = _class_node(source, "LlavaMultimodalProcessor")
    node.bases = []
    node.body = [
        n for n in node.body if getattr(n, "name", None) == "decision_image_tokens"
    ]
    namespace = {}
    _compile(namespace, [node], source)
    wrapper = namespace["LlavaMultimodalProcessor"].__new__(
        namespace["LlavaMultimodalProcessor"]
    )
    wrapper.inner = SimpleNamespace(decision_image_tokens=lambda image: 42)
    assert wrapper.decision_image_tokens("data:image/png;base64,AA==") == 42
    wrapper.inner = SimpleNamespace()
    assert getattr(wrapper, "decision_image_tokens", None) is None
