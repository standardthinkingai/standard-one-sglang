"""Native control tokens and Pixtral patch offsets must survive image expansion."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/multimodal/decision_images.py"
)
SPEC = importlib.util.spec_from_file_location("decision_images_under_test", SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def expand(original, processed, offsets, **markers):
    return module.rebuild_decision_image_ids(
        original,
        processed,
        offsets,
        **({"image_token_id": 10, "image_break_id": 12, "image_end_id": 13} | markers),
    )


def test_preserves_native_chat_with_two_differently_sized_images():
    original = [1, 3, 901, 902, 10, 10, 4]
    processed = [1, 10, 10, 12, 10, 10, 13, 10, 10, 10, 13, 2]
    offsets = [[(1, 2), (4, 5)], [(7, 9)]]
    rebuilt, translated = expand(original, processed, offsets)
    assert rebuilt == [1, 3, 901, 902, 10, 10, 12, 10, 10, 13, 10, 10, 10, 13, 4]
    assert translated == [[(4, 5), (7, 8)], [(10, 12)]]
    assert original == [1, 3, 901, 902, 10, 10, 4]
    assert offsets == [[(1, 2), (4, 5)], [(7, 9)]]


def test_images_can_be_interleaved_with_native_text():
    rebuilt, offsets = expand(
        [1, 3, 10, 21, 10, 22, 4],
        [1, 10, 13, 10, 10, 13],
        [[(1, 1)], [(3, 4)]],
    )
    assert rebuilt == [1, 3, 10, 13, 21, 10, 10, 13, 22, 4]
    assert offsets == [[(2, 2)], [(5, 6)]]


@pytest.mark.parametrize(
    ("processed", "offsets"),
    [
        ([1, 10, 10], [[(1, 2)]]),
        ([1, 10, 12, 13], [[(1, 1)]]),
        ([1, 10, 99, 13], [[(1, 1)]]),
        ([1, 12, 10, 13], [[(2, 2)]]),
        ([1, 10, 13, 13], [[(1, 1)]]),
        ([1, 10, 13, 10, 13], [[(1, 1)]]),
        ([1, 10, 13], []),
        ([1, 10, 10, 13], [[(1, 1)]]),
        ([1, 10, 13], [[(1, 2)]]),
        ([1, 10, 13], [[(True, 1)]]),
    ],
)
def test_rejects_incomplete_missing_or_misaligned_images(processed, offsets):
    with pytest.raises(ValueError):
        expand([1, 3, 10, 4], processed, offsets)


@pytest.mark.parametrize("original", [[], [1, True, 10], [1, 12, 10], [1, 3, 4]])
def test_rejects_invalid_native_image_input(original):
    with pytest.raises(ValueError):
        expand(original, [1, 10, 13], [[(1, 1)]])
