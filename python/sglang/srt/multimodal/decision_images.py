"""Preserve native instruction tokens while expanding Pixtral image patches.

The image processor sees an image-only placeholder prompt. Its patch rows are
spliced into the caller's token IDs without decoding or retokenizing the chat.
This module deliberately has no GPU or SGLang imports.
"""


def rebuild_decision_image_ids(
    original_ids,
    processed_ids,
    offsets_by_image,
    *,
    image_token_id,
    image_break_id,
    image_end_id,
):
    """Return native IDs with expanded images and translated inclusive offsets.

    When supplied, offsets must provide one item per image with complete,
    ordered patch-row coverage. Processors may pass None before offsets are
    computed from the rebuilt IDs. Unexpected layouts fail instead of silently
    changing the text on which the decision is conditioned.
    """
    markers = (image_token_id, image_break_id, image_end_id)
    if (
        any(type(token) is not int or token < 0 for token in markers)
        or len(set(markers)) != 3
    ):
        raise ValueError("decision image markers must be distinct token IDs")
    for tokens in (original_ids, processed_ids):
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ValueError("decision images require nonempty integer token IDs")
    if any(token in (image_break_id, image_end_id) for token in original_ids):
        raise ValueError("decision image inputs must contain unexpanded placeholders")

    # Each image consists of one or more IMG rows separated by IMG_BREAK and
    # terminated by IMG_END. Other processor-added tokens (e.g. BOS) are unused.
    blocks, row_groups = [], []
    index = 0
    while index < len(processed_ids):
        token = processed_ids[index]
        if token in (image_break_id, image_end_id):
            raise ValueError("decision processor returned an orphan image marker")
        if token != image_token_id:
            index += 1
            continue
        start, rows = index, []
        while True:
            row_start = index
            while index < len(processed_ids) and processed_ids[index] == image_token_id:
                index += 1
            if index == row_start or index == len(processed_ids):
                raise ValueError("decision processor returned an incomplete image row")
            rows.append((row_start, index - 1))
            separator = processed_ids[index]
            index += 1
            if separator == image_end_id:
                break
            if separator != image_break_id:
                raise ValueError(
                    "decision processor returned an invalid image separator"
                )
        blocks.append((start, index))
        row_groups.append(rows)

    count = original_ids.count(image_token_id)
    if not count or len(blocks) != count:
        raise ValueError("decision image placeholders, blocks, and features must align")
    if offsets_by_image is not None and len(offsets_by_image) != count:
        raise ValueError("decision image placeholders, blocks, and features must align")
    for actual, expected in zip(offsets_by_image or row_groups, row_groups):
        if (
            not isinstance(actual, (list, tuple))
            or any(
                not isinstance(row, (list, tuple))
                or len(row) != 2
                or any(type(value) is not int for value in row)
                for row in actual
            )
            or [tuple(row) for row in actual] != expected
        ):
            raise ValueError("decision image feature offsets do not match patch rows")

    rebuilt, translated = [], []
    next_image = 0
    for token in original_ids:
        if token != image_token_id:
            rebuilt.append(token)
            continue
        start, end = blocks[next_image]
        shift = len(rebuilt) - start
        translated.append(
            [(left + shift, right + shift) for left, right in row_groups[next_image]]
        )
        rebuilt.extend(processed_ids[start:end])
        next_image += 1
    return rebuilt, translated
