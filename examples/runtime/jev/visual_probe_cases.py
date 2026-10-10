"""Small deterministic visual probes for an untrained multimodal decision model.

    python visual_probe_cases.py --output-dir /tmp/jev-visual-probes

Each category has four images, an identical question and candidate dictionary,
and exactly one example of each answer. Swapping images within a category is a
counterfactual check: the correct decision changes while all prompt text stays
fixed. This is a pipeline smoke test, not a measure of general visual ability.

The evaluator must keep id, filename, expected, and contact-sheet annotations
out of the model prompt. Send only image pixels, question, and criteria.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

SIZE = 448
BACKGROUND = "#f6f8fb"
INK = "#172334"
COLORS = {
    "red": "#df3434",
    "green": "#20964b",
    "blue": "#2469d4",
    "yellow": "#f1ca24",
}


def _font(size: int) -> Any:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(name).is_file():
            return ImageFont.truetype(name, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow before 10.1.
        return ImageFont.load_default()


def _center(
    draw: ImageDraw.ImageDraw,
    text: str,
    center: tuple[int, int],
    size: int = 26,
    fill: str = INK,
) -> None:
    font = _font(size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (center[0] - (right - left) / 2 - left,
         center[1] - (bottom - top) / 2 - top),
        text,
        font=font,
        fill=fill,
    )


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    return image, ImageDraw.Draw(image)


def _render(category: str, variant: int) -> Image.Image:
    image, draw = _canvas()
    if category == "color":
        color = list(COLORS.values())[variant]
        draw.rectangle((114, 114, 334, 334), fill=color)
    elif category == "count":
        positions = {
            1: [(224, 224)],
            2: [(134, 224), (314, 224)],
            3: [(224, 130), (130, 308), (318, 308)],
            4: [(134, 134), (314, 134), (134, 314), (314, 314)],
        }
        for x, y in positions[variant + 1]:
            draw.ellipse((x - 37, y - 37, x + 37, y + 37), fill=COLORS["blue"])
    elif category == "spatial":
        draw.rectangle((184, 184, 264, 264), fill=COLORS["blue"])
        x, y = [(88, 224), (360, 224), (224, 88), (224, 360)][variant]
        draw.ellipse((x - 36, y - 36, x + 36, y + 36), fill=COLORS["red"])
    elif category == "ocr":
        draw.rounded_rectangle(
            (54, 146, 394, 302), radius=14, fill="white", outline="#68798c", width=3
        )
        _center(draw, ["NORTH", "SOUTH", "EAST", "WEST"][variant], (224, 224), 52)
    elif category == "bar_chart":
        _center(draw, "ITEM SALES", (224, 37), 25)
        for tick in range(0, 101, 25):
            y = 354 - int(tick * 2.5)
            draw.line((63, y, 412, y), fill="#d3dbe5", width=1)
            _center(draw, str(tick), (33, y), 16)
        draw.line((63, 93, 63, 354, 412, 354), fill=INK, width=2)
        # Rotate the same height pattern; every bar is tallest in one variant.
        heights = [90, 35, 60, 20]
        heights = heights[-variant:] + heights[:-variant] if variant else heights
        for index, height in enumerate(heights):
            x = 85 + index * 82
            draw.rectangle(
                (x, 354 - int(height * 2.5), x + 47, 353), fill=COLORS["blue"]
            )
            _center(draw, "ABCD"[index], (x + 24, 384), 24)
    elif category == "ui_status":
        draw.rounded_rectangle(
            (28, 28, 420, 420), radius=17, fill="white", outline="#bcc9d8", width=2
        )
        draw.rounded_rectangle((29, 29, 419, 89), radius=15, fill="#24364d")
        draw.rectangle((29, 61, 419, 89), fill="#24364d")
        _center(draw, "FILE EXPORT", (224, 59), 25, "white")
        _center(draw, "Monthly report", (224, 129), 27)
        badge, badge_color, detail, action, enabled, progress = [
            ("Ready", "#486581", "Waiting to start", "Start export", True, 0),
            ("Exporting", "#2469d4", "Creating your file...", "Please wait", False, 0.56),
            ("Complete", "#20964b", "Your file is ready", "Download", True, 1),
            ("Failed", "#c72f36", "Connection interrupted", "Retry export", True, 0.27),
        ][variant]
        draw.rounded_rectangle((115, 173, 333, 216), radius=21, fill=badge_color)
        _center(draw, badge, (224, 194), 24, "white")
        draw.rounded_rectangle((64, 245, 384, 259), radius=7, fill="#e0e6ee")
        if progress:
            draw.rounded_rectangle(
                (64, 245, 64 + int(320 * progress), 259), radius=7, fill=badge_color
            )
        _center(draw, detail, (224, 288), 20)
        draw.rounded_rectangle(
            (79, 332, 369, 389), radius=9,
            fill="#2469d4" if enabled else "#e0e6ee",
            outline="#2469d4" if enabled else "#d2dae5",
        )
        _center(draw, action, (224, 360), 24, "white" if enabled else "#8592a2")
    else:
        raise ValueError(f"Unknown probe category: {category}")
    return image


def validate_cases(cases: list[dict[str, Any]]) -> None:
    """Check balancing and actual image files before running model inference."""
    if len(cases) != 24 or len({case["id"] for case in cases}) != 24:
        raise ValueError("Expected 24 probes with unique identifiers")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[case["category"]].append(case)
        path = Path(case["image_path"])
        if not path.is_absolute():
            raise ValueError(f"Image path must be absolute: {path}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (SIZE, SIZE):
                raise ValueError(f"Expected a {SIZE}x{SIZE} RGB image: {path}")
            image.load()
    if len(groups) != 6:
        raise ValueError("Expected six categories")
    for category, group in groups.items():
        reference = group[0]
        if len(group) != 4 or len(reference["criteria"]) != 4:
            raise ValueError(f"Expected four probes and four choices: {category}")
        if any(
            case["question"] != reference["question"]
            or case["criteria"] != reference["criteria"]
            for case in group
        ):
            raise ValueError(f"Prompt text must be fixed within {category}")
        if Counter(case["expected"] for case in group) != Counter(reference["criteria"].keys()):
            raise ValueError(f"Answers must be balanced within {category}")


def build_cases(output_dir: Path) -> list[dict[str, Any]]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = [
        ("color", "What color is the square?", {
            color: f"The square is {color}." for color in COLORS
        }),
        ("count", "How many blue circles are in the image?", {
            "one": "There is one blue circle.",
            "two": "There are two blue circles.",
            "three": "There are three blue circles.",
            "four": "There are four blue circles.",
        }),
        ("spatial", "Where is the red circle relative to the blue square?", {
            "left": "The red circle is to the left of the blue square.",
            "right": "The red circle is to the right of the blue square.",
            "above": "The red circle is above the blue square.",
            "below": "The red circle is below the blue square.",
        }),
        ("ocr", "Which word is printed inside the sign?", {
            word.lower(): f"The printed word is {word}."
            for word in ("NORTH", "SOUTH", "EAST", "WEST")
        }),
        ("bar_chart", "Which labeled bar is the tallest?", {
            letter: f"Bar {letter} is the tallest." for letter in "ABCD"
        }),
        ("ui_status", "What is the export task's current status in this application?", {
            "ready": "The task is ready to start and has not started yet.",
            "running": "The export is currently running and has not finished.",
            "complete": "The export completed successfully and the file can be downloaded.",
            "error": "The export failed and needs a retry.",
        }),
    ]
    cases = []
    for category, question, criteria in definitions:
        for variant, expected in enumerate(criteria):
            case_id = f"{category}_{variant + 1:02d}"
            path = output_dir / f"{case_id}.png"
            _render(category, variant).save(path, format="PNG")
            cases.append({
                "id": case_id,
                "category": category,
                "image_path": str(path),
                "question": question,
                "criteria": dict(criteria),
                "expected": expected,
            })
    validate_cases(cases)
    return cases


def _contact_sheet(cases: list[dict[str, Any]], output_path: Path) -> None:
    # This human-only sheet intentionally contains evaluator labels.
    tile, label_height = 224, 34
    sheet = Image.new("RGB", (tile * 4, (tile + label_height) * 6), "#e7edf4")
    draw = ImageDraw.Draw(sheet)
    for index, case in enumerate(cases):
        x, y = (index % 4) * tile, (index // 4) * (tile + label_height)
        with Image.open(case["image_path"]) as source:
            sheet.paste(source.resize((tile, tile), Image.Resampling.LANCZOS), (x, y))
        _center(draw, f"{case['id']}: {case['expected']}", (x + tile // 2, y + tile + 17), 14)
    sheet.save(output_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    cases = build_cases(output_dir)
    (output_dir / "cases.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    _contact_sheet(cases, output_dir / "contact-sheet.png")
    print(f"Created {len(cases)} balanced RGB probes in {output_dir}")


if __name__ == "__main__":
    main()
