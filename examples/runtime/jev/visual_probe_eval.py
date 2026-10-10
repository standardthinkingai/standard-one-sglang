"""Run small, balanced visual diagnostics on an actual local MLX model.

These simple synthetic probes test the vision-to-decision path; they are not
a general-purpose benchmark. Ground truth and file names never enter prompts.
"""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from mlx_systemone import MLXSystemOne
from visual_probe_cases import _contact_sheet, build_cases


def image_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def payload(case, model, image_path, permutations=1):
    return {
        "model": model,
        "state": "Use the supplied image to answer the visual question.",
        "images": [image_url(image_path)] if image_path else [],
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": case["question"],
                "criteria": case["criteria"],
            }
        },
        "options": {"return_logprobs": True, "permutations": permutations},
    }


def summarize(rows):
    summary = {}
    for condition in sorted({r["condition"] for r in rows}):
        items = [r for r in rows if r["condition"] == condition]
        cats = sorted({r["category"] for r in items})
        summary[condition] = {
            "count": len(items),
            "accuracy_vs_original": statistics.mean(r["correct_original"] for r in items),
            "accuracy_vs_visible_image": None if condition == "no_image" else statistics.mean(r["correct_visible"] for r in items),
            "by_category": {
                cat: statistics.mean(r["correct_original"] for r in items if r["category"] == cat)
                for cat in cats
            },
            "mean_ms": statistics.mean(r["response"]["metadata"]["wall_ms"] for r in items),
            "median_ms": statistics.median(r["response"]["metadata"]["wall_ms"] for r in items),
            "mean_correct_probability": statistics.mean(r["probabilities"][r["expected_original"]] for r in items),
            "mean_label_probability_mass": statistics.mean(
                p["label_probability_mass"] for r in items for p in r["response"]["metadata"]["prefills"]
            ),
            "brier_vs_original": statistics.mean(sum((p - (name == r["expected_original"])) ** 2 for name, p in r["probabilities"].items()) for r in items),
        }
    normal = {r["id"]: r for r in rows if r["condition"] == "normal"}
    swaps = [r for r in rows if r["condition"] == "swapped_image" and r["id"] in normal]
    if swaps:
        summary["swapped_image"]["prediction_changed_from_normal"] = statistics.mean(r["prediction"] != normal[r["id"]]["prediction"] for r in swaps)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local MLX-VLM snapshot")
    parser.add_argument("--served-model", default="mlx-community/Qwen3.5-4B-4bit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--conditions", nargs="+", choices=["normal", "swapped_image", "no_image"], default=["normal", "swapped_image", "no_image"])
    parser.add_argument("--permutations", type=int, default=1)
    parser.add_argument("--kev-dev-jsonl", type=Path, help="Optional frozen transfer-v4 development suite for a separate small text probe")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args.output_dir / "images")
    _contact_sheet(cases, args.output_dir / "contact-sheet.png")
    (args.output_dir / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    groups = defaultdict(list)
    for case in cases:
        groups[case["category"]].append(case)
    chosen = cases[:args.limit]
    print("Loading model with vision encoder...", flush=True)
    engine = MLXSystemOne(args.model, args.served_model)
    print(f"Model loaded in {engine.load_seconds:.2f}s", flush=True)
    # One real-image warmup, recorded separately and excluded from timed cases.
    warmup = engine.score(payload(cases[0], args.served_model, cases[0]["image_path"]))
    (args.output_dir / "warmup.json").write_text(json.dumps(warmup, indent=2) + "\n")
    rows = []
    started = time.perf_counter()
    with (args.output_dir / "results.jsonl").open("w") as results:
        for condition in args.conditions:
            for case in chosen:
                group = groups[case["category"]]
                visible = group[(group.index(case) + 1) % len(group)] if condition == "swapped_image" else case
                path = None if condition == "no_image" else visible["image_path"]
                request = payload(case, args.served_model, path, args.permutations)
                response = engine.score(request)
                answer = response["answers"]["decision"]
                if response["usage"]["output_tokens"] != 0:
                    raise RuntimeError("The scoring path decoded tokens")
                if not math.isclose(sum(answer["probabilities"].values()), 1.0, abs_tol=1e-6):
                    raise RuntimeError("Invalid option distribution")
                row = {
                    "id": case["id"],
                    "category": case["category"],
                    "condition": condition,
                    "expected_original": case["expected"],
                    "expected_visible": None if path is None else visible["expected"],
                    "image_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None,
                    "prediction": answer["choice"],
                    "correct_original": answer["choice"] == case["expected"],
                    "correct_visible": path is not None and answer["choice"] == visible["expected"],
                    "probabilities": answer["probabilities"],
                    "response": response,
                }
                rows.append(row)
                results.write(json.dumps(row) + "\n")
                results.flush()
                print(f"{condition:14s} {case['id']:18s} {answer['choice']:12s} original={row['correct_original']} visible={row['correct_visible']} {response['metadata']['wall_ms']:.0f}ms", flush=True)
    summary = {
        "model": args.served_model,
        "model_path": str(Path(args.model).resolve()),
        "model_inference": True,
        "training_performed": False,
        "benchmark_type": "synthetic_pipeline_diagnostic_not_general_accuracy",
        "image_resolution": [448, 448],
        "ground_truth_in_prompt": False,
        "question_option_order": "fixed_with_balanced_answers",
        "permutations": args.permutations,
        "warmup_excluded": True,
        "vision_feature_cache": False,
        "prefix_cache": False,
        "load_seconds": engine.load_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "vision_forward_calls_including_warmup": engine.vision_calls,
        "peak_memory_gb": engine.mx.get_peak_memory() / 1e9,
        "device": engine.mx.device_info(),
        "versions": {p: importlib.metadata.version(p) for p in ["mlx", "mlx-vlm", "transformers", "pillow"]},
        "metrics": summarize(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if args.kev_dev_jsonl:
        from kev_development_probe import run_probe

        probe_summary = run_probe(engine, args.kev_dev_jsonl, args.output_dir)
        print(json.dumps(probe_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
