"""Paired local text versus image-added decision latency, without accuracy scoring.

Identical text prompts are evaluated with no image, one 448px image, and one
896px image. Images are a controlled compute addition, not question evidence.
No answer tokens are generated. Disk I/O and network latency are excluded.
"""

import argparse
import hashlib
import importlib.metadata
import json
import random
import statistics
import time
from pathlib import Path

from PIL import Image

from kev_development_probe import _read_suite, _request
from mlx_systemone import SystemOneRequest, protocol
from visual_probe_cases import build_cases


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
    }


def prepare_cases(suite_path, output):
    records, manifest = _read_suite(suite_path)
    selected = []
    for source in ("mmlu", "sciq"):
        eligible = [r for r in records if r["_meta"]["source"] == source
                    and r["_meta"]["variant"] == "clean"
                    and len(r["questions"]) == 1
                    and all(q["type"] == "choice" and len(q["criteria"]) == 4
                            for q in r["questions"].values())]
        if len(eligible) < 12:
            raise ValueError("Expected twelve four-choice text prompts per source")
        selected.extend(eligible[:12])
    fixtures = build_cases(output / "images")
    cases = []
    for index, record in enumerate(selected):
        request = SystemOneRequest.model_validate(_request(record, "jev-latest"))
        plan = protocol.plan_question(next(iter(request.questions.values())))
        prompt = protocol.build_prompt(request.state, plan, tuple("ABCD"), (0, 1, 2, 3))
        image448 = Image.open(fixtures[index]["image_path"]).convert("RGB")
        image896 = image448.resize((896, 896), Image.Resampling.LANCZOS)
        path896 = output / "images" / f"latency_{index:02d}_896.png"
        image896.save(path896)
        cases.append({
            "id": record["_meta"]["id"], "source": record["_meta"]["source"],
            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "image448_path": fixtures[index]["image_path"], "image896_path": str(path896),
            "images": {"text": [], "image448": [image448], "image896": [image896]},
        })
    (output / "cases.json").write_text(json.dumps([{k: v for k, v in c.items() if k != "images"} for c in cases], indent=2) + "\n")
    return cases, manifest


def summarize(rows):
    phases = ("total_ms", "preprocessing_ms", "embedding_ms", "vision_embedding_ms", "language_prefill_ms")
    result = {}
    for condition in ("text", "image448", "image896"):
        subset = [r for r in rows if r["condition"] == condition]
        result[condition] = {
            "timing": {phase: distribution([r["measurement"][phase] for r in subset]) for phase in phases},
            "total_ms_by_round": {str(round_index): distribution([
                r["measurement"]["total_ms"] for r in subset if r["round"] == round_index
            ]) for round_index in sorted(set(r["round"] for r in subset))},
            "input_tokens": distribution([r["measurement"]["input_tokens"] for r in subset]),
            "image_tokens": sorted(set(r["measurement"]["image_tokens"] for r in subset)),
        }
    text_by_pair = {(r["round"], r["case_index"]): r["measurement"]["total_ms"] for r in rows if r["condition"] == "text"}
    for condition in ("image448", "image896"):
        subset = [r for r in rows if r["condition"] == condition]
        result[condition]["paired_extra_ms"] = distribution([
            r["measurement"]["total_ms"] - text_by_pair[(r["round"], r["case_index"])] for r in subset])
        result[condition]["paired_slowdown"] = distribution([
            r["measurement"]["total_ms"] / text_by_pair[(r["round"], r["case_index"])] for r in subset])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--suite-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases, suite = prepare_cases(args.suite_jsonl, output)
    if args.prepare_only:
        print(json.dumps({"prepared_cases": len(cases), "accuracy_scoring": False}))
        return
    from ministral_latency_backend import MinistralLatencyBackend

    engine = MinistralLatencyBackend(str(args.model.resolve()))
    print(f"Loaded model in {engine.load_seconds:.2f}s", flush=True)
    warmups = []
    for index in (0, 8, 16):
        for condition in ("text", "image448", "image896"):
            m = engine.measure(cases[index]["prompt"], cases[index]["images"][condition])
            warmups.append({"case_index": index, "condition": condition, "measurement": m})
            print(f"warmup {condition}: {m['total_ms']:.1f}ms", flush=True)
    (output / "warmup.json").write_text(json.dumps(warmups, indent=2) + "\n")
    rng = random.Random(20260921)
    rows = []
    started = time.perf_counter()
    with (output / "results.jsonl").open("w") as stream:
        for round_index in range(args.rounds):
            indices = list(range(len(cases)))
            rng.shuffle(indices)
            for index in indices:
                case = cases[index]
                conditions = ["text", "image448", "image896"]
                rng.shuffle(conditions)
                for condition in conditions:
                    measurement = engine.measure(case["prompt"], case["images"][condition])
                    if measurement["output_tokens"] != 0:
                        raise RuntimeError("Unexpected generated tokens")
                    if condition != "text" and not measurement["vision_forward_executed"]:
                        raise RuntimeError("Image did not run through the vision encoder")
                    row = {"round": round_index, "case_index": index, "id": case["id"],
                           "condition": condition, "prompt_sha256": case["prompt_sha256"],
                           "measurement": measurement}
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                if len(rows) % 12 == 0:
                    print(f"Measured {len(rows)}/{len(cases)*3*args.rounds} requests", flush=True)
    manifest_path = args.model.resolve().parent / "model.json"
    model_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if model_manifest and Path(model_manifest["path"]).resolve() != args.model.resolve():
        raise ValueError("Model manifest does not match the loaded directory")
    summary = {
        "model": model_manifest.get("repo", "local Ministral3 checkpoint"),
        "model_path": str(args.model.resolve()),
        "model_revision": model_manifest.get("revision"),
        "model_download_manifest": model_manifest,
        "accuracy_scoring": False, "training_performed": False, "generated_tokens": 0,
        "method": "identical text, paired image addition, randomized interleaving; fresh request KV and vision features",
        "image_content_role": "controlled extra computation; images are unrelated to the text questions",
        "timing_scope": "in-process prompt preparation, image transforms, synchronized vision and language inference, last-position option probabilities; excludes disk/network/HTTP and model loading",
        "load_seconds": engine.load_seconds, "warmup_requests_excluded": len(warmups),
        "rounds": args.rounds, "unique_prompts": len(cases), "requests": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_mlx_memory_gb": engine.mx.get_peak_memory() / 1e9,
        "vision_forward_calls_including_warmup": engine.vision_calls,
        "device": engine.mx.device_info(), "source_suite": suite,
        "versions": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-vlm", "transformers", "pillow")},
        "metrics": summarize(rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"metrics": summary["metrics"], "peak_mlx_memory_gb": summary["peak_mlx_memory_gb"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
