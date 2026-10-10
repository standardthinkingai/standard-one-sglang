"""Run a small, frozen subset of KEV's public text-only development suite.

This reports question-level classification accuracy, not a reproduction of the
published KEV benchmark, its complete transfer suite, or its calibration metrics.
Labels, source annotations, IDs, and privileged metadata are evaluator-only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_SHA256 = "ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e"
SOURCE_REVISION = "a957287d1c502a4e2e3b9d9d1325c2c6f27f181c"
SOURCE_URL = (
    "https://huggingface.co/datasets/jaredpalmer/kev-suites/resolve/"
    f"{SOURCE_REVISION}/v4/transfer-v4/development.jsonl"
)


def _read_suite(suite_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = suite_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Frozen suite SHA256 mismatch: {digest}")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(records) != 764:
        raise ValueError(f"Expected 764 frozen suite records, got {len(records)}")
    sources = Counter(record["_meta"]["source"] for record in records)
    variants = Counter(record["_meta"]["variant"] for record in records)
    clean = [record for record in records if record["_meta"]["variant"] == "clean"]
    clean_sources = Counter(record["_meta"]["source"] for record in clean)
    if len(clean_sources) != 8:
        raise ValueError("Expected eight clean-record sources in the frozen suite")
    manifest = {
        "path": str(suite_path.resolve()),
        "bytes": len(raw),
        "sha256": digest,
        "records": len(records),
        "questions": sum(len(record["questions"]) for record in records),
        "records_by_source": dict(sources),
        "records_by_variant": dict(variants),
        "clean_records_by_source": dict(clean_sources),
        "question_types": dict(Counter(
            question["type"]
            for record in records
            for question in record["questions"].values()
        )),
        "records_by_source_and_variant": {
            source: dict(Counter(
                record["_meta"]["variant"]
                for record in records if record["_meta"]["source"] == source
            ))
            for source in sources
        },
        "clean_unique_groups_by_source": {
            source: len({
                record["_meta"].get("group_id", record["_meta"]["id"])
                for record in clean if record["_meta"]["source"] == source
            })
            for source in clean_sources
        },
        "clean_unique_pairs_by_source": {
            source: len({
                record["_meta"]["pair_id"]
                for record in clean
                if record["_meta"]["source"] == source
                and record["_meta"].get("pair_id") is not None
            })
            for source in clean_sources
        },
    }
    return records, manifest


def _select_records(
    records: list[dict[str, Any]], per_source: int,
) -> list[dict[str, Any]]:
    if isinstance(per_source, bool) or not isinstance(per_source, int) or per_source < 1:
        raise ValueError("per_source must be a positive integer")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["_meta"]["variant"] == "clean":
            by_source[record["_meta"]["source"]].append(record)
    selected = []
    for source_records in by_source.values():
        pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in source_records:
            pair_id = record["_meta"].get("pair_id")
            if pair_id is not None:
                pairs[pair_id].append(record)
        chosen, seen_ids, seen_groups = [], set(), set()
        for record in source_records:
            if len(chosen) >= per_source:
                break
            meta = record["_meta"]
            if meta["id"] in seen_ids:
                continue
            pair_id = meta.get("pair_id")
            unit = pairs[pair_id] if pair_id is not None else [record]
            if pair_id is not None and (
                len(unit) != 2
                or {item["_meta"].get("sibling") for item in unit} != {"a", "b"}
            ):
                raise ValueError(f"Expected a complete a/b policy pair: {pair_id}")
            groups = {
                item["_meta"].get("group_id", item["_meta"]["id"])
                for item in unit
            }
            if groups & seen_groups:
                continue
            chosen.extend(unit)
            seen_ids.update(item["_meta"]["id"] for item in unit)
            seen_groups.update(groups)
        # Preserve complete pairs even if this exceeds an odd per-source limit.
        selected.extend(chosen)
    return selected


def _request(record: dict[str, Any], model: str) -> dict[str, Any]:
    # Whitelist public model inputs: never serialize the labeled source record.
    return {
        "model": model,
        "state": copy.deepcopy(record["state"]),
        "images": [],
        "questions": {
            key: {
                field: copy.deepcopy(question[field])
                for field in ("type", "instructions", "criteria")
                if field in question
            }
            for key, question in record["questions"].items()
        },
        "options": {"return_logprobs": True, "permutations": 1},
    }


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Model probability must be a number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Model probability must be finite and within [0, 1]")
    return float(value)


def _prediction(question: dict[str, Any], answer: dict[str, Any]) -> Any:
    kind = question["type"]
    if kind == "noul":
        return _probability(answer["noul"]) >= 0.5
    if kind == "choice":
        names = list(question["criteria"])
    elif kind == "score":
        names = [str(index) for index in range(len(question["criteria"]))]
    else:
        raise ValueError(f"Unsupported question type: {kind}")
    probabilities = answer["probabilities"]
    if set(probabilities) != set(names):
        raise ValueError("Response probability keys do not match candidate labels")
    values = [_probability(probabilities[name]) for name in names]
    if not math.isclose(math.fsum(values), 1.0, abs_tol=1e-6):
        raise ValueError("Response probabilities must sum to one")
    index = max(range(len(names)), key=values.__getitem__)
    return names[index] if kind == "choice" else index


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    questions = [question for row in rows for question in row["questions"].values()]
    latencies = [row["request_latency_ms"] for row in rows]
    return {
        "record_count": len(rows),
        "question_count": len(questions),
        "correct_questions": sum(question["correct"] for question in questions),
        "question_accuracy": (
            statistics.mean(question["correct"] for question in questions)
            if questions else None
        ),
        "failed_questions": sum(question.get("error") is not None for question in questions),
        "mean_request_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_request_latency_ms": statistics.median(latencies) if latencies else None,
        "total_request_latency_ms": math.fsum(latencies),
    }


def run_probe(
    engine: Any, suite_path: Path, output_dir: Path, per_source: int = 4,
) -> dict[str, Any]:
    """Run real engine.score calls; callers may supply a fake engine for tests."""
    records, manifest = _read_suite(Path(suite_path))
    selected = _select_records(records, per_source)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "kev-development-probe.jsonl"
    summary_path = output_dir / "kev-development-summary.json"
    rows = []
    started = time.perf_counter()
    with result_path.open("w", encoding="utf-8") as results:
        for index, record in enumerate(selected):
            request = _request(record, engine.served_model)
            response, error = None, None
            request_started = time.perf_counter()
            try:
                response = engine.score(request)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - request_started) * 1000
            question_results = {}
            for key, question in record["questions"].items():
                prediction, question_error = None, error
                if question_error is None:
                    try:
                        prediction = _prediction(question, response["answers"][key])
                    except (KeyError, TypeError, ValueError) as exc:
                        question_error = f"{type(exc).__name__}: {exc}"
                question_results[key] = {
                    "type": question["type"],
                    "expected": question["label"],
                    "prediction": prediction,
                    "correct": question_error is None and prediction == question["label"],
                    "error": question_error,
                }
            meta = record["_meta"]
            row = {
                "id": meta["id"],
                "source": meta["source"],
                "group_id": meta.get("group_id"),
                "pair_id": meta.get("pair_id"),
                "variant": meta["variant"],
                "request_latency_ms": latency_ms,
                "questions": question_results,
                "response": response,
                "request_error": error,
            }
            rows.append(row)
            results.write(json.dumps(row, allow_nan=False) + "\n")
            results.flush()
            correct = sum(item["correct"] for item in question_results.values())
            print(
                f"KEV development {index + 1}/{len(selected)} {meta['source']} "
                f"correct={correct}/{len(question_results)} {latency_ms:.0f}ms",
                flush=True,
            )
    summary = {
        "source": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "frozen_sha256": EXPECTED_SHA256,
        "suite_manifest": manifest,
        "model": engine.served_model,
        "partial_not_published_benchmark": True,
        "benchmark_type": "partial_kev_transfer_v4_development_text_only_probe",
        "selection": {
            "variant": "clean",
            "requested_records_per_source": per_source,
            "method": "source file order, distinct groups, complete a/b pairs retained",
            "selected_ids": [record["_meta"]["id"] for record in selected],
            "selected_question_types": dict(Counter(
                question["type"]
                for record in selected
                for question in record["questions"].values()
            )),
        },
        "images": False,
        "training_performed": False,
        "labels_or_metadata_in_prompt": False,
        "permutations": 1,
        "temperature": 1.0,
        "metrics": _metrics(rows),
        "per_source": {
            source: _metrics([row for row in rows if row["source"] == source])
            for source in dict.fromkeys(row["source"] for row in rows)
        },
        "elapsed_seconds": time.perf_counter() - started,
        "results_path": str(result_path),
        "summary_path": str(summary_path),
        "metric_notes": (
            "Question-level classification accuracy only: choice probability argmax, "
            "noul >= 0.5, score probability argmax (not rounded expected score). "
            "Failures remain in the accuracy denominator. No ECE, calibration, "
            "complete-suite, or published-KEV result reproduction is claimed. "
            "Request latency includes engine.score and has no separate warmup exclusion."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local MLX-VLM snapshot directory")
    parser.add_argument("--suite-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=4)
    args = parser.parse_args()
    # Fail provenance validation before allocating or loading a real model.
    _read_suite(args.suite_jsonl)
    from mlx_systemone import MLXSystemOne

    engine = MLXSystemOne(args.model)
    summary = run_probe(engine, args.suite_jsonl, args.output_dir, args.per_source)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
