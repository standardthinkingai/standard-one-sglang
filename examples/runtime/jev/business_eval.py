#!/usr/bin/env python3
"""Synthetic Korean business decisions against /v1/systemone (stdlib only).

These explicit-policy fixtures test rule following and the API contract. They
are not a representative business benchmark. Keep scenario families disjoint
between calibration and evaluation; fit temperature before opening eval results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

DEFAULT_CASES = Path(__file__).with_name("business_cases.jsonl")
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
SCHEMA = "jev-business-evaluation-v1"


def strict_json(text):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(text, parse_constant=reject)


def number(value, name, minimum=None, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return float(value)


def option_names(question):
    kind = question["type"]
    if kind == "noul":
        return ["true", "false"]
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return list(question["criteria"])


def validate_case(case):
    for name in ("id", "family"):
        if not isinstance(case.get(name), str) or not case[name].strip():
            raise ValueError(f"case needs a nonempty {name}")
    if case.get("split") not in ("calibration", "eval"):
        raise ValueError(f"{case['id']}: invalid split")
    if not isinstance(case.get("state"), (str, dict, list)):
        raise TypeError(f"{case['id']}: state must be text, object, or array")
    questions, expected = case.get("questions"), case.get("expected")
    if (
        not isinstance(questions, dict)
        or not questions
        or not isinstance(expected, dict)
    ):
        raise ValueError(f"{case['id']}: questions and expected must be nonempty maps")
    if set(questions) != set(expected):
        raise ValueError(f"{case['id']}: expected keys must match question keys")
    for key, question in questions.items():
        if not isinstance(key, str) or not key or not isinstance(question, dict):
            raise ValueError("invalid question")
        if (
            not isinstance(question.get("instructions"), str)
            or not question["instructions"].strip()
        ):
            raise ValueError(f"{case['id']}/{key}: explicit instructions required")
        kind, criteria, target = (
            question.get("type"),
            question.get("criteria"),
            expected[key],
        )
        if kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError("choice needs 2..255 criteria")
            if any(not isinstance(name, str) or not name for name in criteria):
                raise ValueError("choice names must be nonempty strings")
            if not isinstance(target, str) or target not in criteria:
                raise ValueError("choice target must be an offered choice")
        elif kind == "noul":
            if type(target) is not bool:
                raise ValueError("noul target must be a boolean")
            if criteria is not None and (
                not isinstance(criteria, dict) or set(criteria) != {"true", "false"}
            ):
                raise ValueError("invalid noul criteria")
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("score needs 2..10 criteria")
            if type(target) is not int or not 0 <= target < len(criteria):
                raise ValueError("score target must be an exact valid level index")
        else:
            raise ValueError(f"unsupported type: {kind}")
    json.dumps(case, allow_nan=False)


def load_cases(path):
    content = Path(path).read_bytes()
    cases, seen, families = [], set(), {}
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = strict_json(line)
            validate_case(case)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if case["id"] in seen:
            raise ValueError(f"duplicate case ID: {case['id']}")
        if case["family"] in families and families[case["family"]] != case["split"]:
            raise ValueError(f"scenario family leaks across splits: {case['family']}")
        seen.add(case["id"])
        families[case["family"]] = case["split"]
        cases.append(case)
    if not cases:
        raise ValueError("dataset is empty")
    return cases, hashlib.sha256(content).hexdigest()


def request_for(case, model, temperature):
    temperature = number(temperature, "temperature", minimum=1e-12)
    return {
        "model": model,
        "state": case["state"],
        "questions": case["questions"],
        "options": {"temperature": temperature, "return_logprobs": True},
    }


def normalize(logprobs, temperature):
    top = max(logprobs)
    weights = [math.exp((p - top) / temperature) for p in logprobs]
    total = math.fsum(weights)
    return [p / total for p in weights]


def read_answers(case, response):
    """Validate the contract; never invent a missing probability or token count."""
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise TypeError("response has no answers map")
    usage = response.get("usage", {})
    if type(usage.get("output_tokens")) is not int or usage["output_tokens"] != 0:
        raise ValueError("usage.output_tokens must explicitly equal integer zero")
    if set(response["answers"]) != set(case["questions"]):
        raise ValueError("response question keys do not match request")
    observations = []
    for key, question in case["questions"].items():
        answer = response["answers"][key]
        kind, names = question["type"], option_names(question)
        if answer.get("type") != kind:
            raise ValueError(f"{key}: wrong answer type")
        if kind == "noul":
            p = number(answer.get("noul"), f"{key}.noul", 0, 1)
            probabilities = [p, 1 - p]
            predicted = p >= 0.5
        else:
            distribution = answer.get("probabilities")
            if not isinstance(distribution, dict) or set(distribution) != set(names):
                raise ValueError(f"{key}: probability keys do not match criteria")
            probabilities = [
                number(distribution[name], f"{key}.{name}", 0, 1) for name in names
            ]
            if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-6):
                raise ValueError(f"{key}: probabilities do not sum to one")
            predicted = names[max(range(len(names)), key=probabilities.__getitem__)]
            if kind == "choice" and answer.get("choice") != predicted:
                raise ValueError(f"{key}: choice disagrees with distribution argmax")
            if kind == "score":
                predicted = number(
                    answer.get("score"), f"{key}.score", 0, len(names) - 1
                )
                mean = math.fsum(i * p for i, p in enumerate(probabilities))
                if not math.isclose(predicted, mean, abs_tol=1e-6):
                    raise ValueError(f"{key}: score disagrees with expected level")
        rows = answer.get("logprobs")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{key}: requested per-permutation logprobs are absent")
        vectors = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != set(names):
                raise ValueError(f"{key}: invalid canonical logprob keys")
            vectors.append(
                [number(row[name], f"{key}.logprobs.{name}") for name in names]
            )
        target = case["expected"][key]
        target_index = (
            (0 if target else 1)
            if kind == "noul"
            else (target if kind == "score" else names.index(target))
        )
        observations.append(
            {
                "task": case["family"] + "/" + key,
                "type": kind,
                "names": names,
                "expected": target,
                "target_index": target_index,
                "predicted": predicted,
                "probabilities": probabilities,
                "logprobs": vectors,
            }
        )
    return observations


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def probability_metrics(rows, bins=10):
    if not rows:
        return {"nll": None, "brier": None, "ece": None, "ece_bins": bins}
    # Multiclass Brier uses the sum across categories; noul uses binary Brier.
    nll, brier, buckets = [], [], [[] for _ in range(bins)]
    for row in rows:
        ps, target = row["probabilities"], row["target_index"]
        nll.append(-math.log(max(ps[target], 1e-15)))
        if row["type"] == "noul":
            brier.append((ps[0] - (1 if target == 0 else 0)) ** 2)
        else:
            brier.append(
                math.fsum(
                    (p - (1 if i == target else 0)) ** 2 for i, p in enumerate(ps)
                )
            )
        winner = max(range(len(ps)), key=ps.__getitem__)
        confidence = ps[winner]
        buckets[min(bins - 1, int(confidence * bins))].append(
            (confidence, winner == target)
        )
    ece = 0.0
    for bucket in buckets:
        if bucket:
            ece += abs(
                math.fsum(p for p, _ in bucket) - sum(ok for _, ok in bucket)
            ) / len(rows)
    return {
        "nll": math.fsum(nll) / len(rows),
        "brier": math.fsum(brier) / len(rows),
        "ece": ece,
        "ece_bins": bins,
    }


def macro_f1(rows):
    # Scope classes by task: "approve" in two unrelated policies is not one class.
    labels, tp, fp, fn = set(), Counter(), Counter(), Counter()
    for row in rows:
        truth = (row["task"], str(row["expected"]))
        prediction = (row["task"], str(row["predicted"]))
        labels.update((truth, prediction))
        if truth == prediction:
            tp[truth] += 1
        else:
            fp[prediction] += 1
            fn[truth] += 1
    return (
        math.fsum(
            2 * tp[label] / (2 * tp[label] + fp[label] + fn[label]) for label in labels
        )
        / len(labels)
        if labels
        else None
    )


def summarize(cases, traces):
    expected_counts = Counter(
        q["type"] for case in cases for q in case["questions"].values()
    )
    observations = [
        row
        for trace in traces
        if trace["status"] == "ok"
        for row in trace["observations"]
    ]
    metrics = {}
    for kind in ("choice", "noul", "score"):
        rows = [row for row in observations if row["type"] == kind]
        item = {
            "n_expected": expected_counts[kind],
            "n_scored": len(rows),
            **probability_metrics(rows),
        }
        if kind == "score":
            item["mae"] = (
                math.fsum(abs(row["predicted"] - row["expected"]) for row in rows)
                / len(rows)
                if rows
                else None
            )
        else:
            correct = sum(row["predicted"] == row["expected"] for row in rows)
            item["accuracy"] = (
                correct / expected_counts[kind] if expected_counts[kind] else None
            )
            item["accuracy_valid_responses"] = correct / len(rows) if rows else None
            item["macro_f1_valid_responses"] = macro_f1(rows)
        metrics[kind] = item
    successful = sum(trace["status"] == "ok" for trace in traces)
    latencies = [trace["latency_ms"] for trace in traces]
    return {
        "requests": len(traces),
        "successful_requests": successful,
        "failed_requests": len(traces) - successful,
        "zero_output_tokens_verified": bool(traces) and successful == len(traces),
        "latency_ms_all_attempts": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "metrics": metrics,
        "metric_notes": "Accuracy counts failed requests as incorrect. Other quality metrics use valid responses only; n_scored is reported. ECE is top-label ECE over 10 equal-width bins. Brier is binary for noul and summed multiclass for choice/score. F1 uses observed truth/prediction labels scoped by policy family and question. NLL clips zero probability at 1e-15. Latencies include HTTP and failures.",
    }


def call_endpoint(base_url, payload, timeout):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("SGLANG_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["SGLANG_API_KEY"]
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/systemone",
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return strict_json(response.read().decode("utf-8"))


def run_evaluation(
    cases,
    dataset_sha,
    base_url,
    model,
    split,
    temperature,
    timeout=120,
    transport=call_endpoint,
):
    chosen = [case for case in cases if case["split"] == split]
    if not chosen:
        raise ValueError(f"no cases in split {split}")
    traces = []
    for case in chosen:
        payload = request_for(case, model, temperature)
        trace = {
            "id": case["id"],
            "family": case["family"],
            "split": split,
            "request": payload,
        }
        start = time.perf_counter()
        try:
            response = transport(base_url, payload, timeout)
            trace["response"] = response
            trace["observations"] = read_answers(case, response)
            trace["status"] = "ok"
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            trace["status"] = "error"
            trace["error"] = f"{type(exc).__name__}: {exc}"
        trace["latency_ms"] = (time.perf_counter() - start) * 1000
        traces.append(trace)
    return {
        "schema": SCHEMA,
        "kind": "synthetic_business_diagnostic",
        "dataset_sha256": dataset_sha,
        "split": split,
        "model": model,
        "temperature": temperature,
        "base_url": base_url,
        "summary": summarize(chosen, traces),
        "traces": traces,
    }


def fit_temperature(report, cases, dataset_sha):
    """Fit one global T using only identified calibration examples and raw rows."""
    if report.get("schema") != SCHEMA or report.get("split") != "calibration":
        raise ValueError(
            "temperature fitting requires a calibration report, never eval"
        )
    if report.get("dataset_sha256") != dataset_sha:
        raise ValueError("calibration report does not match this dataset revision")
    lookup = {case["id"]: case for case in cases if case["split"] == "calibration"}
    rows, seen = [], set()
    for trace in report.get("traces", []):
        key = trace.get("id")
        if trace.get("split") != "calibration" or key not in lookup or key in seen:
            raise ValueError("unknown, duplicate, or non-calibration trace")
        if trace.get("family") != lookup[key]["family"] or trace.get("status") != "ok":
            raise ValueError(
                "all calibration traces must be successful and match their family"
            )
        expected_request = request_for(
            lookup[key], report["model"], report["temperature"]
        )
        if trace.get("request") != expected_request:
            raise ValueError(
                "calibration request differs from the fixture or run settings"
            )
        seen.add(key)
        rows.extend(read_answers(lookup[key], trace["response"]))
    if seen != set(lookup) or not rows:
        raise ValueError(
            "calibration report must include every calibration case exactly once"
        )

    def loss(log_temperature):
        temperature = math.exp(log_temperature)
        losses = []
        for row in rows:
            target_ps = [
                normalize(vector, temperature)[row["target_index"]]
                for vector in row["logprobs"]
            ]
            probability = math.fsum(target_ps) / len(target_ps)
            losses.append(-math.log(max(probability, 1e-15)))
        return math.fsum(losses) / len(losses)

    low, high = math.log(0.05), math.log(20.0)
    grid = [low + i * (high - low) / 80 for i in range(81)]
    index = min(range(len(grid)), key=lambda i: loss(grid[i]))
    left, right = grid[max(0, index - 1)], grid[min(80, index + 1)]
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        a, b = right - ratio * (right - left), left + ratio * (right - left)
        if loss(a) <= loss(b):
            right = b
        else:
            left = a
    optimum = min((low, high, (left + right) / 2, 0.0), key=loss)
    temperature = math.exp(optimum)
    return {
        "schema": "jev-business-calibration-v1",
        "model": report["model"],
        "dataset_sha256": dataset_sha,
        "source_split": "calibration",
        "n_questions": len(rows),
        "temperature": temperature,
        "options": {"temperature": temperature},
        "nll_before": loss(0.0),
        "nll_after": loss(optimum),
        "search_bounds": [0.05, 20.0],
        "at_search_boundary": math.isclose(optimum, low) or math.isclose(optimum, high),
        "note": "One global temperature, fitted before averaging permutation probabilities. Validated only on synthetic calibration fixtures; freeze before evaluation. No eval examples used.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=("calibration", "eval"), default="eval")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--fit-temperature", type=Path, metavar="CALIBRATION_REPORT")
    args = parser.parse_args(argv)
    try:
        cases, dataset_sha = load_cases(args.cases)
        number(args.temperature, "temperature", minimum=1e-12)
        number(args.timeout, "timeout", minimum=1e-12)
        if args.validate_only:
            result = {
                "kind": "fixture_validation_only",
                "dataset_sha256": dataset_sha,
                "cases": len(cases),
                "questions": sum(len(case["questions"]) for case in cases),
                "splits": dict(Counter(case["split"] for case in cases)),
                "families": len({case["family"] for case in cases}),
                "model_performance_measured": False,
            }
        elif args.fit_temperature:
            if not args.output:
                parser.error(
                    "--fit-temperature requires --output for the calibration config"
                )
            result = fit_temperature(
                strict_json(args.fit_temperature.read_text()), cases, dataset_sha
            )
        else:
            if not args.output:
                parser.error(
                    "evaluation requires --output to preserve traces and metrics"
                )
            result = run_evaluation(
                cases,
                dataset_sha,
                args.base_url,
                args.model,
                args.split,
                args.temperature,
                args.timeout,
            )
        if args.output:
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                result.get("summary", result),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return int(result.get("summary", {}).get("failed_requests", 0) > 0)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
