"""Warm SystemOne latency/throughput by concurrency, with actual token counts.

Run separately from throughput/soak traffic. Synthetic inputs only.
  python latency_sweep.py --out /path/latency.json --samples 100 --levels 1,4,8,16
"""
import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.error
import urllib.request


CONTEXT = (
    "Customer reports two charges for one monthly subscription, requests a refund of the duplicate charge, and says the "
    "product works correctly. Support verified two transaction IDs for the same period. "
)
QUESTIONS = [
    {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {"billing": "Payments and refunds", "technical": "Product defects"}},
    {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    {"type": "noul", "instructions": "Was the customer charged twice for one subscription period?"},
    {"type": "noul", "instructions": "Does the customer report a product defect?"},
]


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30010/v1/systemone")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--levels", default="1,4,8,16", help="Concurrent request levels, tested sequentially")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    levels = [int(value) for value in args.levels.split(",")]
    if not levels or any(level < 1 for level in levels):
        parser.error("--levels must contain positive integers")
    cases = [
        ("short-1q", "Please refund my duplicate charge.", {"refund": QUESTIONS[1]}),
        ("ticket-1q", {"ticket": "Bench request small-4q-0. " + CONTEXT}, {"refund": QUESTIONS[1]}),
        ("short-4q", "Please refund my duplicate charge.", {f"q{i+1:02d}": q for i, q in enumerate(QUESTIONS)}),
        ("bench-4q", {"ticket": "Bench request small-4q-0. " + CONTEXT}, {
            f"q{i+1:02d}": {**q, "instructions": q["instructions"] + f" Evaluation {i+1}."}
            for i, q in enumerate(QUESTIONS)
        }),
    ]
    results = []
    for name, state, questions in cases:
        body = {"model": "jev-latest", "state": state, "questions": questions}
        data = json.dumps(body).encode()
        def call(_):
            request = urllib.request.Request(args.url, data=data, headers={"Content-Type": "application/json"})
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.load(response)
                    status = response.status
                valid = set(result.get("answers", {})) == set(questions)
                tokens = result.get("usage", {}).get("input_tokens")
            except urllib.error.HTTPError as error:
                status, valid, tokens = error.code, False, None
            except (urllib.error.URLError, TimeoutError):
                status, valid, tokens = 0, False, None
            elapsed = (time.perf_counter() - start) * 1000
            return {"status": status, "valid_answers": valid, "input_tokens": tokens, "ms": round(elapsed, 4)}
        for level in levels:
            for i in range(8):
                call(i)
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(level) as pool:
                rows = list(pool.map(call, range(args.samples)))
            wall = time.perf_counter() - start
            ok = [row for row in rows if row["status"] == 200 and row["valid_answers"]]
            values = [row["ms"] for row in ok]
            summary = {"case": name, "concurrency": level, "warmup": 8, "samples": len(rows), "ok": len(ok),
                       "input_tokens": sorted({row["input_tokens"] for row in ok}, key=str),
                       "req_s": round(len(ok) / wall, 2), "questions_s": round(len(ok) * len(questions) / wall, 2)}
            if values:
                summary.update({"p50_ms": round(statistics.median(values), 2),
                                "p95_ms": round(percentile(values, .95), 2),
                                "min_ms": round(min(values), 2), "max_ms": round(max(values), 2)})
            print("LATENCY " + json.dumps(summary), flush=True)
            results.append({"summary": summary, "request": body, "rows": rows})
            with open(args.out, "w") as output:
                json.dump(results, output, indent=2)
    if any(item["summary"]["ok"] != args.samples for item in results):
        raise SystemExit("One or more requests failed; inspect the saved rows.")


if __name__ == "__main__":
    main()
