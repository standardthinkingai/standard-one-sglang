"""Summarize a parity_run.py output against the parity dataset gold labels."""

import argparse
import json
import math
import statistics
from collections import defaultdict


def vector(answer, option_names):
    if answer["type"] == "noul":
        p = answer["noul"]
        return {"true": p, "false": 1.0 - p}
    return {name: answer["probabilities"][name] for name in option_names}


def option_names(question):
    if question["type"] == "noul":
        return ["true", "false"]
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return list(question["criteria"])


def argmax(vec, names):
    return max(names, key=lambda name: vec[name])


def pct(values, q):
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--a", default="ref")
    parser.add_argument("--b", default="fork")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    data = {row["id"]: row for row in map(json.loads, open(args.data))}
    records = [json.loads(line) for line in open(args.run)]

    per = []
    for record in records:
        row = data[record["id"]]
        question = row["request"]["questions"][row["question"]]
        names = option_names(question)
        item = {"id": row["id"], "suite": row["suite"], "type": row["type"],
                "gold": row["gold"], "tokens": {}}
        ok = True
        for side in (args.a, args.b):
            result = record[side]
            if result["status"] != 200:
                ok = False
                item[side] = {"error": result["response"]}
                continue
            answer = result["response"]["answers"][row["question"]]
            vec = vector(answer, names)
            item[side] = {"vec": vec, "pred": argmax(vec, names),
                          "latency_ms": result["latency_s"] * 1000}
            item["tokens"][side] = result["response"]["usage"]["input_tokens"]
            if "logprobs" in answer:
                item[side]["logprobs"] = answer["logprobs"][0]
        item["ok"] = ok
        if ok:
            a, b = item[args.a]["vec"], item[args.b]["vec"]
            diffs = [abs(a[n] - b[n]) for n in names]
            item["max_abs_dp"] = max(diffs)
            item["mean_abs_dp"] = sum(diffs) / len(diffs)
            item["agree"] = item[args.a]["pred"] == item[args.b]["pred"]
            item["margin_a"] = sorted(a.values())[-1] - sorted(a.values())[-2]
        per.append(item)

    good = [i for i in per if i["ok"]]
    def summary(items):
        out = {"n": len(items)}
        if not items:
            return out
        out["argmax_agreement_pct"] = 100 * sum(i["agree"] for i in items) / len(items)
        out["mean_abs_dp"] = statistics.fmean(i["mean_abs_dp"] for i in items)
        out["mean_max_abs_dp"] = statistics.fmean(i["max_abs_dp"] for i in items)
        out["max_abs_dp"] = max(i["max_abs_dp"] for i in items)
        out["p99_max_abs_dp"] = pct([i["max_abs_dp"] for i in items], 0.99)
        for side in (args.a, args.b):
            out[f"acc_{side}_pct"] = 100 * sum(i[side]["pred"] == i["gold"] for i in items) / len(items)
            lat = [i[side]["latency_ms"] for i in items]
            out[f"lat_{side}_p50_ms"] = statistics.median(lat)
            out[f"lat_{side}_p95_ms"] = pct(lat, 0.95)
            out[f"lat_{side}_mean_ms"] = statistics.fmean(lat)
        out["same_input_tokens"] = sum(
            len(set(i["tokens"].values())) == 1 for i in items
        )
        return out

    groups = defaultdict(list)
    for item in good:
        groups[item["suite"]].append(item)
        groups["type:" + item["type"]].append(item)
    result = {"all": summary(good), "errors": len(per) - len(good)}
    result.update({k: summary(v) for k, v in sorted(groups.items())})
    disagreements = [
        {k: i[k] for k in ("id", "suite", "type", "gold", "max_abs_dp", "margin_a")}
        | {args.a: i[args.a]["vec"], args.b: i[args.b]["vec"]}
        for i in good if not i["agree"]
    ]
    result["disagreements"] = disagreements
    print(json.dumps({k: v for k, v in result.items() if k != "disagreements"}, indent=1))
    print("disagreements:")
    for d in disagreements:
        print(json.dumps(d))
    if args.json_out:
        json.dump({"summary": result, "per_item": per}, open(args.json_out, "w"))


if __name__ == "__main__":
    main()
