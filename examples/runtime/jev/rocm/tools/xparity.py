"""Cross-host parity for the decision fork: send the same requests to one engine, save the answers;
then compare two saved runs (e.g. MI350X vs MI325X).

  xparity.py run  <requests.jsonl> <out.jsonl> [url] [concurrency]
  xparity.py diff <a.jsonl> <b.jsonl>

requests.jsonl rows: {"id": ..., "body": {...}} (a /v1/systemone body). Stdlib only.
"""
import concurrent.futures as cf
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


def call(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    s = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.load(r), time.monotonic() - s
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.monotonic() - s


def run(src, out, url="http://127.0.0.1:30010/v1/systemone", conc="16"):
    rows = [json.loads(line) for line in open(src)]
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(int(conc)) as pool, open(out, "w") as f:
        for row, (status, body, sec) in zip(rows, pool.map(lambda r: call(url, r["body"]), rows)):
            f.write(json.dumps({"id": row["id"], "status": status, "sec": round(sec, 4),
                                "answers": body.get("answers"), "usage": body.get("usage"),
                                "error": (body.get("error") or {}).get("code")}) + "\n")
    print(json.dumps({"rows": len(rows), "seconds": round(time.monotonic() - t0, 1)}))


def probs(answer):
    if answer["type"] == "noul":
        return [answer["noul"]]
    return list(answer["probabilities"].values())


def pick(answer):
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    p = answer["probabilities"]
    return max(p, key=p.get)


def diff(a_path, b_path):
    a = {r["id"]: r for r in map(json.loads, open(a_path))}
    b = {r["id"]: r for r in map(json.loads, open(b_path))}
    ids = [i for i in a if i in b]
    status_diff = [i for i in ids if a[i]["status"] != b[i]["status"]]
    usage_diff = [i for i in ids if a[i]["status"] == 200 and a[i]["usage"] != b[i]["usage"]]
    by_set = {}
    dps = []
    for i in ids:
        if a[i]["status"] != 200 or b[i]["status"] != 200:
            continue
        name = i.split("/")[0]
        s = by_set.setdefault(name, {"questions": 0, "same": 0, "max_abs_dp": 0.0})
        for q, x in a[i]["answers"].items():
            y = b[i]["answers"][q]
            d = max(abs(u - v) for u, v in zip(probs(x), probs(y)))
            s["questions"] += 1
            s["same"] += pick(x) == pick(y)
            s["max_abs_dp"] = max(s["max_abs_dp"], round(d, 4))
            dps.append(d)
    for name, s in sorted(by_set.items()):
        print("XPARITY", name, json.dumps({**s, "same_pct": round(100 * s["same"] / max(1, s["questions"]), 2)}))
    dps.sort()
    total = sum(s["questions"] for s in by_set.values())
    same = sum(s["same"] for s in by_set.values())
    print("XPARITY ALL", json.dumps({
        "requests": len(ids), "status_mismatch": len(status_diff), "usage_mismatch": len(usage_diff),
        "questions": total, "same": same, "same_pct": round(100 * same / max(1, total), 2),
        "median_abs_dp": round(statistics.median(dps), 5) if dps else None,
        "p99_abs_dp": round(dps[int(0.99 * (len(dps) - 1))], 4) if dps else None,
        "max_abs_dp": round(dps[-1], 4) if dps else None,
        "status_examples": [(i, a[i]["status"], a[i]["error"], b[i]["status"], b[i]["error"]) for i in status_diff[:5]]}))


if __name__ == "__main__":
    {"run": run, "diff": diff}[sys.argv[1]](*sys.argv[2:])
