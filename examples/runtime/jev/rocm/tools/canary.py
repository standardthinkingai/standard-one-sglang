"""Startup canary for eng-decision: replay single-question text records one at a
time and require bit-identical answers to a golden run from a healthy server.

Usage: canary.py [--url http://127.0.0.1:30010] [--golden /opt/engine/data/canary-golden.jsonl] [--n 100]
The default golden was recorded from a verified triton-backend eng-decision
(canary-golden-aiter.jsonl: from a verified aiter start).
Exit 0 = identical; 1 = mismatch (restart the container and re-check); 2 = error.
"""
import argparse, http.client, json, sys, urllib.parse

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:30010")
parser.add_argument("--golden", default="/opt/engine/data/canary-golden.jsonl")
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--record", help="write this server's answers as a new golden file")
args = parser.parse_args()
u = urllib.parse.urlparse(args.url)
conn = http.client.HTTPConnection(u.hostname, u.port, timeout=120)
same = total = 0
recorded = open(args.record, "w") if args.record else None
worst = 0.0
for line in open(args.golden):
    if total >= args.n:
        break
    row = json.loads(line)
    conn.request("POST", "/v1/systemone", body=json.dumps(row["request"]),
                 headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    body = json.loads(response.read())
    if response.status != 200:
        print("error", row["id"], response.status, body); sys.exit(2)
    got, want = body["answers"], row["answers"]
    if recorded:
        recorded.write(json.dumps({**row, "answers": got}) + "\n")
    total += 1
    if got == want:
        same += 1
        continue
    for key, answer in want.items():
        a, b = answer, got[key]
        if a["type"] == "noul":
            worst = max(worst, abs(a["noul"] - b["noul"]))
        else:
            worst = max(worst, max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]))
print(json.dumps({"checked": total, "identical": same, "max_abs_dp": worst}))
sys.exit(0 if same == total else 1)
