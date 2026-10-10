"""Determinism canary for a stock SGLang server: the released adapter's exact
/generate payloads (native prompt IDs + label token_ids_logprob), one at a time.
--record writes a golden; otherwise compares against --golden bit-for-bit."""
import argparse, http.client, json, sys, urllib.parse, uuid
p = argparse.ArgumentParser()
p.add_argument("--url", required=True); p.add_argument("--ids", default="/opt/engine/runs/prompt-ids/released.jsonl")
p.add_argument("--n", type=int, default=150); p.add_argument("--golden"); p.add_argument("--record")
a = p.parse_args()
u = urllib.parse.urlparse(a.url); conn = http.client.HTTPConnection(u.hostname, u.port, timeout=120)
golden = [json.loads(l) for l in open(a.golden)] if a.golden else None
out = open(a.record, "w") if a.record else None
same = worst = 0; n = 0
for i, line in enumerate(open(a.ids)):
    if i >= a.n: break
    r = json.loads(line)
    payload = {"input_ids": r["ids"], "sampling_params": {"max_new_tokens": 0, "temperature": 1, "top_p": 1, "top_k": -1},
               "return_logprob": True, "logprob_start_len": -1, "token_ids_logprob": r["label_token_ids"],
               "top_logprobs_num": 0, "return_text_in_logprobs": False, "stream": False, "rid": "canary-" + uuid.uuid4().hex}
    conn.request("POST", "/generate", body=json.dumps(payload), headers={"Content-Type": "application/json"})
    res = conn.getresponse(); body = json.loads(res.read())
    if res.status != 200: print("error", res.status, str(body)[:300]); sys.exit(2)
    vals = {str(t): v for v, t, *_ in body["meta_info"]["output_token_ids_logprobs"][0]}
    n += 1
    if out: out.write(json.dumps({"id": r["id"], "logprobs": vals}) + "\n")
    if golden:
        g = golden[i]["logprobs"]
        if g == vals: same += 1
        else: worst = max(worst, max(abs(g[k] - vals[k]) for k in g))
print(json.dumps({"checked": n, "identical": same if golden else None, "max_abs_dlogprob": worst}))
sys.exit(0 if not golden or same == n else 1)
