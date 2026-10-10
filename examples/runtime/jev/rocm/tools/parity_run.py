"""Sequential text parity: released adapter (stock SGLang) vs decision-mode fork.

One request at a time. Each record is sent to both endpoints; the order
alternates per record. Raw responses and client wall latency are saved.
Stdlib only (http.client keep-alive connections).
"""

import argparse
import http.client
import json
import time
import urllib.parse


class Client:
    def __init__(self, url, timeout=180):
        parsed = urllib.parse.urlparse(url)
        self.host, self.port = parsed.hostname, parsed.port
        self.timeout = timeout
        self.conn = None

    def post(self, path, body):
        data = json.dumps(body).encode()
        for attempt in (0, 1):
            if self.conn is None:
                self.conn = http.client.HTTPConnection(
                    self.host, self.port, timeout=self.timeout
                )
            try:
                started = time.perf_counter()
                self.conn.request(
                    "POST", path, body=data, headers={"Content-Type": "application/json"}
                )
                response = self.conn.getresponse()
                raw = response.read()
                elapsed = time.perf_counter() - started
                return response.status, json.loads(raw), elapsed
            except (http.client.HTTPException, ConnectionError, OSError):
                self.conn.close()
                self.conn = None
                if attempt:
                    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref-url", default="http://127.0.0.1:30121")
    parser.add_argument("--fork-url", default="http://127.0.0.1:30010")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--only", choices=("ref", "fork"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    clients = {"ref": Client(args.ref_url), "fork": Client(args.fork_url)}
    targets = [args.only] if args.only else ["ref", "fork"]

    def body(row, target):
        request = {"model": args.model, **row["request"]}
        if target == "ref":
            request["options"] = {"return_logprobs": True}
        return request

    for row in rows[: args.warmup]:
        for target in targets:
            clients[target].post("/v1/systemone", body(row, target))

    with open(args.out, "w") as out:
        for index, row in enumerate(rows):
            order = targets if index % 2 == 0 else list(reversed(targets))
            record = {"id": row["id"], "index": index, "order": order}
            for target in order:
                status, response, elapsed = clients[target].post(
                    "/v1/systemone", body(row, target)
                )
                record[target] = {
                    "status": status,
                    "latency_s": elapsed,
                    "response": response,
                }
            out.write(json.dumps(record) + "\n")
            out.flush()
            if index % 50 == 0:
                print(index, {t: round(record[t]["latency_s"] * 1000, 1) for t in order},
                      flush=True)


if __name__ == "__main__":
    main()
