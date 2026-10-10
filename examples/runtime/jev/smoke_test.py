"""Exercise a running GPU server's text, image, and 255-choice contracts.

This verifies runtime contracts, not benchmark accuracy. No mock server is
started here: --base-url must point at a real SGLang server with this fork.
"""

import argparse
import base64
import json
import math
import os
import time
import urllib.request

from visual_game import make_game, render_png, seed_for


def check_response(body, request):
    assert body["usage"]["output_tokens"] == 0, "server generated output tokens"
    assert set(body["answers"]) == set(request["questions"])
    for key, question in request["questions"].items():
        answer = body["answers"][key]
        assert answer["type"] == question["type"]
        if answer["type"] == "noul":
            assert 0 <= answer["noul"] <= 1
            continue
        probabilities = answer["probabilities"]
        names = (
            list(question["criteria"])
            if answer["type"] == "choice"
            else [str(i) for i in range(len(question["criteria"]))]
        )
        assert set(probabilities) == set(names)
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
        assert math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6)
        assert 0 <= answer["confidence"] <= 1
        if answer["type"] == "choice":
            assert answer["choice"] == max(names, key=probabilities.__getitem__)
        else:
            assert math.isclose(
                answer["score"],
                sum(i * probabilities[str(i)] for i in range(len(names))),
                abs_tol=1e-6,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    args = parser.parse_args()
    layout, state = make_game(seed_for("train", 0))
    image = (
        "data:image/png;base64," + base64.b64encode(render_png(layout, state)).decode()
    )
    cases = {
        "mixed_text": {
            "state": "The customer was charged twice and requests a refund.",
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Choose the team.",
                    "criteria": {"billing": "Payments", "technical": "Software errors"},
                },
                "refund": {
                    "type": "noul",
                    "instructions": "Does the customer request a refund?",
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate the financial issue.",
                    "criteria": [
                        "No financial issue",
                        "One duplicate charge",
                        "Multiple duplicate charges",
                    ],
                },
            },
        },
        "visual": {
            "state": "Judge only the attached screenshot.",
            "images": [image],
            "questions": {
                "player": {
                    "type": "choice",
                    "instructions": "What color is the player marked YOU?",
                    "criteria": {"blue": None, "red": None, "green": None},
                }
            },
        },
        "choice_255": {
            "state": "The requested item ID is item_017.",
            "questions": {
                "id": {
                    "type": "choice",
                    "instructions": "Select the requested item ID.",
                    "criteria": {f"item_{i:03d}": None for i in range(255)},
                }
            },
        },
    }
    headers = {"Content-Type": "application/json"}
    if os.environ.get("SGLANG_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["SGLANG_API_KEY"]
    for name, body in cases.items():
        body["model"] = args.model
        body["options"] = {"return_logprobs": True}
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/systemone",
            json.dumps(body).encode(),
            headers,
        )
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.load(response)
        check_response(result, body)
        print(
            json.dumps(
                {
                    "case": name,
                    "contract_passed": True,
                    "latency_s": time.perf_counter() - started,
                    "usage": result["usage"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
