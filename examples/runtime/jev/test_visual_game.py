"""Run with: python -m unittest discover -s examples/runtime/jev -p test_visual_game.py"""

import base64
import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from PIL import Image

spec = importlib.util.spec_from_file_location(
    "jev_visual_game", Path(__file__).with_name("visual_game.py")
)
game = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = game
spec.loader.exec_module(game)


class VisualGameTests(unittest.TestCase):
    def test_live_requires_explicit_zero_output_tokens(self):
        invalid_responses = [{"answers": {}}, {"answers": {}, "usage": None}]
        for usage in (
            {},
            {"output_tokens": None},
            {"output_tokens": 1},
            {"output_tokens": False},
            {"output_tokens": True},
            {"output_tokens": 0.0},
            {"output_tokens": "0"},
        ):
            invalid_responses.append({"answers": {}, "usage": usage})
        client = game.LiveClient("http://localhost:30000", "mock-model")
        for result in invalid_responses:
            with (
                self.subTest(result=result),
                mock.patch.object(
                    game.urllib.request,
                    "urlopen",
                    return_value=io.BytesIO(json.dumps(result).encode()),
                ),
                self.assertRaises((RuntimeError, TypeError)),
            ):
                client.ask(b"test-png", [])
        valid = {"answers": {}, "usage": {"output_tokens": 0}}
        with mock.patch.object(
            game.urllib.request,
            "urlopen",
            return_value=io.BytesIO(json.dumps(valid).encode()),
        ):
            self.assertEqual(client.ask(b"test-png", []), valid)

    def test_live_optional_bearer_header(self):
        client = game.LiveClient("http://localhost:30000", "mock-model")
        for api_key in ("", "test-only-api-key"):
            with (
                self.subTest(api_key=bool(api_key)),
                mock.patch.dict(game.os.environ, {"SGLANG_API_KEY": api_key}),
                mock.patch.object(
                    game.urllib.request,
                    "urlopen",
                    return_value=io.BytesIO(
                        b'{"answers": {}, "usage": {"output_tokens": 0}}'
                    ),
                ) as request,
            ):
                client.ask(b"test-png", [])
                authorization = request.call_args.args[0].get_header("Authorization")
                self.assertEqual(
                    authorization, f"Bearer {api_key}" if api_key else None
                )

    def test_malformed_action_records_episode_error(self):
        for action in (None, [], {"choice": []}, {"choice": 1}, {"choice": "teleport"}):
            with (
                self.subTest(action=action),
                tempfile.TemporaryDirectory() as directory,
            ):
                client = mock.Mock(model="mock-model")
                client.ask.return_value = {
                    "answers": {"action": action},
                    "usage": {"output_tokens": 0},
                }
                result = game.run_episode(
                    game.seed_for("eval", 0), "eval", "live", Path(directory), 1, client
                )
                self.assertEqual(result["outcome"], "error")
                self.assertEqual(result["steps"], 0)
                self.assertFalse(result["success"])

    def test_solvable_deterministic_and_split_isolation(self):
        seeds = {
            split: {game.seed_for(split, i) for i in range(20)}
            for split in game.SPLIT_OFFSETS
        }
        self.assertEqual(len(set.union(*seeds.values())), 60)
        for seed in set.union(*seeds.values()):
            layout, state = game.make_game(seed)
            self.assertEqual((layout, state), game.make_game(seed))
            path = game.oracle_actions(layout, state)
            self.assertGreaterEqual(len(path), 8)
            for action in path:
                state, invalid = game.transition(layout, state, action)
                self.assertFalse(invalid)
                self.assertFalse(state.lost)
            self.assertTrue(state.won)
            self.assertTrue(state.door_open)
        with self.assertRaises(ValueError):
            game.seed_for("eval", 1_000_000)

    def test_controls_inventory_locked_door_hazard_and_terminal(self):
        layout = game.Layout(
            5, frozenset({(0, 1)}), frozenset({(1, 2)}), (1, 0), (2, 1), (3, 1)
        )
        state = game.State((1, 1))
        self.assertEqual(game.transition(layout, state, "left"), (state, True))
        self.assertEqual(game.transition(layout, state, "right"), (state, True))
        self.assertEqual(game.transition(layout, state, "wait"), (state, False))
        key_state, invalid = game.transition(layout, state, "up")
        self.assertTrue(key_state.has_key)
        self.assertFalse(invalid)
        key_state, _ = game.transition(layout, key_state, "down")
        open_state, _ = game.transition(layout, key_state, "right")
        self.assertTrue(open_state.door_open)
        self.assertFalse(open_state.has_key)
        win_state, _ = game.transition(layout, open_state, "right")
        self.assertTrue(win_state.won)
        self.assertEqual(game.transition(layout, win_state, "left"), (win_state, True))
        lost_state, invalid = game.transition(layout, state, "down")
        self.assertTrue(lost_state.lost)
        self.assertFalse(invalid)
        self.assertIsNone(game.oracle_actions(layout, lost_state))

    def test_screenshot_request_has_no_privileged_state(self):
        layout, state = game.make_game(game.seed_for("eval", 0))
        screenshot = game.render_png(layout, state)
        with Image.open(io.BytesIO(screenshot)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (480, 608))
        self.assertEqual(screenshot, game.render_png(layout, state))
        request = game.build_request(screenshot, ["up"] * 20, "test-model", 1.5)
        self.assertEqual(
            set(request), {"model", "state", "images", "questions", "options"}
        )
        self.assertEqual(
            request["state"],
            game.RULES + "\nRecent actions (oldest first): " + ", ".join(["up"] * 6),
        )
        self.assertEqual(
            base64.b64decode(request["images"][0].split(",", 1)[1]), screenshot
        )
        self.assertEqual(request["questions"]["action"]["type"], "choice")
        self.assertEqual(
            set(request["questions"]["action"]["criteria"]), set(game.ACTIONS)
        )
        self.assertEqual(
            request["options"], {"temperature": 1.5, "return_logprobs": True}
        )
        for forbidden in (
            "seed",
            "coordinates",
            "oracle",
            "optimal_steps",
            "player_position",
        ):
            self.assertNotIn(forbidden, json.dumps(request))
        diagnostic = game.build_request(
            screenshot, [], "test-model", 1, diagnostic=True
        )
        self.assertEqual(set(diagnostic["questions"]), {"goal_reached"})

    def test_oracle_loop_writes_replay_and_honest_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = game.run_episode(
                game.seed_for("calibration", 0), "calibration", "oracle", output, 64
            )
            self.assertTrue(report["success"])
            self.assertEqual(report["steps"], report["optimal_steps"])
            self.assertEqual(report["invalid_actions"], 0)
            self.assertFalse(report["model_inference"])
            summary = game.summarize([report], "oracle")
            self.assertEqual(summary["success_rate"], 1)
            self.assertIn("no model inference", summary["note"])
            game.write_replay(output, [report])
            self.assertIn(
                "data:image/png;base64,", (output / "replay.html").read_text()
            )
            self.assertTrue((output / "calibration-1000000" / "trace.json").is_file())

    def test_live_http_control_contract_and_isolated_diagnostic(self):
        """A fake server tests transport/control, and is explicitly not an LLM."""
        seed = game.seed_for("eval", 2)
        layout, initial = game.make_game(seed)
        path = game.oracle_actions(layout, initial)
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                received.append((self.path, request))
                if "goal_reached" in request["questions"]:
                    answers = {"goal_reached": {"type": "noul", "noul": 1.0}}
                else:
                    action = path[len(received) - 1]
                    answers = {
                        "action": {
                            "type": "choice",
                            "choice": action,
                            "probabilities": {action: 1.0},
                        }
                    }
                body = json.dumps(
                    {"answers": answers, "usage": {"output_tokens": 0}}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                client = game.LiveClient(
                    f"http://127.0.0.1:{server.server_port}", "fake-contract-server"
                )
                report = game.run_episode(
                    seed, "eval", "live", Path(directory), 64, client, True
                )
                self.assertTrue(report["success"])
                self.assertEqual(report["steps"], len(path))
                self.assertTrue(report["goal_diagnostic"]["correct_at_0_5"])
                self.assertEqual(len(received), len(path) + 1)
                self.assertTrue(
                    all(endpoint == "/v1/systemone" for endpoint, _ in received)
                )
                self.assertNotEqual(received[0][1]["images"], received[1][1]["images"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
