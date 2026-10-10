"""Closed-loop screenshot/action benchmark for the Jev endpoint.

Requires Python 3.10+ and Pillow; no SGLang import or GPU is needed locally.

    python visual_game.py --mode oracle --episodes 5 --output-dir /tmp/jev-oracle
    python visual_game.py --mode live --base-url http://localhost:30000 \
        --model Qwen/Qwen3.6-35B-A3B --episodes 5 --output-dir /tmp/jev-live

Oracle mode validates the game, renderer, controls and evaluator, NOT model
inference. Live mode sends only screenshots, fixed rules and recent actions to
/v1/systemone. It never sends map arrays, coordinates, seeds or oracle labels.
Use --split calibration for calibration data and reserve --split eval for the
final evaluation. Trace metadata and oracle labels are local evaluation data.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "wait": (0, 0),
}
SPLIT_OFFSETS = {"train": 0, "calibration": 1_000_000, "eval": 2_000_000}
RULES = (
    "Play the game shown in the screenshot. The blue circle marked YOU is your "
    "player. Collect the yellow KEY, then enter the brown DOOR to unlock it "
    "automatically; the key is consumed. Reach the purple GOAL. Gray WALL cells "
    "cannot be entered. Entering a red X hazard ends the game in failure. "
    "Each action moves one cell in the screen direction: up, down, left, right. "
    "There is no diagonal movement. Walking into a wall or locked door does "
    "nothing. An open door is passable. The wait action does nothing. "
    "Choose a safe next move that helps reach the goal. Inventory and terminal "
    "status are visible above the board. The board changes only after an action."
)


@dataclass(frozen=True)
class Layout:
    size: int
    walls: frozenset[tuple[int, int]]
    hazards: frozenset[tuple[int, int]]
    key: tuple[int, int]
    door: tuple[int, int]
    goal: tuple[int, int]


@dataclass(frozen=True)
class State:
    player: tuple[int, int]
    has_key: bool = False
    door_open: bool = False
    won: bool = False
    lost: bool = False


def seed_for(split: str, index: int) -> int:
    if split not in SPLIT_OFFSETS or not 0 <= index < 1_000_000:
        raise ValueError("Use a valid split and an episode index from 0 to 999999")
    return SPLIT_OFFSETS[split] + index


def transition(layout: Layout, state: State, action: str) -> tuple[State, bool]:
    """Return (next state, invalid action); a hazard is a legal, losing move."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")
    if state.won or state.lost:
        return state, True
    dx, dy = ACTIONS[action]
    target = (state.player[0] + dx, state.player[1] + dy)
    if not all(0 <= v < layout.size for v in target) or target in layout.walls:
        return state, True
    if target == layout.door and not state.door_open and not state.has_key:
        return state, True
    has_key = state.has_key or (target == layout.key and not state.door_open)
    door_open = state.door_open or (target == layout.door and has_key)
    return (
        State(
            player=target,
            has_key=has_key and not door_open,
            door_open=door_open,
            won=target == layout.goal,
            lost=target in layout.hazards,
        ),
        False,
    )


def oracle_actions(layout: Layout, state: State) -> list[str] | None:
    """Shortest safe path; privileged evaluator-only access, never model input."""
    if state.lost:
        return None
    queue = deque([(state, [])])
    visited = {state}
    while queue:
        current, path = queue.popleft()
        if current.won:
            return path
        for action in ("up", "down", "left", "right"):
            following, invalid = transition(layout, current, action)
            if invalid or following.lost or following in visited:
                continue
            visited.add(following)
            queue.append((following, path + [action]))
    return None


def make_game(seed: int) -> tuple[Layout, State]:
    """Generate a deterministic solvable board with a mandatory locked door."""
    rng = random.Random(seed)
    size = 7
    for _ in range(1000):
        border = {
            (x, y)
            for x in range(size)
            for y in range(size)
            if x in (0, 6) or y in (0, 6)
        }
        door = (3, rng.randrange(1, 6))
        walls = border | {(3, y) for y in range(1, 6) if (3, y) != door}
        left = [(x, y) for x in (1, 2) for y in range(1, 6)]
        right = [(x, y) for x in (4, 5) for y in range(1, 6)]
        start, key = rng.sample(left, 2)
        goal = rng.choice(right)
        available = [p for p in left + right if p not in (start, key, goal)]
        obstacles = rng.sample(available, 5)
        walls.update(obstacles[:2])
        layout = Layout(
            size, frozenset(walls), frozenset(obstacles[2:]), key, door, goal
        )
        state = State(start)
        path = oracle_actions(layout, state)
        if path and len(path) >= 8:
            return layout, state
    raise RuntimeError("Could not generate a solvable board")


def render_png(layout: Layout, state: State) -> bytes:
    """Render the actual observable game screen, including visible inventory."""
    from PIL import Image, ImageDraw, ImageFont

    cell, margin, top = 64, 16, 66
    width, height = layout.size * cell + margin * 2, top + layout.size * cell + 94
    image = Image.new("RGB", (width, height), "#101827")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=15)
        title_font = ImageFont.load_default(size=20)
    except TypeError:  # Pillow before 10.1
        font = title_font = ImageFont.load_default()
    status = (
        "GOAL REACHED" if state.won else "GAME OVER" if state.lost else "KEY & DOOR"
    )
    draw.text((margin, 12), status, font=title_font, fill="#ffffff")
    draw.text(
        (margin, 40),
        f"Inventory: {'KEY' if state.has_key else 'empty'}",
        font=font,
        fill="#ffd24a",
    )

    def centered(text: str, x: int, y: int, color: str = "#ffffff") -> None:
        bounds = draw.textbbox((0, 0), text, font=font)
        draw.text(
            (x + (cell - bounds[2]) / 2, y + (cell - bounds[3]) / 2),
            text,
            font=font,
            fill=color,
        )

    for row in range(layout.size):
        for col in range(layout.size):
            position = (col, row)
            x, y = margin + col * cell, top + row * cell
            draw.rectangle((x + 2, y + 2, x + cell - 2, y + cell - 2), fill="#e6eaf0")
            if position in layout.walls:
                draw.rectangle(
                    (x + 2, y + 2, x + cell - 2, y + cell - 2), fill="#586479"
                )
                centered("WALL", x, y)
            elif position in layout.hazards:
                draw.rectangle(
                    (x + 2, y + 2, x + cell - 2, y + cell - 2), fill="#ed4c55"
                )
                draw.line((x + 17, y + 17, x + 47, y + 47), fill="white", width=6)
                draw.line((x + 47, y + 17, x + 17, y + 47), fill="white", width=6)
            elif position == layout.goal:
                draw.rectangle(
                    (x + 5, y + 5, x + cell - 5, y + cell - 5), fill="#8b46c5"
                )
                centered("GOAL", x, y)
            elif position == layout.door:
                draw.rectangle(
                    (x + 9, y + 5, x + cell - 9, y + cell - 5),
                    fill="#428565" if state.door_open else "#9c5f2b",
                )
                centered("OPEN" if state.door_open else "DOOR", x, y)
            elif position == layout.key and not (state.has_key or state.door_open):
                draw.ellipse((x + 6, y + 6, x + cell - 6, y + cell - 6), fill="#ffd24a")
                centered("KEY", x, y, "#302819")
            if position == state.player:
                draw.ellipse(
                    (x + 7, y + 7, x + cell - 7, y + cell - 7),
                    fill="#1667cf",
                    outline="white",
                    width=3,
                )
                centered("YOU", x, y)
    draw.text(
        (margin, height - 77),
        "YOU (blue) > KEY (yellow) > DOOR > GOAL",
        font=font,
        fill="white",
    )
    draw.text(
        (margin, height - 53),
        "Avoid red X. Gray WALL blocks movement.",
        font=font,
        fill="#cbd5e1",
    )
    draw.text(
        (margin, height - 29),
        "Actions: up / down / left / right / wait",
        font=font,
        fill="#cbd5e1",
    )
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def build_request(
    image: bytes,
    history: list[str],
    model: str,
    temperature: float,
    diagnostic: bool = False,
) -> dict[str, Any]:
    """This boundary deliberately cannot receive Layout, State or seed values."""
    recent = [a for a in history[-6:] if a in ACTIONS]
    question = (
        {
            "goal_reached": {
                "type": "noul",
                "instructions": "Does the screenshot show GOAL REACHED?",
            }
        }
        if diagnostic
        else {
            "action": {
                "type": "choice",
                "instructions": "Which action should the player take next to safely reach GOAL?",
                "criteria": {
                    "up": "Move one cell toward the top of the screen",
                    "down": "Move one cell toward the bottom of the screen",
                    "left": "Move one cell toward the left of the screen",
                    "right": "Move one cell toward the right of the screen",
                    "wait": "Stay in the current cell",
                },
            }
        }
    )
    return {
        "model": model,
        "state": RULES
        + "\nRecent actions (oldest first): "
        + (", ".join(recent) or "none"),
        "images": ["data:image/png;base64," + base64.b64encode(image).decode("ascii")],
        "questions": question,
        "options": {"temperature": temperature, "return_logprobs": True},
    }


class LiveClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 1.0,
        timeout: float = 120.0,
    ):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.model, self.temperature, self.timeout = model, temperature, timeout

    def ask(
        self, image: bytes, history: list[str], diagnostic: bool = False
    ) -> dict[str, Any]:
        payload = build_request(
            image, history, self.model, self.temperature, diagnostic
        )
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("SGLANG_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(f"Jev HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Jev request failed: {exc}") from exc
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise TypeError("Jev response must contain an answers object")
        usage = result.get("usage")
        if not isinstance(usage, dict):
            raise TypeError("Jev response must contain a usage object")
        output_tokens = usage.get("output_tokens")
        if type(output_tokens) is not int or output_tokens != 0:
            raise RuntimeError("Expected explicit integer output_tokens=0")
        return result


def run_episode(
    seed: int,
    split: str,
    mode: str,
    output_dir: Path,
    max_steps: int,
    client: LiveClient | None = None,
    diagnostic_goal: bool = False,
) -> dict[str, Any]:
    layout, state = make_game(seed)
    optimal = oracle_actions(layout, state)
    episode_dir = output_dir / f"{split}-{seed}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    history: list[str] = []
    trace: list[dict[str, Any]] = []
    error = None
    invalid_count = 0
    for step in range(max_steps):
        screen = render_png(layout, state)
        filename = f"{step:03d}.png"
        (episode_dir / filename).write_bytes(screen)
        decision_start = time.perf_counter()
        answer: dict[str, Any] = {}
        try:
            if mode == "oracle":
                path = oracle_actions(layout, state)
                if not path:
                    raise RuntimeError("Oracle has no legal path")
                action = path[0]
            elif mode == "live" and client is not None:
                answer = client.ask(screen, history)
                action_answer = answer["answers"].get("action")
                if not isinstance(action_answer, dict):
                    raise TypeError("Jev action answer must be an object")
                action = action_answer.get("choice")
                if not isinstance(action, str) or action not in ACTIONS:
                    raise RuntimeError(f"Invalid model action: {action!r}")
            else:
                raise ValueError(
                    "Live mode needs a client; mode must be oracle or live"
                )
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            error = str(exc)
            trace.append({"step": step, "screenshot": filename, "error": error})
            break
        latency_ms = (time.perf_counter() - decision_start) * 1000
        following, invalid = transition(layout, state, action)
        invalid_count += int(invalid)
        trace.append(
            {
                "step": step,
                "screenshot": filename,
                "action": action,
                "invalid": invalid,
                "decision_latency_ms": latency_ms,
                "response": answer if mode == "live" else None,
            }
        )
        history.append(action)
        state = following
        if state.won or state.lost:
            break
    final_screen = render_png(layout, state)
    (episode_dir / "final.png").write_bytes(final_screen)
    diagnostic = None
    if diagnostic_goal and mode == "live" and client is not None and error is None:
        try:
            result = client.ask(final_screen, history, diagnostic=True)
            probability = float(result["answers"]["goal_reached"]["noul"])
            if not 0 <= probability <= 1:
                raise ValueError("noul must be a probability")
            diagnostic = {
                "true_goal_reached": state.won,
                "predicted_probability": probability,
                "correct_at_0_5": (probability >= 0.5) == state.won,
                "response": result,
            }
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            diagnostic = {"error": str(exc)}
    outcome = (
        "error"
        if error
        else "success"
        if state.won
        else "hazard"
        if state.lost
        else "step_limit"
    )
    report = {
        "seed": seed,
        "split": split,
        "mode": mode,
        "model_inference": mode == "live",
        "model": client.model if mode == "live" and client else None,
        "success": state.won,
        "outcome": outcome,
        "steps": len(history),
        "invalid_actions": invalid_count,
        "optimal_steps": len(optimal or []),
        "path_efficiency": len(optimal or []) / len(history)
        if state.won and history
        else 0.0,
        "error": error,
        "trace": trace,
        "final_screenshot": "final.png",
        "goal_diagnostic": diagnostic,
    }
    (episode_dir / "trace.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def summarize(reports: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    latencies = sorted(
        t["decision_latency_ms"]
        for r in reports
        for t in r["trace"]
        if "decision_latency_ms" in t
    )
    return {
        "mode": mode,
        "model_inference": mode == "live",
        "note": "Live visual model control"
        if mode == "live"
        else "Oracle environment/control validation only; no model inference",
        "episodes": len(reports),
        "success_rate": statistics.mean(r["success"] for r in reports)
        if reports
        else 0,
        "mean_steps": statistics.mean(r["steps"] for r in reports) if reports else 0,
        "invalid_actions": sum(r["invalid_actions"] for r in reports),
        "error_episodes": sum(r["outcome"] == "error" for r in reports),
        "mean_decision_latency_ms": statistics.mean(latencies) if latencies else None,
        "p95_decision_latency_ms": latencies[
            max(0, (95 * len(latencies) + 99) // 100 - 1)
        ]
        if latencies
        else None,
    }


def write_replay(output_dir: Path, reports: list[dict[str, Any]]) -> None:
    """Self-contained screenshot replay; inspect what the controller actually saw."""
    sections = []
    data = []
    for number, report in enumerate(reports):
        episode_dir = output_dir / f"{report['split']}-{report['seed']}"
        frames = []
        for item in report["trace"] + [
            {"screenshot": "final.png", "action": "END: " + report["outcome"]}
        ]:
            frames.append(
                {
                    "image": "data:image/png;base64,"
                    + base64.b64encode(
                        (episode_dir / item["screenshot"]).read_bytes()
                    ).decode(),
                    "caption": f"{item.get('action', 'ERROR')} | invalid={item.get('invalid', False)}",
                }
            )
        data.append(frames)
        sections.append(
            f'<section><h2>{html.escape(report["mode"])} · seed {report["seed"]} · {html.escape(report["outcome"])}</h2><input type="range" min="0" max="{len(frames) - 1}" value="0" oninput="show({number},Number(this.value))"><p id="caption-{number}"></p><img id="frame-{number}" alt="Game screenshot"></section>'
        )
    payload = json.dumps(data).replace("</", "<\\/")
    document = (
        '<!doctype html><meta charset="utf-8"><title>Jev visual game replay</title><style>body{font:16px system-ui;background:#101827;color:white;max-width:1050px;margin:32px auto;padding:16px}section{display:inline-block;vertical-align:top;margin:12px}h2{font-size:16px}img{width:360px;max-width:100%;display:block}input{width:100%}p{color:#ccd6e5}</style><h1>Jev screenshot → action replay</h1><p>Oracle runs validate the environment only. Live runs use visual model inference. Move each slider to inspect an observation and the action applied to it.</p>'
        + "".join(sections)
        + "<script>const episodes="
        + payload
        + ';function show(i,j){document.getElementById("frame-"+i).src=episodes[i][j].image;document.getElementById("caption-"+i).textContent="Frame "+j+": "+episodes[i][j].caption;}episodes.forEach((_,i)=>show(i,0));</script>'
    )
    (output_dir / "replay.html").write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("oracle", "live"), required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_OFFSETS), default="eval")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Calibration temperature, not sampling randomness",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("jev-visual-game-results")
    )
    parser.add_argument(
        "--diagnostic-goal",
        action="store_true",
        help="Separate final-screen noul request; never controls the environment",
    )
    args = parser.parse_args()
    if (
        args.episodes <= 0
        or args.max_steps <= 0
        or args.temperature <= 0
        or args.timeout <= 0
    ):
        parser.error("episodes, max-steps, temperature and timeout must be positive")
    try:
        seeds = [
            seed_for(args.split, args.start_index + i) for i in range(args.episodes)
        ]
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = (
        LiveClient(args.base_url, args.model, args.temperature, args.timeout)
        if args.mode == "live"
        else None
    )
    reports = []
    for seed in seeds:
        report = run_episode(
            seed,
            args.split,
            args.mode,
            args.output_dir,
            args.max_steps,
            client,
            args.diagnostic_goal,
        )
        reports.append(report)
        print(
            f"{args.mode}: seed={seed} outcome={report['outcome']} steps={report['steps']} invalid={report['invalid_actions']}",
            flush=True,
        )
    summary = summarize(reports, args.mode)
    summary["configuration"] = {
        "split": args.split,
        "start_index": args.start_index,
        "max_steps": args.max_steps,
        "model": args.model if args.mode == "live" else None,
        "temperature": args.temperature,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_replay(args.output_dir, reports)
    print(json.dumps(summary, indent=2))
    print(f"Replay: {(args.output_dir / 'replay.html').resolve()}")
    return 1 if summary["error_episodes"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
