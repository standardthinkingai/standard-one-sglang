"""Test this extension without importing CUDA dependencies or model weights.

Only namespace package initialization is bypassed. The protocol, endpoint,
HTTP router, evaluation tools, and game code under test are the real modules.
The serving tests explicitly fake the GPU backend, not model accuracy.
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for name in ("sglang", "sglang.srt", "sglang.srt.entrypoints"):
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT / "python" / name.replace(".", "/"))]
    sys.modules[name] = module

if __name__ == "__main__":
    files = [
        ROOT / "test/registered/unit/entrypoints/test_systemone_protocol.py",
        ROOT / "test/registered/unit/entrypoints/test_serving_systemone.py",
        ROOT / "test/registered/unit/entrypoints/test_systemone_tokenizer.py",
        ROOT / "examples/runtime/jev/test_visual_game.py",
        ROOT / "examples/runtime/jev/test_business_eval.py",
    ]
    raise SystemExit(
        pytest.main(
            [str(path) for path in files if path.exists()] + ["-q"] + sys.argv[1:]
        )
    )
