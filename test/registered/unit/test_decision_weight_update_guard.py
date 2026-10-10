"""Exercise the public update methods without importing GPU SGLang dependencies."""

import ast
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
CONTRACT_SOURCE = SRT / "decision_contract.py"
SPEC = importlib.util.spec_from_file_location(
    "decision_contract_guard_test", CONTRACT_SOURCE
)
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def _subject(method_name, source, decision_path):
    """Compile the actual method body with deferred annotations and stub globals."""
    tree = ast.parse(source.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(child, ast.AsyncFunctionDef) and child.name == method_name
            for child in node.body
        )
    )
    method = next(
        child
        for child in owner.body
        if isinstance(child, ast.AsyncFunctionDef) and child.name == method_name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            ast.ClassDef(
                name="Subject", bases=[], keywords=[], body=[method], decorator_list=[]
            ),
        ],
        type_ignores=[],
    )
    namespace = {
        "get_model": lambda: SimpleNamespace(decision_readout_path=decision_path),
        "DECISION_WEIGHT_UPDATE_MESSAGE": contract.DECISION_WEIGHT_UPDATE_MESSAGE,
    }
    exec(  # noqa: S102 - compile the checked-in method body under CPU-only stubs
        compile(ast.fix_missing_locations(module), str(source), "exec"), namespace
    )
    return namespace["Subject"]()


@pytest.mark.parametrize(
    ("source", "method_name", "result_size"),
    [
        (SRT / "managers/tokenizer_manager.py", "update_weights_from_disk", 3),
        (
            SRT / "managers/tokenizer_control_mixin.py",
            "update_weights_from_distributed",
            2,
        ),
        (
            SRT / "managers/tokenizer_control_mixin.py",
            "update_weights_from_tensor",
            2,
        ),
        (
            SRT / "managers/tokenizer_control_mixin.py",
            "update_weights_from_ipc",
            2,
        ),
    ],
)
def test_decision_mode_rejects_weight_update_before_side_effects(
    monkeypatch, source, method_name, result_size
):
    # The disk method imports this lazily. Its parent SGLang package is intentionally
    # absent from this CPU-only test process.
    monkeypatch.setitem(sys.modules, "sglang.srt.decision_contract", contract)
    subject = _subject(method_name, source, "/models/decision.npz")
    calls = []

    def unexpected_handle_loop():
        calls.append("handle_loop")
        raise AssertionError("weight-update side effect reached")

    class UntouchableRequest:
        def __getattribute__(self, name):
            raise AssertionError(f"weight-update request accessed: {name}")

    subject.auto_create_handle_loop = unexpected_handle_loop
    result = asyncio.run(getattr(subject, method_name)(UntouchableRequest()))
    assert result[:2] == (False, contract.DECISION_WEIGHT_UPDATE_MESSAGE)
    assert len(result) == result_size
    if result_size == 3:
        assert result[2] == 0
    assert calls == []

    # Outside decision mode, control still enters the preexisting update path.
    normal_subject = _subject(method_name, source, None)
    normal_subject.auto_create_handle_loop = unexpected_handle_loop
    with pytest.raises(AssertionError, match="side effect reached"):
        asyncio.run(getattr(normal_subject, method_name)(UntouchableRequest()))
    assert calls == ["handle_loop"]
