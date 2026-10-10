"""Final, read-only server contract check for GPU decision serving."""

from __future__ import annotations

from typing import Any

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.decision_contract import validate_decision_server


def handle_decision_readout_validation(server_args: Any) -> None:
    cfg = resolving_view(server_args)
    if cfg.decision_readout_path is None:
        # The decision-only settings (label scheme, deadline, queue bound)
        # would otherwise be accepted and silently ignored.
        validate_decision_server(cfg)
        return
    from sglang.srt.environ import envs

    if envs.SGLANG_RUST_SERVER.get():
        raise ValueError("decision readout does not support SGLANG_RUST_SERVER")
    validate_decision_server(cfg)
