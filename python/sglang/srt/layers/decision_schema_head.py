"""Joint schema head for decision serving: one prefill scores every question.

The head reads the hidden states of every prompt token at its taps (the final
post-norm state and decoder-layer outputs), pools question and option spans
into nodes, routes evidence from the state tokens and scores each option. It
runs in fp32 on the worker's device, one request at a time.

Module and parameter names match the training code (schema-head series,
``head.py``) so ``schema_head.safetensors`` loads strictly; dropout and
gradient checkpointing exist only in training and are omitted here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

SCHEMA_HEAD_CONFIG = "schema_head_config.json"
SCHEMA_HEAD_WEIGHTS = "schema_head.safetensors"
SCHEMA_TEMPLATES = ("schema-v1",)

QTYPE_ID = {"choice": 0, "noul": 1, "score": 2}
ROLE_OTHER, ROLE_STATE, ROLE_PREVIEW, ROLE_QUESTION, ROLE_OPTION = range(5)
# Relation of node i to node j in field-interaction attention (training order).
(
    REL_SELF,
    REL_SIBLING,
    REL_OPT_TO_OWN_Q,
    REL_Q_TO_OWN_OPT,
    REL_OPT_TO_OTHER_OPT,
    REL_OPT_TO_OTHER_Q,
    REL_Q_TO_OTHER_Q,
    REL_Q_TO_OTHER_OPT,
) = range(8)
N_REL = 8
ORD_FEATS = 33


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class _SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, kv, bias=None):
        n, m = x.shape[0], kv.shape[0]
        q = self.q(x).view(n, self.h, self.dh).transpose(0, 1).unsqueeze(0)
        k = self.k(kv).view(m, self.h, self.dh).transpose(0, 1).unsqueeze(0)
        v = self.v(kv).view(m, self.h, self.dh).transpose(0, 1).unsqueeze(0)
        mask = None if bias is None else bias.unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.squeeze(0).transpose(0, 1).reshape(n, -1))


class _RouteBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int):
        super().__init__()
        self.n1, self.n2, self.nkv = _RMSNorm(d), _RMSNorm(d), _RMSNorm(d)
        self.cross = _Attention(d, heads)
        self.ffn = _SwiGLU(d, ffn)

    def forward(self, x, state):
        x = x + self.cross(self.n1(x), self.nkv(state))
        return x + self.ffn(self.n2(x))


class _InteractBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int):
        super().__init__()
        self.n1, self.n2, self.n3, self.nkv = (_RMSNorm(d) for _ in range(4))
        self.self_attn = _Attention(d, heads)
        self.cross = _Attention(d, heads)
        self.rel_bias = nn.Parameter(torch.zeros(N_REL, heads))
        self.ffn = _SwiGLU(d, ffn)

    def forward(self, x, state, rel):
        bias = self.rel_bias[rel].permute(2, 0, 1)
        y = self.n1(x)
        x = x + self.self_attn(y, y, bias=bias)
        x = x + self.cross(self.n2(x), self.nkv(state))
        return x + self.ffn(self.n3(x))


def _segment_sum(values: torch.Tensor, segment: torch.Tensor, n: int) -> torch.Tensor:
    """Sum ``values`` rows per segment id; segments are contiguous and sorted.

    ``index_add`` on the GPU adds with atomics, so the same request summed in a
    different order from one call to the next (about 1e-8 per probability) and
    no two passes were bit-identical. An inclusive scan has one order.
    Scan in float64: a difference of two running totals over a long segment
    would otherwise lose float32 digits.
    """
    ids, counts = torch.unique_consecutive(segment, return_counts=True)
    ends = counts.cumsum(0) - 1
    scan = values.double().cumsum(0)
    sums = scan.index_select(0, ends)
    sums = sums - torch.cat([torch.zeros_like(sums[:1]), sums[:-1]])
    out = torch.zeros((n,) + values.shape[1:], dtype=values.dtype, device=values.device)
    return out.index_copy(0, ids, sums.to(values.dtype))


def ordinal_features(k: int, count: int) -> List[float]:
    """Features of score level k of count levels (2 <= count <= 10)."""
    position = k / (count - 1)
    feats = [position, count / 10.0]
    feats += [1.0 if i == k else 0.0 for i in range(10)]
    feats += [1.0 if count == i else 0.0 for i in range(2, 11)]
    for f in range(1, 7):
        feats += [math.sin(math.pi * f * position), math.cos(math.pi * f * position)]
    return feats


class DecisionSchemaHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        taps: Sequence[Any],
        d: int,
        heads: int,
        ffn: int,
        route_blocks: int,
        interact_blocks: int,
        n_roles: int,
        ord_hidden: int,
        max_nodes: int,
        template: str,
        preview: bool,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.taps = list(taps)
        self.max_nodes = max_nodes
        self.template = template
        self.preview = preview
        n_taps = len(self.taps)
        self.tap_norms = nn.ModuleList([_RMSNorm(hidden_size) for _ in range(n_taps)])
        self.tap_mix = nn.Parameter(torch.zeros(n_taps))
        self.tap_gain = nn.Parameter(torch.ones(1))
        self.in_proj = nn.Linear(hidden_size, d, bias=False)
        self.role_emb = nn.Embedding(n_roles, d)
        self.qtype_emb = nn.Embedding(3, d)
        self.kind_emb = nn.Embedding(2, d)
        self.pool_query = nn.Parameter(torch.zeros(2, d))
        self.pool_key = nn.Linear(d, d, bias=False)
        self.pool_out = nn.Linear(2 * d, d, bias=False)
        self.ord_mlp = nn.Sequential(
            nn.Linear(ORD_FEATS, ord_hidden), nn.GELU(), nn.Linear(ord_hidden, d)
        )
        self.route = nn.ModuleList(
            [_RouteBlock(d, heads, ffn) for _ in range(route_blocks)]
        )
        self.interact = nn.ModuleList(
            [_InteractBlock(d, heads, ffn) for _ in range(interact_blocks)]
        )
        self.final_norm = _RMSNorm(d)
        self.scorer = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.log_temp = nn.Parameter(torch.zeros(3))
        self.d = d

    @property
    def tap_layers(self) -> List[int]:
        """Decoder layers whose outputs the head reads, in ascending order."""
        return sorted(int(tap) for tap in self.taps if tap != "final")

    @classmethod
    def from_dir(
        cls, path: str | Path, *, hidden_size: int, num_layers: int
    ) -> DecisionSchemaHead:
        from safetensors.torch import load_file

        folder = Path(path)
        try:
            config = json.loads((folder / SCHEMA_HEAD_CONFIG).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read {SCHEMA_HEAD_CONFIG} in {folder}") from exc
        head = config.get("head")
        if not isinstance(head, dict):
            raise ValueError("schema head config needs a 'head' object")
        if head.get("template") not in SCHEMA_TEMPLATES:
            raise ValueError(f"unsupported schema template {head.get('template')!r}")
        if head.get("hidden_size") != hidden_size:
            raise ValueError("schema head hidden_size differs from the loaded model")
        taps = head.get("taps")
        if (
            not isinstance(taps, list)
            or not taps
            or len(taps) != head.get("n_taps")
            or len(set(map(str, taps))) != len(taps)
            or any(
                tap != "final" and (type(tap) is not int or not 0 <= tap < num_layers - 1)
                for tap in taps
            )
        ):
            # The last decoder layer's output is never materialized before the
            # final norm, so only earlier layers can be tapped.
            raise ValueError(
                f"schema head taps must be 'final' or layer indices below {num_layers - 1}"
            )
        # Built while the worker's default dtype is the model's (bf16); the head
        # and its loaded weights must stay fp32.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            module = cls(
                hidden_size=hidden_size,
                taps=taps,
                d=head["d"],
                heads=head["heads"],
                ffn=head["ffn"],
                route_blocks=head["route_blocks"],
                interact_blocks=head["interact_blocks"],
                n_roles=head["n_roles"],
                ord_hidden=head["ord_hidden"],
                max_nodes=head["max_nodes"],
                template=head["template"],
                preview=bool(config.get("preview", True)),
            )
        finally:
            torch.set_default_dtype(default_dtype)
        state = load_file(str(folder / SCHEMA_HEAD_WEIGHTS))
        module.load_state_dict(state, strict=True)
        if not all(torch.isfinite(t).all() for t in state.values()):
            raise ValueError("schema head weights must be finite")
        return module.float().eval()

    # ------------------------------------------------------------------ plan
    def _plan(self, plan: Dict[str, Any], n_tokens: int, device) -> Dict[str, Any]:
        """Index tensors for one request's node graph (training ``plan``)."""
        kinds, node_q, node_qtype, node_ord = [], [], [], []
        pool_node, pool_tok, last_tok, q_node, opt_nodes = [], [], [], [], []
        roles = [ROLE_OTHER] * n_tokens
        for a, b in plan["state_spans"]:
            roles[a:b] = [ROLE_STATE] * (b - a)
        for i, qtype in enumerate(plan["qtype"]):
            index = len(kinds)
            q_node.append(index)
            kinds.append(0)
            node_q.append(i)
            node_qtype.append(qtype)
            node_ord.append(None)
            qa, qb = plan["q_span"][i]
            tokens = list(range(qa, qb))
            preview = plan["p_span"][i]
            if preview is not None:
                roles[preview[0] : preview[1]] = [ROLE_PREVIEW] * (preview[1] - preview[0])
                tokens = list(range(preview[0], preview[1])) + tokens
            roles[qa:qb] = [ROLE_QUESTION] * (qb - qa)
            pool_node += [index] * len(tokens)
            pool_tok += tokens
            last_tok.append(qb - 1)
            count = len(plan["o_span"][i])
            ids = []
            for j, (a, b) in enumerate(plan["o_span"][i]):
                index = len(kinds)
                ids.append(index)
                kinds.append(1)
                node_q.append(i)
                node_qtype.append(qtype)
                node_ord.append((j, count) if qtype == "score" else None)
                roles[a:b] = [ROLE_OPTION] * (b - a)
                pool_node += [index] * (b - a)
                pool_tok += list(range(a, b))
                last_tok.append(b - 1)
            opt_nodes.append(ids)
        n = len(kinds)
        if n > self.max_nodes:
            raise ValueError(f"schema request has {n} nodes; the head allows {self.max_nodes}")
        kind = torch.tensor(kinds)
        question = torch.tensor(node_q)
        same_q = question[:, None] == question[None, :]
        ki, kj = kind[:, None], kind[None, :]
        rel = torch.empty(n, n, dtype=torch.long)
        rel[(ki == 1) & (kj == 1) & same_q] = REL_SIBLING
        rel[(ki == 1) & (kj == 0) & same_q] = REL_OPT_TO_OWN_Q
        rel[(ki == 0) & (kj == 1) & same_q] = REL_Q_TO_OWN_OPT
        rel[(ki == 1) & (kj == 1) & ~same_q] = REL_OPT_TO_OTHER_OPT
        rel[(ki == 1) & (kj == 0) & ~same_q] = REL_OPT_TO_OTHER_Q
        rel[(ki == 0) & (kj == 0)] = REL_Q_TO_OTHER_Q
        rel[(ki == 0) & (kj == 1) & ~same_q] = REL_Q_TO_OTHER_OPT
        rel.fill_diagonal_(REL_SELF)
        ords = [ordinal_features(*o) if o else [0.0] * ORD_FEATS for o in node_ord]
        state_tok = [t for a, b in plan["state_spans"] for t in range(a, b)]

        def to(values, dtype=torch.long):
            return torch.tensor(values, dtype=dtype).to(device, non_blocking=True)

        return {
            "n": n,
            "kind": kind.to(device, non_blocking=True),
            "qtype": to([QTYPE_ID[t] for t in node_qtype]),
            "is_score": to([o is not None for o in node_ord], torch.bool),
            "ord": to(ords, torch.float32),
            "rel": rel.to(device, non_blocking=True),
            "pool_node": to(pool_node),
            "pool_tok": to(pool_tok),
            "last_tok": to(last_tok),
            "state_tok": to(state_tok),
            "roles": to(roles),
            "q_node": q_node,
            "opt_nodes": opt_nodes,
            "qtypes": list(plan["qtype"]),
        }

    # ------------------------------------------------------------------ forward
    def _score_one(self, taps: List[torch.Tensor], plan: Dict[str, Any]) -> torch.Tensor:
        p = self._plan(plan, taps[0].shape[0], taps[0].device)
        weights = torch.softmax(self.tap_mix, 0)
        mixed = 0
        for i, tap in enumerate(taps):
            mixed = mixed + weights[i] * self.tap_norms[i](tap.float())
        x_tok = self.in_proj(mixed * self.tap_gain) + self.role_emb(p["roles"])
        state = x_tok.index_select(0, p["state_tok"])
        toks = x_tok.index_select(0, p["pool_tok"])
        node_of = p["pool_node"]
        query = self.pool_query.index_select(0, p["kind"].index_select(0, node_of))
        scores = (self.pool_key(toks) * query).sum(-1) / math.sqrt(self.d)
        peak = torch.full((p["n"],), -1e30, device=scores.device, dtype=scores.dtype)
        peak = peak.scatter_reduce(0, node_of, scores, reduce="amax", include_self=True)
        e = torch.exp(scores - peak.index_select(0, node_of))
        den = _segment_sum(e, node_of, p["n"])
        w = e / den.index_select(0, node_of)
        pooled = _segment_sum(toks * w[:, None], node_of, p["n"])
        last = x_tok.index_select(0, p["last_tok"])
        x = self.pool_out(torch.cat([pooled, last], -1))
        x = x + self.qtype_emb(p["qtype"]) + self.kind_emb(p["kind"])
        x = x + self.ord_mlp(p["ord"]) * p["is_score"][:, None].to(x.dtype)
        for block in self.route:
            x = block(x, state)
        for block in self.interact:
            x = block(x, state, p["rel"])
        x = self.final_norm(x)
        out = []
        for i, q in enumerate(p["q_node"]):
            options = x[p["opt_nodes"][i][0] : p["opt_nodes"][i][-1] + 1]
            question = x[q].expand_as(options)
            z = self.scorer(torch.cat([options, question, options * question], -1))
            out.append(z.squeeze(-1) / torch.exp(self.log_temp[QTYPE_ID[p["qtypes"][i]]]))
        return torch.cat(out)

    @torch.no_grad()
    def forward(
        self,
        *,
        final_states: torch.Tensor,
        layer_states: Optional[List[torch.Tensor]],
        plans: List[Dict[str, Any]],
        seq_lens: List[int],
    ) -> torch.Tensor:
        """Per-request option logits in plan order, NaN-padded to [B, max options].

        Rows of each request are contiguous from the start of the token
        dimension (prefill-only, no prefix); padding rows past them are ignored.
        """
        by_layer = dict(zip(self.tap_layers, layer_states or []))
        if len(by_layer) != len(self.tap_layers):
            raise RuntimeError("schema head is missing captured decoder-layer states")
        widths = [sum(len(o) for o in plan["o_span"]) for plan in plans]
        out = torch.full(
            (len(plans), max(widths)),
            float("nan"),
            dtype=torch.float32,
            device=final_states.device,
        )
        start = 0
        with torch.autocast(device_type=final_states.device.type, enabled=False):
            for i, (plan, length) in enumerate(zip(plans, seq_lens)):
                rows = slice(start, start + length)
                taps = [
                    final_states[rows] if tap == "final" else by_layer[int(tap)][rows]
                    for tap in self.taps
                ]
                out[i, : widths[i]] = self._score_one(taps, plan)
                start += length
        return out
