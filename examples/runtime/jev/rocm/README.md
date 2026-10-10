# Decision mode on AMD ROCm (MI350X gfx950, MI325X gfx942) and the NVIDIA image

Brought up 2026-09-30 on one AMD Instinct MI350X (288 GB HBM, ROCm 7.2) with
`StandardOne-8B` (BF16, `StandardThinking/StandardOne-8B` revision
`36d7999c63403acc1fd5861d5b139eee275e26d8`); in production there since
2026-10-05 with v2.2. Verified 2026-10-06 on one MI325X (gfx942), see
`BRINGUP-mi325x-20261006.md`.

## GPU profiles

`launch-decision.sh` reads the GPU family from the host (`GPU_PROFILE=auto`;
AMD: `gfx_target_version` in the kfd topology, NVIDIA: `nvidia-smi`
compute capability) and applies one launch profile:

| profile | GPUs | base image (`BASE_IMAGE`) | attention | notes |
|---|---|---|---|---|
| `gfx950` | MI350X, MI355X | `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260929` | `triton`, `SGLANG_USE_AITER=0` | production |
| `gfx942` | MI300X, MI325X | `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi30x-20260929` | `triton`, `SGLANG_USE_AITER=0` | verified 2026-10-06; no source change |
| `sm90` | H100, H200 | `lmsysorg/sglang:nightly-dev-cu13-20260929-79cafec0` | `fa3` + KV-free prefill (`KVFREE=0` for the paged pool) | verified 2026-10-06 on H100, images included; see `BRINGUP-h100-20261006.md` |
| `sm100` | B200, B300 | same CUDA image | `fa4` + KV-free prefill (`KVFREE=0` for the paged pool) | to be validated on hardware |

KV-free prefill (`--prefill-only-disable-kv-cache`) is the fork's fastest
path: the FA backend's raw-K/V branch (`fa_skip_kv_cache`) serves the whole
decision prefill without a paged KV pool. It exists for fa3 on Hopper and
fa4 on Blackwell; no ROCm backend has a raw-K/V path, so the AMD profiles
use the paged pool. Since 2026-10-06 it serves **images too**: the earlier
refusal in `decision_contract` was a guard, not a kernel limit (the vision
encoder has its own attention and the image embeddings reach the language
model as ordinary token embeddings). Both are owner decisions of
2026-10-06: every GPU family serves text and images through its fastest
path, one engine per host, no routing by request type.

All four base images are the same SGLang release (`v0.5.20`, sglang-kernel
0.4.7, the fork's pin), so one `Dockerfile.decision` builds every image:
`--build-arg BASE_IMAGE=...`. The ROCm images keep their Python in
`/opt/venv` (3.10), the CUDA image in `/opt/sglang` (3.12); the Dockerfile
asks the interpreter for its site-packages and repoints whichever editable
finder or `.pth` it finds there. Everything else in the launch (flags, env,
mounts, label scheme, readout) is identical across profiles; `MEM_MB`
(default 40000) is turned into `--mem-fraction-static` from the GPU's total
memory so the KV pool is the same size on an 80 GB and a 288 GB card.

The `tools/` directory holds the bring-up scripts: `canary.py` (bit-identical
start check), `parity_run.py`/`parity_analyze.py` (fork vs production),
`xparity.py` (the same request file answered on two hosts or two profiles,
compared per question), `restart_loop.sh` + `stock_canary.py` (aiter start
classification).

## Image

`Dockerfile.decision` builds `st-sglang:decision-<sha8>` from
`lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260929` (SGLang main
`98fce73d5b`, Python 3.10, torch 2.9.1+rocm7.2, aiter, sgl-kernel 0.4.7). It
keeps every compiled component and only repoints the base image's editable
`sglang` finder to this fork's `python/sglang` (base `0024efa0de` plus the
decision changes), adds a `.pth` for the jev-adapter package, and optionally
bakes in the aiter kernels that are otherwise JIT-compiled on first use.

## Compatibility changes

- **This fork's Python package: none.** It imports and serves unchanged on the
  base image's kernels. `sglang.srt.models.falcon_mamba`/`mamba` and the
  inkling/mimo processors log ignored import errors (`causal_conv1d_fn`,
  `cutlass`, `torchcodec`); those models are unused here.
- **jev-adapter (branch `feat/rocm-compat`): the decision deadline used
  `asyncio.timeout`, which Python 3.10 lacks**, so every `/v1/systemone`
  request would fail. On 3.10 the service now waits with
  `asyncio.wait(timeout=...)` under the same contract (504 only for the
  deadline, inner `TimeoutError` passes through, cancellation aborts engine
  work). Tests cover the 3.10 branch on any Python.
- **jev-adapter: serving defaults from the environment.** The in-process
  service had fixed defaults (temperature 1.0, alias `decision-model`).
  `JEV_DEFAULT_TEMPERATURE`, `JEV_TEMPERATURE_BY_TYPE`, `JEV_PROMPT_WORDING`
  and `JEV_MODEL_ALIAS` now set them, with the standalone CLI's meaning, so
  the engine can serve the released configuration (served wording, 1.65,
  `jev-latest`). Not ROCm-specific.

## What runs on ROCm

- LLM attention: **`triton`** (set by `launch-decision.sh`); Pixtral encoder:
  `aiter_attn`. SGLang's gfx950 default, `aiter`, is not safe here, see below.
- The KV-free prefill (`--prefill-only-disable-kv-cache`) uses fa3/SM90
  or fa4/SM100 and is refused at startup on ROCm. Decision prefill writes the paged KV
  pool (95,103 tokens, 12.3 GB at `--mem-fraction-static 0.15`).
- Prefill and decode graphs are disabled (`--disable-*-cuda-graph`), as in
  the first-run recipe.
- The GPU readout (255 LM rows, BF16 `F.linear`) runs on the `cuda` (HIP)
  device, unchanged.

## aiter attention: corrupt starts (open)

With `--attention-backend aiter` (the default), a container start is either
healthy (bit-identical answers run to run and across healthy starts, within
1-2 BF16 logit ulps of the production server) or corrupt from its first
request: probabilities off by up to 0.15, varying between identical
requests, then `Memory access fault by GPU` and a scheduler abort. Fork, 10
aiter starts: 4 healthy, 6 corrupt or crashed. The unmodified base image
serving the same model with the production flags on /generate, 6 starts:
start 1 corrupt (label-logit gaps off by up to 5.3 versus production), starts
3-6 aborted during the 150-request canary (start 6 log: the same memory
fault), start 2 not classified. With `--attention-backend triton` all 6 fork
starts were bit-identical and matched production to within BF16 rounding. Root cause
not isolated; the paged `mha_batch_prefill` path that SGLang forces for
multimodal models is the prime suspect. `tools/restart_loop.sh` +
`tools/canary.py` (fork) and `tools/stock_canary.py` (stock /generate)
reproduce it.

## Measured performance settings

H100 uses `GPU_PROFILE=sm90`, fa3 attention and `KVFREE=1` by default.
The measured v2.2 readout engine delivered 89.3 req/s for 4q at c16,
versus 80.9 with paged KV, with identical answers on all 6,125 compared
questions. This is the verified H100 optimization; the host was removed
after the tests. The schema-head engine is a separate model path: it
delivered 9.6 req/s for 64q versus the readout engine's 2.93, but was slower
for 4q. See `BRINGUP-h100-20261006.md` for the model and full validation.

MI325X defaults remain triton attention and `AITER_OPS=0`.
`AITER_OPS=1` enables the experimental AMD GEMM/norm/activation kernels;
`ATTENTION_BACKEND=aiter` additionally changes attention. The combination
reduced 4q c1 latency from 36.85 to 31.30 ms in the shared-GPU comparison,
but lowered MI350X-reference answer agreement from 98.53% to 97.50%.
It is an opt-in performance tradeoff, not a numerical drop-in replacement.
For a synthetic 52-token 1q request it measured 14.67 ms p50 / 15.14 ms
p95 at c1, or 249.18 req/s / 30.78 ms p50 at c8. These are warm local
HTTP measurements on a shared GPU, not production-network or dedicated
GPU guarantees. See `BRINGUP-mi325x-20261006.md`.

`MAX_RUNNING_REQUESTS` defaults to 32 and controls engine evaluations,
which differ from external requests when a request has several questions.
Values 64/128 have not been benchmarked in this update and are not
recommended defaults. All settings are printed before container launch.

Reproduce the synthetic 1q/4q concurrency sweep on a running engine:

```bash
python3 tools/latency_sweep.py --out latency.json --samples 100 --levels 1,4,8,16
python3 tools/test_launch_decision.py
```

Each case/level has eight sequential warmups followed by 100 measured
requests; successful throughput, response coverage, actual token counts,
p50/p95, and individual timings are saved. Disable prefix caching as in
the launch profile and run without other benchmark/soak traffic.

`launch-decision.sh` is the exact launch used for `eng-decision`; `tools/`
holds the parity, canary and prompt-ID scripts used for the bring-up.
