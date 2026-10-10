# Development verification — 2026-09-20

The development host was macOS arm64 with Python 3.12.14. No GPU server was
available, and no Qwen language-model or vision-encoder inference was run.

## Passed

- **48 CPU tests, 590 subtests** using `run_cpu_tests.py` with
  `JEV_TOKENIZER_DIR` set. Includes isolated native-route HTTP tests and local
  loopback transport tests. The language-model backend is explicitly faked.
- Actual Qwen3.6-35B-A3B tokenizer and chat template at revision
  `995ad96eacd98c81ed38be0c5b274b04031597b0`: **255 distinct single-token labels**,
  with all 255 checked at both text and image non-thinking answer boundaries.
  Only tokenizer files were downloaded, not model weights.
- Korean business fixture validation: **24 cases / 72 decisions**, 12 cases
  each for calibration and evaluation, eight disjoint scenario families.
  Fixture SHA-256:
  `a9cebb67049499c49c0b5c8e64a75f609700557fe9aeac9fbce7ad94320c3db7`.
- Visual environment oracle: **5/5 episodes succeeded**, 9.8 mean steps, zero
  invalid moves, seeds 2000000–2000004. This verifies game/control code only;
  it is not an LLM success rate or latency result.
- Python compilation, extension Ruff checks, import ordering, registered-test
  lint scripts, and `git diff --check`.

CPU tests cover invalid schemas, all primitive outputs, structured legends,
temperature stability, exact rotation counts and inverse mappings, independent
question prompts, multimodal forwarding through the chat adapter interface,
zero-token request construction, cancellation/abort, missing/nonfinite backend
scores, nonzero completion rejection, and HTTP error handling.

The evaluation-tool tests verify explicit-policy ground truth, split isolation,
calibration-input provenance checks, probability metrics, authenticated HTTP
contracts, screenshot-only model input, deterministic solvable games, movement
rules, and malformed responses. Temporary test HTTP servers return fixtures;
they do not run an LLM. The final test run emitted two upstream Starlette test
client deprecation warnings and no failures.

## Required when the GPU server is available

1. Install and launch this fork with the pinned Qwen model revision; keep
   speculation/MTP disabled.
2. Run `smoke_test.py` against that actual server. This must verify all three
   primitives, multimodal processing, 255 choices, and zero completion tokens.
3. Collect the calibration split, fit temperature, then run held-out business
   evaluation. Extend the small diagnostic fixtures with real labeled data
   before making quality/calibration claims.
4. Run `visual_game.py --mode live` on unseen evaluation seeds. Inspect saved
   screenshots and traces; report success rate, invalid moves, and latency.
5. Record GPU, memory type, model/engine revision, precision, input/image sizes,
   batch concurrency, and cache state before comparing throughput or bandwidth.

Neither the full CUDA SGLang application startup nor real multimodal processing,
model accuracy, throughput, HBM-versus-non-HBM performance, or production
probability calibration has been validated by the CPU checks.
