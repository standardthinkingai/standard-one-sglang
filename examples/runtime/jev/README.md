# Jev-style classification with Qwen3.6 and SGLang

This fork adds a native `POST /v1/systemone` endpoint to SGLang. It evaluates
label-token log-probabilities with `max_new_tokens=0` and constructs Choice,
Noul, and Score answers in Python. The target is `Qwen/Qwen3.6-35B-A3B`, including
image inputs for a screenshot-to-action game loop.

The endpoint implements Jev's three output primitives using existing Qwen
weights. It does not reproduce TypeSafe's proprietary model, accuracy,
calibration, confidence formula, or constant-latency multi-question behavior.

For a training-free Qwen3.5-4B vision backend on Apple Silicon, see
[the local MLX guide](LOCAL_MLX.md). It uses the same scoring protocol with
MLX-VLM and includes an actual-image diagnostic. Public KEV data, benchmark
methods, and reproduction details are in [KEV benchmark notes](KEV_BENCHMARK_NOTES.md).
For a paired text-only versus image-added timing experiment with Ministral 3,
see [Ministral latency measurement](MINISTRAL_LATENCY.md).

## Implementation and compatibility

- `python/sglang/srt/entrypoints/systemone_protocol.py`: validated schemas,
  independent question prompts, temperature scaling, and answer construction.
- `python/sglang/srt/entrypoints/serving_systemone.py`: existing chat template and
  image adapter → native zero-token engine requests → label probabilities.
- `http_server.py`: handler initialization and router registration. Existing
  `/v1/score`, chat, and generation endpoints keep their implementations.

SGLang was chosen because its generation path already returns selected-token
log-probabilities for a zero-token request. This avoids adding a new scheduler or
sampler to vLLM. Upstream base: `0024efa0de38794ee309ba10ab00ebb891a3d050`.

| Type | Input | Output |
| --- | --- | --- |
| `choice` | 2–255 named options | `choice`, `probabilities`, `confidence` |
| `noul` | A yes/no question; optional `true`/`false` descriptions | `noul=P(true)` |
| `score` | 2–10 ordered descriptions | `score`, `legend`, `probabilities`, `confidence` |

Score is `sum(index * probability)`, with indices starting at zero. Confidence
is `1 - entropy(probabilities) / log(option_count)`, matching the linked
llama.cpp fork, not a published TypeSafe formula. It is not the probability of
being correct. Temperature defaults to 1; no temperature is claimed calibrated
until you fit it on a separate labeled calibration split.

String, object, and array states/instructions/descriptions are supported. Each
question receives only the shared state and its own instructions/options.
Question IDs are response keys, not model input. The runtime can batch the
independent sequences and reuse cached prefixes when its cache configuration
allows it; a shared state is not guaranteed to be processed only once when the
first questions arrive simultaneously. Report image preprocessing and cache
conditions alongside latency.

Request options:

| Option | Default | Meaning |
| --- | --- | --- |
| `temperature` | 1.0 | Positive finite calibration temperature |
| `temperature_scaling` | true | Apply temperature before normalization |
| `permutations` | 1 | 1–16 cyclic rotations, each a separate evaluation |
| `return_logprobs` | false | Canonical-order vocabulary logprobs for each rotation |
| `assistant_prefix` | null | Optional explicit final-answer prefix |

`return_logits` is deliberately rejected: the runtime returns log-probabilities,
not unnormalized logits. These contain the relative scores needed for
temperature calibration. Other unknown request fields are rejected. Schema
errors use a local 422 error envelope; this is not byte-for-byte TypeSafe
error compatibility. The actual served model name is returned; `jev-latest` is
accepted as an alias for it.

Labels begin with A–Z. Beyond Z the label scheme decides (`JEV_LABEL_SCHEME`;
`--decision-label-scheme` on a decision worker): `upper` (default) continues
with uppercase AA, AB, … then AAA, …, keeping only verified single tokens;
`served` keeps the original a–z, 0–9, then AA, … order. At most 26 options are
labelled A–Z under both. Responses report the scheme in `metadata.label_scheme`.
A decision worker refuses a `jev-gpu-readout-v2` artifact whose `label_scheme`
differs from the configured one (v1 artifacts count as `served`), so pin
`--decision-label-scheme` (or `JEV_LABEL_SCHEME`) in every launch script and
re-export v1 readouts with `--label-scheme upper` before switching to `upper`.
The flag is refused without `--decision-readout-path`; the no-readout
`/v1/systemone` path reads only `JEV_LABEL_SCHEME`.
The real Qwen tokenizer supports 255 distinct labels. Tokenization is checked at the
assistant boundary; an unsupported tokenizer fails rather than silently scoring
the wrong labels. More choices may still reduce model accuracy. Qwen's
`enable_thinking=False` is always applied. Other model templates may require an
explicit assistant prefix and have not been validated.

`images` accepts HTTP(S) URLs or image data URLs. This extends the text-only
TypeSafe interface following the linked llama.cpp fork. Actual model image
preprocessing is delegated to SGLang's chat adapter. A text-only loaded model
rejects image requests. Raw bare-base64 images are not accepted; use a data URL.
A `--decision-readout-path` worker is stricter: it accepts only base64 PNG, JPEG
or WebP data URLs and never fetches a URL or reads a local path.

## Start the GPU server

Platform: Linux with NVIDIA CUDA and enough GPU memory for the selected Qwen
checkpoint plus vision processing, cache, and runtime workspace. No GPU server
was available during development; the commands below are the handoff for a real
GPU smoke test, not a claim that model execution has passed.

Install this fork using the upstream source-install prerequisites. From the
repository root in a fresh environment:

```bash
git clone --branch feat/jev-systemone https://github.com/myeongho12/sglang-jev.git
cd sglang-jev
uv pip install -e ./python
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --host 127.0.0.1 --port 30000 \
  --tp 1 --context-length 8192 --mem-fraction-static 0.9
```

Adjust `--tp` to the available GPU layout. This example uses the model's standard
weights; it does not select a quantized checkpoint. Do not enable speculative
decoding/MTP: the endpoint rejects that configuration because the upstream
zero-token optimization depends on speculation being disabled. A dedicated
classification server makes compute measurements easier to interpret than a
server mixing classification with long generation requests.

```bash
curl http://localhost:30000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B",
    "state": "결제가 두 번 청구됐습니다. 중복 결제를 환불해주세요.",
    "questions": {
      "team": {
        "type": "choice", "instructions": "담당 부서를 선택하세요.",
        "criteria": {"billing": "결제·환불", "technical": "기술 오류"}
      },
      "refund": {"type": "noul", "instructions": "환불을 요청하고 있나요?"},
      "severity": {
        "type": "score", "instructions": "결제 피해를 분류하세요.",
        "criteria": ["피해 없음", "한 건의 중복 결제", "여러 건의 중복 결제"]
      }
    }
  }'
```

Run the real-server smoke checks before collecting benchmark results:

```bash
python examples/runtime/jev/smoke_test.py --base-url http://localhost:30000
```

This checks the three output types, an image request, 255 choices, and zero
reported output tokens. It checks contracts, not model accuracy.
For an authenticated server, set `SGLANG_API_KEY` in the client environment;
the smoke, business, and visual-game clients send a Bearer header.

## CPU verification

```bash
uv venv .venv-jev
uv pip install --python .venv-jev/bin/python -r examples/runtime/jev/requirements-cpu.txt
.venv-jev/bin/python examples/runtime/jev/run_cpu_tests.py
```

The runner bypasses SGLang's CUDA package initialization while importing the real
extension modules. It tests validation, numerical scoring, HTTP response
contracts, image forwarding, cancellation, the evaluation harness, and the game
environment. The GPU engine is faked explicitly. One game transport test opens a
temporary loopback HTTP server.

Optional tokenizer verification: download only `tokenizer.json` and
`tokenizer_config.json` from Qwen revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`, then run:

```bash
JEV_TOKENIZER_DIR=/path/to/tokenizer-files \
  .venv-jev/bin/python examples/runtime/jev/run_cpu_tests.py
```

This checks all 255 labels at the real non-thinking chat-template boundary for
text and image prompts. It does not run the vision encoder or language model.

## Korean business decisions

`business_cases.jsonl` contains 24 synthetic Korean cases with explicit policies
and three decisions each. Calibration/evaluation families are disjoint. These
are rule-following diagnostics, not representative production accuracy or a
reproduction of Jev's benchmark.

```bash
python examples/runtime/jev/business_eval.py --validate-only
python examples/runtime/jev/business_eval.py \
  --base-url http://localhost:30000 --split calibration \
  --output /tmp/jev-business-calibration.json
python examples/runtime/jev/business_eval.py \
  --fit-temperature /tmp/jev-business-calibration.json \
  --output /tmp/jev-business-temperature.json
```

Then pass the reported temperature to a held-out evaluation:

```bash
python examples/runtime/jev/business_eval.py \
  --base-url http://localhost:30000 --split eval \
  --temperature TEMPERATURE_FROM_CALIBRATION \
  --output /tmp/jev-business-eval.json
```

The harness records raw responses and per-rotation logprobs; reports accuracy,
choice macro-F1, NLL/Brier/ECE, score MAE, and request latency; and rejects
nonzero output-token counts. Use its `--help` for fitting a temperature from a
calibration report, then supply the fitted temperature to evaluation. Do not fit
or select a temperature from the evaluation split. The included small corpus
does not establish probability calibration for other tasks or games.

## Screenshot-to-action game

`visual_game.py` renders a small grid game as PNG. A player must collect a key,
open a door, and reach the goal while avoiding hazards. The model receives only
the screenshot, fixed rules, and a short history of action names. Coordinates,
map state, seed, oracle path, and success labels are never sent to it. The chosen
up/down/left/right/wait action advances the game and produces the next screenshot.

Check the environment without a model:

```bash
python examples/runtime/jev/visual_game.py \
  --mode oracle --split eval --episodes 5 --output-dir /tmp/jev-game-oracle
```

Run Qwen against screenshots:

```bash
python examples/runtime/jev/visual_game.py \
  --mode live --base-url http://localhost:30000 \
  --model Qwen/Qwen3.6-35B-A3B \
  --split eval --episodes 20 --output-dir /tmp/jev-game-live \
  --diagnostic-goal
```

Each run writes screenshots, episode traces, `summary.json`, and a self-contained
`replay.html`. Metrics include success rate, invalid moves, step count, and
decision latency. The optional Noul question judges the final screenshot in a
separate request and cannot control the environment. Train, calibration, and
evaluation use disjoint seed ranges. Oracle runs are explicitly marked
`model_inference: false`; their latency is never an LLM speed measurement.

This initial game evaluates visual decisions in a controlled turn-based loop.
It is not a Doom benchmark or evidence of real-time keyboard/mouse control in
an arbitrary commercial game. A game-specific screenshot/control adapter and
action-duration policy are the next integration point.

## Benchmark provenance

See [BENCHMARK_SOURCES.md](BENCHMARK_SOURCES.md) for official TypeSafe workflow
evaluations, the currently unrecovered 1,191/358-question llama.cpp corpus,
reusable third-party NPC/chess tests, and leakage-aware comparison protocols.
See [VALIDATION.md](VALIDATION.md) for what was actually run in development.
