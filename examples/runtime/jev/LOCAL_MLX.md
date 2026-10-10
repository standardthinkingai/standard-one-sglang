# Training-free local vision decisions on Apple Silicon

`mlx_systemone.py` loads a Qwen3.5 multimodal checkpoint through MLX-VLM and
exposes the existing Choice/Noul/Score protocol at `POST /v1/systemone`.
It runs the vision encoder, evaluates the last prompt position, and normalizes
the selected label probabilities. It generates zero tokens and trains nothing.

This is a local experimental backend for the protocol, not a reproduction of
TypeSafe's proprietary model or calibration. It supports up to 62 choices,
four base64 image data URLs, and 8192 input tokens. Remote image URLs are not
accepted. Requests run serially with independent caches. The server binds only
to `127.0.0.1`.

## Install and download

From the repository root, create a separate environment:

```bash
uv venv --python 3.12 .venv-jev-mlx
uv pip install --python .venv-jev-mlx/bin/python \
  -r examples/runtime/jev/requirements-mlx.txt
HF_HUB_DISABLE_XET=1 .venv-jev-mlx/bin/hf download \
  mlx-community/Qwen3.5-4B-4bit \
  --revision 0e7ffd5c629ef7719d4cbc04069232580bfa9d9c
```

The download prints a snapshot directory. Use that directory as `MODEL_PATH`
below. The measured run's `model.json` records its actual path and revision.
Do not use `mlx_lm.load`: that is a language-only path for this model family.
Use an environment that can access the Mac's Metal GPU.

## Run the visual diagnostic

```bash
.venv-jev-mlx/bin/python examples/runtime/jev/visual_probe_eval.py \
  --model "$MODEL_PATH" --output-dir /tmp/qwen-vision-probe
```

The diagnostic contains 24 balanced 448x448 images across color, counting,
spatial relations, OCR, bar charts, and UI status. Each category has four
identical questions/option lists and one example of each correct answer.
Only image pixels, the question, and choices enter the model. Neither file
names nor ground truth enter the prompt.

It runs three conditions, with a separate recorded warmup:

- `normal`: the intended image.
- `swapped_image`: another image in the same category with a different correct
  answer. Compare both original-answer and visible-image accuracy, plus whether
  the prediction changed from the normal condition.
- `no_image`: identical question and choices without an image. This is a
  diagnostic control, not a valid visual task with an answerable question.

Outputs include every option probability and response in `results.jsonl`,
aggregate metrics and runtime versions in `summary.json`, images, and a human
contact sheet. Reported latency includes image preparation and synchronized
inference; model loading and warmup are separate. No vision-feature or prefix
cache is used. `vision_embedding_ms` also includes embedding/position setup.

These synthetic probes only test the local vision-to-decision path. They do
not measure general visual ability, production performance, probability
calibration, or comparability to KEV/Jev published benchmark scores.

Verified on 2026-09-21 with an Apple M5 (32 GiB): all 24 original images and
24 swapped-image controls were classified correctly; the no-image control
matched 6/24 original answers. Mean original-image request time was 345 ms,
including preprocessing; MLX peak allocated memory was 3.70 GB (not total
process memory). These are easy synthetic fixtures, not a general benchmark.

To also run the separate 32-record KEV text probe, add
`--kev-dev-jsonl /path/to/transfer-v4/development.jsonl`. The helper verifies
the frozen file hash, selects four clean records per source, preserves policy
pairs, and strips labels and metadata from requests. The measured subset
scored 25/32; it must not be compared to published full-suite KEV accuracy.
See [KEV benchmark notes](KEV_BENCHMARK_NOTES.md) for the download source.

## Start the local API

```bash
.venv-jev-mlx/bin/python examples/runtime/jev/mlx_systemone.py \
  --model "$MODEL_PATH" --port 30110
```

Then existing clients can use `http://127.0.0.1:30110`, with
`model="mlx-community/Qwen3.5-4B-4bit"` or the alias `jev-latest`. For example:

```bash
python examples/runtime/jev/visual_game.py \
  --mode live --base-url http://127.0.0.1:30110 \
  --model mlx-community/Qwen3.5-4B-4bit \
  --split eval --episodes 1 --max-steps 12 \
  --output-dir /tmp/qwen-visual-game
```

Choice/Score `confidence` is one minus normalized option entropy, not the
probability of correctness. Raw vocabulary label mass is recorded separately
because probabilities conditional on the offered labels can appear confident
even when the model gives those labels little total probability.

Each question and rotation performs an independent prefill, including image
encoding. Multi-question requests therefore repeat vision work. This adapter
does not implement a shared-state constant-latency multi-question inference.
