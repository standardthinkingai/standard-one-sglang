# Ministral text and image decision latency

This measures local latency only. It does not grade answers or train a model.
The model is `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, a multimodal
Ministral 3 checkpoint with the vision tower retained. It uses the existing
`requirements-mlx.txt` environment on Apple Silicon.

## Inputs and timing

The harness selects 24 four-choice text questions from the frozen KEV transfer-v4
development corpus (12 MMLU and 12 SciQ). Ground-truth labels are stripped.
Every prompt runs in three conditions: text only, text plus one 448×448 image,
and text plus one 896×896 image. Image content is unrelated to the questions:
this controlled addition measures image processing cost, not visual accuracy
or semantic equivalence between a text description and a picture.

Two rounds give 48 observations per condition. Condition and prompt order are
randomized with a fixed seed. Nine separate warmups cover all three conditions
at three prompt lengths; they are excluded. Round-specific summaries help
identify residual first-use or thermal effects. P95 describes these measured
samples, not a production serving guarantee.

Every request uses fresh KV and vision features. The real vision encoder,
patch merger, and projector run on each image request. The decoder evaluates
the prompt and projects only the last position to A/B/C/D probabilities;
no answer tokens are generated. Standard model chat templates, including the
default Mistral system prompt, remain active. Reported token counts include it.

Timing includes prompt preparation, CPU image transforms, synchronized vision
embeddings, language prefill and the final option probability calculation.
It excludes loading model weights, disk reads, image-file decoding, networking,
HTTP, and request queueing. Images are already in memory. Stage boundaries are
synchronized to separate preprocessing, vision embeddings, and language work.
The vision timing includes projector and embedding insertion overhead.

## Run

Use [the local MLX installation guide](LOCAL_MLX.md) to create the environment.
Download a fixed model revision:

```bash
HF_HUB_DISABLE_XET=1 .venv-jev-mlx/bin/hf download \
  mlx-community/Ministral-3-3B-Instruct-2512-4bit \
  --revision a962dcb09eee4169c890e544c9eb938f1113fdee
```

Set `MODEL_PATH` to the resulting snapshot directory. The KEV development
JSONL source and verified hash are in [KEV benchmark notes](KEV_BENCHMARK_NOTES.md).

```bash
HF_HUB_OFFLINE=1 .venv-jev-mlx/bin/python examples/runtime/jev/ministral_latency_eval.py \
  --model "$MODEL_PATH" \
  --suite-jsonl /path/to/transfer-v4/development.jsonl \
  --output-dir /tmp/ministral-latency
```

The output directory contains the exact prompts and image paths, raw per-request
measurements, excluded warmups, and aggregate `summary.json`. The local measured
download additionally has a sibling `model.json` that records its revision and
verified weight SHA-256; the harness includes it when available.

Sources: [Mistral model documentation](https://docs.mistral.ai/models/ministral-3-3b-25-12),
[MLX checkpoint](https://huggingface.co/mlx-community/Ministral-3-3B-Instruct-2512-4bit).
