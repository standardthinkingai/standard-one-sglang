# Standard One decision engine (SGLang fork)

This repository is the inference engine that serves Standard Thinking's decision models
([StandardOne-8B-SH](https://huggingface.co/StandardThinking/StandardOne-8B-SH),
[StandardOne-3B-SH](https://huggingface.co/StandardThinking/StandardOne-3B-SH) and the Standard One letter-readout
models) through the System One / Jev API (`POST /v1/systemone`). It is SGLang with a **decision mode**: a request is
one prefill, no tokens are generated, and the answer is read from the model's hidden states.

It is a snapshot of the code we run in production (fork commit `ad26e8b179`), published as a single commit.

## What the fork adds

- **Prefill-only decisions.** A System One request (`state` + typed `questions`) is rendered once and answered from
  one prefill; output tokens are zero.
- **Schema-head readout** (`--decision-readout-path <dir with schema_head_config.json>`): the joint schema head of the
  SH models reads the final layer and one tapped layer and returns a distribution for every question at once
  (`python/sglang/srt/layers/decision_schema_head.py`). Letter-readout models use the same flag with a readout file.
- **Images** in decisions (Pixtral vision tower), preprocessed once per request.
- **Batch-invariant float32 head** so that answers do not depend on which requests share a batch.
- Request validation before GPU work (`decision_contract`), a deadline per request, and the Jev-compatible HTTP API
  through the adapter package [standard-one-adapter](https://github.com/standardthinkingai/standard-one-adapter).

The full list of changed files against upstream is in `STANDARD_ONE_CHANGES.txt`.

## Build

One Dockerfile builds every GPU family on top of a stock SGLang `v0.5.20` image; only the pure-Python `sglang` package
is replaced. The build context needs this repository and the adapter repository side by side:

```bash
mkdir ctx && cd ctx
git clone https://github.com/standardthinkingai/standard-one-sglang sglang-jev
git clone https://github.com/standardthinkingai/standard-one-adapter jev-adapter
mkdir -p jit-cache/aiter jit-cache/root-cache
# NVIDIA Blackwell / Hopper:
docker build -f sglang-jev/examples/runtime/jev/rocm/Dockerfile.decision \
  --build-arg BASE_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260929-79cafec0 \
  -t standard-one-engine:cu13 .
# AMD MI350X / MI355X: BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260929
```

## Run a schema-head model

```bash
hf download StandardThinking/StandardOne-3B-SH --local-dir StandardOne-3B-SH
docker run -d --gpus all --network host --ipc host --shm-size 32g \
  -v $PWD/StandardOne-3B-SH:/models/m:ro \
  standard-one-engine:cu13 python3 -m sglang.launch_server \
    --model-path /models/m --decision-readout-path /models/m/head \
    --served-model-name standardthinking/standard-one-3b-sh \
    --host 127.0.0.1 --port 30000 --tp-size 1 --model-impl sglang --dtype bfloat16 --attention-backend triton \
    --context-length 32768 --max-running-requests 64 --mem-fraction-static 0.3 \
    --chunked-prefill-size -1 --model-config-parser hf --load-format safetensors \
    --disable-radix-cache --disable-decode-cuda-graph --disable-prefill-cuda-graph --decision-deadline-seconds 110
curl -s localhost:30000/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"message": "Please close my card, I lost it yesterday."},
  "questions": {"urgent": {"type": "noul", "instructions": "The request is urgent."}}}'
```

On AMD use `--device /dev/kfd --device /dev/dri --group-add video` instead of `--gpus all` and add
`-e SGLANG_USE_AITER=0`. `examples/runtime/jev/rocm/launch-decision.sh` wraps these per GPU family.

## Validation

| Hardware | Image | Check |
|---|---|---|
| AMD MI350X | ROCm `v0.5.20-rocm720-mi35x` base | production since 2026-10-05; StandardOne-8B-SH: 12,028 evaluation questions 99.04 % same answer as the PyTorch reference, two passes bit-identical |
| NVIDIA B300 | CUDA 13 base, `triton` attention | 2026-10-11, 966 questions each (JevBench public tiers and held-out sets) against the PyTorch reference: StandardOne-8B-SH v2 99.17 % same answer (all 231 JevBench answers identical), StandardOne-3B-SH 98.96 %; one small request, one at a time: 8B 28 ms, 3B 24 ms median |

Differences against the PyTorch reference server come from bfloat16 kernel order, not from the readout.

## License and provenance

Apache-2.0, as upstream. This fork starts from SGLang upstream commit `0024efa0de38794ee309ba10ab00ebb891a3d050`
(2026-09-20, https://github.com/sgl-project/sglang); the decision-mode changes are by Standard Thinking. Upstream's
`LICENSE` and notices are kept unchanged.
