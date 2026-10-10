#!/usr/bin/env bash
# Launch the decision-mode SGLang fork (StandardOne-8B v2.2) as container
# eng-decision on :30010, with the launch profile of the GPU it runs on.
#
#   GPU_PROFILE=auto|gfx950|gfx942|sm90|sm100   (auto: read from the host)
#
#   gfx950  MI350X/MI355X  triton attention, aiter ops off   (production, 2026-10-05)
#   gfx942  MI300X/MI325X  triton attention, aiter ops off   (verified 2026-10-06)
#   sm90    H100/H200      fa3 attention, KV-free prefill (text and images)
#   sm100   B200/B300      fa4 attention, KV-free prefill (hardware validation pending)
#
# Why triton on both AMD families: SGLang's ROCm default, aiter attention,
# came up corrupted on about half of the container starts on the MI350X test
# host (fork and unmodified base image alike: wrong, run-to-run varying
# probabilities, later "Memory access fault by GPU"); see README.md. triton
# was stable on every start on both gfx950 and gfx942. Check each start with
# tools/canary.py (or the relaxed v2.2 canary) before sending traffic.
#
# Overrides: IMAGE, NAME, PORT, MODEL_DIR, READOUT, MEM_MB (static KV pool in
# MB, default 40000, turned into --mem-fraction-static from the GPU's memory),
# MEM_FRACTION (used verbatim instead), ATTENTION_BACKEND, GPU (NVIDIA device
# index). Extra arguments are appended to launch_server (a later flag wins).
# AITER_OPS=1 opts into AMD aiter GEMM/norm/activation kernels; this changes
# probabilities and is experimental. MAX_RUNNING_REQUESTS caps concurrent
# evaluations inside the engine, not the number of external API requests.
set -euo pipefail
NAME=${NAME:-eng-decision}
PORT=${PORT:-30010}
MODEL_DIR=${MODEL_DIR:-/opt/di/models/StandardOne-8B-v22}
READOUT=${READOUT:-/opt/engine/readout-v22/standardone-8B-v22-gpu-readout.npz}
REVISION=${REVISION:-ac495d1c221297f0ae951941eff27bcf58b5ce52}
MEM_MB=${MEM_MB:-40000}
AITER_OPS=${AITER_OPS:-0}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-32}
case "$AITER_OPS" in
  0|1) ;;
  *) echo "AITER_OPS must be 0 or 1" >&2; exit 1 ;;
esac
if ! [[ "$MAX_RUNNING_REQUESTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RUNNING_REQUESTS must be a positive integer" >&2; exit 1
fi

detect_profile() {
  local target
  if [ -d /sys/class/kfd/kfd/topology/nodes ]; then
    target=$(cat /sys/class/kfd/kfd/topology/nodes/*/properties | awk '$1=="gfx_target_version" && $2!=0 {print $2; exit}')
    case "$target" in
      905*) echo gfx950 ;;
      904*) echo gfx942 ;;
      *) echo "unknown AMD gfx_target_version $target" >&2; exit 1 ;;
    esac
  elif command -v nvidia-smi >/dev/null; then
    target=$(nvidia-smi -i "${GPU:-0}" --query-gpu=compute_cap --format=csv,noheader)
    case "$target" in
      9.0) echo sm90 ;;
      10.*) echo sm100 ;;
      *) echo "unknown NVIDIA compute capability $target" >&2; exit 1 ;;
    esac
  else
    echo "no AMD kfd topology and no nvidia-smi: set GPU_PROFILE" >&2; exit 1
  fi
}

gpu_total_mb() {
  if [ -d /sys/class/kfd/kfd/topology/nodes ]; then
    # The first GPU node's first memory bank (HBM); CPU nodes have no banks with a GPU heap.
    for node in /sys/class/kfd/kfd/topology/nodes/*; do
      awk '$1=="gfx_target_version" && $2!=0 {found=1} END {exit !found}' "$node/properties" || continue
      awk '$1=="size_in_bytes" {printf "%d\n", $2/1048576; exit}' "$node/mem_banks/0/properties"; return
    done
  else
    nvidia-smi -i "${GPU:-0}" --query-gpu=memory.total --format=csv,noheader,nounits
  fi
}

PROFILE=${GPU_PROFILE:-auto}
[ "$PROFILE" = auto ] && PROFILE=$(detect_profile)
case "$PROFILE" in
  gfx950)
    IMAGE=${IMAGE:-st-sglang:decision-mi35x}
    ATTENTION_BACKEND=${ATTENTION_BACKEND:-triton}
    DEVICE_ARGS=(--device /dev/kfd --device /dev/dri --group-add video)
    ENV_ARGS=(-e SGLANG_USE_AITER="$AITER_OPS")
    PROFILE_ARGS=() ;;
  gfx942)
    IMAGE=${IMAGE:-st-sglang:decision-mi30x}
    ATTENTION_BACKEND=${ATTENTION_BACKEND:-triton}
    DEVICE_ARGS=(--device /dev/kfd --device /dev/dri --group-add video)
    ENV_ARGS=(-e SGLANG_USE_AITER="$AITER_OPS")
    PROFILE_ARGS=() ;;
  sm90)
    IMAGE=${IMAGE:-st-sglang:decision-cu13}
    ATTENTION_BACKEND=${ATTENTION_BACKEND:-fa3}
    DEVICE_ARGS=(--gpus "device=${GPU:-0}")
    ENV_ARGS=()
    # The fork's KV-free prefill (no paged KV pool, FA raw-K/V path), text and
    # images alike. KVFREE=0 serves through the paged KV pool instead.
    PROFILE_ARGS=(); [ "${KVFREE:-1}" = 1 ] && PROFILE_ARGS=(--prefill-only-disable-kv-cache) ;;
  sm100)
    IMAGE=${IMAGE:-st-sglang:decision-cu13}
    ATTENTION_BACKEND=${ATTENTION_BACKEND:-fa4}
    DEVICE_ARGS=(--gpus "device=${GPU:-0}")
    ENV_ARGS=()
    PROFILE_ARGS=(); [ "${KVFREE:-1}" = 1 ] && PROFILE_ARGS=(--prefill-only-disable-kv-cache) ;;
  *) echo "unknown GPU_PROFILE $PROFILE" >&2; exit 1 ;;
esac
case "$PROFILE:$AITER_OPS" in
  sm90:1|sm100:1) echo "AITER_OPS is only supported on AMD profiles" >&2; exit 1 ;;
esac
if [ -z "${MEM_FRACTION:-}" ]; then
  TOTAL_MB=$(gpu_total_mb)
  MEM_FRACTION=$(python3 -c "print(round($MEM_MB / $TOTAL_MB, 3))")
fi
echo "profile=$PROFILE image=$IMAGE attention=$ATTENTION_BACKEND aiter_ops=$AITER_OPS max_running_requests=$MAX_RUNNING_REQUESTS mem_fraction=$MEM_FRACTION" >&2

docker run -d --name "$NAME" \
  "${DEVICE_ARGS[@]}" \
  --network host --ipc private --shm-size 32g --security-opt seccomp=unconfined \
  -v "$MODEL_DIR":/models/StandardOne-8B-v22:ro \
  -v "$(dirname "$READOUT")":/opt/engine/readout:ro \
  -e HF_HUB_OFFLINE=1 \
  -e JEV_DEFAULT_TEMPERATURE=1.65 \
  -e JEV_PROMPT_WORDING=served \
  -e JEV_MODEL_ALIAS=jev-latest \
  -e JEV_LABEL_SCHEME=upper \
  ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /models/StandardOne-8B-v22 \
    --revision "$REVISION" \
    --served-model-name standardthinking/standard-one-8b \
    --decision-readout-path "/opt/engine/readout/$(basename "$READOUT")" \
    --decision-label-scheme upper \
    --host 127.0.0.1 --port "$PORT" \
    --tp-size 1 --model-impl sglang --dtype bfloat16 \
    --attention-backend "$ATTENTION_BACKEND" \
    --context-length 32768 --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --mem-fraction-static "$MEM_FRACTION" \
    --chunked-prefill-size -1 \
    --model-config-parser hf --load-format safetensors \
    --disable-radix-cache --disable-decode-cuda-graph --disable-prefill-cuda-graph \
    --decision-deadline-seconds 110 \
    ${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"} \
    "$@"
