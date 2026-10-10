#!/usr/bin/env bash
# restart_loop.sh <label> <n_starts> <golden> [extra launch args...]
label=$1; n=$2; golden=$3; shift 3
for i in $(seq 1 "$n"); do
  docker rm -f eng-decision >/dev/null 2>&1
  /opt/engine/launch-eng-decision.sh "$@" >/dev/null
  timeout 400 bash -c "until curl -sf http://127.0.0.1:30010/health >/dev/null || ! docker ps --format {{.Names}} | grep -qx eng-decision; do sleep 2; done"
  if [ ! -s "$golden" ]; then
    /opt/di/venv/bin/python /opt/engine/tools/canary.py --n 150 --golden /opt/engine/data/canary-golden.jsonl --record "$golden" >/dev/null 2>&1
    echo "$label start $i: recorded golden $golden"; continue
  fi
  echo -n "$label start $i: "
  /opt/di/venv/bin/python /opt/engine/tools/canary.py --n 150 --golden "$golden" 2>&1 | tail -1
done
