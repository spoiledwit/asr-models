#!/usr/bin/env bash
# Queue all model evals sequentially. Safe to start while another eval is
# already running (it waits), and safe to rerun (skips models whose
# results/<model>.json already exists; interrupted ones resume from checkpoint).
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[queue] waiting for any in-flight eval to finish ..."
while pgrep -f "scripts/run_eval.py" > /dev/null; do sleep 30; done

RUNS=(
  "voxtral voxtral-mini-realtime"
  "voxtral ark-asr-3b"
  "nemo    canary-1b-v2"
  "nemo    parakeet-tdt-0.6b-v3"
  "whisper whisper-large-v3"
  "whisper whisper-large-v3-turbo"
  "qwen    qwen3-asr-1.7b"
  "qwen    qwen3-asr-0.6b"
)

for run in "${RUNS[@]}"; do
  set -- $run
  venv=$1 model=$2
  if ls "results/$model"*.json >/dev/null 2>&1; then
    echo "[queue] skip $model (results exist)"
    continue
  fi
  echo "[queue] === $model ==="
  ".venvs/$venv/bin/python" scripts/run_eval.py --model "$model" \
    || echo "[queue] !!! $model failed, continuing with the rest"
done

.venvs/whisper/bin/python scripts/report.py
