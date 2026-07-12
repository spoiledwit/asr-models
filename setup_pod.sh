#!/usr/bin/env bash
# One-time pod setup: creates one venv per serving stack (their deps conflict,
# NeMo and vLLM in particular pin different torch/transformers versions).
#
#   bash setup_pod.sh            # all stacks
#   bash setup_pod.sh whisper    # just one
set -euo pipefail
cd "$(dirname "$0")"

# system deps: ffmpeg shared libs (torchcodec), build tools
command -v ffmpeg >/dev/null || (apt-get update && apt-get install -y ffmpeg) || true

STACKS=("${@:-whisper qwen voxtral nemo vllm}")
[ $# -eq 0 ] && STACKS=(whisper qwen voxtral nemo vllm)

for stack in "${STACKS[@]}"; do
  echo "=== setting up .venvs/$stack ==="
  python -m venv ".venvs/$stack"
  ".venvs/$stack/bin/pip" install -U pip wheel
  ".venvs/$stack/bin/pip" install -r "requirements/$stack.txt"
done

echo
echo "Done. Usage:"
echo "  .venvs/whisper/bin/python scripts/prepare_data.py --langs en,ar,fr --n 200"
echo "  .venvs/whisper/bin/python scripts/run_eval.py --model whisper-large-v3-turbo"
echo "  .venvs/qwen/bin/python    scripts/run_eval.py --model qwen3-asr-1.7b"
echo "  .venvs/voxtral/bin/python scripts/run_eval.py --model voxtral-mini-realtime"
echo "  .venvs/voxtral/bin/python scripts/run_eval.py --model ark-asr-3b"
echo "  .venvs/nemo/bin/python    scripts/run_eval.py --model canary-1b-v2"
echo "  .venvs/nemo/bin/python    scripts/run_eval.py --model parakeet-tdt-0.6b-v3"
echo "  .venvs/whisper/bin/python scripts/report.py"
