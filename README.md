# STT Model Benchmark Harness

Compares multilingual speech-to-text models on quality (WER/CER), noise robustness,
and speed (RTFx), using FLEURS test data with synthetic noise conditions — no dataset
of your own needed. Model background and selection rationale: [stt-models-shortlist.md](stt-models-shortlist.md).

**Models wired up:** Whisper large-v3 / v3-turbo, Qwen3-ASR 0.6B / 1.7B,
Voxtral Mini 4B Realtime, Canary-1B-v2, Parakeet-TDT-0.6B-v3, ARK-ASR-3B.

## Quickstart on a RunPod pod (24 GB GPU recommended)

Open a **terminal** in Jupyter (File → New → Terminal) — long runs belong in a
terminal (ideally under `tmux`), use the notebook only to inspect results.

```bash
git clone <your-repo-url> && cd models-test

# 1. One venv per serving stack (~10-20 min; NeMo is the slow one)
bash setup_pod.sh                    # or: bash setup_pod.sh whisper qwen

# 2. Build the eval set (FLEURS + noise). ~200 clips/lang ≈ stable WER.
.venvs/whisper/bin/python scripts/prepare_data.py \
    --langs en,ar,fr --n 200 --conditions clean,babble@10,babble@5

# 3. Run each model in its own venv
.venvs/whisper/bin/python scripts/run_eval.py --model whisper-large-v3-turbo
.venvs/whisper/bin/python scripts/run_eval.py --model whisper-large-v3
.venvs/qwen/bin/python    scripts/run_eval.py --model qwen3-asr-1.7b
.venvs/voxtral/bin/python scripts/run_eval.py --model voxtral-mini-realtime
.venvs/voxtral/bin/python scripts/run_eval.py --model ark-asr-3b
.venvs/nemo/bin/python    scripts/run_eval.py --model canary-1b-v2
.venvs/nemo/bin/python    scripts/run_eval.py --model parakeet-tdt-0.6b-v3

# 4. Comparison table -> results/summary.md
.venvs/whisper/bin/python scripts/report.py
```

Each eval writes `results/<model>.json` with per-utterance ref/hyp pairs (grep those
to see *what kind* of errors a model makes, not just how many).

## Notes

- **Languages**: edit `--langs`; supported codes live in `stt_eval/langs.py` (all 13
  Voxtral languages pre-mapped). NeMo models only cover the European subset and skip
  the rest automatically.
- **Noise conditions**: `babble@SNR` (summed speakers, cafe-like), `white@SNR`, or
  `musan@SNR` with `--musan-dir` if you download the real MUSAN corpus. Lower SNR =
  noisier; the interesting read is how *slowly* each model degrades from clean → 5 dB.
- **Fair comparison**: one shared text normalizer (Whisper's `BasicTextNormalizer`)
  for every model; CER instead of WER for zh/ja/ko.
- **RTFx caveat**: offline batch throughput, not conversational latency. Voxtral's
  streaming latency (its headline feature) needs `vllm serve
  mistralai/Voxtral-Mini-4B-Realtime-2602` and its `/v1/realtime` endpoint —
  planned as phase 2 after the WER pass picks the top 2-3 models.
- **HF auth**: `huggingface-cli login` is not needed for FLEURS or any wired model;
  it is needed only if you add gated datasets like Common Voice.
- **Verify one clip first**: after setup, run with `--n 5` once end-to-end before
  the full pass — the two newest models (Voxtral, Qwen3-ASR) move fast and their
  APIs may have shifted since this was written.

## Layout

```
stt_eval/           package: langs, metrics, data prep, model adapters
  adapters/         one module per serving stack, common load()/transcribe() interface
scripts/            prepare_data.py, run_eval.py, report.py
requirements/       base + one file per stack (they conflict -> separate venvs)
setup_pod.sh        creates .venvs/{whisper,qwen,voxtral,nemo}
data/, results/     generated, gitignored
```
# asr-models
