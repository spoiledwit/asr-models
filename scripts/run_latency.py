#!/usr/bin/env python
"""Measure Qwen3-ASR streaming latency under real-time-paced audio.

Feeds each clip in chunk_ms slices at wall-clock pace (like a live caller),
and records per utterance:
  - first_partial_s: speech start -> first partial text visible
  - finalize_s:      last audio chunk fed -> final transcript returned
  - chunk_rtf:       p95 chunk processing time / chunk duration (must be <1
                     or the model can't keep up with a live stream)
  - streaming WER vs the reference (the quality tax of this chunk size)

Run in the qwen venv (needs qwen-asr; uses the vLLM backend if installed,
else transformers):
  .venvs/qwen/bin/python scripts/run_latency.py --chunk-ms 1000 --n 100
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_eval.langs import lang_info
from stt_eval.metrics import error_rate

SR = 16000


def pctl(xs, p):
    return round(statistics.quantiles(xs, n=100)[p - 1], 3) if len(xs) >= 2 else (xs[0] if xs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--chunk-ms", type=int, default=1000)
    ap.add_argument("--n", type=int, default=100, help="clips per condition")
    ap.add_argument("--langs", default="en,ar,fr")
    ap.add_argument("--conditions", default="clean,babble@5")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    from qwen_asr import Qwen3ASRModel

    try:
        asr = Qwen3ASRModel.LLM(model=args.model, gpu_memory_utilization=0.8, max_new_tokens=32)
        backend = "vllm"
    except Exception as e:
        print(f"[latency] vLLM backend unavailable ({e}); falling back to transformers")
        import torch

        asr = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
        backend = "transformers"
    print(f"[latency] backend: {backend}, chunk: {args.chunk_ms}ms")

    items = [json.loads(l) for l in open(args.manifest)]
    langs = set(args.langs.split(","))
    conds = set(args.conditions.split(","))
    by_key = {}
    for it in items:
        if it["lang"] in langs and it["condition"] in conds:
            by_key.setdefault((it["lang"], it["condition"]), []).append(it)

    chunk_n = int(SR * args.chunk_ms / 1000)
    report = {}
    for (lang, cond), pool in sorted(by_key.items()):
        pool = pool[: args.n]
        first_partials, finalizes, chunk_times, refs, hyps = [], [], [], [], []
        for k, it in enumerate(pool):
            audio, _ = sf.read(it["audio_path"], dtype="float32")
            state = asr.init_streaming_state(unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=2.0)
            t0 = time.perf_counter()
            first_partial = None
            n_chunks = (len(audio) + chunk_n - 1) // chunk_n
            for i in range(n_chunks):
                due = t0 + (i + 1) * args.chunk_ms / 1000  # chunk exists once fully "spoken"
                wait = due - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                tp = time.perf_counter()
                asr.streaming_transcribe(audio[i * chunk_n:(i + 1) * chunk_n], state)
                chunk_times.append(time.perf_counter() - tp)
                if first_partial is None and (state.text or "").strip():
                    first_partial = time.perf_counter() - t0
            t_end = time.perf_counter()  # last chunk fed and processed
            asr.finish_streaming_transcribe(state)
            finalizes.append(time.perf_counter() - t_end)
            if first_partial is not None:
                first_partials.append(first_partial)
            refs.append(it["text"])
            hyps.append((state.text or "").strip())
            print(f"\r[latency] {lang}/{cond}: {k + 1}/{len(pool)}", end="", flush=True)
        print()
        wer = error_rate(refs, hyps, lang_info(lang)["cer_based"])
        report[f"{lang}/{cond}"] = {
            "n": len(pool),
            "first_partial_p50": pctl(first_partials, 50),
            "first_partial_p95": pctl(first_partials, 95),
            "finalize_p50": pctl(finalizes, 50),
            "finalize_p95": pctl(finalizes, 95),
            "chunk_proc_p95_s": pctl(chunk_times, 95),
            "chunk_rtf_p95": round(pctl(chunk_times, 95) / (args.chunk_ms / 1000), 3),
            "streaming_wer_pct": round(wer * 100, 2),
        }
        for key, val in report[f"{lang}/{cond}"].items():
            print(f"[latency]   {key}: {val}")

    out = Path(args.out_dir) / f"latency-qwen-{args.chunk_ms}ms.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": args.model, "backend": backend, "chunk_ms": args.chunk_ms,
                   "report": report}, f, indent=1)
    print(f"[latency] wrote {out}")


if __name__ == "__main__":
    main()
