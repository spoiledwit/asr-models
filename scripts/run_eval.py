#!/usr/bin/env python
"""Run one model over the manifest and write per-utterance results + summary.

Run inside the venv for the model's stack (see README):
  python scripts/run_eval.py --model whisper-large-v3-turbo
  python scripts/run_eval.py --model canary-1b-v2 --langs en,fr
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_eval.adapters import MODELS, get_adapter
from stt_eval.langs import lang_info
from stt_eval.metrics import error_rate

BATCH = 16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--manifest", default="data/manifest.jsonl")
    p.add_argument("--langs", default=None, help="restrict to comma-separated langs")
    p.add_argument("--conditions", default=None, help="restrict to comma-separated conditions")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    items = [json.loads(l) for l in open(args.manifest)]
    if args.langs:
        keep = set(args.langs.split(","))
        items = [i for i in items if i["lang"] in keep]
    if args.conditions:
        keep = set(args.conditions.split(","))
        items = [i for i in items if i["condition"] in keep]

    # Skip languages the model's framework doesn't cover (NeMo models).
    adapter, model_id = get_adapter(args.model)
    _, _, stack = MODELS[args.model]
    if stack == "nemo":
        skipped = sorted({i["lang"] for i in items if lang_info(i["lang"])["nemo"] is None})
        if skipped:
            print(f"[eval] {args.model} does not support: {', '.join(skipped)} — skipping those")
        items = [i for i in items if lang_info(i["lang"])["nemo"] is not None]

    if not items:
        sys.exit("[eval] nothing to evaluate after filtering")

    print(f"[eval] loading {args.model} ({model_id}) ...")
    handle = adapter.load(model_id)

    # Warmup (excluded from timing) so model compile/caches don't skew RTFx.
    adapter.transcribe(handle, items[:1])

    print(f"[eval] transcribing {len(items)} utterances ...")
    hyps, t_total = [], 0.0
    for i in range(0, len(items), BATCH):
        batch = items[i : i + BATCH]
        t0 = time.perf_counter()
        hyps.extend(adapter.transcribe(handle, batch))
        t_total += time.perf_counter() - t0
        done = min(i + BATCH, len(items))
        print(f"\r[eval] {done}/{len(items)}", end="", flush=True)
    print()

    audio_sec = sum(i["duration"] for i in items)
    rtfx = audio_sec / t_total if t_total else 0.0

    # Per (lang, condition) error rates
    summary = {}
    for lang in sorted({i["lang"] for i in items}):
        cer_based = lang_info(lang)["cer_based"]
        for cond in sorted({i["condition"] for i in items}):
            idx = [k for k, it in enumerate(items) if it["lang"] == lang and it["condition"] == cond]
            if not idx:
                continue
            er = error_rate([items[k]["text"] for k in idx], [hyps[k] for k in idx], cer_based)
            summary[f"{lang}/{cond}"] = {
                "metric": "CER" if cer_based else "WER",
                "value": round(er * 100, 2),
                "n": len(idx),
            }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.model}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "model_id": model_id,
            "rtfx": round(rtfx, 1),
            "audio_seconds": round(audio_sec, 1),
            "wall_seconds": round(t_total, 1),
            "summary": summary,
            "utterances": [
                {"id": it["id"], "lang": it["lang"], "condition": it["condition"],
                 "ref": it["text"], "hyp": h}
                for it, h in zip(items, hyps)
            ],
        }, f, ensure_ascii=False, indent=1)

    print(f"[eval] RTFx: {rtfx:.1f}  ({audio_sec/60:.1f} min audio in {t_total:.1f}s)")
    for key, s in summary.items():
        print(f"[eval] {key}: {s['metric']} {s['value']}%  (n={s['n']})")
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
