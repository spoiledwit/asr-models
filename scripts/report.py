#!/usr/bin/env python
"""Aggregate results/*.json into a markdown comparison table (results/summary.md)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    results_dir = Path("results")
    # Merge multiple files for the same model (parallel --tag runs on different langs).
    groups: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            r = json.load(f)
        g = groups.setdefault(r["model"], {
            "model": r["model"], "audio_seconds": 0.0, "wall_seconds": 0.0, "summary": {},
        })
        g["audio_seconds"] += r["audio_seconds"]
        g["wall_seconds"] += r["wall_seconds"]
        g["summary"].update(r["summary"])
    runs = list(groups.values())
    for r in runs:
        r["rtfx"] = round(r["audio_seconds"] / r["wall_seconds"], 1) if r["wall_seconds"] else 0.0
    if not runs:
        sys.exit("No results/*.json found — run scripts/run_eval.py first.")

    keys = sorted({k for r in runs for k in r["summary"]})
    lines = [
        "# STT Eval Summary",
        "",
        "Values are WER% (CER% for zh/ja/ko). Lower is better. RTFx = audio-seconds per wall-second (higher is faster).",
        "",
        "| Model | RTFx | " + " | ".join(keys) + " |",
        "|---|---|" + "---|" * len(keys),
    ]
    for r in runs:
        cells = []
        for k in keys:
            s = r["summary"].get(k)
            cells.append(f"{s['value']}" if s else "—")
        lines.append(f"| {r['model']} | {r['rtfx']} | " + " | ".join(cells) + " |")

    out = results_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
