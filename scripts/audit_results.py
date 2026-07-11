#!/usr/bin/env python
"""Audit results/*.json for broken-transcription signatures:
empty hyps, truncation (hyp much shorter than ref), repetition loops.

  .venvs/whisper/bin/python scripts/audit_results.py            # all result files
  .venvs/whisper/bin/python scripts/audit_results.py qwen       # files matching 'qwen'
"""

import json
import sys
from collections import Counter
from pathlib import Path


def flags(ref: str, hyp: str) -> list[str]:
    out = []
    h = hyp.strip()
    r = ref.strip()
    if not h:
        return ["EMPTY"]
    rw, hw = r.split(), h.split()
    if len(rw) >= 8 and len(hw) < 0.5 * len(rw):
        out.append("SHORT")  # possible truncation
    if len(hw) >= 10:
        top = Counter(hw).most_common(1)[0]
        if top[1] > max(4, 0.4 * len(hw)):
            out.append("REPEAT")  # hallucination loop
    return out


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    paths = [p for p in sorted(Path("results").glob("*.json")) if pattern in p.name]
    if not paths:
        sys.exit("no matching results/*.json")
    for path in paths:
        r = json.load(open(path))
        utts = r.get("utterances", [])
        counts = Counter()
        examples = []
        for u in utts:
            for f in flags(u["ref"], u["hyp"]):
                counts[f] += 1
                if len(examples) < 5:
                    examples.append((f, u["lang"], u["condition"], u["id"],
                                     u["ref"][:70], u["hyp"][:70]))
        total = len(utts)
        status = "CLEAN" if not counts else " ".join(f"{k}:{v}" for k, v in counts.items())
        print(f"{path.name}: {total} utts -> {status}")
        for f, lang, cond, uid, ref, hyp in examples:
            print(f"   [{f}] {uid}|{cond}\n      REF: {ref}\n      HYP: {hyp}")
    print("\nNote: a few SHORT/REPEAT flags in noisy conditions (babble@5) are normal —"
          "\nmodels genuinely fail there. Worry if flags cluster in 'clean'.")


if __name__ == "__main__":
    main()
