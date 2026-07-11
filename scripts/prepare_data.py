#!/usr/bin/env python
"""Build the eval set: FLEURS subsets + noise conditions -> data/manifest.jsonl

Examples:
  python scripts/prepare_data.py --langs en,ar,fr --n 200
  python scripts/prepare_data.py --langs en,ar --n 100 --conditions clean,babble@10,babble@5,white@10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_eval.data import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", default="en,ar,fr", help="comma-separated ISO codes (see stt_eval/langs.py)")
    p.add_argument("--n", type=int, default=200, help="utterances per language")
    p.add_argument("--conditions", default="clean,babble@10,babble@5",
                   help="comma list: clean | babble@SNR | white@SNR | musan@SNR")
    p.add_argument("--out", default="data", help="output directory")
    p.add_argument("--musan-dir", default=None, help="path to unpacked MUSAN noise/ dir (optional)")
    args = p.parse_args()

    prepare(
        langs=args.langs.split(","),
        n_per_lang=args.n,
        conditions=args.conditions.split(","),
        out_dir=Path(args.out),
        musan_dir=Path(args.musan_dir) if args.musan_dir else None,
    )


if __name__ == "__main__":
    main()
