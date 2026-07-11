#!/usr/bin/env python
"""Transcribe a few specific clips through a model's real adapter code path.

Use to verify adapter fixes on known-problem clips before a full run:
  .venvs/voxtral/bin/python scripts/debug_model.py voxtral-mini-realtime data/clean/en/en_362.wav ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_eval.adapters import get_adapter


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: debug_model.py <model-name> <wav> [wav ...]")
    model_name, paths = sys.argv[1], sys.argv[2:]
    adapter, model_id = get_adapter(model_name)
    handle = adapter.load(model_id)
    items = [{"audio_path": p, "lang": "en"} for p in paths]
    for p, text in zip(paths, adapter.transcribe(handle, items)):
        status = "EMPTY!" if not text.strip() else "ok"
        print(f"[{status}] {p}\n        {text[:200]!r}\n")


if __name__ == "__main__":
    main()
