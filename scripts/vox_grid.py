#!/usr/bin/env python
"""Diagnostic grid for Voxtral realtime empties: per-clip audio stats plus
transcription of raw/lead-silence/gain/trail-silence variants against a
locally running vllm realtime server (port 8991)."""

import base64
import json

import numpy as np
import soundfile as sf
from websockets.sync.client import connect

MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
SR = 16000


def transcribe(arr):
    pcm16 = (np.clip(arr, -1, 1) * 32767).astype(np.int16).tobytes()
    with connect("ws://127.0.0.1:8991/v1/realtime", max_size=None) as ws:
        assert json.loads(ws.recv(timeout=30)).get("type") == "session.created"
        ws.send(json.dumps({"type": "session.update", "model": MODEL}))
        ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        for i in range(0, len(pcm16), 4096):
            ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16[i:i + 4096]).decode(),
            }))
        ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        parts = []
        while True:
            ev = json.loads(ws.recv(timeout=120))
            if ev["type"] == "transcription.delta":
                parts.append(ev["delta"])
            elif ev["type"] == "transcription.done":
                return (ev.get("text") or "".join(parts)).strip(), ev.get("usage")
            elif ev["type"] == "error":
                return f"ERROR {ev}", None


for name in ["en_362", "en_183", "en_129"]:
    x, _ = sf.read(f"data/clean/en/{name}.wav", dtype="float32")
    nz = np.where(np.abs(x) > 0.001)[0]
    lead = nz[0] / SR if len(nz) else -1
    trail = (len(x) - nz[-1]) / SR if len(nz) else -1
    print(f"\n{name}: dur={len(x)/SR:.1f}s peak={np.abs(x).max():.3f} "
          f"rms={np.sqrt((x**2).mean()):.4f} lead_sil={lead:.2f}s trail_sil={trail:.2f}s")
    for label, arr in [
        ("raw     ", x),
        ("lead+1s ", np.concatenate([np.zeros(SR), x])),
        ("gain0.7 ", np.clip(x * 0.7 / max(np.abs(x).max(), 1e-6), -1, 1)),
        ("trail+2s", np.concatenate([x, np.zeros(2 * SR)])),
    ]:
        text, usage = transcribe(arr.astype(np.float32))
        print(f"  {label} -> {text[:100]!r}  usage={usage}")
