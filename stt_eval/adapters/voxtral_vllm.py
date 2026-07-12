"""Voxtral Mini 4B Realtime via vLLM's /v1/realtime websocket endpoint.

This is the deployment path Mistral ships (the offline transformers path
produced deterministic empty transcripts on some clips). load() spawns
`vllm serve` from the dedicated vllm venv and waits for readiness; the client
side needs only `websockets` + numpy/soundfile, so this adapter runs from any
venv. Audio is streamed as base64 PCM16@16kHz per vLLM's realtime protocol.
"""

import atexit
import base64
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

PORT = 8991
VLLM_BIN = Path(".venvs/vllm/bin/vllm")
STARTUP_TIMEOUT_S = 900
CLIP_TIMEOUT_S = 300
CHUNK_BYTES = 4096


def _wait_ready(proc, log_path):
    deadline = time.time() + STARTUP_TIMEOUT_S
    url = f"http://127.0.0.1:{PORT}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve died at startup — see {log_path}")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"vllm serve not ready after {STARTUP_TIMEOUT_S}s — see {log_path}")


def load(model_id: str):
    if not VLLM_BIN.exists():
        raise RuntimeError("vllm venv missing — run: bash setup_pod.sh vllm")
    log_path = Path("logs/vllm-serve.log")
    log_path.parent.mkdir(exist_ok=True)
    log = open(log_path, "a")
    env = {**os.environ,
           "VLLM_DISABLE_COMPILE_CACHE": "1",
           # old flashinfer builds don't recognize Blackwell (sm_120) and fail
           # their arch check backwards; native sampling is fine for our use
           "VLLM_USE_FLASHINFER_SAMPLER": "0",
           # venv bin first so pip-installed build tools (ninja for flashinfer
           # JIT) are found even though we never "activate" the venv
           "PATH": f"{VLLM_BIN.parent.resolve()}:{os.environ.get('PATH', '')}"}
    proc = subprocess.Popen(
        [str(VLLM_BIN), "serve", model_id,
         "--tokenizer-mode", "mistral",
         # vLLM's own realtime examples serve this model with --enforce-eager;
         # the compiled/cudagraph path produced empty transcripts (model consumed
         # audio frames but every delta decoded to "").
         "--enforce-eager",
         "--port", str(PORT),
         # modest slice so parallel evals on the same GPU keep working
         "--gpu-memory-utilization", "0.35",
         "--max-model-len", "16384"],
        stdout=log, stderr=subprocess.STDOUT, env=env,
    )
    atexit.register(proc.terminate)
    print(f"[voxtral-vllm] waiting for server (log: {log_path}) ...")
    _wait_ready(proc, log_path)
    print("[voxtral-vllm] server ready")
    return {"proc": proc, "model": model_id}


def _transcribe_one(handle, audio_path: str) -> str:
    from websockets.sync.client import connect

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    # Peak-normalize: the realtime model's VAD gates near-silent recordings
    # (FLEURS has clips peaking at ~0.008) and returns empty transcripts.
    # Adds no information — equivalent to the AGC any voice pipeline runs.
    peak = np.abs(audio).max()
    if 0 < peak < 0.3:
        audio = audio * (0.7 / peak)
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()

    text_parts = []
    with connect(f"ws://127.0.0.1:{PORT}/v1/realtime", max_size=None) as ws:
        created = json.loads(ws.recv(timeout=30))  # wait for session.created
        if created.get("type") != "session.created":
            raise RuntimeError(f"unexpected first event: {created}")
        ws.send(json.dumps({"type": "session.update", "model": handle["model"]}))
        ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        for i in range(0, len(pcm16), CHUNK_BYTES):
            ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16[i : i + CHUNK_BYTES]).decode(),
            }))
        ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        deadline = time.time() + CLIP_TIMEOUT_S
        while time.time() < deadline:
            event = json.loads(ws.recv(timeout=CLIP_TIMEOUT_S))
            etype = event.get("type", "")
            if etype == "transcription.delta":
                text_parts.append(event.get("delta", ""))
            elif etype == "transcription.done":
                return (event.get("text") or "".join(text_parts)).strip()
            elif etype == "error":
                raise RuntimeError(f"realtime error: {event.get('error')}")
    return "".join(text_parts).strip()


def transcribe(handle, items: list[dict]) -> list[str]:
    # Concurrent websocket sessions — vLLM batches them server-side, so this
    # multiplies throughput without multiple servers fighting over the port.
    from concurrent.futures import ThreadPoolExecutor

    def one(item):
        try:
            return _transcribe_one(handle, item["audio_path"])
        except Exception as e:
            print(f"\n[voxtral-vllm] {item['audio_path']}: {e}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, items))
