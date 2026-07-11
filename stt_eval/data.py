"""Dataset preparation: FLEURS subsets + synthetic noise conditions.

Produces 16 kHz mono WAVs under data/<condition>/<lang>/ and one manifest at
data/manifest.jsonl with rows:
  {"id", "audio_path", "text", "lang", "condition", "duration"}

Noise conditions need no external corpus:
  - babble@SNR: sum of 6 other utterances from the same language pool (cafe-like)
  - white@SNR:  gaussian noise
Optionally point --musan-dir at an unpacked MUSAN noise/ folder for real noise.
"""

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def _to_mono16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def _fit_length(noise: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    if len(noise) < n:
        reps = n // len(noise) + 1
        noise = np.tile(noise, reps)
    start = rng.randint(0, len(noise) - n)
    return noise[start : start + n]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = noise * (_rms(speech) / _rms(noise)) / (10 ** (snr_db / 20))
    mixed = speech + noise
    peak = np.abs(mixed).max()
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed


def make_noise(kind: str, n: int, pool: list[np.ndarray], rng: random.Random,
               musan_files: list[Path]) -> np.ndarray:
    if kind == "white":
        return np.random.default_rng(rng.randint(0, 2**31)).standard_normal(n).astype(np.float32)
    if kind == "babble":
        voices = [_fit_length(pool[rng.randrange(len(pool))], n, rng) for _ in range(6)]
        return np.sum(voices, axis=0)
    if kind == "musan":
        if not musan_files:
            raise ValueError("musan condition requested but --musan-dir not given/empty")
        audio, sr = sf.read(musan_files[rng.randrange(len(musan_files))])
        return _fit_length(_to_mono16k(audio, sr), n, rng)
    raise ValueError(f"Unknown noise kind '{kind}'")


def prepare(langs: list[str], n_per_lang: int, conditions: list[str],
            out_dir: Path, musan_dir: Path | None, seed: int = 17) -> Path:
    from datasets import load_dataset

    from .langs import lang_info

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    musan_files = sorted(musan_dir.rglob("*.wav")) if musan_dir else []

    rows = []
    for lang in langs:
        info = lang_info(lang)
        print(f"[data] loading FLEURS {info['fleurs']} test split ...")
        ds = load_dataset("google/fleurs", info["fleurs"], split="test", trust_remote_code=True)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:n_per_lang]

        clips = []
        for i in idxs:
            ex = ds[i]
            audio = _to_mono16k(ex["audio"]["array"], ex["audio"]["sampling_rate"])
            clips.append((f"{lang}_{i}", audio, ex["transcription"]))
        pool = [c[1] for c in clips]

        for cond in conditions:
            cond_dir = out_dir / cond.replace("@", "_snr") / lang
            cond_dir.mkdir(parents=True, exist_ok=True)
            for clip_id, audio, text in clips:
                if cond == "clean":
                    wav = audio
                else:
                    kind, snr = cond.split("@")
                    noise = make_noise(kind, len(audio), pool, rng, musan_files)
                    wav = mix_at_snr(audio, noise, float(snr))
                path = cond_dir / f"{clip_id}.wav"
                sf.write(path, wav, SR)
                rows.append({
                    "id": clip_id,
                    "audio_path": str(path),
                    "text": text,
                    "lang": lang,
                    "condition": cond,
                    "duration": round(len(audio) / SR, 3),
                })
            print(f"[data] {lang} / {cond}: {len(clips)} clips")

    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"[data] wrote {len(rows)} utterances ({hours:.2f} h) -> {manifest_path}")
    return manifest_path
