# Speech-to-Text Model Shortlist (Multilingual · Latency vs Quality)

Curated list of top-tier STT/ASR models to benchmark on a RunPod GPU pod, later compared on a common dataset.
Compiled 2026-07-11 from the Hugging Face Open ASR Leaderboard and recent releases.

---

## Tier 1 — Open-weight, multilingual, must-test

| Model | Params | Languages | License | Strength | Serving | VRAM (fp16) |
|---|---|---|---|---|---|---|
| [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) (Mistral, Feb 2026) | 4B | 13 major langs (en, zh, hi, es, ar, fr, pt, ru, de, ja, ko, it, nl) | Apache 2.0 | Best accuracy of open models (~5.9% avg WER on FLEURS langs) **and** native streaming, configurable 80 ms–2.4 s latency (480 ms sweet spot ≈ offline quality) | vLLM (official recipe) | ~16 GB |
| [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 0.6B / 1.7B (Alibaba, Jan 2026) | 0.6B / 1.7B | 52 langs & dialects, auto language ID | Apache 2.0 | One model for both streaming and offline; very robust to noise/music/singing; huge throughput (0.6B: ~2000× RTF at concurrency 128); companion ForcedAligner for word timestamps | vLLM / transformers | ~4–8 GB |
| [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) + [large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (OpenAI) | 1.55B / 0.8B | 99 langs — widest coverage of the quality models | MIT | The multilingual baseline everything is measured against (~7.4% avg WER mixed benchmarks); turbo is ~6× faster with minor quality loss | faster-whisper (CTranslate2), transformers, WhisperX (timestamps/diarization) | ~10 GB / ~6 GB |
| [Canary-1B-v2](https://huggingface.co/nvidia/canary-1b-v2) (NVIDIA) | 1B | 25 European langs, ASR + en↔X translation | CC-BY-4.0 | Matches models 3× its size at ~10× less compute; trained with anti-hallucination data | NeMo | ~6 GB |
| [Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (NVIDIA) | 0.6B | 25 European langs, auto language detect | CC-BY-4.0 | Speed king — RTFx in the thousands; the latency/throughput reference point | NeMo | ~4 GB |
| [ARK-ASR-3B](https://huggingface.co/AutoArk-AI/ARK-ASR-3B) (AutoArk) | 3B | 19 langs (en, zh, de, ja, fr, ko, es, pl, it + Central/Northern European) | open weights (check card) | #1 on Open ASR Leaderboard English short-form: 5.04% avg WER, RTFx ~491; Whisper encoder + Qwen decoder | transformers (remote code) | ~8 GB |

## Tier 2 — Worth including if scope allows

| Model | Params | Languages | Notes |
|---|---|---|---|
| [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) (OpenMOSS) | 0.9B | zh/en focus | Joint transcription + speaker diarization in one pass; best pick if the eval set is long-form multi-speaker (meetings, calls, podcasts) |
| [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) (NVIDIA) | 2.5B | **English only** | ~5.6% avg WER, long-time leaderboard leader; useful as an English quality ceiling, not for the multilingual comparison |
| [Granite Speech 3.3 8B](https://huggingface.co/ibm-granite/granite-speech-3.3-8b) (IBM) | 8B | en + a few (fr, de, es, pt) | ~5.85% WER English; heavy for its coverage — include only if quality-at-any-cost matters |
| [Kyutai STT](https://kyutai.org/stt/) (stt-1b-en_fr) | 1B | **en + fr only** | True streaming with 0.5 s delay and semantic VAD; batches hundreds of concurrent streams on one GPU — great voice-agent architecture reference, weak language coverage |
| [Moonshine](https://huggingface.co/UsefulSensors) Tiny/Base | 27M–60M | en (+ small multilingual variants) | Edge/CPU class; only relevant if an on-device tier is part of the comparison |
| [Voxtral Small 24B](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) | 24B | multilingual + audio understanding | Audio-LLM (Q&A/summarization over audio, not just ASR); needs ~48 GB+; skip for a pure ASR benchmark |

## API baselines (closed, for reference numbers only)

- **ElevenLabs Scribe** — tops several third-party multilingual accuracy comparisons; batch only.
- **OpenAI gpt-4o-transcribe / gpt-4o-mini-transcribe** — strong multilingual quality, realtime API available.
- **Deepgram Nova-3** — the latency/streaming benchmark among APIs (~sub-300 ms).
- **[Cohere Transcribe](https://cohere.com/blog/transcribe)** (Mar 2026) — near top of leaderboard (5.42% WER), closed.
- **Mistral Voxtral API** — $0.003/min batch, $0.006/min realtime; sanity check vs the self-hosted weights.

---

## Testing plan notes (RunPod)

**GPU sizing:** everything in Tier 1 fits on a single 24 GB card (RTX 4090 / L4 / A5000). An A100/H100 80GB pod only pays off for batched throughput runs or Voxtral Small 24B. Suggest one pod with a 24 GB GPU for the full sweep.

**Serving stacks to install:** `faster-whisper` (Whisper family), `vllm` (Voxtral Realtime, Qwen3-ASR, ARK-ASR), `nemo_toolkit[asr]` (Canary, Parakeet). Three docker images or one fat image with separate venvs.

**Metrics to record per model:**
- WER/CER per language (normalize text consistently — use the Whisper/`evaluate` normalizer for all models, this is the #1 source of unfair comparisons)
- RTFx (audio seconds processed per wall-clock second, batch mode)
- First-token / partial-result latency for the streaming models (Voxtral Realtime, Qwen3-ASR streaming, Kyutai)
- VRAM peak + throughput at batch 1 / 8 / 32

**Candidate eval datasets (multilingual):**
- **FLEURS** — 100+ langs, short-form, the standard multilingual WER benchmark
- **Common Voice 17+** — broad language coverage, noisier/user-recorded
- **Multilingual LibriSpeech (MLS)** — 8 European langs, clean long-form read speech
- Open ASR Leaderboard multilingual track currently covers 5 languages — pick the intersection of languages supported by *all* Tier 1 models (roughly: en, de, es, fr, it, pt, nl + zh/ja/ko/hi/ar/ru for the non-NVIDIA ones) so every model competes on every sample.

**Recommended first pass:** Whisper large-v3-turbo (baseline), Voxtral Mini 4B Realtime, Qwen3-ASR 1.7B, Canary-1B-v2, Parakeet-TDT-0.6B-v3, ARK-ASR-3B — 6 models, one 24 GB pod, FLEURS subset.

---

## Sources

- [HF Open ASR Leaderboard — trends, multilingual & long-form tracks](https://huggingface.co/blog/open-asr-leaderboard) · [leaderboard space](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) · [repo](https://github.com/huggingface/open_asr_leaderboard)
- [Mistral — Voxtral Transcribe 2 announcement](https://mistral.ai/news/voxtral-transcribe-2/) · [Voxtral-Mini-4B-Realtime model card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) · [vLLM recipe](https://recipes.vllm.ai/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR) · [technical report](https://arxiv.org/html/2601.21337v2)
- [Canary-1B-v2 & Parakeet-TDT-0.6B-v3 paper](https://arxiv.org/abs/2509.14128) · [NVIDIA Granary/models blog](https://blogs.nvidia.com/blog/speech-ai-dataset-models/)
- [ARK-ASR-3B model card](https://huggingface.co/AutoArk-AI/ARK-ASR-3B) · [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) ([paper](https://arxiv.org/pdf/2601.01554))
- [Kyutai STT](https://kyutai.org/stt/) · [delayed-streams-modeling repo](https://github.com/kyutai-labs/delayed-streams-modeling)
- Comparison articles: [Northflank 2026 STT benchmarks](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks) · [Gladia open-source STT roundup](https://www.gladia.io/blog/best-open-source-speech-to-text-models) · [Voxtral vs Whisper 2026](https://weesperneonflow.ai/en/blog/2026-03-31-voxtral-whisper-open-source-speech-models-comparison-2026/)
