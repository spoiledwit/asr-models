"""Voxtral Mini 4B Realtime via transformers (offline/batch WER mode).

Streaming latency is measured separately by serving with vLLM
(`vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602`, /v1/realtime endpoint)
— see scripts/run_latency.py. This adapter runs the model offline so WER is
comparable with the other models.
"""

import torch


def load(model_id: str):
    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    return (processor, model)


def transcribe(handle, items: list[dict]) -> list[str]:
    import numpy as np
    from mistral_common.tokens.tokenizers.audio import Audio

    processor, model = handle
    sr = processor.feature_extractor.sampling_rate
    texts = []
    for item in items:
        audio = Audio.from_file(item["audio_path"], strict=False)
        audio.resample(sr)
        # Trailing silence flushes the realtime model's delayed text output,
        # so the last words of the clip are actually emitted.
        arr = np.concatenate([audio.audio_array,
                              np.zeros(int(sr * 2.0), dtype=audio.audio_array.dtype)])
        inputs = processor(arr, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=model.dtype)
        # Greedy decoding (Mistral recommends temperature 0.0) — the default
        # sampling config can emit an immediate EOS -> empty transcript.
        # Explicit max_new_tokens: the default max_length truncates transcripts.
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        texts.append(decoded[0].strip())
    return texts
