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
    chunk_samples = int(sr * 25.0)  # long clips come back empty in one pass

    def _generate(arr):
        # Trailing silence flushes the realtime model's delayed text output,
        # so the last words of the clip are actually emitted.
        arr = np.concatenate([arr, np.zeros(int(sr * 2.0), dtype=arr.dtype)])
        inputs = processor(arr, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=model.dtype)
        # Greedy decoding (Mistral recommends temperature 0.0) with an explicit
        # cap — the default max_length truncates transcripts.
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    texts = []
    for item in items:
        audio = Audio.from_file(item["audio_path"], strict=False)
        audio.resample(sr)
        arr = audio.audio_array
        parts = [_generate(arr[i : i + chunk_samples])
                 for i in range(0, len(arr), chunk_samples)]
        texts.append(" ".join(p for p in parts if p).strip())
    return texts
