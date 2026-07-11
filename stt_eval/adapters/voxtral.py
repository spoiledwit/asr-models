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
    from mistral_common.tokens.tokenizers.audio import Audio

    processor, model = handle
    texts = []
    for item in items:
        audio = Audio.from_file(item["audio_path"], strict=False)
        audio.resample(processor.feature_extractor.sampling_rate)
        inputs = processor(audio.audio_array, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=model.dtype)
        outputs = model.generate(**inputs)
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        texts.append(decoded[0].strip())
    return texts
