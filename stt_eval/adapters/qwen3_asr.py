"""Qwen3-ASR via the official qwen-asr package (transformers backend)."""

import torch

from ..langs import lang_info


def load(model_id: str):
    from qwen_asr import Qwen3ASRModel

    return Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=32,
        max_new_tokens=256,
    )


def transcribe(model, items: list[dict]) -> list[str]:
    results = model.transcribe(
        audio=[i["audio_path"] for i in items],
        language=[lang_info(i["lang"])["qwen"] for i in items],
    )
    return [r.text for r in results]
