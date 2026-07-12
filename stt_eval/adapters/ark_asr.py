"""ARK-ASR-3B via transformers with remote code."""

import torch


def load(model_id: str):
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    return (processor, tokenizer, model)


def transcribe(handle, items: list[dict]) -> list[str]:
    processor, tokenizer, model = handle
    texts = []
    for item in items:
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "path": item["audio_path"]},
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        }]
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True,
            return_tensors="pt", sampling_rate=16000,
        ).to(model.device, torch.bfloat16)  # cast float features to model dtype
        outputs = model.generate(**inputs, max_new_tokens=256)
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        texts.append(tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip())
    return texts
