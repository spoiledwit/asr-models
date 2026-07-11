"""Model adapter registry.

Every adapter exposes:
    load(model_id: str) -> handle
    transcribe(handle, items: list[dict]) -> list[str]
where each item is a manifest row ({"audio_path", "lang", ...}).

Adapters are imported lazily so each venv only needs its own stack's deps.
"""

MODELS = {
    # name:                 (adapter module,  model id,                                stack venv)
    "whisper-large-v3":       ("whisper_fw", "large-v3",                               "whisper"),
    "whisper-large-v3-turbo": ("whisper_fw", "large-v3-turbo",                         "whisper"),
    "qwen3-asr-0.6b":         ("qwen3_asr",  "Qwen/Qwen3-ASR-0.6B",                    "qwen"),
    "qwen3-asr-1.7b":         ("qwen3_asr",  "Qwen/Qwen3-ASR-1.7B",                    "qwen"),
    "voxtral-mini-realtime":  ("voxtral",    "mistralai/Voxtral-Mini-4B-Realtime-2602", "voxtral"),
    "canary-1b-v2":           ("nemo_asr",   "nvidia/canary-1b-v2",                    "nemo"),
    "parakeet-tdt-0.6b-v3":   ("nemo_asr",   "nvidia/parakeet-tdt-0.6b-v3",            "nemo"),
    "ark-asr-3b":             ("ark_asr",    "AutoArk-AI/ARK-ASR-3B",                  "voxtral"),
}


def get_adapter(name: str):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Options: {', '.join(MODELS)}")
    module_name, model_id, stack = MODELS[name]
    import importlib

    module = importlib.import_module(f"stt_eval.adapters.{module_name}")
    return module, model_id
