"""NVIDIA Canary / Parakeet via NeMo.

Canary needs source/target language; Parakeet auto-detects. Languages not in
the NeMo 25-language set are skipped upstream (see run_eval.py lang filter).
"""

from ..langs import lang_info


def load(model_id: str):
    from nemo.collections.asr.models import ASRModel

    model = ASRModel.from_pretrained(model_id)
    model.eval()
    return model


def _text(hyp) -> str:
    return hyp.text if hasattr(hyp, "text") else str(hyp)


def transcribe(model, items: list[dict]) -> list[str]:
    paths = [i["audio_path"] for i in items]
    is_canary = "canary" in model.__class__.__name__.lower() or hasattr(model, "prompt_format")
    if is_canary:
        # Canary requires one language per call — group items by language.
        texts = [None] * len(items)
        by_lang: dict[str, list[int]] = {}
        for idx, item in enumerate(items):
            by_lang.setdefault(item["lang"], []).append(idx)
        for lang, idxs in by_lang.items():
            code = lang_info(lang)["nemo"]
            hyps = model.transcribe(
                [paths[i] for i in idxs],
                source_lang=code, target_lang=code, batch_size=16,
            )
            for i, hyp in zip(idxs, hyps):
                texts[i] = _text(hyp)
        return texts
    hyps = model.transcribe(paths, batch_size=16)
    return [_text(h) for h in hyps]
