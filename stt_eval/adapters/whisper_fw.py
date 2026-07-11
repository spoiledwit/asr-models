"""Whisper via faster-whisper (CTranslate2)."""

from ..langs import lang_info


def load(model_id: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_id, device="cuda", compute_type="float16")


def transcribe(model, items: list[dict]) -> list[str]:
    texts = []
    for item in items:
        segments, _ = model.transcribe(
            item["audio_path"],
            language=lang_info(item["lang"])["whisper"],
            beam_size=5,
            condition_on_previous_text=False,
        )
        texts.append(" ".join(s.text for s in segments).strip())
    return texts
