"""Text normalization + WER/CER.

One normalizer for every model — inconsistent normalization is the #1 source of
unfair ASR comparisons. We use Whisper's BasicTextNormalizer (lowercase, strip
punctuation/diacritics-aware) for all languages, and CER instead of WER for
space-free scripts (zh/ja/ko).
"""

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import jiwer

_normalizer = BasicTextNormalizer()


def normalize(text: str) -> str:
    return _normalizer(text or "").strip()


def error_rate(refs: list[str], hyps: list[str], cer_based: bool) -> float:
    refs = [normalize(r) for r in refs]
    hyps = [normalize(h) for h in hyps]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]  # drop empty refs
    if not pairs:
        return float("nan")
    refs, hyps = zip(*pairs)
    if cer_based:
        return jiwer.cer(list(refs), list(hyps))
    return jiwer.wer(list(refs), list(hyps))
