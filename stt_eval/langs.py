"""Language code mappings between our short codes and each framework's convention.

Our canonical code is ISO 639-1 (e.g. "en", "ar"). Add rows here to test more languages.
All 13 Voxtral Realtime languages are pre-filled; Canary/Parakeet only cover the
European subset (`nemo` is None where unsupported — the adapter will skip those).
"""

LANGS = {
    #  code:  (fleurs_config,     whisper, qwen_name,    nemo,  cer_based)
    "en": ("en_us",           "en", "English",    "en", False),
    "zh": ("cmn_hans_cn",     "zh", "Chinese",    None, True),
    "hi": ("hi_in",           "hi", "Hindi",      None, False),
    "es": ("es_419",          "es", "Spanish",    "es", False),
    "ar": ("ar_eg",           "ar", "Arabic",     None, False),
    "fr": ("fr_fr",           "fr", "French",     "fr", False),
    "pt": ("pt_br",           "pt", "Portuguese", "pt", False),
    "ru": ("ru_ru",           "ru", "Russian",    "ru", False),
    "de": ("de_de",           "de", "German",     "de", False),
    "ja": ("ja_jp",           "ja", "Japanese",   None, True),
    "ko": ("ko_kr",           "ko", "Korean",     None, True),
    "it": ("it_it",           "it", "Italian",    "it", False),
    "nl": ("nl_nl",           "nl", "Dutch",      "nl", False),
}

FIELDS = ("fleurs", "whisper", "qwen", "nemo", "cer_based")


def lang_info(code: str) -> dict:
    if code not in LANGS:
        raise ValueError(f"Unknown language '{code}'. Add it to stt_eval/langs.py LANGS.")
    return dict(zip(FIELDS, LANGS[code]))
