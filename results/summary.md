# STT Eval Summary

Values are WER% (CER% for zh/ja/ko). Lower is better. RTFx = audio-seconds per wall-second (higher is faster).

| Model | RTFx | ar/babble@10 | ar/babble@5 | ar/clean | en/babble@10 | en/babble@5 | en/clean | fr/babble@10 | fr/babble@5 | fr/clean |
|---|---|---|---|---|---|---|---|---|---|---|
| ark-asr-3b | 8.9 | — | — | — | 13.88 | 12.04 | 6.59 | — | — | — |
| canary-1b-v2 | 45.6 | — | — | — | 6.84 | 16.09 | 4.91 | 10.09 | 24.09 | 4.61 |
| parakeet-tdt-0.6b-v3 | 266.8 | — | — | — | 11.72 | 22.88 | 5.32 | 14.96 | 27.44 | 5.17 |
| qwen3-asr-0.6b | 31.8 | 30.01 | 44.25 | 24.48 | 6.91 | 14.16 | 4.91 | 14.28 | 30.29 | 7.71 |
| qwen3-asr-1.7b | 54.5 | 18.29 | 26.22 | 15.11 | 4.52 | 8.63 | 3.84 | 7.41 | 14.86 | 4.81 |
| voxtral-mini-realtime | 7.8 | 20.33 | 27.67 | 15.77 | 11.04 | 20.11 | 5.41 | 19.75 | 31.69 | 7.06 |
| whisper-large-v3-turbo | 36.0 | 19.21 | 28.43 | 16.1 | 6.09 | 10.61 | 4.89 | 10.91 | 20.98 | 5.28 |
| whisper-large-v3 | 10.5 | 17.81 | 24.89 | 15.52 | 5.41 | 10.79 | 4.54 | 8.54 | 18.42 | 5.04 |
