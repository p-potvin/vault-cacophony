---
license: apache-2.0
language:
- ar
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- gguf
- conformer
- arabic
- crispasr
base_model: CohereLabs/cohere-transcribe-arabic-07-2026
---

# cohere-transcribe-arabic-07-2026 — GGUF

GGUF weights for **[CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)** — Cohere Labs' 2B-parameter Arabic ASR model (a FastConformer encoder + Transformer decoder), released July 2026 (~11% WER on FLEURS Arabic per the source card).

These GGUFs run on CPU/Metal/CUDA/Vulkan via **[CrispASR](https://github.com/CrispStrobe/CrispASR)** — a C++ runtime for the Cohere Conformer-encoder / Transformer-decoder architecture.

> **License**: Apache 2.0 (inherited from source model). See the [original model card](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) for full terms.

---

## Files

| File | Size | Type |
|------|------|------|
| `cohere-transcribe-arabic-f16.gguf` | 4.1 GB | F16 (reference precision) |
| `cohere-transcribe-arabic-q8_0.gguf` | 2.4 GB | Q8_0 |
| `cohere-transcribe-arabic-q4_k.gguf` | 1.5 GB | Q4_K |
| `cohere-transcribe-arabic-q4_k-imatrix.gguf` | 1.5 GB | Q4_K + importance matrix (Arabic-calibrated) |
| `cohere-transcribe-arabic-ref.gguf` | small | per-stage reference activations for `crispasr-diff` |

All quants keep LayerNorm weights / biases at F32. The `-imatrix` build is
calibrated on CC0 Common Voice Arabic (`fsicoli/common_voice_17_0`, `ar/dev`) and
recovers quality at the 4-bit budget; prefer it over plain `q4_k`.

On an M1 (Metal), end-to-end encode+decode on an 11 s clip is on par with the
`transcribe.cpp` reference runtime (~1.5 s, load excluded).

---

## Quick start

```bash
# Build CrispASR (see the repo for full instructions)
git clone --recursive https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Shorthand: auto-downloads the recommended imatrix GGUF and defaults
# the language to Arabic — no -m / --hf-repo / -l needed.
build/bin/crispasr --backend cohere-ar audio.wav
```

`cohere-ar` is a CLI alias for the `cohere` backend: it routes to the same
runtime, resolves `cohere-transcribe-arabic-q4_k-imatrix.gguf` via `-m auto`,
and sets `-l ar` unless you pass an explicit `-l` (which always wins — useful
if you want to run this model with the LID pre-step or force another
language for testing). It's equivalent to:

```bash
build/bin/crispasr --backend cohere \
    --hf-repo cstr/cohere-transcribe-arabic-07-2026-GGUF:cohere-transcribe-arabic-q4_k-imatrix.gguf \
    audio.wav -l ar
```

Or point `-m` at a locally downloaded GGUF:

```bash
build/bin/crispasr --backend cohere -m cohere-transcribe-arabic-q4_k.gguf audio.wav -l ar
```

---

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | 48-layer FastConformer, d=1280, heads=8, head_dim=160, ffn=5120, conv_kernel=9 |
| **Decoder** | 8-layer causal Transformer, d=1024, heads=8, head_dim=128, ffn=4096 (ReLU), max_ctx=1024 |
| **Vocab** | 16,384 SentencePiece tokens |
| **Audio** | 16 kHz mono, 128 mel bins, n_fft=512, hop=160, win=400, 8× time subsampling |
| **Parameters** | ~2B |

Prompt tokens (decoder): the runtime prepends `decoder_start_token_id` (`▁`,
13764) to the control-token prompt, matching the reference `decoder_input_ids`.
Mel frame count is `floor(n/hop)+1` (NeMo FilterbankFeatures), giving `T_enc =
calc_length(T_mel)`; both are required for the cross-attention context to line up.

---

## Validation (`crispasr-diff`)

`cohere-transcribe-arabic-ref.gguf` holds per-stage reference activations dumped
from the transformers model by `tools/dump_reference.py`. Reproduce the
per-layer parity check with:

```bash
build/bin/crispasr-diff cohere cohere-transcribe-arabic-f16.gguf \
    cohere-transcribe-arabic-ref.gguf audio.wav
```

---

## Related

- **Source model**: [CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
- **English sibling**: [cstr/cohere-transcribe-03-2026-GGUF](https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF) — Cohere Transcribe 2B (lowest English WER)
- **C++ runtime**: [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)
