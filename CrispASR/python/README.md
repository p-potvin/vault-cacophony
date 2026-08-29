# crispasr

Python bindings for [CrispASR](https://github.com/CrispStrobe/CrispASR) — lightweight on-device speech recognition via ggml.

Supports the ASR backends compiled into the linked CrispASR library, including Whisper, Qwen3-ASR, FastConformer, Canary, Parakeet, Cohere, Granite-Speech, Voxtral, wav2vec2, GLM-ASR, Kyutai-STT, Moonshine, FireRed, OmniASR, and VibeVoice-ASR.

## Install

```bash
pip install crispasr
```

Platform wheels **bundle the native `libcrispasr`** — nothing else to install —
for Linux (x86_64, arm64), macOS (Apple Silicon, Metal-accelerated), and
Windows (x86_64).

### GPU wheels

CUDA and Vulkan builds are published to a separate index (llama-cpp-python
style — pass `--extra-index-url`):

```bash
# NVIDIA CUDA (Linux + Windows)
pip install crispasr --extra-index-url https://crispstrobe.github.io/CrispASR/whl/cuda/
# Vulkan (Windows)
pip install crispasr --extra-index-url https://crispstrobe.github.io/CrispASR/whl/vulkan/
```

### Other platforms / bring-your-own library

Where no prebuilt wheel matches, pip installs the pure-Python **sdist**, which
loads a `libcrispasr` you supply. Build/install it from source and, if it lands
in a non-standard location, point `CRISPASR_LIB_PATH` at the file:

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR && cmake -B build && cmake --build build -j && sudo cmake --install build
export CRISPASR_LIB_PATH=/usr/local/lib/libcrispasr.so   # only if non-standard
```

## Quick start

```python
from crispasr import CrispASR

model = CrispASR("ggml-base.en.bin")
for seg in model.transcribe("audio.wav"):
    print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
model.close()
```

Or use the unified `Session` API for non-Whisper backends (Qwen3-ASR, FastConformer, Parakeet, …):

```python
from crispasr import Session

s = Session("qwen3-asr-0.6b-q4_k.gguf")
for seg in s.transcribe_pcm(pcm_f32, sample_rate=16000):
    print(seg.text)
```

## API

- `CrispASR` — Whisper-compatible high-level API
- `Session` — unified API across all backends compiled into `libcrispasr`
- `align_words(...)` — word-level CTC alignment
- `diarize_segments(...)` — speaker diarization (energy / xcorr / vad-turns / pyannote)
- `SpeakerEmbedder(spec)` — pluggable embedder ("auto"/"titanet", "indextts"/"ecapa", or a `.gguf` path)
- `PyannoteCache(pcm, model)` — pre-computed pyannote-seg posteriors for cross-slice consistency
- `agglomerative_cluster(embeddings, ...)` — single-linkage cosine clustering for globally stable speaker IDs
- `TitaNet` / `SpeakerDB` — standalone speaker verification + closed-roster profile matching (consent-gated; requires `expected_names` + `consent=True`, see docs/diarization-speakers.md)
- `detect_language_pcm(...)` — language ID
- `registry_lookup(...)` — auto-download known models from the model hub
- `registry_default_bundle(...)` — enumerate the exact primary, companion, and extra files used by `-m auto`, including licence-acceptance policy

See the [main repo](https://github.com/CrispStrobe/CrispASR) for full documentation, model registry, and CLI.

## License

MIT — see [LICENSE](LICENSE).
