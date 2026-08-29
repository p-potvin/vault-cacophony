# Audio Flamingo 3 on ggml

Tue, 25 Aug 2026

NVIDIA's Audio Flamingo 3 running quantized on the RTX 3060, through llama.cpp
rather than PyTorch. This is the record of how it was built, what it costs, and
the one limitation that will bite anyone who uses it for transcription.

---

## The thing worth knowing first

**llama.cpp already supports AF3.** This was not obvious and it inverts the
plan. The session started from the assumption that running AF3 outside PyTorch
meant writing a ggml implementation from scratch and would be a first. It is
not. Upstream has it:

| where | what |
|---|---|
| `conversion/qwen.py:53` | `AudioFlamingo3ForConditionalGeneration` registered as a Qwen2 text model |
| `conversion/ultravox.py:199` | the audio tower, with `@ModelBase.example("nvidia/audio-flamingo-3-hf")` |
| `tools/mtmd/clip.cpp` | `PROJECTOR_TYPE_MUSIC_FLAMINGO`, handled at six sites |
| `tools/mtmd/models/whisper-enc.cpp` | the encoder graph, shared with Ultravox and Voxtral |

The reason it was already there is structural. AF3's audio tower is *exactly*
whisper-large-v3 encoder geometry — 1280 hidden, 32 layers, 20 heads, 128 mel
bins, 1500 source positions — so it drops straight into the encoder llama.cpp
already carried for the Ultravox family. AF3 is a Whisper encoder, a two-layer
projector, and Qwen2.5-7B. Nothing about it is novel to ggml.

What is still novel: **no AF3 GGUF is published on the Hub.** `music-flamingo`
(the `audio-flamingo-next` sibling) has been quantized by others, complete with
mmproj files — which is what proves the runtime path works — but AF3 itself has
not been. The conversion below appears to be the first.

## Architecture, from the weight map

830 tensors, three top-level modules:

| module | tensors | what |
|---|---|---|
| `audio_tower` | 487 | whisper-large-v3 encoder, 635M params |
| `multi_modal_projector` | 4 | two linear layers, 1280 -> 3584 |
| `language_model` | 339 | Qwen2.5-7B, vocab 151672 |

## Build

Everything lives on `D:` because the working set is ~37 GB.

```bash
# 1. weights. NOTE the allow_patterns: the repo ships BOTH a single 16.5 GB
#    model.safetensors AND a 16.5 GB sharded copy of the same weights, so a
#    naive snapshot_download pulls 34.5 GB and leaves the converter with two
#    copies of every tensor to glob over.
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/audio-flamingo-3-hf',
    local_dir='D:/HuggingFace/models/audio-flamingo-3-hf',
    allow_patterns=['*.json','*.jinja','merges.txt','tokenizer*','model-*-of-*.safetensors'],
    ignore_patterns=['think/*','static/*'])"
```

```bash
# 2. converter + prebuilt CUDA 13.3 binaries
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git D:/HuggingFace/llama.cpp
# llama-b10630-bin-win-cuda-13.3-x64.zip + cudart-llama-bin-win-cuda-13.3-x64.zip
```

```bash
# 3. two conversions: the audio tower, then the text model
python convert_hf_to_gguf.py D:/HuggingFace/models/audio-flamingo-3-hf \
  --mmproj --outtype f16 --outfile D:/HuggingFace/gguf/mmproj-af3-f16.gguf

python convert_hf_to_gguf.py D:/HuggingFace/models/audio-flamingo-3-hf \
  --outtype f16 --outfile D:/HuggingFace/gguf/af3-f16.gguf
```

```bash
# 4. quantize the text model only. The encoder stays f16 -- it is 1.3 GB and
#    quantizing a Whisper encoder is where audio quality goes to die.
llama-quantize D:/HuggingFace/gguf/af3-f16.gguf D:/HuggingFace/gguf/af3-Q4_K_M.gguf Q4_K_M 8
```

Artifacts:

| file | size |
|---|---|
| `af3-Q4_K_M.gguf` | 4.68 GB |
| `mmproj-af3-f16.gguf` | 1.32 GB |
| `af3-f16.gguf` (intermediate) | 15.2 GB |

## Cost

Measured on the RTX 3060 (12,288 MiB, ~1,900 MiB held by the desktop):

| | |
|---|---|
| resident, `-c 8192` | ~6.4 GB |
| resident, default `-c 32768` | ~7.7 GB (the extra 1.9 GB is all KV cache) |
| model buffer on CUDA0 | 4,167 MiB |
| prompt eval | 1,239 tok/s |
| generation | 61 tok/s |
| audio encode | ~210 ms per 30 s window |
| quantize time | 173 s |

Pass `-c 8192`. AF3 caps audio at ten minutes and the default 32k context buys
nothing but KV cache.

## The thirty-second rule

**AF3 transcribes reliably only inside a single 30 s window.** Past that, ASR
fails — and it fails *quietly*, returning fluent text instead of an error.

| clip | windows | result |
|---|---|---|
| 29.0 s | 1 | correct |
| 29.9 s | 1 | correct |
| 29.99 s | 2 | `"the"` |
| 30.0 s | 2 | `"the"` |
| 60.0 s | 3 | `"to get the money to get the money to get the money..."` |

Two details make this worse than it looks:

- **The split fires just under 30 s, not at it.** 29.9 s is one window; 29.99 s
  is two. A chunker written to the obvious 30 s boundary lands exactly on the
  broken case. `scripts/audio_flamingo.py` uses 29.0 s for margin.
- **It is not a quantization artifact.** The unquantized f16 fails the same 60 s
  clip, hallucinating *"they're going to be able to get the money to get the
  equipment they need to get the job done"* against speech that says nothing of
  the kind. This is the mtmd audio path, which prints an experimental-stage
  warning on every run.

**Captioning and understanding are unaffected.** A 90 s clip across four windows
is described correctly and identifies genre and instrumentation. Only
transcription needs chunking, which is why only `--task asr` chunks.

## Gotchas

**`hf download --exclude "a" "b" "c"` silently downloads the wrong thing.**
argparse binds only the first pattern to `--exclude` and treats the rest as
positional filenames, so the command fetches *exactly the directories you meant
to exclude* and none of the weights. It exits 0. Use `snapshot_download` with
explicit `allow_patterns`.

**The chat template is a separate file and it is easy to miss.** AF3-hf carries
`chat_template.jinja`, not a `chat_template` key in `tokenizer_config.json`
(that key is absent). Any `allow_patterns` list built from `*.json` misses it,
and the resulting GGUF has no template. llama.cpp then **hard-errors** with
`Model does not have chat template` rather than guessing — which is the good
outcome, since a wrong guess would have been a silent quality loss. Fix without
re-converting:

```bash
python gguf-py/gguf/scripts/gguf_new_metadata.py in.gguf out.gguf \
  --chat-template-file D:/HuggingFace/models/audio-flamingo-3-hf/chat_template.jinja
```

The template is plain ChatML with `<sound>` substituted for audio content, so
`--chat-template chatml` is an equivalent runtime workaround. Note that
`llama-mtmd-cli` accepts `--chat-template` but **not** `--chat-template-file`.

**CUDA 13 DLLs.** The `cudart-llama-bin-win-cuda-13.3-x64.zip` release drops
`cublas64_13.dll` and friends next to the binaries, which sidesteps the
`bin\x64`-not-on-PATH problem documented in `HANDOFF.md` entirely. No PATH edit
needed if you extract both zips into the same folder.

**`llama-mtmd-cli` splits its streams.** Every log line goes to stderr and only
the model's answer to stdout. Merging them with `2>&1` drags the startup banner
and the chat-template example into what looks like model output.

## Usage

```bash
# transcription, chunked automatically
python scripts/audio_flamingo.py --audio clip.wav --task asr --timestamps

# understanding, whole file
python scripts/audio_flamingo.py --audio song.wav --task ask \
  --prompt "What genre is this and what instruments do you hear?"
```

Verified against `samples/sample_suitcase.wav` — word-perfect — and
`samples/ItJustDoesntMatter16k.wav`, which returns `"the"` as a single call and
transcribes correctly through the chunker.

## Licence

AF3 is released under the **NVIDIA OneWay Noncommercial License** — research use
only. Whether that permits redistributing quantized derivatives is an open
question and has not been checked; do not publish these GGUFs without reading
`static/NVIDIA_OneWay_Noncommercial_License.docx` in the model repo first.
