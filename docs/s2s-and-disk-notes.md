# VoiceChat S2S import, and the disk-dedup reality check

Wed, 10 Sep 2026.

## S2S is imported and builds

Upstream `a5b6953` ("feat(s2s): NemotronLabs VoiceChat support") is vendored in
and builds on Windows.

- `NEMO_SPEECH_BUILD_S2S` defaults **OFF**, and upstream only wired it into
  `configure.sh` / CMakePresets. `scripts/windows/build.ps1` now has a `-S2S`
  switch that mirrors `-Flashlight` and forces NMT on with it (CMakeLists gates
  the llama backend on `WITH_NMT OR BUILD_S2S`).
- Built to `build-cuda-s2s` — 482 targets, exit 0.
- There is no separate subcommand. S2S rides on `serve`:
  `--s2s-model-dir DIR`, `--s2s-max-streams N`, `--s2s.* VALUE`, exposed at
  `/v1/realtime` and `/realtime`.

### Q4_K_M is the default, not an aspiration

`conversion/s2s.py` sets `DEFAULT_PROFILE = "q4_k_m"` and accepts
`bf16 | q4_k_m | nvfp4`. It also quantizes per component rather than uniformly:
`perception` and `eartts_side` stay `q8_0`, and the EarTTS backbone is held at
`q4_k_m` even under `nvfp4` because it is speech-critical and the extra ~82 MiB
is cheap. So the requested Q4_K_M path is the supported default.

### Custom tools are plain Python functions

`clients/voicechat/nemotron-voicechat-client.py` ships worked examples —
`_tool_convert_currency`, `_tool_get_news_headlines`, `_tool_calculate_bmi`,
`_tool_get_current_datetime`. On the engine side `src/s2s/pipeline.h` carries
`max_tool_tokens`, `fn_call_timeout_frames`, an `extracting_tool` state and
`tool_ack_messages` parsed from the system prompt; the model emits sanitized
tool-call JSON in `function_text`. Defining our own tools means writing the
functions and describing them in the system prompt.

### The blocker is the checkpoint, not the code

`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` ships a single
**`model.safetensors` of 44.4 GB** (fp32). The converter wants that HF
checkpoint directory — it looks for `perception.safetensors` or an HF/legacy
bundle — and then writes the quantized bundle.

Free space right now: **C: 52.9 GB, D: 127 GB, G: 200 GB.** So the download must
target D: or G:; C: cannot hold the source plus the ~7 GB output.

A third-party `hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF` exists with q4_0
and q8_0, but it is a llama.cpp-style single-file GGUF. This pipeline needs its
own multi-component bundle (perception, llm_backbone, eartts_backbone, codec),
so that repo is very unlikely to drop in. Not tested.

PersonaPlex would need the same treatment and does not use this checkpoint
layout at all, so it is a separate conversion problem rather than a slot-in.

## Disk dedup: the number was much smaller than it looked

**`du` is lying to you, and so was my first estimate.** `uv` already hardlinks
package files from its global cache into each venv, so most "duplicate" CUDA
DLLs were already sharing storage. A naive `size x (copies - 1)` calculation
counted that sharing as waste.

Measured properly, by unique `(device, inode)`:

| venv | apparent (`du`) | actually unique |
|---|---|---|
| vault-cacophony | 8.55 G | 7.75 G |
| vault-explorer | 4.98 G | 4.04 G |
| vault-streaming | 4.90 G | **0.27 G** |
| vault-inference | 4.59 G | 4.51 G |
| ColONEL-KFC | 4.88 G | **1.17 G** |
| vaultwares-studio | 2.75 G | 2.17 G |
| vaultwares-api | 0.68 G | 0.61 G |
| python-zipper | 0.76 G | 0.64 G |
| **total** | **32.09 G** | **21.16 G** |

10.93 GiB was already shared before this session. The dedup pass linked 71 files
with 0 failures and reclaimed roughly **2 GB** of genuinely duplicated data —
not the 13.98 GiB the naive arithmetic claimed. `torch_cuda.dll` now shows
`nlink=5` on one inode across vault-explorer, vault-streaming and ColONEL-KFC.
All five CUDA venvs still import torch with `cuda=True`.

The model files told the same story: `nemotron-3.5-asr-q8_0.gguf` looked like
three copies but was already one inode with `nlink=3`; likewise parakeet-tdt and
bs-roformer. Only the `htdemucs-f16.gguf` set (5 copies, ~84 MB each) and two
HTDemucs-ft HF snapshots are genuinely duplicated, and those failed to link
because the HF cache paths exceed `MAX_PATH`. Every failed attempt rolled back;
no orphaned `.dedupe-bak` files and all copies verified intact.

### Where the remaining space actually is

Torch versions are **not** all different:

| venv | torch |
|---|---|
| vault-explorer, vault-streaming, ColONEL-KFC | **2.13.0+cu126** (already shared) |
| vaultwares-studio | 2.13.0 |
| vault-cacophony | 2.8.0+cu129 |
| vault-inference | 2.5.1+cu121 |
| vaultwares-api | 2.12.1 (Python 3.13, not 3.12) |

Three already agree and cost 5.5 GB between them instead of 14.7 GB. The real
remaining duplication is the two outliers: **vault-cacophony at 2.8.0+cu129 and
vault-inference at 2.5.1+cu121, together about 12.3 GB unique.** Aligning those
two onto 2.13.0+cu126 would fold them into the shared set and is worth roughly
**10 GB** — far more than any further hardlinking.

That is a behaviour change to two working projects, so it needs a decision
rather than a script. `vaultwares-api` is on Python 3.13 and cannot join the
3.12 set without its own migration.

A single global VaultWares venv would save little beyond that alignment: uv's
cache is already doing most of the sharing.

---

## Follow-up: the two venvs are aligned

Wed, 10 Sep 2026. `vaultwares-api` left alone as instructed (Python 3.13, CPU
torch, no CUDA payload).

`vault-inference` turned out to be a latent bug rather than a preference: its
`pyproject.toml` had pinned `torch==2.13.0` for a while but sourced it from the
cu121 index, which has no 2.13.0 build, so the venv sat on **2.5.1+cu121 while
claiming 2.13.0**. Repointing the index to cu126 satisfies the pin that was
already declared.

`vault-cacophony` moved 2.8.0+cu129 to 2.13.0+cu126. That broke `torchaudio`,
which is compiled against a specific torch ABI — `libtorchaudio.pyd` refused to
load, and `silero_vad/utils_vad.py` imports torchaudio at module level, so the
VAD segmenter went down with it. **torchaudio does not track torch's version
number**: there is no torchaudio 2.13.0, and 2.11.0 is the build that pairs with
torch 2.13.0. `requirements-vad-segment.txt` now pins all three explicitly.

| venv | unique before | unique after |
|---|---|---|
| vault-cacophony | 7.75 G | 6.78 G |
| vault-explorer | 4.04 G | 1.24 G |
| vault-streaming | 0.27 G | 0.27 G |
| vault-inference | 4.51 G | **0.29 G** |
| ColONEL-KFC | 1.17 G | 1.00 G |
| vaultwares-studio | 2.17 G | 2.17 G |
| vaultwares-api | 0.61 G | 0.61 G |
| python-zipper | 0.64 G | 0.64 G |
| **total unique** | **21.16 G** | **13.00 G** |

**8.16 GiB of unique venv data eliminated**, and C: free space went from 52.9 GB
to 71.7 GB — the larger figure because dropping the two odd torch builds also
released their entries in uv's shared cache.

Verified after the change: all five CUDA venvs import torch with `cuda=True`;
vault-inference's transformers, accelerate, bitsandbytes 0.50.0, safetensors,
sentencepiece, fastapi, uvicorn and pydantic all import; and `vad_segment.py`
reproduces its previous output exactly on the French file (36 speech spans,
98.8% speech).

Rollback freezes for both venvs are in the session scratchpad under
`venv-rollback/`.

## VoiceChat is out of reach on this box

Not pursued further. Even at Q4 the pipeline needs two ~5 GB models resident
side by side plus KV cache, which this GPU cannot hold. The 44.4 GB fp32
download was never started. A Q4 PersonaPlex is the remaining idea and does not
use this checkpoint layout, so it would be its own conversion problem rather
than a drop-in.
