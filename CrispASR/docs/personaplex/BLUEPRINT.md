# PersonaPlex — blueprint

Reference source read for HARD RULE #1 before any C++. Everything below is
taken from the Python and from the checkpoint header, not from a handover.

**Model.** `nvidia/personaplex-7b-v1` — a full-duplex speech-to-speech
conversational model on the Moshi architecture, with persona control via a
text role prompt and voice conditioning via an audio prompt.

**Provenance and licence.** The Python is Kyutai's Moshi adapted by NVIDIA;
`LICENSE.moshi` is MIT (Copyright Kyutai) and the repo code carries NVIDIA's
MIT header. CrispASR is MIT. All compatible. The **weights** are under the
NVIDIA Open Model License — that constrains redistributing a converted GGUF,
not building or running one.

**Reference checkout.** `D:\moshi-docker\personaplex` (weights cached under
`.cache/huggingface/hub/models--nvidia--personaplex-7b-v1`).

---

## 1. Hyperparameters

From `moshi/models/loaders.py` `_lm_kwargs`, with `dep_q` overridden 8 → 16
by `get_moshi_lm` at line 187.

| | temporal transformer | depformer |
|---|---|---|
| `dim` | 4096 | 1024 |
| layers | 32 | 6 |
| heads | 32 | 16 |
| `dim_feedforward` | `int(4.125 * 4096)` = 16896 | `int(4.125 * 1024)` = 4224 |
| **SwiGLU hidden** | **11264** | **2816** |
| norm | `rms_norm_f32` | `rms_norm_f32` |
| gating | silu | silu |
| positional | rope, θ=10000 | **none** |
| context | 3000 | 8 |
| causal | yes | yes |
| `weights_per_step` | — | 16 |

Streams: `n_q = 16`, `dep_q = 16`, `card = 2048`, `text_card = 32000`,
`existing_text_padding_id = 3`, `layer_scale = None`, `bias_proj` false —
**there are no biases anywhere in the LM**.

Audio: Mimi at 24 kHz / 12.5 Hz → **frame_size = 1920 samples**.

### The delay pattern

```
delays = [0,  0,1,1,1,1,1,1,1,  0,1,1,1,1,1,1,1]
          ^   \--- own audio ---/  \-- other audio --/
        text        8 codebooks       8 codebooks
```

17 streams = 1 text + 8 own + 8 other. This is the full-duplex mechanism:
the model predicts its own audio *and* the user's audio *and* an inner text
monologue at every frame. `offline.py` decodes `tokens[:, 1:9]` — channels
1..8, the agent's own audio.

The delay logic is **not baked into the weights**. `_delay_sequence` /
`_undelay_sequence` are plain tensor rolls at `models/lm.py:70-97`, and
`LMGen` keeps `max_delay` / `delays_cuda` at `lm.py:690-693`. It ports
directly.

In-tree prior art: the mini-omni2 layershift deinterleave (commit
`1a9e7b0c`) is the same class of transform.

---

## 2. Tensor map

475 tensors, all BF16, 8.37 B parameters, 15.59 GB. Dumped from the
safetensors header — this is ground truth, not inferred from the modules.

| tensor | count | shape |
|---|---|---|
| `transformer.layers.{0..31}.self_attn.in_proj_weight` | 32 | `[12288, 4096]` |
| `transformer.layers.{0..31}.self_attn.out_proj.weight` | 32 | `[4096, 4096]` |
| `transformer.layers.{0..31}.gating.linear_in.weight` | 32 | `[22528, 4096]` |
| `transformer.layers.{0..31}.gating.linear_out.weight` | 32 | `[4096, 11264]` |
| `transformer.layers.{0..31}.norm1.alpha` | 32 | `[1, 1, 4096]` |
| `transformer.layers.{0..31}.norm2.alpha` | 32 | `[1, 1, 4096]` |
| `out_norm.alpha` | 1 | `[1, 1, 4096]` |
| `emb.{0..15}.weight` | 16 | `[2049, 4096]` |
| `text_emb.weight` | 1 | `[32001, 4096]` |
| `text_linear.weight` | 1 | `[32000, 4096]` |
| `depformer_in.{0..15}.weight` | 16 | `[1024, 4096]` |
| `depformer_emb.{0..14}.weight` | 15 | `[2049, 1024]` |
| `depformer_text_emb.weight` | 1 | `[32001, 1024]` |
| `depformer.layers.{0..5}.self_attn.in_proj_weight` | 6 | `[49152, 1024]` |
| `depformer.layers.{0..5}.self_attn.out_proj.weight` | 6 | `[16384, 1024]` |
| `depformer.layers.{0..5}.gating.{0..15}.linear_in.weight` | 96 | `[5632, 1024]` |
| `depformer.layers.{0..5}.gating.{0..15}.linear_out.weight` | 96 | `[1024, 2816]` |
| `linears.{0..15}.weight` | 16 | `[2048, 1024]` |

`in_proj_weight` is a **combined QKV** — 12288 = 3 × 4096 — which is exactly
the layout `kyutai_stt`'s `lm_layer::attn_in_w` already uses.

---

## 3. Four things that will bite

**1. The SwiGLU hidden size is not `dim_feedforward`.** `ActivationGating`
(`modules/gating.py:56-66`):

```python
if dim_feedforward == 4 * dim:
    hidden = (21 * dim) // 8
else:
    hidden = (2 * dim_feedforward) // 3
```

so 16896 → 11264 and 4224 → 2816, and `linear_in` emits `2 * hidden`. A port
that assumes `4 * dim` silently takes the other branch and mis-sizes every
FFN. The shapes in §2 confirm which branch this checkpoint used.

**2. The depformer is per-step asymmetrically.** Attention is per-step but
*packed into one tensor* — `in_proj_weight` is `[3 * 1024 * 16, 1024]`,
`out_proj` is `[1024 * 16, 1024]`. Gating is per-step as **separate modules**
(6 layers × 16 steps = 96 tensors each). But `norm1.alpha` / `norm2.alpha`
appear **6 times, not 96** — the norms are *shared across steps*. Getting
this backwards produces a model that loads cleanly and generates noise.

**3. The `copy_missing_weights` patches are no-ops here — do not port them.**
`loaders.py` patch 1 expands depformer self-attn and patch 2 copies
`gating`/`linears`/`depformer_in`/`depformer_emb` from indices 0..7 into
8..15. Those exist for checkpoints trained at `dep_q = 8`. This checkpoint
already ships all 16 sets and `in_proj_weight` is already `[49152, 1024]`, so
both patches do nothing. The converter reads straight through.

**4. `<system>` is not a special token — and only one of its two oddities is
a bug.** It encodes to three ordinary SPM pieces — `▁<`, `system`, `>`
(ids 607, 4831, 578); `</system>` would be four.

`wrap_with_system_tags` emits `<system> … <system>` with the *same* tag at
both ends, identically in `server.py:79` and `offline.py:83`. **That is
intentional — reproduce it byte-for-byte.** It is not a missing `</system>`:
the tags are ordinary text the model was trained against, the two files
agree, and "fixing" the closer would change the tokenization and can only
hurt persona adherence.

The actual defect is the guard:

```python
if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
    return cleaned
```

A prompt that opens with the tag but does not close with one fails the test
and gets wrapped again — `<system> <system> foo <system>` — and the bare
string `<system>` satisfies both halves and passes through unwrapped. Our
port fixes the guard and keeps the symmetric tags.

Cross-checks that pass: SPM `pad_id = 3` matches `existing_text_padding_id`;
SPM vocab 32000 matches `text_card`; `text_linear` is `[32000, 4096]`,
i.e. `extra_text = 0`, because `existing_text_padding_id` is not None.

---

## 4. What CrispASR already has

The codec half is done, because Kyutai STT *is* Moshi-architecture and CSM
uses the same codec.

| piece | where | note |
|---|---|---|
| Mimi encoder (SEANet, ratios 4/5/6/8, dim 512, 64 filters) | `src/kyutai_stt.cpp:171` | config is bit-identical to `_seanet_kwargs` |
| Mimi encoder transformer, 8L, `layer_scale` | `src/kyutai_stt.cpp` | matches `_transformer_kwargs` |
| 25 → 12.5 Hz stride-2 downsample | `kyutai_model::downsample` | |
| Split RVQ, 1 semantic + 31 acoustic, bins 2048, dim 256 | `rvq_first` / `rvq_rest` | matches `_quantizer_kwargs` |
| Mimi decoder | `src/core/seanet_decoder.h`, `src/csm_tts.cpp` | |
| Moshi-shaped LM layer (RMSNorm alpha, combined QKV, SwiGLU) | `kyutai_stt.cpp` `lm_layer` | same field-for-field layout |
| Depth decoder over codebooks with per-position heads | `src/csm_tts.cpp` | template for the depformer, not a drop-in |

Missing: the depformer's `weights_per_step`, the dual-stream delay pattern,
4096/32L geometry, and voice-prompt conditioning.

Multi-GPU note: the scheduler arrays are hardcoded
`ggml_backend_t backends[2] = {backend, backend_cpu}`
(`kyutai_stt.cpp:550`, `csm_tts.cpp:1082`). A second CUDA device means
widening those to 3. Layer-boundary activations are 4096 × 2 B = 8 KB per
frame, so a layer split is cheap even over a slow link.

---

## 5. Target: `offline.py`, not `server.py`

`offline.py` streams an input wav through and writes a matched-length output
wav plus `output.json`. That removes the WebSocket work from the critical
path **and** is itself the HARD RULE #3 decoded-output acceptance gate, so
the gate does not have to be built.

Voice conditioning has two entry points (`offline.py:236-240`): a `.pt` of
precomputed embeddings via `load_voice_prompt_embeddings`, or a `.wav` via
`load_voice_prompt`, which needs Mimi encode — which we will already have.

---

## 6. The GPU topology this targets (measured 2026-08-16)

```
cuDeviceGetCount -> 2                   # nvcuda.dll, driver CUDA 13.3
  [0] NVIDIA GeForce RTX 3060 sm_86 12.88 GB
  [1] NVIDIA GeForce RTX 2060 sm_75  6.44 GB      19.32 GB combined
```

Both cards are CUDA-visible, so the 15.59 GB bf16 reference **fits across the
pair** and phase 2 needs no CPU offload. The `#105` meta-device patch stays in
the tree as insurance for the offload path, but it is not on the critical
path.

Getting here cost real time, and the two causes are worth recording because
both fail *silently*:

**1. NVIDIA Control Panel → Manage 3D Settings → CUDA - GPUs.** This restricts
which physical cards the driver exposes to CUDA. While it excluded the 2060,
`cuDeviceGetCount` returned **1** and asking for device 1 returned *zero*
devices — indistinguishable from the card being absent, even though
`nvidia-smi -L` listed it, both reported Compute Mode Default / WDDM, and
Turing `sm_75` is CUDA 13's *minimum* supported arch (13 dropped
Maxwell/Pascal/Volta, not Turing). Vulkan saw it throughout —
`vulkaninfo` listed 3060 / AMD iGPU / 2060, and ollama bound the 3060 through
CUDA and the 2060 through Vulkan in the same session. **A card visible to
NVML and Vulkan but not CUDA points here first.**

**2. `CUDA_VISIBLE_DEVICES=0` set at machine level**, combined with
`cli.cpp:364` only setting the variable when it is *not already present* —
which makes `--device 1` a silent no-op. Removed. Two follow-on traps: check
the *process* environment rather than the registry, because a shell started
before the change keeps the stale value and every child inherits it; and note
that `--device N` sets both `CUDA_VISIBLE_DEVICES` and
`GGML_VK_VISIBLE_DEVICES` (`cli.cpp:362-368`), with `--gpu-backend` taking
`cuda|vulkan|metal|cpu`, so Vulkan remains a viable fallback for the second
instance if CUDA ever regresses.

RAM is not a constraint either: 31 GB total, a single process measured to
~26 GB.

## 7. Open

- RTF risk is the depformer, not the 8 B backbone: 16 sequential 6-layer
  1024-dim steps per 80 ms frame = 96 tiny graph executions per frame,
  1200/s, launch-latency-bound. `weights_per_step` blocks the easy batching.
  Measured at the end of phase 3.
- KV cache at the full 3000-frame context is 32 × 3000 × 4096 × 2 × 2 B ≈
  1.6 GB. At 750 frames (60 s) it is ~400 MB. The 6 GB card needs the cap.
