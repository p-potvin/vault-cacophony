# text-style-transfer

**Goal (re-scoped 4 Sep 2026):** start from a picture that contains text in some
style, and write a *different short text* in that exact visual style onto another
picture.

That is not font-file generation. It is *reference-styled visual text generation /
scene-text editing*: font, colour, gradient, outline, shadow, texture — the whole
look — copied from picture A onto picture B. The earlier VecGlypher/SVG track
(train a model that emits font outlines) solves a different problem and is parked.

---

## 1. Landscape (checked 4 Sep 2026)

### Purpose-built research models

| Method | Venue | What it does | Usable now? |
| --- | --- | --- | --- |
| **StyleTextGen** | CVPR 2026 | style-reference image → new text in that style onto a target; dual-branch style encoder, DiT inpainting; CN+EN | **no code/weights released** |
| Self-Prompting DiT STE ("mstedit") | ICML 2026 | scene-text editing preserving the *source image's* text style via MM-DiT in-context | no release found; style must come from the same image |
| SceneTextStylizer | 2025/26 | training-free, *prompt-described* style change of existing text | different task (prompt, not reference) |
| TextCtrl | NeurIPS 2024 | SD1.5 scene-text editing with style disentanglement | weights released; low-res, same-image style |
| AnyText2 | 2024 | visual text gen/edit with font+colour attributes from reference glyphs | released; SD1.5-era quality |
| FontAdapter | 2025 | font adaptation from a reference glyph in T2I | released; font only, not full look |
| FontCrafter | CVPR 2026 | element-driven *decorative* fonts (flames, flowers as texture) | relevant only for that sub-case |

Bottom line: the one paper that matches the task exactly (StyleTextGen) has no
release, and the released academic models are SD1.5-class.

### General instruction-edit models — the pragmatic SOTA

| Model | Params | Text-edit reputation | GGML |
| --- | --- | --- | --- |
| **Qwen-Image-Edit-2511** | 20B MMDiT | bilingual text add/replace *preserving original font, size and style*; multi-image (up to 3 refs) | GGUF Q2–Q8 on the Hub; **stable-diffusion.cpp supports 2509/2511** (`--model-args qwen_image_zero_cond_t=true`) |
| FLUX.1 Kontext dev | 12B | good text replacement, weaker than Qwen on font fidelity in published comparisons | GGUF + sd.cpp |
| FLUX.2 Klein 4B/9B | 4/9B | fast edit/T2I | GGUF on disk here |

Decision: **Qwen-Image-Edit-2511.** It is the model whose documented strength is
exactly "replace the text, keep the font/colour/shadow", it takes a second image
as a style reference natively, and — critically — the fp8 Lightning-8-step build,
the Qwen2.5-VL text encoder, VAE, Lightning LoRAs and the InstantX inpainting
ControlNet were **already on this machine** under `D:\ComfyUI\resources\ComfyUI\models`.

---

## 2. Architecture

```
style.jpg  ──crop──►  style_crop ─────────────────────────┐
                                                          ▼  image2 (reference)
target.jpg ──fit 1MP──► target ──paste crop in bbox──► composite  image1
                             │                                   │
                             │            TextEncodeQwenImageEditPlus(prompt, image1, image2)
                             │              → FluxKontextMultiReferenceLatentMethod(index_timestep_zero)
                             │            KSampler(euler/simple, 8 steps, cfg 1.0)  ← ModelSamplingAuraFlow 3.1 + CFGNorm
                             │            VAEDecode
                             └──── ImageCompositeMasked(original, decoded, feathered bbox mask) ──► out.png
```

The graph is lifted verbatim from ComfyUI's own `image_qwen_image_edit_2511`
template (read out of the installed `comfyui-workflow-templates` package), not
reconstructed from memory. Three strategies share it:

| strategy | image1 | image2 | prompt | notes |
| --- | --- | --- | --- | --- |
| composite | target with the style crop pasted into the bbox | style crop | *replace the text with "…", keep exactly the same font/colours/outline/shadow* | leans on 2511's best-documented ability: in-place text replacement |
| **multiref** (default) | clean target | style crop | *write "…" on image 1 at ‹where›, in the style of the text in image 2* | nothing foreign is pasted into the target — this is the one that works |
| inpaint-cn | clean target | style crop | as multiref + InstantX inpainting ControlNet | ControlNet was trained on base Qwen-Image, not Edit-2511 → experimental |

Region control is pixel-space by default: an instruction-edit model changes what
it is told to, and `ImageCompositeMasked` pastes the result back onto the
untouched original through a grown+feathered bbox mask, so everything outside the
box is pixel-identical. `--latent-mask` additionally freezes the latent outside the
box during sampling (`SetLatentNoiseMask`).

Sizing is done in PIL (≈1 MP, both sides a multiple of 16) instead of the
template's `FluxKontextImageScale`, so the decoded image has exactly the input
size and can be composited back 1:1.

### Files

| Path | Purpose |
| --- | --- |
| `sheet.py` | your own handwriting: print a template, write 68 glyphs, photograph it, get a .ttf. No model involved |
| `capture.py` | read the letters a photo actually contains and score them, so only the missing ones are generated |
| `features.py` | measures stroke weight, monoline-ness and ink colour from a reference; `--selftest` checks it against known widths and the Arial family |
| `handwriting.py` | phone photo -> paper -> rectified, illumination-flattened page -> ink mask -> letter boxes |
| `pipeline.py` | CLI: `--style --target --text [--bbox x,y,w,h] [--strategy] [--latent-mask]` |
| `workflows.py` | API-format graph builders: `build_graph` (Qwen-2511) and `build_graph_klein` (FLUX.2 Klein) |
| `comfy_client.py` | upload / queue / **WebSocket completion** / fetch. One connection per job — no `/history` polling loop |
| `app.py` | Gradio UI on :7870: style-crop box (cyan) + placement box (yellow), both previewed live |
| `run.ps1` | picks a live ComfyUI (:8188 Desktop → :8189 headless, starting it if needed) then launches the UI |
| `run-comfy-headless.ps1` | core-only ComfyUI on :8189 from `..\.venv-comfy` against the D: model library |
| `comfy-runtime/` | headless server's `extra_model_paths.yaml` + its own input/output/user/temp dirs |
| `torch-constraints.txt` | pins torch 2.8.0+cu129 so ComfyUI's unpinned requirements can't swap in a CPU wheel |
| `examples/` | synthetic test pair: `style_summer.jpg` (Jokerman, orange gradient, purple outline, shadow) and `target_waves.jpg` |
| `out/`, `logs/` | gitignored |

---

## 3. Running

```powershell
.\font-cloning\text-style-transfer\run.ps1          # UI at http://127.0.0.1:7870
```

CLI:

```powershell
..\..\.venv\Scripts\python.exe pipeline.py `
  --style examples\style_summer.jpg --style-bbox 0.17,0.35,0.63,0.28 `
  --target examples\target_waves.jpg --text "HELLO" --bbox 0.15,0.35,0.70,0.30 `
  --out out\hello.png
```

| flag | why it matters |
| --- | --- |
| `--style-bbox x,y,w,h` | **crop the style reference tight around the text** — background inside the crop contaminates the reference |
| `--bbox x,y,w,h` | where the new text goes, and the region composited back |
| `--strategy` | `multiref` (default) / `composite` / `inpaint-cn` |
| `--backend` | `qwen` (default, Q4_0 GGUF) / `klein` (FLUX.2 Klein 4B, ~2× faster, weaker letterforms) |
| `--unet`, `--no-gguf` | override the model; together they reproduce the broken fp8 file |
| `--latent-mask` | also freeze the latent outside the bbox during sampling |

Two ComfyUI back-ends are supported; the client auto-detects :8188 then :8189.

**Comfy Desktop (:8188)** — the user's GUI install. Runs
`--base-directory D:\Comfyui\resources\ComfyUI` with ~125 custom nodes, including
`comfyui-multigpu` / `DisTorch` which patch core model loading.

**Headless core-only (:8189)** — `run-comfy-headless.ps1`. Same D: source tree
and models, `--disable-all-custom-nodes`, own runtime dirs, Python from
`font-cloning\.venv-comfy` (uv, 3.12, torch 2.8.0+cu129 hard-linked from cache;
`comfyui-frontend-package` had to be added — it is shipped separately from the
source tree's `requirements.txt`).

Why a second venv again: the D: install's own `.venv` is dead (its base
interpreter `pythoncore-3.14-64` was uninstalled), and the repo `.venv` carries the
ASR stack that must not be touched.

---

## 4. Verification log (4 Sep 2026)

| # | server | test | wall | result |
| --- | --- | --- | --- | --- |
| 1 | :8188 Desktop | composite, bbox mask, 8 steps cfg 1.0 | 237 s (cold: 19.5 GB fp8 streamed) | outside box = original ✓ · **inside box = pure noise ✗** |
| 2 | :8188 Desktop | same, **no mask, no composite-back** (control) | 217 s | **entire frame noise** → not a mask/composite bug |
| 3 | :8188 Desktop | VAE encode→decode round-trip only | 27 s | perfect reconstruction → decoder fine |
| 4 | :8188 Desktop | multiref | 10 s | sampler "finished" 8 steps in 10 s — physically impossible for 20B on a 3060; aborted |
| 5 | :8189 core-only, 0.22.3, `--disable-all-custom-nodes` | identical control graph | 214 s | **byte-identical noise** (2 042 735 B, same as run 2) |
| 6 | — | safetensors header of the fp8 file | — | 840 × `F8_E4M3` + 1933 × `BF16`; **no `scaled_fp8`, no `*.scale_weight`** |
| 7 | — | raw tensor read (no mmap) | — | every fp8 tensor: **min −448, max +448, mean\|x\| 73–98**; bf16 tensors normal (\|x\| ~0.1) |
| 8 | :8188 Desktop | **FLUX.2 Klein 4B fp8**, composite, whole style image pasted, 4 steps cfg 1 | 131 s | **legible "HELLO"**, orange gradient + dark outline transferred; letterforms blocky (not Jokerman); pasted style panel left visible → plumbing OK, model choice + crop matter |
| 9 | :8188 | **Qwen-Image-Edit-2511 Q4_0 GGUF** + 4-step Lightning LoRA, `multiref`, tight style crop, 4 steps cfg 1 | 284 s cold / ~128 s warm | **WORKS.** "HELLO" in the reference's quirky letterforms, orange fill, dark-purple outline, drop shadow; background untouched |
| 10 | :8188 | same, text `BONJOUR`, seed 123 | 130 s | style perfect, content wrong: renders **`SUMJOUR`** |
| 11 | :8188 | `BONJOUR` after rewriting both prompts to spell the target out and forbid copying image 2 | 127 s | **still `SUMJOUR`** at the same seed |
| 12 | :8188 | `BONJOUR`, `multiref`, seed 4242 | 125 s | **`SUMMUR`** — even more of the reference word |
| 13 | :8188 | `BONJOUR`, `composite`, seed 123 | 127 s | **`SUMJOUR`** + the pasted style panel still visible |
| 14 | :8188 | **`WELCOME`** (EN, 7 letters), same word reference, same seed 123 | 180 s | **CLEAN** — isolates the variable: the bleed is lexical |
| 15 | :8188 | **3D extruded gold** reference → `SHINE` on a feather photo | 130 s | **excellent** — extrusion depth, bevel, metallic gradient, specular, dark contour, plus a new cast shadow onto the feathers |
| 16 | :8188 | **photo-textured extruded** reference → `DREAM` | 124 s | **excellent** — per-letter photographic fill re-synthesised as texture (not a pasted patch), depth and outline kept |
| 17 | — | measure changed-region vs requested box on runs 15/16 | — | model overran the box by **53 px vertically (SHINE)** and **52 px horizontally (DREAM)** → the feathered bbox was clipping the glyphs |
| 18 | :8188 | `SHINE`, Lightning-4 cfg 1.0, **new diff mask** | 128 s | clipping and haze gone; extrusion, contour and shadow fully visible |
| 19 | :8188 | `SHINE`, **no LoRA**, 20 steps cfg 2.5 | **616 s** | matte, desaturated |
| 20 | :8188 | `SHINE`, **no LoRA**, 40 steps cfg 3.0 | **1183 s** | more matte still |
| 21 | :8188 | `SHINE`, `composite` strategy, Lightning-4 | 127 s | text rendered smaller; gradient no closer |

Server log for run 1 confirms the diffusion model really executed:
`model weight dtype torch.float8_e4m3fn, manual cast: torch.bfloat16` ·
`Model QwenImage prepared for dynamic VRAM loading. 19483MB Staged` ·
`First sampler step … total_steps=8 cfg=1.0` · `Prompt executed in 236.30 seconds`.
So on the Desktop instance the sampler runs and returns garbage; the VAE is fine.
The suspects are the memory-management custom nodes hooking model loading
(`[MultiGPU Core Patching] …` appears in the log) interacting with scaled-fp8
dequantisation.

## 5. Verdict — the checkpoint is broken, not the servers

Runs 2 and 5 produce byte-identical garbage on two different ComfyUI versions,
one with 125 custom nodes and one with none. That exonerates the Desktop
instance and its memory patches.

Run 7 is the smoking gun. `float8_e4m3fn` saturates at ±448; a real transformer
weight matrix has |x| around 0.01–0.1. Every fp8 tensor in
`qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors` is
pinned to the saturation bounds with a mean magnitude near 80. That is what a
**scaled-fp8** export looks like *after the per-tensor scale tensors have been
dropped*: the weights were multiplied up to fill the fp8 range, and the
`scaled_fp8` / `*.scale_weight` entries that would divide them back down are not
in the file (run 6). ComfyUI therefore dequantises with an implicit scale of 1,
every weight is ~10³× too large, and the denoiser emits noise — deterministically,
so the two servers agree to the byte. The bf16 tensors (embeddings, norms) are
normal, which is why the VAE and text encoder paths looked healthy.

Consequences:

* The file is unusable in ComfyUI as-is; the scales cannot be recovered.
* **Fix, confirmed by run 9:** `unsloth/Qwen-Image-Edit-2511-GGUF` **Q4_0 (11.0 GB)**,
  downloaded once into `D:\comfyUI\resources\ComfyUI\models\diffusion_models`
  (visible to both servers), loaded through `ComfyUI-GGUF`, with the on-disk
  `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16` LoRA for 4-step sampling.
  Its load line reads `gguf qtypes: F32 (1088), BF16 (6), Q4_0 (720), Q4_1 (120)`
  — a normal quant mix, unlike the saturated fp8 above. This is now the default
  `ModelSet`; the fp8 file stays reachable via `--unet ... --no-gguf` for repro.
* Plumbing is independently validated on a *different* model: FLUX.2 Klein 4B
  (`build_graph_klein`), which is also kept as a zero-download fallback backend.

### Lessons from run 8 (first healthy model)

* **The style reference must be a tight crop of the text.** Passing the whole
  style picture pastes its background into the target and the model treats it as
  a photo inset. The UI and CLI now expose a style crop box (`--style-bbox`).
* **`multiref` is the safer default** for cross-picture transfer: the target stays
  untouched as image1, the crop rides along only as image2 — nothing foreign is
  pasted in. `composite` remains useful when the model's in-place text
  replacement is stronger than its cross-reference styling (Qwen-2511 claims
  exactly that), so both are kept.
* Klein 4B copies colour, gradient and outline but not the letterforms. It is a
  fast, fully-resident (3.8 GB) fallback, not the primary.

## 5a. Resolved: the reference-word bleed is lexical

Asking for `BONJOUR` against a style crop reading `SUMMER` yields `SUMJOUR` /
`SUMMUR` — the reference's *content* leaks into the output. Run 14 settled why:

| text | language | letters | seed | result |
| --- | --- | --- | --- | --- |
| `HELLO` | EN | 5 | 7 | clean |
| **`WELCOME`** | **EN** | **7** | **123** | **clean** |
| `BONJOUR` | FR | 7 | 123 | `SUMJOUR` |
| `BONJOUR` | FR | 7 | 4242 | `SUMMUR` |

`WELCOME` is the same length, the same reference crop, the same strategy and the
same seed that failed for `BONJOUR`, and it renders perfectly. So it is not
length, seed, strategy, or prompt wording (runs 10-13 ruled those out
individually, including a rewrite that spells the target `B-O-N-J-O-U-R` and
forbids copying from image 2).

**Cause:** Qwen-Image-Edit-2511 is documented as an English/Chinese text renderer.
Given a string it cannot confidently spell, it falls back on the word it can see
in the reference. This is a model-capability limit, not a pipeline bug, and no
prompt engineering fixes it.

**Implication:** the pipeline is production-usable for English. For French or
other non-EN/CN scripts a different model is required — and there is currently no
obvious one (see below).

### No drop-in upgrade exists (checked 6 Sep 2026)

* `Qwen-Image-Edit-2511` is still the **newest Qwen *edit* model**. `Qwen-Image-2512`
  and `Qwen-Image-2.0` are text-to-image *generation* models, not edit models, so
  their broader multilingual claims do not transfer to this pipeline.
* `FLUX.2-dev` (32B, multi-reference editing) makes **no text-rendering or
  multilingual claim at all** on its model card. Switching to it for French would
  be a gamble, not a known win.
* Worth an experiment before committing: FLUX.2-dev / Klein on a French string,
  since Klein is already on disk and `build_graph_klein` already exists.

## 5b. Texture and depth transfer well

Tested with two synthetic references carrying real depth cues — extrusion built
from offset copies, a bevelled face gradient, a specular band along the top edge,
a dark contour and a cast shadow — written onto a busy feather photograph that
contains no text and no comparable relief.

| reference | asked for | kept |
| --- | --- | --- |
| extruded metallic **GOLD** (Impact) | `SHINE` | extrusion depth and direction, bevel, metallic face gradient, specular highlight, dark contour — **and it added a plausible new cast shadow onto the feathers** |
| photo-textured extruded **NEON** (Cooper Black) | `DREAM` | the photographic fill inside the glyphs, varying per letter, plus depth and outline |

Two things worth noting. The texture is **re-synthesised, not copied** — the
letters carry fluid-art colour in the same character as the reference without
being the same pixels, which is what you want when the new word has different
letters. And the model relights: the cast shadow onto the feathers is not present
in either input, so it is inferring the lighting rather than pasting a style
patch. On this evidence depth and texture are a **strength**, not a risk area.

## 5c. The composite mask: why the lettering looked clipped and hazy

The old mask was a rectangle from `--bbox`, grown 24 px and Gaussian-blurred 16 px,
composited in-graph. Measuring the actual changed region against the requested box
(run 17) showed the problem:

| run | requested box | region the model actually drew |
| --- | --- | --- |
| SHINE | (185, 270, 1142, 481) | (150, 217, 1182, **534**) — **+53 px vertically** |
| DREAM | (185, 270, 1142, 481) | (**132**, 224, **1193**, 532) — **+52 px horizontally** |

The model sizes the lettering itself and routinely overruns the box, in whichever
axis the glyphs push hardest — which is exactly why SHINE looked short of height
and DREAM short of width. The 16 px feather then faded the overrun out, taking the
extrusion, drop shadow and outer glow with it and leaving a hazy band.

**The fix is not a bigger box.** The mask is now derived from what changed:

```python
diff_mask(original, generated, box, threshold=45, grow=12, feather=2.0,
          smooth=6.0, pad=70)
```

One thing had to be got right. A plain pixel-diff threshold does **not** work: the
model returns a full VAE round-trip, and on a detailed background (feather barbs)
the round-trip alone moves pixels by p99 = 127, max = 204 — indistinguishable from
lettering by magnitude. What separates them is *structure*: the lettering is a
single large connected region while round-trip error is scattered speckle. So the
difference is **blurred (σ 6) before thresholding**, and bounded to the box padded
by 70 px purely to reject far-field noise. Measured far-field leak: **0.00%**.

Feather drops 16 px → 2 px, which is seam anti-aliasing only. `--mask-mode bbox`
restores the old behaviour, `none` returns the raw decode.

## 5d. What actually controls the gradient (and what does not)

Simple styles were the weak case: `SHINE` kept the *idea* of the GOLD reference but
not its gradient. Four configurations, same seed, same reference, measured against
the reference glyph bodies — normalised top-to-bottom luminance profile (shape) and
RGB spread (colour richness):

| run | wall | profile L2 vs ref | mean luminance | saturation |
| --- | --- | --- | --- | --- |
| **reference** | — | 0.000 | 188.5 | **130.9** |
| A Lightning-4, cfg 1.0 | 128 s | 1.544 | 216.5 *(too bright)* | **96.2** *(closest)* |
| B no LoRA, 20 st, cfg 2.5 | 616 s | **1.270** *(closest)* | 190.9 | 80.7 |
| C no LoRA, 40 st, cfg 3.0 | 1183 s | 1.343 | **188.3** *(closest)* | 82.3 |
| D composite, Lightning-4 | 127 s | 1.746 | 192.6 | 93.2 |

**No configuration reproduces the reference gradient.** Every run sits ≥ 1.27 from
it in profile and loses 26–38% of its saturation. The knobs move *where the error
lands*, they do not remove it:

* **cfg / steps trade glossiness for tonal accuracy.** cfg 1.0 with the 4-step
  Lightning LoRA keeps colour richest (sat 96) but runs too bright and glossy;
  dropping the LoRA and raising steps and cfg lands the overall luminance and
  profile shape almost exactly but goes matte and desaturated.
* **Steps are not the lever.** 40 steps costs **9×** the wall clock of 4 and is
  *worse* than 20 on profile and no better on saturation.
* **`composite` does not help.** Giving the model the real style pixels in image 1
  produced the *worst* profile match and shrank the lettering.

The reason is structural: the model **re-renders the material from its own prior**
rather than sampling the reference's pixels. That is precisely the behaviour that
makes the textured case excellent (5b: `DREAM`'s fill is re-synthesised per letter,
which is what you want when the glyphs differ) and the simple-gradient case
unreliable.

### Consequence: two lanes, and generative is the wrong tool for one of them

* **Complex / photographic styles — keep the generative path.** Variation is a
  feature; the model relights and re-textures coherently. Defaults are already right
  (Lightning-4, cfg 1.0, `multiref`, diff mask).
* **Simple parametric styles — do not use the model for the fill.** A gradient +
  outline + extrusion + shadow is a handful of numbers. Fit those from the reference
  crop and re-render the target word deterministically — `examples/style3d_gold.jpg`
  was *generated* that way by `extruded()`, so the renderer already exists. That gives
  pixel-exact, seed-independent, sub-second output for exactly the case where the
  model's creativity is unwanted. The model would then be used only for placement
  and for lighting the result into the scene, or not at all.

Not built. Recorded because the sweep shows tuning cannot close this gap.

## 5e. Client robustness fix

Long jobs killed the WebSocket with `keepalive ping timeout`: ComfyUI's event loop
is blocked while the sampler runs on the GPU, so it never answers the library's
pings. `comfy_client.run()` now passes `ping_interval=None` and, if the socket
dies anyway, reads the outcome from `/history` rather than losing a job that
actually completed.

### Memory note

During this session the box hit its Windows commit limit (RAM free 0.4 GB,
84.5 / 89.7 GB committed; `OSError 1455 paging file too small`). A headless
ComfyUI that has streamed the 19.5 GB fp8 model keeps ~24 GB private after the
job ("RAM pressure cache"). Do not run two ComfyUI servers with 20B-class models
on this 32 GB machine at the same time; stop one (`POST /free` or kill) first.

Related: after ~10 back-to-back 20B jobs the Comfy Desktop instance on :8188
wedged - it accepted a prompt, staged the text encoder, then stopped logging, GPU
idle at 8%, HTTP unresponsive. Nothing was lost (every completed job is in
`outputs/tst/`) but the app needs a restart. Using the headless :8189 server keeps
the GUI instance out of that state.

---

## 6. GGML path (for later)

`stable-diffusion.cpp` runs Qwen-Image-Edit-2511 from GGUF with the Qwen2.5-VL
encoder as GGUF + mmproj; its CLI stores reference images in a vector
(`gen_params.ref_images.push_back(...)`), so multiple `-r` are accepted even though
the docs only show one. Sizes that matter on a 12 GB card:

| file | GB |
| --- | --- |
| qwen-image-edit-2511 Q3_K_M / Q4_0 / Q4_K_M (unsloth) | 9.2 / 11.0 / 12.3 |
| Qwen2.5-VL-7B Q4_K_S + mmproj Q8_0 (already on D:) | 4.2 + 0.8 |
| qwen_image_vae | 0.24 |

Windows CUDA builds are published per commit (`master-841-…`, 30 Aug 2026). That is
the "fast precompiled binary" route once the pipeline's prompts and strategies are
settled here.

---

## 7. Photo → font (added 6 Sep 2026)

`clone_font.py` runs the whole path: a picture of handwriting → one letter taken
as the style reference → the charset generated glyph by glyph → normalised → TTF.

```powershell
..\..\.venv\Scripts\python.exe clone_font.py --photo hand.jpg          # lists letter candidates
..\..\.venv\Scripts\python.exe clone_font.py --photo hand.jpg --ref-index 3 --chars letters --colour
```

Two things it does that the A–Z probe did not:

**A gate after the first glyph.** A bad reference wastes an hour, so glyph one is
generated alone and checked: is there ink at all, is the amount sane, is it one
connected piece, and — the failure that actually bit us — is it just a copy of the
reference? That last check is an IoU against the reference silhouette, because
`O` and `Q` came out as the reference `D` and nothing in the output flagged it.
`--gate ask` stops for a human look, `auto` (default) halts on a failed check,
`none` forces through.

**The batch is queued in one go.** ComfyUI keeps the model resident across a queue
run, so the ~60 s GGUF dequantisation is paid once rather than per letter. On the
A–Z probe that overhead was roughly half of each 120 s job.

### Backgrounds: two paths, both supported

| | clean plate (default) | no clean plate |
| --- | --- | --- |
| when | we generated the target, so the untouched plate exists | cloning off a photo someone else made |
| extraction | difference against the plate — says *exactly* which pixels the model drew, then a warm/bright test splits face from extrusion and shadow | segmentation; `SAM3_Detect` + `sam3.1_multiplex_fp16` are installed locally |
| accuracy | exact | estimated |

A flat or transparent target makes extraction trivial and is the right choice when
building a font atlas. It is the wrong choice for in-scene work: §5b showed the
model *relights* the lettering to match the scene — the cast shadow on the feathers
exists in neither input — and a flat grey ground gives it nothing to relight
against. Keep both.

### Normalisation: cap height, not bounding box

`Q` came out ~5 % small because scaling used the full bounding box, which for `Q`
includes the descending tail. Row-ink profiles separate them cleanly (Q's tail is
25 px of a 485 px bbox; O/I/J have ~0), so `normalise()` now scales the letter
*body* to a fixed cap height and lets descenders hang below the baseline.

### Colour font

`--colour` additionally emits an **OT-SVG** font carrying the artwork as rendered —
extrusion, gradient, specular, contour, cast shadow. The monochrome `glyf`
outlines remain as the fallback, so one file serves both.

Renders in Firefox, Photoshop and Illustrator. **Chrome does not support OT-SVG** —
verified here, it falls back to the outlines. If Chrome/Android support is needed
the format to add is `CBDT/CBLC` (colour bitmaps), not OT-SVG.

### LoRA dataset

`lora_dataset.py` emits control→target pairs in Ref2Font's layout
(`has_control: true`) but not its content: style, background and cap height are
held identical while only the text changes. See `REF2FONT-NOTES.md` for why that
construction targets the three measured defects directly.

---

## 8. Measured features in the prompt (added 6 Sep 2026)

The reference letter is pasted into the target at box scale, so **it carries no
information about how heavy the original stroke was**. A 62 px letter with a 3 px
stroke and a 620 px letter with a 30 px stroke are the same picture after the
paste. Whatever the model infers about the medium, it infers from the shape alone.

`features.py` measures the reference and the prompt says what it found.

### How weight is measured

Distance transform, no skeletonisation (OpenCV has no thinning outside contrib).
For a stroke of uniform width *w* the distance-to-edge of its interior is uniform
on `(0, w/2]`, so the mean is `w/4` and

    w = 4 * mean(distanceTransform(ink))

`features.py --selftest` checks this end to end:

| shape | true width | measured |
| --- | --- | --- |
| bar | 20 | 21.5 |
| bar | 40 | 39.8 |
| ring | 20 | 22.2 |
| L (corners) | 20 | 21.7 |

What goes in the prompt is the **ratio** of stroke width to cap height, since that
is what survives being rescaled into the box. Bands are calibrated so real faces
land where their names say (capital M at 400 px):

| font | ratio | band |
| --- | --- | --- |
| Arial | 0.126 | regular weight |
| Arial Bold | 0.193 | bold |
| Arial Black | 0.245 | very bold |
| Times | 0.120 | regular weight |

Two more numbers fall out of the same distance transform for free:

- **modulation** — the spread of local width along the stroke. Arial reads 0.09,
  Times 0.43, so the 0.28 cut is the monoline/modulated distinction. This turned
  out to matter more than weight (below).
- **ink colour** — the median of the darkest 35 % of the ink, after a per-channel
  white balance. Without the white balance the sample photo, which has a blue
  shadow across half the card, reports *violet* ink for a grey-blue pen.

### What it changed in the prompt

The old wording ended `Keep exactly the same font, letter shapes, colors,
gradient, outline, shadow and lettering style`. Against a plain pen that list is
an invitation — it names four effects the reference does not have. It is replaced
by the measurement:

    ...which is light handwriting in even strokes of constant width, each stroke
    about a tenth as thick as the letter is tall, drawn in grey. Every stroke is
    one solid line of flat, even colour, filled in completely and drawn in a
    single pass, like ordinary handwriting on paper.

**No negations.** The graph samples at `cfg 1.0` with the distilled Lightning
LoRA and `negative=""`, so there is no negative branch for the sampler to
subtract — everything written here lands in the *positive* prompt, where a phrase
like "no outline" is just the token *outline*. That is reason enough to avoid it.

(An earlier version of this section claimed the negations had actually *produced*
outlined letters. They had not: the glyphs read as outlines in the normalised
silhouette sheet, but in the colour output they are the same thin ragged stroke
as every other letter in those runs. The negations are worth avoiding on the cfg
argument alone; there is no evidence they caused a distinct failure.)

The display-lettering path is untouched: with no measured features, or with
`--no-features`, the original wording is used verbatim.

### Prompt templates (`--prompt`)

`pipeline.PROMPT_STYLES` holds **only the style clause**. The head (`Replace the
text in image 1 with the word "X". The new text must read exactly "X" (X) and
nothing else.`) and the tail (`Blend the new letters naturally into the
background. Do not change anything else in the image.`) are shared, so the
variants are directly comparable and a run difference is attributable to the
clause alone. `default` is asserted byte-identical to what the pipeline sent
before templates existed, which makes it a valid control.

| name | style clause |
| --- | --- |
| `default` | Keep exactly the same font, letter shapes, colors, gradient, outline, shadow and lettering style as the existing text. |
| `minimal` | Keep exactly the same lettering style as the existing text. |
| `pen` | Write it by hand with the same pen and the same handwriting as the existing text. |
| `erase` | Erase the old letter completely so no trace of it remains, then write the new one in its place in exactly the same lettering style. |

All four are phrased as an edit to perform rather than a caption of the wanted
result, because Qwen-Image-Edit is instruction-tuned. `default` is the odd one
out: its list of "colors, gradient, outline, shadow" is a description, and against
handwriting it describes four things the reference does not have.


### Sweep result: deleting the effects list is the single best change

A/E/O, 4x reference, seed 77, everything but the style clause held identical.

| style clause | stroke ratio | modulation | saturation | components |
| --- | --- | --- | --- | --- |
| `default` (control) | 0.052 | 1.10 | 0.75 | 3, 1, 1 |
| **`minimal`** | **0.064** | **0.65** | 0.54 | **1, 1, 1** |
| `pen` | 0.044 | 0.89 | **0.50** | 1, 1, 1 |
| `flat` (removed) | 0.050 | 0.80 | 0.60 | 1, 3, 1 |
| **her hand** | **0.102** | **0.12** | **~0.16** | 1 |

`minimal` wins on stroke weight and modulation and ties on coherence; `pen` is
marginally better on colour only. Note what this says: the two clauses that *add*
words (`pen`, `flat`) both do worse than the one that only *deletes* them. The
effects list was not merely inert, it was steering the model toward display
lettering. `flat` was dropped outright — worse than `minimal` on every axis.

Weight still only reaches 0.064 against 0.102, so the conclusion that wording
cannot fully fix weight stands; but wording was doing real harm, and stopping that
recovered about a third of the gap.

### Two artefacts that are not the model's fault

**A cast shadow behind every glyph.** Measured offsets of (+12,+40), (+43,+2) and
(+75,-23) px on A/E/O, so a genuine drop shadow rather than an antialiasing rim.
`diff_mask()` kept it, because it masks everything that changed inside the box —
correct for the scene-text path, where a shadow helps lettering sit in a
photograph, and wrong for a font. `clone_font.py` now composites through
`glyphs.body_mask` instead, which keeps 0% of that residual. The font never
contained the shadow; only the preview PNGs did. `--shadow` restores the old
behaviour.

**The reference letter survives into the output.** This is the bigger one. The
composite strategy pastes the reference letter into the target box, and the model
reads "replace the text" as "add text": the raw output for the `pen` run's O
contains the pasted M at *full strength* (amplitude 147 against the O's 96) with
the O drawn over it. The M then reaches the saved PNG because it, too, is
"something that changed".

Detecting it by blurriness does not work — measured crispness (local Laplacian
energy per unit of ink amplitude) is 0.094 on the leftover M against 0.085 on the
newly drawn O, i.e. indistinguishable; the leftover is not a soft ghost but a
fully rendered letter. The reliable handle is that **we paste the reference
ourselves, so its mask is already known** and needs no detection at all. Two fixes
follow from that and are tested rather than assumed:

- `--no-paste` — leave the box empty, so there is no reference in the target area
  to fail to erase. Structural; cannot fail the way a prompt can.
- `--prompt erase` — keep the paste, but state the removal as its own operation.

### Result: colour responds, weight does not

Three runs on Carmen's page, A/E/O, same seed, same box, same reference letter:

| run | stroke ratio | modulation | saturation | components per glyph |
| --- | --- | --- | --- | --- |
| plain instruction, 87px reference | 0.059 | 0.99 | 0.75 | 4, 4, 2 |
| measured features, band word | 0.023 | 0.60 | 0.03 | 18, 7, 8 |
| measured features, explicit fraction | 0.025 | 0.95 | 0.04 | 11, 7, 8 |
| **plain instruction, 4x reference** | **0.052** | **1.10** | **0.75** | **3, 1, 1** |
| **her hand** | **0.102** | **0.12** | **~0.16** | 1 |

- **Reference resolution fixes the letterforms.** This is the one large win.
  Run 1 returned a bevelled A, a malformed E and a broken crescent O; with a 4x
  Real-ESRGAN upscale of the same cut-out and the *same* prompt, all three are
  correct, legible letters. Connected components per glyph went 4/4/2 → 3/1/1.
  The 87x90 crop was being blown up ~4x into the box, and the model was faithfully
  copying its staircased edges.
- **Ink colour responds to the prompt, and only to the prompt.** Saturation 0.75 →
  0.03 when the measured colour was named, and straight back to 0.75 without it.
  The reference itself is only ~0.20 saturated (a blue-black pen on violet-tinted
  paper — `handwriting.neutralise()` anchors on the paper beside the strokes to
  measure that), so the model is amplifying the tint roughly 4x on its own.
- **Weight is not prompt-steerable here.** Two different wordings — a band word,
  then an explicit "about a tenth as thick as the letter is tall" — landed on the
  same ratio, about a quarter of the target, *further* from her hand than the plain
  instruction. Reference resolution did not move it either (0.059 → 0.052).
- **Modulation is unstable**, 0.60 / 0.95 / 1.10 across runs, i.e. noise rather than
  control. Nothing tried so far touches it.

So the two failures are separable: **letterform coherence was a reference-quality
problem and is now largely fixed; weight, modulation and colour amplification are
the model's display-lettering prior and survive everything except naming the
colour.**

Two reasons, and the second is the one that matters.

**Structural.** At `cfg 1.0` there is no guidance scale, so text conditioning has
no leverage to raise against the composite image; the pasted reference and the
model's own lettering prior win regardless of wording.

**The prompts had drifted off-model.** Qwen-Image-Edit is instruction-tuned: it
expects a direct edit operation on the image, not a caption of the wanted result.
"light handwriting in even strokes of constant width, each stroke about a tenth as
thick as the letter is tall" is a description, not an instruction — and the
fraction is worse than useless once BPE has split "a tenth as thick as" into
pieces cross-attention has to reassemble into a quantity. Both feature runs scored
worse than the plain instruction they replaced. `--features` is therefore **off by
default**; the measurement stays because it is how runs get scored, not because it
writes good prompts.

The remaining levers for weight are raising cfg (dropping the Lightning LoRA for
~40 steps at cfg 3.0, roughly 20x the compute per glyph) or a LoRA.

Real-ESRGAN on the cut-out (`--ref-image`) is worth keeping for edge quality — the
87x90 crop is visibly blocky at box scale — but it cannot move the weight either:
the stroke-to-height ratio is invariant under any resize.

### Why this was needed — the run-1 gap, measured

Carmen's page against the 26 glyphs it produced:

| | her hand | generated (median of 26) |
| --- | --- | --- |
| stroke / cap height | 0.102 | 0.056 |
| modulation | 0.12 | 0.94 |
| ink saturation | ~0.16 (near neutral) | 0.80 |

So the model returned **half the relative weight at four times the width
variation** — a dry brush, not a pen. The earlier reading of this ("a thin stroke
upscaled 5.8x becomes a thick one") was wrong: the stroke-to-height ratio is
invariant under any resize, and the generated strokes are relatively *thinner*,
not thicker. What actually diverged was modulation and colour.

---

## 9. Scope: what this is for (decided 6 Sep 2026)

**Deriving someone's personal handwriting from one letter is not a real use case,
and not achievable in principle.** Letter shapes are not recoverable from a single
letter — an M says nothing about the bowl of a B — so the most the model can do is
supply plausible letterforms in a compatible texture. That is a handwritten-*looking*
font, not the writer's own hand.

The day's measurements say the same thing. From Carmen's M we reached stroke ratio
0.056 against her 0.102 and modulation 0.94 against her 0.12, and no prompt,
reference resolution or feature description closed it. What the run *did* reproduce
reliably was texture, colour and medium — the things a single sample genuinely
carries. The gap was never a tuning failure; it was information that was not in the
input.

It also does not need solving. If you want your own handwriting as a font, write
the 52 glyphs and process them: `handwriting.py` already does paper detection,
rectification, illumination flattening, ink thresholding and letter segmentation
(including splitting touching letters), and `glyphs.py` already does normalisation,
vectorisation and TTF assembly. No diffusion model is involved in any of that.

### The use case that works

> I like the lettering on this poster and will never find its name. Photograph it,
> get a font.

Here a single reference is legitimate, because the goal is the *style* — texture,
depth, colour, gradient, medium — and the letterforms are allowed to be the model's
invention. That is exactly the strength measured in section 5b, and the failure of
sections 8's weight experiments is not a defect for this purpose.

### Consequence for the architecture

Both paths want the same first step: **read every letter that is actually present,
and only generate the ones that are missing.** Carmen's card yields 10 of 26 (C A R
M E N B O I V); a poster title typically yields 8-15 unique letters. Generating the
remainder from many same-hand references is a far better-conditioned problem than
generating 25 from one, and the extracted letters are ground truth rather than
inference.

---

## 10. The two paths, built 7 Sep 2026

### `sheet.py` — your own handwriting, no model

    python sheet.py make --out sheet.png      # print it, write in the boxes
    python sheet.py read --photo photo.jpg    # -> .ttf

Two decisions carry the reliability.

**Registration marks, not grid lines.** Four filled corner squares map the photo
onto template coordinates with one homography; every cell is then located
arithmetically. Detecting printed cell borders in a phone photo is far less
dependable, and a border the writer has drawn over is unrecoverable.

Finding those marks needs Otsu on an illumination-flattened copy, *not* an
adaptive threshold: a solid 50 px square is bigger than any sensible adaptive
window, so the window sees only black and the middle of the mark is classified as
background. That silently lost one of the four marks in testing.

**Cyan guides, read through the blue channel.** Cyan reflects blue, so the printed
grid and its labels are nearly invisible in the blue channel while black or
graphite stays dark. The guides are removed exactly rather than by threshold luck.
The cost: the pen must not be blue.

**One scale for the whole alphabet.** `normalise()` scales each glyph's body to cap
height individually, which is right for an all-caps set and wrong for a mixed-case
one — every lowercase x-height becomes a cap height. `glyphs.place_on_baseline()`
plus `glyphs.shared_scale()` apply a single scale taken from the capitals and sit
each glyph on a common baseline. The baseline is *measured* from the ink of
non-descender characters, using the printed line only as the anchor for the
descenders, because writers do not reliably sit on it.

Tested end to end on a synthetic fill (real font into the cells, then perspective,
lighting falloff, blur and noise):

| | recovered | source font |
| --- | --- | --- |
| cells read | 68/68 | |
| cap height | 1.00x | 1.00x |
| x-height | 0.67x | 0.68x |
| ascender | 1.00x | 1.03x |
| descender | 0.19x | 0.36x |

One real limitation, measured rather than assumed: a writer who centres each glyph
in its box instead of sitting it on the line **loses the descenders entirely**
(0.02x instead of 0.19x). The information is not on the page. The printed sheet
says so explicitly.

### `capture.py` — read what is there, generate only the rest

    python capture.py --photo poster.jpg --text CANADADRY --source colour

An extracted letter is ground truth; a generated one is a guess. Carmen's card
yields eight usable letters, a poster title typically yields eight to fifteen.

**Two ink finders, neither a fallback for the other.** `--source colour` finds
lettering as the colour furthest from the background median in CIELAB, which is
what works for red on a white shield or gold on green glass. `--source paper` runs
the document pipeline instead. Colour distance does *not* work on Carmen's card:
the blue shadow across half that photo is a coloured difference, so flattening
lightness cannot remove it, and the largest colour distance in the frame is the
shadow — 13% of the frame came back as "ink".

**Candidate scoring, because a photograph is not a scan.** On a curved bottle the
letters at the ends of a word are foreshortened and rotated, and copying one into
a font bakes the curvature in permanently. Score components:

| term | what it catches |
| --- | --- |
| `sharpness` | edge energy per unit contrast — soft or out-of-focus letters |
| `height_dev` | cap height is constant across capitals whatever the letter, so a deviation is the camera |
| `baseline_dev` | distance off the line fitted through all letter bottoms — a cylinder curves it |
| `gap` | distance to the nearest neighbour, in cap heights — touching letters make poor references |
| `clipped` | touching the frame edge |

Two earlier metrics were **discarded after measurement, not on taste**. Run on flat
paper with no distortion at all, the first scoring rejected 10 of 12 good letters:
`slant` from second-moment orientation read -130° for an I and +37° for an O — that
is the letter's own shape, not slant — and `squeeze` from letter-centre spacing
read 0.46 for the end of BOIVIN, because I and V are narrow and there is a word
gap. Both measured the alphabet rather than the camera.

`height_dev` is only safe for typeset lettering. Carmen's swash C and B stand 73%
and 54% over her median, which is her hand and not a defect, so `--source paper`
loosens the tolerance from 0.25 to 0.60.

---

## 11. Canada Dry, and a runtime without PyTorch (11 Sep 2026)

### Capture on a real label: three fixes, all found in the previews

`capture.py` on `IMG_1369.jpg` now reads **A C D N R Y** — six real letters, best
A scoring 0.73, lettering measured *very bold, monoline, red*.

- **"Background is the median colour" is false on a word crop.** The red letters
  and the lilac shield are near-equal in area, so the median fell *between* them
  and only the deepest-red edges survived. `ink_by_colour` now k-means clusters the
  crop (k=4, CIELAB with L flattened) and keeps the cluster whose mask segments
  into the number of letters `--text` says there are. We know that number; use it.
  `--ink-colour R,G,B` overrides when a picture is unusual.
- **The D and final A touch at a serif.** Same idea: fewer boxes than letters →
  `split_to_count` splits the widest box at the deepest column-ink valley in its
  middle half and repeats. The old width>1.4×height rule never fires on a wide
  serif pair (DA is 1.3×).
- **Two scoring bugs.** `clipped` compared against the crop edge — a `--bbox` puts
  letters at the crop edge by design — it now compares against the photo edge. And
  the ink-colour label came from a white-balanced copy: on a red-dominated crop
  the local mean *is* red, so red read "pale grey". It is read off the raw crop.

### stable-diffusion.cpp: the ggml path, and it already covers our models

[leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) runs
Qwen-Image-Edit 2511 and FLUX.2 klein 4B/9B in pure C/C++ on ggml — no PyTorch,
no ComfyUI. Installed at `D:\AI\sdcpp\sd` from release `master-853-b68d586`
(10 Sep 2026, `sd-master-b68d586-bin-win-cuda12-x64.zip`, 336 MB). The 563 MB
`cudart` bundle was **not** downloaded: the three DLLs it provides —
`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll` — were copied from the
torch wheel already in `vault-cacophony\.venv`. `sd-cli --list-devices` sees
`CUDA0 NVIDIA GeForce RTX 3060`.

Every model file the Qwen path needs was already on `D:`:

| role | file | size |
| --- | --- | --- |
| diffusion | `qwen-image-edit-2511-Q4_0.gguf` | 11.85 GB |
| text encoder (`--llm`) | `Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_S.gguf` | 4.46 GB |
| vision tower (`--llm_vision`) | `Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf` | 0.85 GB |
| VAE | `qwen_image_vae.safetensors` | 0.25 GB |
| Lightning | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` via `<lora:…:1>` | 0.85 GB |

Flags that matter, from the [qwen_image_edit doc](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_edit.md):
`--model-args qwen_image_zero_cond_t=true` (the 2511 mode — "quality will degrade
significantly" without it), `--flow-shift 3`, `-r` once per reference image
(image1 = composite, image2 = extra reference), `--offload-to-cpu`,
`--diffusion-fa`. A community sd-server report measured **19.5 GB VRAM without
offload**, so offload is mandatory on 12 GB — the same memory pressure that stopped
ComfyUI keeping the model resident. Do not expect Qwen to get faster here; expect
it to get *simpler*.

FLUX.2 klein ([flux2 doc](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux2.md)):
`--llm qwen_3_4b.safetensors` for 4B, `qwen_3_8b` for 9B, `--steps 4 --cfg-scale
1.0`, `-r` per reference. On disk: `flux-2-klein-4b-fp8.safetensors` (4.07 GB),
`flux-2-klein-9b-Q8_0.gguf` (9.98 GB), `qwen_3_4b.safetensors` (8.04 GB),
`qwen_3_8b_fp8mixed.safetensors` (8.66 GB). Whether sd.cpp loads ComfyUI's fp8 /
fp8mixed safetensors is being tested, not assumed.

**No smaller Qwen edit exists** — 20B is the only size; Q4_0 *is* the lite option.
Under 6B the credible open editors are klein 4B (on disk), VIBE (Qwen3-VL 2B +
Sana 1.6B, [arXiv 2601.02242](https://arxiv.org/html/2601.02242)) and Z-Image
Turbo 6B; klein is the only one with a ggml runtime today.

### First timing: klein 4B, and where the 127 s goes

One E on the Canada Dry references (image1 = plate with the real A pasted,
image2 = the real N), 1328x752, 4 steps, cfg 1.0, seed 77, `--offload-to-cpu
--diffusion-fa`:

| phase | klein 4B |
| --- | --- |
| load text encoder | 31.2 s |
| encode the prompt (qwen3 4B) | 35.9 s |
| load diffusion model | 38.2 s |
| **sampling, 4 steps** | **73.9 s** |
| VAE decode | 3.3 s |
| **wall** | **127 s** |

**Most of that is not sampling.** 69 s is reading weights off disk and 36 s is
text encoding, and both are paid *again* on every invocation of `sd-cli`. This is
the same cost that made ComfyUI re-dequantise for every glyph (README 5e memory
note) — a 26-glyph set pays it 26 times either way.

`sd-server` is the answer to that: it holds the model and exposes both an
A1111-style API (`/sdapi/v1/txt2img`, `/sdapi/v1/img2img`) and an OpenAI-style one
(`/v1/images/edits`, `/v1/images/generations`), found by scanning the binary's
strings. A resident klein 4B should cost ~75 s per glyph rather than 127 s, with
no reload — and unlike ComfyUI it is a single 336 MB binary holding one 4 GB model
instead of a Python stack holding 20 GB.

VRAM measured on the 3060, with weights held in RAM by `--offload-to-cpu`:
diffusion compute buffer 2.78 GB, VAE compute buffer 6.34 GB peak, text-encoder
compute 65 MB. The *params* sat in host RAM: 6.68 GB for the encoder, 3.88 GB for
the diffusion model. So the binding constraint on this machine is **32 GB of RAM,
not 12 GB of VRAM** — with ComfyUI's server also resident (~7 GB idle), free RAM
hit zero and the Qwen run had nothing left to offload into.

### Three-way result, one E, identical inputs and seed

| runtime + model | wall | encode | sampling | letterform | ink RGB | sat | drawn size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sd.cpp klein 4B | 127 s | 35.9 s | 73.9 s | no slab serifs | (142, 28, 37) | 0.77 | 339x347 |
| **sd.cpp klein 9B Q8_0** | **144 s** | **1.9 s** | 134.6 s | **slab serifs** | **(143, 28, 37)** | 0.77 | **323x329** |
| sd.cpp Qwen 2511 Q4_0 | 309 s | — | 256.2 s | correct slab serifs | (103, 18, 29) | 0.80 | 568x686 |
| ComfyUI Qwen 2511 Q4_0 | 163 s | — | — | serifs, but a fuzzy 3D object with a cast shadow | — | — | — |
| **the real label's A** | — | — | — | — | **(133, 38, 44)** | **0.71** | box is 319x361 |

Three things this says that were not obvious:

- **sd.cpp's Qwen beats ComfyUI's Qwen on the same model and seed.** The ComfyUI
  edit came back as a fuzzy three-dimensional object with a cast shadow; the sd.cpp
  edit is flat printed ink with clean edges. Same 2511 weights, same Lightning LoRA,
  same 4 steps. The difference is in how each applies 2511 mode
  (`--model-args qwen_image_zero_cond_t=true` versus the ComfyUI node graph).
- **klein 4B is the closest on colour**, (142,28,37) against the label's
  (133,38,44), where Qwen renders it markedly darker at (103,18,29).
- **klein respects the target box and Qwen does not.** The box is 319x361; klein
  drew 339x347, Qwen drew 568x686 — nearly twice too large. That is the composite
  sizing problem from README 5c, and it survives the change of runtime, so it is
  the model's behaviour and not ComfyUI's.

**klein 9B Q8_0 is the pick.** It gets the slab serifs that 4B misses, matches the
label's colour as closely as 4B does, and fits the target box better than anything
else tried — 323x329 against a box of 319x361, where Qwen overflows to 568x686.
It costs 17 s more than 4B and less than half of Qwen.

### The bf16 text encoder was a third of klein 4B's runtime

klein 9B encodes the prompt in **1.9 s**; klein 4B takes **35.9 s** for the same
prompt. The difference is not the model, it is the file: 9B was given
`Qwen3-8B-Q5_K_M.gguf` while 4B was given `qwen_3_4b.safetensors`, a bf16
ComfyUI file. A *larger* encoder in GGUF is roughly twenty times faster to run
than a smaller one in bf16 safetensors.

So klein 4B's 127 s is really ~93 s waiting to happen: pair it with a GGUF
Qwen3-4B and its wall time becomes sampling-bound like 9B's. Worth doing before
concluding anything about 4B versus 9B on speed.

Memory on the 3060 for klein 9B: diffusion compute buffer 3.64 GB VRAM (peak
across 34 segments), VAE compute 6.34 GB, params 13.58 GB held in host RAM.

### klein 9B is blocked on its text encoder, not on itself

`flux-2-klein-9b-Q8_0.gguf` (9.98 GB) loads fine — the failure is the encoder:

    GGML_ASSERT(scale_nelements == 1 || scale_nelements == out_features) failed

Isolated by swapping models rather than guessed: **klein 4B, which works, fails
identically when given `qwen_3_8b_fp8mixed.safetensors`.** That file carries 311
`weight_scale` tensors of shape `[]` — 0-dimensional scalars — which satisfy
neither branch of the assert. It is a ComfyUI-specific fp8 layout. Feeding klein 9B
the 4B encoder instead fails differently and correctly, on
`GGML_ASSERT(ggml_can_mul_mat(a, b))`: Qwen3-4B and Qwen3-8B have different hidden
sizes, so the 9B really does need an 8B encoder.

The [unsloth klein-9B discussion](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/discussions/6)
reports the same shape mismatch for both `qwen_3_8b_fp8mixed` and
`qwen_3_8b_fp4mixed`, and points at GGUF instead. `Qwen3-8B-Q5_K_M.gguf` from
unsloth is the file to pair with klein 9B.
