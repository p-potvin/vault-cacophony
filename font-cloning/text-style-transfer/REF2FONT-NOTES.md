# What is worth taking from Ref2Font

[SnJake/Ref2Font](https://github.com/SnJake/Ref2Font) is a contextual LoRA for
**FLUX.2 Klein 9B** that turns a single `Aa` reference into a 1280×1280 atlas of
the whole charset, then vectorises it to a TTF. Read 6 Sep 2026 from the clone in
`font-cloning/Ref2Font` plus the upstream repo.

It solves a *different* problem to ours (it makes font files; we write styled text
onto photos) and its inputs are explicitly **strict black & white, no gray, no
shadows, no volume** — the exact opposite of the texture/depth case where our
pipeline is strongest. So the repo is not directly reusable. Three things in it
are.

---

## 1. It proves Klein can copy letterform style from a reference

That was an open question. It answers it with a rank-64 LoRA and ~7.4 h of
training, which is a tractable budget on one consumer GPU.

## 2. The full training recipe, recovered from the safetensors metadata

| | |
| --- | --- |
| base | `flux-2-klein-base-9b.safetensors` (**base**, not the instruct/edit variant) |
| trainer | `networks.lora_flux_2` (sd-scripts / musubi lineage) |
| rank / alpha | **64 / 32** |
| lr / optimiser / schedule | **2e-4**, `torch.optim.adamw.AdamW`, cosine, 0 warmup |
| steps | **1500**, 1 epoch, grad-accum 1 |
| batch | 4 per device |
| precision | bf16 mixed, **fp8 base**, gradient checkpointing on |
| timestep sampling | `flux2_shift`, discrete flow shift 1.0, guidance 1.0 |
| resolution | 1280×1280, bucketing on |
| wall clock | **7.41 h** |
| adapted modules | `img_attn_qkv/proj`, `img_mlp_0/2`, `txt_attn_qkv/proj`, `txt_mlp_0/2`, `linear1/2` — 112 down-projections |

Dataset config is the important part:

```json
{"resolution": [1280,1280], "batch_size_per_device": 4, "num_repeats": 1,
 "enable_bucket": true, "image_directory": "targets",
 "control_directory": "controls", "has_control": true}
```

`has_control: true` — this is a **control→target image-to-image LoRA**. The
reference is the control, the atlas is the target. That is structurally the same
shape as our task, which is why the recipe transfers even though the content does
not.

`scripts/generate_flux_atlas_dataset.py` builds pairs entirely procedurally:
index local `.ttf`/`.otf` files, render `Aa` as the control and the charset grid
as the target, attach one fixed caption. No manual annotation anywhere.

## 3. Inference settings we had wrong for Klein

Their workflow runs the **base** Klein 9B with no distillation LoRA:

| setting | Ref2Font | what `build_graph_klein` used |
| --- | --- | --- |
| steps | **35** | 4 |
| cfg | **5.0** | 1.0 |
| negative prompt | *"jpeg compression, jpeg artifacts, bad quality, poor details, low quality, low bitrate artifacts, noise…"* | empty |
| `weight_dtype` | `fp8_e4m3fn` | `default` |
| text encoder | `qwen_3_8b_fp8mixed` | `qwen_3_4b` |

Our Klein run used the *distilled* settings (4 steps, cfg 1, empty negative) on a
non-distilled model. That is very likely why its letterforms came out blocky, and
it means the earlier "Klein is weaker at letterforms" conclusion is **not safe** —
it was tested at the wrong settings.

---

## The interesting idea: their formulation without their constraints

Ref2Font's `Aa` requirement is not a property of the method, it is just the
cheapest control image they could render from a font file. The same is true of the
black-and-white rule: you cannot render volume or texture from a `.ttf` alone, so
their dataset could not contain any.

We are not limited that way, because we already generate styled lettering
procedurally — `examples/style3d_gold.jpg` and `style3d_texture.jpg` were built by
`extruded()` with extrusion, bevel, specular, contour, cast shadow and photographic
face fill.

So a better-shaped dataset for our task, using their exact training recipe:

* **control** = word A rendered in style S (any word, any length, full colour,
  texture and depth allowed)
* **target** = word **B ≠ A** in the same style S, composited onto a random photo
  background

Three of our known problems fall out of that construction rather than needing to be
fixed separately:

1. **No `Aa` constraint** — the control is any word, because we sample it.
2. **No reference-word bleed** — every training pair has control word ≠ target
   word, so copying the control's letters is precisely what the loss punishes.
   This is the one open defect of the current pipeline (README §5a) and it is
   structural, not promptable.
3. **Texture and depth are in-distribution**, not excluded.

Cyrillic support is irrelevant to us and can be dropped; the charset is whatever we
sample.

Cost estimate by analogy: rank 64, 1500 steps at 1280² took them 7.4 h. Our target
sizes are ~1 MP, so comparable or cheaper. The unknown is their GPU — `fp8_base`
plus gradient checkpointing suggests consumer hardware, but the RTX 3060's 12 GB is
tight for a 9B and would likely need lower resolution or rank.

**Not started.** Recorded so the option is costed, not because it is queued.

---

## Files worth reading, none worth vendoring

| upstream path | why |
| --- | --- |
| `scripts/generate_flux_atlas_dataset.py` | the procedural control/target pair builder — the pattern to copy |
| `scripts/index_local_fonts.py` | walks local fonts and records glyph coverage |
| `Example Workflow/Flux 2 Klein 9B Fonts.json` | the corrected Klein inference settings above |
| `flux_grid_to_ttf.py`, `flux_pipeline.py` | atlas → TTF vectorisation; only relevant to the **parked** font-file track |

The clone can stay where it is as a reference; nothing needs to be copied into
`text-style-transfer/`.
