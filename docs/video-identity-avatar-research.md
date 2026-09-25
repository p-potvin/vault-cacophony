# Video to identity to avatar: what fits this machine

Sun, 13 Sep 2026

Research pass after the local VSS pipeline landed. The question was which
pieces of "tag people in videos, pick exemplars, rebuild the face, animate
it" are doable on the RTX 3060 12 GB, and where the free inference budgets
(HF PRO credits, NVIDIA API catalog) actually help. Everything below was
checked against what is on disk and reachable from this box today.

## Inventory

| Thing | State |
|---|---|
| RTX 3060 12 GB, Ryzen 5 7600X, 31 GB RAM | free disk: D 91 GB, E 155 GB, F 214 GB, G 139 GB |
| colonel-kfc | InsightFace antelopev2 (SCRFD-10G, GlintR100 512-d, 2D-106 landmarks), SQLite gallery, `reconstruct_face_mesh.py` (sparse 68-point median mesh), `docs/face-reconstruction-2026.md` already recommends MICA-to-FLAME |
| colonel-kfc/trellis.cpp | ggml port of TRELLIS.2-4B image-to-3D, built for sm86 (this card), weights on D: (LocalAI-io GGUFs) |
| vaultwares-studio | remote-first 3DGS (COLMAP or MASt3R then splatfacto on HF Jobs), OpenUSD export incl. `UsdVol.ParticleField3DGaussianSplat`; `cosmos-reason2/` is a checkout of NVIDIA's repo, no weights; roadmap M4 puts Cosmos annotation on a remote 24 GB card |
| D:\3D Reconstruction | local MASt3R-SfM works on the 3060 (500 frames, 100 min); da3-gsplat build attempts |
| Bonsai | only the demo repo under D:\llama\Bonsai-demo; no GGUF downloaded |
| FLAME 2020 assets, MICA, DECA, PyTorch3D | none on disk (searched D: and the repos) |
| Ollama | qwen3-vl:2b, gemma4:e2b, minicpm-v4.6, llama3.2:3b, qwen3:0.6b personas |
| HF PRO | `clopeux`, isPro, prepaid billing, $2/month Inference Providers credit; router works (142 models, tested Qwen3.5-9B via together) |
| NVIDIA API catalog | `nvapi-` key works against `integrate.api.nvidia.com`; 82 models; tested `nvidia/nemotron-3-ultra-550b-a55b` (HTTP 200) |

## The two free budgets

**HF PRO, $2/month.** Pass-through provider pricing, no markup, credit then
pay-as-you-go. At DeepSeek-V4-Flash or Qwen3.5-9B rates that is roughly 1 to
4 M tokens a month; at Kimi-K3 or GLM-5.3 closer to 0.5 M. It also serves
`prism-ml/Ternary-Bonsai-27B-gguf` through together, and vision models
(`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, `google/gemma-4-31B-it`). No
billing API exposes the remaining credit; the settings page does.

**NVIDIA catalog, free tier.** About 1000 credits at sign-up, one per call,
40 requests a minute, and NVIDIA does not raise it on request. The catalog
has `nemotron-3-ultra-550b-a55b`, `nemotron-3-super-120b-a12b`,
`cosmos-reason2-8b`, `nemotron-3-nano-omni-30b-a3b-reasoning`, Llama 3.2 90B
vision, and `vila`. The unit is calls, not tokens, so it rewards few, large
requests: one 100 k-token summary costs the same as one "say ok".

Where they plug into what exists:

- `scripts/vss_pipeline.py` already talks OpenAI-shaped chat. Pointing the
  aggregation at either endpoint is `--ollama-url` plus a bearer header, a
  ten-line change. The captions stay local (images are the expensive part);
  the two merge passes and the final summary go remote. That is the highest
  quality per credit: Nemotron Ultra reads 200 captions in one call.
- Cosmos-Reason2-8B on the NVIDIA catalog is the VSS VLM without the 20 GB.
  One call per chunk burns the free tier in 1000 chunks, so use it for the
  `ask` path and for re-captioning the chunks BM25 flagged, not for every
  chunk.
- The HF credit is the right place for the text-only embedding leg VSS calls
  rt-embed, if search ever needs to be semantic rather than BM25.

## Bonsai

Ternary-Bonsai-27B is a Qwen3.6-27B derivative at 1.58 bits: 7.17 GB PQ2_0 on
the PrismML llama.cpp fork, or `Ternary-Bonsai-27B-Q2_g64.gguf` on upstream
llama.cpp and Ollama (`ollama run hf.co/prism-ml/Ternary-Bonsai-27B-gguf`).
262 k context, a 0.63 GB mmproj for vision, 94.6% of the parent's benchmark
average. It fits the 3060 beside qwen3-vl:2b, and it is a 27B-class
summarizer where llama3.2:3b is the one now merging captions. Worth pulling
as the default `llm_model` in `config/vss/lvs-local.yaml`; the vision tower
makes it a candidate VLM too, at 8 frames per chunk it will be slower than the
2B and the KV budget at 16 k should be re-measured.

## Faces: recognition to reconstruction to animation

Three distinct stages, and the machine handles them very differently.

### 1. Who is in the video, and which frames to keep

Already solved by colonel-kfc's stack and it is all ONNX on CUDA 12.6:
SCRFD detection, GlintR100 embeddings, the SQLite gallery. What is missing
is the video front end, and the VSS pipeline just built it: chunk plan,
frame extraction, a per-chunk record in the tag store. Run SCRFD over the
sampled frames, cluster GlintR100 embeddings per track, and write a
`face` track beside `caption` and `speaker`. Exemplar picking is then a
query: frames where the identity's embedding is closest to its centroid,
the 2D-106 landmarks say the yaw is under some threshold, and the
detection box is largest. Fully local, no new models, and this is the part
that makes `search "person X"` work across a media folder.

### 2. Rebuilding the face

`docs/face-reconstruction-2026.md` in colonel-kfc already picked the
answer, and nothing in 2026 changes it: MICA to FLAME for a metric identity
mesh, fused over 8 to 12 exemplars, with DECA/EMOCA/SMIRK as per-frame
expression estimators. What is new is the cost on this box:

| Route | Needs | Fits the 3060? | Windows? |
|---|---|---|---|
| MICA + FLAME (identity mesh) | FLAME 2020 assets (registration, non-commercial), MICA checkpoint, PyTorch3D | yes, under 4 GB | PyTorch3D on Windows is a source build; the usual blocker |
| flame-head-tracker (per-frame FLAME params from video) | FLAME 2020, DECA, MICA, MediaPipe; 8 GB VRAM minimum | yes; 0.9 s/frame landmark mode, 1.9 s/frame photometric on a 4090, so 3 to 5 s/frame here | author: untested on Windows |
| GAGAvatar (one image to drivable Gaussian head) | FLAME, custom diff-gaussian-rasterization, EMICA, VGGHead | yes | bash + conda install, no Windows notes |
| LAM (one image to Gaussian head in 1.4 s, WebGL player, 100+ FPS) | its own assets; Apache-2.0; one-click Windows installer | yes | yes, the only one with a Windows path |
| trellis.cpp (image to textured mesh, ggml) | already built and weights on D: | yes, sm86 build exists | yes |

The honest constraint is not VRAM, it is that every FLAME-based tool is a
Linux research repo with PyTorch3D and a registration-gated model file.
LAM is the exception and is also the only one that hands back something
animatable and viewable (WebGL, OpenAvatarChat) without further work.
trellis.cpp gives a mesh from one exemplar today with zero setup, but a
generic mesh, not a face model: no rig, no identity coefficients.

### 3. Animating it

Two realistic outputs. Gaussian heads (LAM, GAGAvatar, GAF) animate from a
driving video or from audio through an Audio2Expression model, and they are
the same primitive vaultwares-studio already exports to USD via the
`ParticleField3DGaussianSplat` schema. So an avatar head could sit inside a
splat room from the same pipeline, in USD. A FLAME mesh animates by
parameters and plays in any DCC, but looks like a mask.

fVDB (OpenVDB's PyTorch extension, NanoVDB underneath) is where the
Gaussian-splat work in NVIDIA's ecosystem is heading: fvdb-reality-capture
trains splats from sensor data with USD export and claims 50% better
throughput than gsplat. But `fvdb-core` 0.5.1 on conda-forge ships
`linux-64` and `linux-aarch64` only, no `win-64`, and the studio's own
roadmap rules out WSL2. On this machine fVDB is a remote-worker technology,
exactly where the studio already runs splatfacto. OpenVDB/NanoVDB
themselves are not the tool for faces; their value here is the volumetric
container, and the Gaussian schema already covers that.

## What is doable now, in order

1. **Face track in the tag store.** colonel-kfc's detector and embedder over
   the VSS chunk frames; a `face` track, identity clusters, exemplar
   scoring. Local, a day of work, and it turns the media folder into a
   people index. This is the step everything else needs and the one with
   no model risk.
2. **Bonsai as the local summarizer**, and the remote endpoints behind a
   flag for the merge passes. Spends the two free budgets on the calls
   that benefit from a big model.
3. **LAM for an animatable head from the best exemplar.** The one route
   with a Windows installer, a viewer, and an export the studio can place
   in USD. Measure VRAM and reconstruction time on the 3060 before
   committing; the paper numbers are A100.
4. **MICA/FLAME for the metric identity mesh**, when the biometric use in
   colonel-kfc needs it. Budget for FLAME registration and a PyTorch3D
   build, or run it in the HF Jobs worker the studio already has, where it
   is a Linux container and none of the Windows friction applies.
5. **fVDB, only remote.** Swap splatfacto for fvdb-reality-capture in the
   worker image when its USD export matters; not on this PC.

Not recommended: Cosmos 3 locally (Nano is 16B, two 8B halves; Edge is 4B
and aimed at robots), FaceLift or any multi-view diffusion route for
identity work (hallucinates views), and any attempt to port fVDB to Windows.

## Sources

- HF Inference Providers pricing: https://huggingface.co/docs/inference-providers/pricing
- NVIDIA catalog free tier: https://forums.developer.nvidia.com/t/api-credit-rate-limit-increase-request-for-build-nvidia-com-free-tier/382934
- Ternary-Bonsai-27B GGUF: https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf
- flame-head-tracker: https://github.com/peizhiyan/flame-head-tracker
- GAGAvatar: https://github.com/xg-chu/GAGAvatar
- LAM: https://github.com/aigc3d/LAM
- GAF: https://tangjiapeng.github.io/projects/GAF/
- FLAME resources: https://github.com/TimoBolkart/FLAME-Universe
- fVDB: https://fvdb.ai/ and https://anaconda.org/conda-forge/fvdb-core
- fvdb-reality-capture: https://github.com/openvdb/fvdb-reality-capture
- Cosmos 3: https://huggingface.co/blog/nvidia/cosmos-3-for-physical-ai
