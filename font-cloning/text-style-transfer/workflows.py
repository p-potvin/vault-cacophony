"""API-format ComfyUI graphs for reference-styled text writing with Qwen-Image-Edit-2511.

The graph follows ComfyUI's own `image_qwen_image_edit_2511` template exactly
(read from the installed comfyui-workflow-templates package, not guessed):

    UNETLoader -> ModelSamplingAuraFlow(shift 3.1) -> CFGNorm(1.0) [-> Lightning LoRA]
    CLIPLoader(type=qwen_image)  VAELoader
    TextEncodeQwenImageEditPlus(clip, prompt, vae, image1, image2, image3)
        -> FluxKontextMultiReferenceLatentMethod(index_timestep_zero)      # the "2511 mode"
    negative = same with prompt ""
    latent   = VAEEncode(image1)  [-> SetLatentNoiseMask(mask) only if latent_mask=True]
    KSampler(euler, simple) -> VAEDecode [-> ImageCompositeMasked(original, decoded, mask)] -> SaveImage

Region control: the default is pixel-space only -- the edit model changes what
the instruction asks for, and ImageCompositeMasked pastes the result back onto
the untouched original so everything outside the (feathered) box is identical.
`latent_mask` additionally freezes the latent outside the box during sampling.

Sampler settings from the template: full model 40 steps @ cfg 3.0; Lightning
4-step LoRA 4 @ 1.0. Default here: unsloth Q4_0 GGUF + the 4-step Lightning LoRA,
4 steps @ cfg 1.0.

Sizing: we pre-scale in PIL to a multiple of 16 instead of using
FluxKontextImageScale, so the decoded image has exactly the input size and can
be composited back onto the untouched original pixel-for-pixel.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelSet:
    # Default = known-good GGUF (unsloth) + the on-disk 4-step Lightning LoRA.
    # The fp8 "scaled" safetensors on this box lacks its scale tensors and emits
    # noise (README section 5); it stays selectable only for reproduction.
    unet: str = "qwen-image-edit-2511-Q4_0.gguf"
    gguf: bool = True                # True -> UnetLoaderGGUF (ComfyUI-GGUF), False -> UNETLoader
    clip: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    vae: str = "qwen_image_vae.safetensors"
    lora: str | None = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    # Set lora=None and raise steps/cfg for maximum fidelity to the reference; the
    # 4-step distillation is the main source of "creative" deviation (README 5d).
    lora_strength: float = 1.0
    controlnet: str = "qwen-image-instantX-inpainting-controlnet.safetensors"
    steps: int = 4
    cfg: float = 1.0
    shift: float = 3.1
    sampler: str = "euler"
    scheduler: str = "simple"
    weight_dtype: str = "default"


@dataclass
class EditRequest:
    image1: str                         # uploaded name of the image being edited
    prompt: str
    refs: list[str] = field(default_factory=list)   # up to 2 more uploaded names -> image2/image3
    mask: str | None = None             # uploaded grayscale mask (white = edit region)
    latent_mask: bool = False           # also confine denoising via SetLatentNoiseMask (opt-in)
    composite_in_graph: bool = False    # False -> return the raw decode, composite client-side
    original: str | None = None         # uploaded untouched target for exact composite-back
    seed: int = 0
    denoise: float = 1.0
    use_inpaint_controlnet: bool = False
    controlnet_strength: float = 1.0
    negative: str = ""
    filename_prefix: str = "tst/text_style"


def build_graph(req: EditRequest, m: ModelSet = ModelSet()) -> dict:
    g: dict[str, dict] = {}
    n = 0

    def add(class_type: str, **inputs) -> str:
        nonlocal n
        n += 1
        nid = str(n)
        g[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    def out(nid: str, slot: int = 0):
        return [nid, slot]

    # -- models ------------------------------------------------------------------
    if m.gguf:
        unet = add("UnetLoaderGGUF", unet_name=m.unet)
    else:
        unet = add("UNETLoader", unet_name=m.unet, weight_dtype=m.weight_dtype)
    model = add("ModelSamplingAuraFlow", model=out(unet), shift=m.shift)
    model = add("CFGNorm", model=out(model), strength=1.0)
    if m.lora:
        model = add("LoraLoaderModelOnly", model=out(model), lora_name=m.lora,
                    strength_model=m.lora_strength)
    clip = add("CLIPLoader", clip_name=m.clip, type="qwen_image", device="default")
    vae = add("VAELoader", vae_name=m.vae)

    # -- images ------------------------------------------------------------------
    img1 = add("LoadImage", image=req.image1)
    ref_nodes = [add("LoadImage", image=r) for r in req.refs[:2]]

    def enc(prompt: str) -> str:
        kw = {"clip": out(clip), "prompt": prompt, "vae": out(vae), "image1": out(img1)}
        for i, rn in enumerate(ref_nodes, start=2):
            kw[f"image{i}"] = out(rn)
        c = add("TextEncodeQwenImageEditPlus", **kw)
        return add("FluxKontextMultiReferenceLatentMethod", conditioning=out(c),
                   reference_latents_method="index_timestep_zero")

    pos = enc(req.prompt)
    neg = enc(req.negative)

    latent = add("VAEEncode", pixels=out(img1), vae=out(vae))

    mask_node = None
    if req.mask:
        mimg = add("LoadImage", image=req.mask)
        mask_node = add("ImageToMask", image=out(mimg), channel="red")
        if req.latent_mask:
            latent = add("SetLatentNoiseMask", samples=out(latent), mask=out(mask_node))
        if req.use_inpaint_controlnet:
            cn = add("ControlNetLoader", control_net_name=m.controlnet)
            cn_apply = add("ControlNetInpaintingAliMamaApply",
                           positive=out(pos), negative=out(neg), control_net=out(cn),
                           vae=out(vae), image=out(img1), mask=out(mask_node),
                           strength=req.controlnet_strength, start_percent=0.0, end_percent=1.0)
            pos, neg = (cn_apply, 0), (cn_apply, 1)
        else:
            pos, neg = (pos, 0), (neg, 0)
    else:
        pos, neg = (pos, 0), (neg, 0)

    # -- sample ------------------------------------------------------------------
    ks = add("KSampler", model=out(model), seed=req.seed, steps=m.steps, cfg=m.cfg,
             sampler_name=m.sampler, scheduler=m.scheduler,
             positive=list(pos), negative=list(neg), latent_image=out(latent),
             denoise=req.denoise)
    decoded = add("VAEDecode", samples=out(ks), vae=out(vae))

    final = decoded
    if mask_node is not None and req.composite_in_graph:
        base = add("LoadImage", image=req.original) if req.original else img1
        final = add("ImageCompositeMasked", destination=out(base), source=out(decoded),
                    x=0, y=0, resize_source=False, mask=out(mask_node))

    add("SaveImage", images=out(final), filename_prefix=req.filename_prefix)
    return g


# -- FLUX.2 Klein backend --------------------------------------------------------
#
# Graph lifted from ComfyUI's `image_flux2_klein_image_edit_4b_base` /
# `image_flux2_klein_9b_kv_image_edit` templates. Klein edits by conditioning on
# reference latents and sampling from an EMPTY latent of the target size:
#
#   UNETLoader  CLIPLoader(type=flux2, qwen_3_4b)  VAELoader(flux2-vae)
#   pos = CLIPTextEncode(prompt); neg = ConditioningZeroOut(pos)          (distilled, cfg 1)
#   for each image: lat = VAEEncode(image); pos = ReferenceLatent(pos, lat); neg = ReferenceLatent(neg, lat)
#   SamplerCustomAdvanced(RandomNoise, CFGGuider(model,pos,neg,cfg), KSamplerSelect(euler),
#                         Flux2Scheduler(steps,W,H), EmptyFlux2LatentImage(W,H))
#   VAEDecode [-> ImageCompositeMasked(original, decoded, mask)] -> SaveImage

@dataclass
class KleinModelSet:
    unet: str = "flux-2-klein-4b-fp8.safetensors"       # distilled: 4 steps, cfg 1
    clip: str = "qwen_3_4b.safetensors"
    vae: str = "flux2-vae.safetensors"
    steps: int = 4
    cfg: float = 1.0
    sampler: str = "euler"
    weight_dtype: str = "default"


def build_graph_klein(req: EditRequest, size: tuple[int, int], m: KleinModelSet = KleinModelSet()) -> dict:
    """`size` = (width, height) of image1 (already a multiple of 16)."""
    g: dict[str, dict] = {}
    n = 0

    def add(class_type: str, **inputs) -> str:
        nonlocal n
        n += 1
        g[str(n)] = {"class_type": class_type, "inputs": inputs}
        return str(n)

    def out(nid: str, slot: int = 0):
        return [nid, slot]

    model = add("UNETLoader", unet_name=m.unet, weight_dtype=m.weight_dtype)
    clip = add("CLIPLoader", clip_name=m.clip, type="flux2", device="default")
    vae = add("VAELoader", vae_name=m.vae)

    img1 = add("LoadImage", image=req.image1)
    imgs = [img1] + [add("LoadImage", image=r) for r in req.refs[:2]]

    pos = add("CLIPTextEncode", text=req.prompt, clip=out(clip))
    neg = (add("CLIPTextEncode", text=req.negative, clip=out(clip)) if req.negative
           else add("ConditioningZeroOut", conditioning=out(pos)))
    for im in imgs:
        lat = add("VAEEncode", pixels=out(im), vae=out(vae))
        pos = add("ReferenceLatent", conditioning=out(pos), latent=out(lat))
        neg = add("ReferenceLatent", conditioning=out(neg), latent=out(lat))

    w, h = size
    noise = add("RandomNoise", noise_seed=req.seed)
    guider = add("CFGGuider", model=out(model), positive=out(pos), negative=out(neg), cfg=m.cfg)
    sampler = add("KSamplerSelect", sampler_name=m.sampler)
    sigmas = add("Flux2Scheduler", steps=m.steps, width=w, height=h)
    empty = add("EmptyFlux2LatentImage", width=w, height=h, batch_size=1)
    samp = add("SamplerCustomAdvanced", noise=out(noise), guider=out(guider), sampler=out(sampler),
               sigmas=out(sigmas), latent_image=out(empty))
    decoded = add("VAEDecode", samples=out(samp, 0), vae=out(vae))

    final = decoded
    if req.mask:
        mimg = add("LoadImage", image=req.mask)
        mask = add("ImageToMask", image=out(mimg), channel="red")
        base = add("LoadImage", image=req.original) if req.original else img1
        final = add("ImageCompositeMasked", destination=out(base), source=out(decoded),
                    x=0, y=0, resize_source=False, mask=out(mask))
    add("SaveImage", images=out(final), filename_prefix=req.filename_prefix)
    return g
