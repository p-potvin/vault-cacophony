"""Write a short text onto a target picture in the exact visual style of text in
another picture.

    style image  (has text in style S)  +  target image  +  new text  ->  target with text in S

Three strategies, all Qwen-Image-Edit-2511 through ComfyUI (see workflows.py):

  composite   Paste the style text crop into the target's bbox, then ask the model
              to REPLACE that text with the new string. Leans on 2511's strongest
              documented behaviour (font/colour/shadow-preserving text replacement).
              The result is composited back onto the untouched target through a
              feathered bbox mask (add --latent-mask to also freeze the latent
              outside the box during sampling).                             [default]
  multiref    image1 = clean target, image2 = style crop; the prompt asks for the
              text to be written at a described position in image2's style.
              Optional bbox mask for region constraint.
  inpaint-cn  `multiref` + InstantX inpainting ControlNet. The ControlNet was
              trained on base Qwen-Image, not Edit-2511 -> experimental.

Usage:
    ..\\..\\.venv\\Scripts\\python.exe pipeline.py --style examples\\style_summer.jpg ^
        --target examples\\target_waves.jpg --text "HELLO" --bbox 0.15,0.35,0.70,0.30 ^
        --out out\\hello.png
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_client import ComfyClient  # noqa: E402
from workflows import EditRequest, KleinModelSet, ModelSet, build_graph, build_graph_klein  # noqa: E402

MULT = 16  # Qwen-Image VAE stride 8 x patch 2


# -- geometry helpers -----------------------------------------------------------

def fit_megapixels(img: Image.Image, mp: float = 1.0) -> Image.Image:
    """Scale to ~mp megapixels with both sides a multiple of MULT."""
    w, h = img.size
    s = (mp * 1_000_000 / (w * h)) ** 0.5
    nw = max(MULT, round(w * s / MULT) * MULT)
    nh = max(MULT, round(h * s / MULT) * MULT)
    return img.resize((nw, nh), Image.LANCZOS)


def parse_box(spec: str | None, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """'x,y,w,h' either as 0-1 fractions or absolute pixels -> absolute (x0,y0,x1,y1)."""
    if not spec:
        return None
    x, y, w, h = [float(v) for v in spec.split(",")]
    W, H = size
    if max(x, y, w, h) <= 1.0:
        x, y, w, h = x * W, y * H, w * W, h * H
    x0, y0 = int(max(0, x)), int(max(0, y))
    x1, y1 = int(min(W, x + w)), int(min(H, y + h))
    return x0, y0, x1, y1


def bbox_mask(size: tuple[int, int], box, grow: int = 24, feather: int = 16) -> Image.Image:
    m = Image.new("L", size, 0)
    x0, y0, x1, y1 = box
    from PIL import ImageDraw
    ImageDraw.Draw(m).rectangle((x0 - grow, y0 - grow, x1 + grow, y1 + grow), fill=255)
    if feather:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return m


def diff_mask(original: Image.Image, generated: Image.Image, box=None,
              threshold: int = 45, grow: int = 12, feather: float = 2.0,
              smooth: float = 6.0, pad: int = 70) -> Image.Image:
    """Mask covering the lettering the model drew, with a clean edge.

    Not a plain pixel-diff threshold: the generated image is a full VAE
    round-trip, and on detailed backgrounds the round-trip alone shifts pixels by
    p99~127, which no magnitude threshold can separate from real lettering. The
    lettering is distinguishable by *structure* instead -- it is one large
    connected region, while round-trip error is scattered speckle -- so the diff
    is blurred by `smooth` before thresholding.

    `box` (x0,y0,x1,y1), padded by `pad`, bounds the mask. The model was measured
    overrunning the requested box by ~50 px in both axes, so the padding must be
    generous; its only job is to reject far-field noise, not to shape the glyphs.

    `feather` stays small: enough to anti-alias the seam, not enough to fade the
    lettering out the way the old 16 px feathered bbox did.
    """
    import numpy as np
    a = np.asarray(original.convert("RGB"), np.int16)
    b = np.asarray(generated.convert("RGB"), np.int16)
    d = np.abs(a - b).max(axis=2).astype(np.uint8)

    dm = Image.fromarray(d, mode="L")
    if smooth:
        dm = dm.filter(ImageFilter.GaussianBlur(smooth))
    m = np.asarray(dm, np.uint8) > threshold

    if box is not None:
        keep = np.zeros(m.shape, bool)
        x0, y0, x1, y1 = box
        # Scale the padding with the box. A fixed 70 px was tuned for a wide word
        # box; for a small single-letter box the reference is scaled up more and
        # the glyph overruns proportionally further, so a constant pad clips it.
        p = max(pad, int(0.35 * min(x1 - x0, y1 - y0)))
        keep[max(0, y0 - p):y1 + p, max(0, x0 - p):x1 + p] = True
        m &= keep

    out = Image.fromarray(m.astype(np.uint8) * 255, mode="L")
    r = int(grow)
    while r > 0:                       # MaxFilter only accepts odd sizes <= 9 reliably
        step = min(4, r)
        out = out.filter(ImageFilter.MaxFilter(2 * step + 1))
        r -= step
    if feather:
        out = out.filter(ImageFilter.GaussianBlur(feather))
    return out


def flatten_ref(img: Image.Image, bg=(30, 28, 44)) -> Image.Image:
    """RGB copy for the uploaded style reference (image2).

    The alpha version exists for compositing onto the target; the reference the
    model *looks at* needs an opaque ground to be read against.

    `bg` matters more than it looks. Matting dark pencil onto a dark ground tells
    the model the style is *light strokes on dark paper*, and it duly returned an
    inverted glyph on a dark panel. Pass the target's own colour so the reference
    and the composite agree about which is ink and which is paper.
    """
    if img.mode != "RGBA":
        return img.convert("RGB")
    flat = Image.new("RGB", img.size, bg)
    flat.paste(img.convert("RGB"), mask=img.getchannel("A"))
    return flat


def paste_fit(target: Image.Image, crop: Image.Image, box,
              fit: str = "height", zoom: float = 1.0) -> Image.Image:
    """Paste `crop` into `box`, aspect preserved, centred.

    `fit` decides what sets the size, and it matters: the model replaces the pasted
    lettering *at the size it was pasted*. The original min() rule made a 4-letter
    reference (aspect 1.78) in a wide box (aspect 4.54) height-limited, so it filled
    only 39% of the box width and the output came out small to match.

      height  match the box height - the reference's cap height sets the scale
      width   match the box width  - the word fills the box horizontally
      box     min(w, h), the old behaviour
    """
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ratios = {"height": bh / crop.height,
              "width": bw / crop.width,
              "box": min(bw / crop.width, bh / crop.height)}
    if fit not in ratios:
        raise SystemExit(f"unknown paste fit {fit!r}; pick one of {sorted(ratios)}")
    s = ratios[fit] * max(0.05, zoom)
    nw, nh = max(1, int(crop.width * s)), max(1, int(crop.height * s))
    c = crop.resize((nw, nh), Image.LANCZOS)
    out = target.copy()
    cx, cy = x0 + bw // 2, y0 + bh // 2
    pos = (cx - nw // 2, cy - nh // 2)
    # An RGBA crop pastes only its lettering, so the reference's own background
    # does not travel into the scene (the edit model would keep it otherwise).
    out.paste(c.convert("RGB"), pos, c.getchannel("A") if c.mode == "RGBA" else None)
    return out


# -- prompts --------------------------------------------------------------------

def _spelled(text: str) -> str:
    """'HELLO' -> 'H-E-L-L-O'. Spelling it out curbs word-level bleed from the reference."""
    return "-".join(text)


# The style clause only. Head and tail are shared, so these are directly
# comparable; see README section 8 for what each one scored.
PROMPT_STYLES: dict[str, str] = {
    "default": ("Keep exactly the same font, letter shapes, colors, gradient, outline, "
                "shadow and lettering style as the existing text."),
    "minimal": "Keep exactly the same lettering style as the existing text.",
    "pen": ("Write it by hand with the same pen and the same handwriting as the "
            "existing text."),
    "erase": ("Erase the old letter completely so no trace of it remains, then write "
              "the new one in its place in exactly the same lettering style."),
}


def _style_clause(style: str | None, constraints: str | None) -> tuple[str, str]:
    """(what to keep, what to forbid) - measured wording when we have it.

    The default wording lists "gradient, outline, shadow", which is right for the
    display lettering this started with and actively harmful for a pen: it names
    four effects the reference does not have, and the model duly adds them.
    features.StrokeFeatures.describe()/constraints() supply the replacement.
    """
    if not style:
        return (PROMPT_STYLES["default"], "")
    return (f"Keep exactly the same font, letter shapes, colours and lettering style as "
            f"the existing text, which is {style}.", (constraints or "").strip())


def prompt_composite(text: str, style: str | None = None,
                     constraints: str | None = None, template: str = "default") -> str:
    keep, forbid = _style_clause(style, constraints)
    if style is None:
        if template not in PROMPT_STYLES:
            raise SystemExit(f"unknown prompt template {template!r}; "
                             f"choose from {sorted(PROMPT_STYLES)}")
        keep = PROMPT_STYLES[template]
    return (f'Replace the text in image 1 with the word "{text}". '
            f'The new text must read exactly "{text}" ({_spelled(text)}) and nothing else. '
            f"{keep} {forbid} Blend the new letters naturally into "
            f"the background. Do not change anything else in the image.").replace("  ", " ")


def prompt_multiref(text: str, where: str, style: str | None = None,
                    constraints: str | None = None) -> str:
    if style:
        copy = (f"Copy ONLY the visual lettering style from image 2 - its font, letter "
                f"shapes and colours. The lettering is {style}. {constraints or ''}")
    else:
        copy = ("Copy ONLY the visual lettering style from image 2 - its font, letter "
                "shapes, colors, gradient, outline and shadow.")
    return (f'Write the word "{text}" on image 1, {where}. '
            f'The text must read exactly "{text}" ({_spelled(text)}) and nothing else. '
            f"{copy} Do NOT copy any word, letter or content "
            f"from image 2, and do not place image 2 into the picture. "
            f"Keep everything else in image 1 unchanged.").replace("  ", " ")


# -- main -----------------------------------------------------------------------

def run(style_path: str, target_path: str, text: str, out_path: str,
        strategy: str = "composite", bbox: str | None = None, style_bbox: str | None = None,
        where: str = "centered", seed: int | None = None, steps: int | None = None,
        cfg: float | None = None, mp: float = 1.0, lora: str | None = None,
        cn_strength: float = 1.0, url: str | None = None, verbose: bool = True,
        latent_mask: bool = False, backend: str = "qwen", unet: str | None = None,
        gguf: bool | None = None, mask_mode: str = "diff", no_lora: bool = False,
        diff_threshold: int = 45, diff_grow: int = 12, diff_feather: float = 2.0,
        paste_fit_mode: str = "height", paste_zoom: float = 1.0,
        features: bool = False, template: str = "default") -> Image.Image:
    client = ComfyClient(url) if url else ComfyClient()
    if not client.is_alive():
        raise SystemExit(f"ComfyUI not reachable at {client.base_url}. Start Comfy Desktop, "
                         f"or run run.ps1 which launches a headless server on :8189.")

    # Keep the alpha channel if the reference has one: paste_fit uses it to drop
    # only the lettering onto the target. Converting to RGB here (as this did)
    # silently reinstated the reference's own background, which the edit model
    # then kept as a dark panel behind the result.
    _style_src = Image.open(style_path)
    style = (_style_src.convert("RGBA") if "A" in _style_src.getbands()
             else _style_src.convert("RGB"))
    target = fit_megapixels(Image.open(target_path).convert("RGB"), mp)
    sbox = parse_box(style_bbox, style.size)
    style_crop = style.crop(sbox) if sbox else style

    box = parse_box(bbox, target.size)
    if strategy == "composite" and box is None:
        # default: a centred band covering 70% width, 30% height
        box = parse_box("0.15,0.35,0.70,0.30", target.size)

    m = KleinModelSet() if backend == "klein" else ModelSet()
    if unet:
        m.unet = unet
    if gguf is not None and backend == "qwen":
        m.gguf = gguf
    if steps:
        m.steps = steps
    if cfg is not None:
        m.cfg = cfg
    if lora and backend == "qwen":
        m.lora = lora
    if no_lora and backend == "qwen":
        m.lora = None
    seed = random.randint(0, 2**31 - 1) if seed is None else seed
    tag = f"{int(time.time())}_{seed}"

    original_name = client.upload_image(target, f"target_{tag}.png")
    mask_name = None
    if box is not None:
        mask_name = client.upload_image(bbox_mask(target.size, box).convert("RGB"),
                                        f"mask_{tag}.png")

    # Measure the style crop and name what it is. The reference is pasted at box
    # scale, so the picture alone says nothing about how heavy the original stroke
    # was; this is the only channel that carries it.
    style_desc = forbid = None
    if features:
        import features as _F
        _f = _F.stroke_features(style_crop, rgb=style_crop)
        if _f:
            style_desc, forbid = _f.describe(), _f.constraints()
            if verbose:
                print(f"[features] {_f}")
                print(f"[features] prompt style: {style_desc}")

    if strategy == "composite":
        comp = paste_fit(target, style_crop, box, paste_fit_mode, paste_zoom)
        image1 = client.upload_image(comp, f"composite_{tag}.png")
        refs = [client.upload_image(flatten_ref(style_crop), f"style_{tag}.png")]
        prompt = prompt_composite(text, style_desc, forbid, template)
        use_cn = False
    elif strategy in ("multiref", "inpaint-cn"):
        image1 = original_name
        refs = [client.upload_image(flatten_ref(style_crop), f"style_{tag}.png")]
        prompt = prompt_multiref(text, where, style_desc, forbid)
        use_cn = strategy == "inpaint-cn"
        if use_cn and mask_name is None:
            raise SystemExit("inpaint-cn needs --bbox")
    else:
        raise SystemExit(f"unknown strategy {strategy}")

    req = EditRequest(image1=image1, prompt=prompt, refs=refs, mask=mask_name,
                      latent_mask=latent_mask or use_cn,
                      composite_in_graph=(mask_mode == "bbox"),
                      original=original_name if mask_name else None, seed=seed,
                      use_inpaint_controlnet=use_cn, controlnet_strength=cn_strength)
    graph = build_graph_klein(req, target.size, m) if backend == "klein" else build_graph(req, m)

    if verbose:
        print(f"backend={backend} unet={m.unet} strategy={strategy} size={target.size} box={box} "
              f"seed={seed} steps={m.steps} cfg={m.cfg} lora={getattr(m, 'lora', None)}")
        print("prompt:", prompt)

    t0 = time.time()
    last = {"v": -1}

    def prog(v, mx, node):
        if verbose and v != last["v"]:
            last["v"] = v
            print(f"  step {v}/{mx}", end="\r", flush=True)

    res = client.run(graph, progress=prog)
    dt = time.time() - t0
    if not res.images:
        raise SystemExit(f"no output images (prompt_id={res.prompt_id})")
    img = res.images[0]

    if mask_mode == "diff":
        # Keep the background bit-exact, but let the lettering be whatever size
        # the model drew it -- see diff_mask().
        dm = diff_mask(target, img, box, diff_threshold, diff_grow, diff_feather)
        img = Image.composite(img, target, dm)
        if verbose:
            bb = dm.getbbox()
            print(f"diff mask: changed region {bb} vs requested box {box}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)
    if verbose:
        print(f"\nwrote {out_path}  {img.size}  in {dt:.1f}s  (prompt_id {res.prompt_id})")
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", required=True, help="image containing text in the wanted style")
    ap.add_argument("--target", required=True, help="image to write onto")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out/result.png")
    ap.add_argument("--strategy", choices=["composite", "multiref", "inpaint-cn"], default="composite")
    ap.add_argument("--bbox", help="x,y,w,h on target (fractions 0-1 or pixels)")
    ap.add_argument("--style-bbox", help="x,y,w,h crop of the text region in the style image")
    ap.add_argument("--where", default="centered", help="position phrase for multiref prompt")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--cfg", type=float)
    ap.add_argument("--mp", type=float, default=1.0, help="working megapixels")
    ap.add_argument("--lora", help="optional LoRA name, e.g. the 4-step Lightning")
    ap.add_argument("--cn-strength", type=float, default=1.0)
    ap.add_argument("--backend", choices=["qwen", "klein"], default="qwen",
                    help="qwen = Qwen-Image-Edit-2511 (GGUF); klein = FLUX.2 Klein 4B fp8 fallback")
    ap.add_argument("--unet", help="override diffusion model filename")
    ap.add_argument("--no-gguf", action="store_true",
                    help="qwen backend: load --unet with UNETLoader instead of UnetLoaderGGUF")
    ap.add_argument("--mask-mode", choices=["diff", "bbox", "none"], default="diff",
                    help="diff (default): composite back exactly what changed, no clipping; "
                         "bbox: old in-graph feathered-box composite; none: raw decode")
    ap.add_argument("--no-lora", action="store_true",
                    help="drop the 4-step Lightning LoRA (raise --steps/--cfg) for max fidelity")
    ap.add_argument("--diff-threshold", type=int, default=45,
                    help="on the blurred difference, not raw pixels")
    ap.add_argument("--diff-grow", type=int, default=12)
    ap.add_argument("--diff-feather", type=float, default=2.0,
                    help="seam anti-aliasing only; raise for a softer blend")
    ap.add_argument("--paste-fit", choices=["height", "width", "box"], default="height",
                    help="composite: what sets the pasted reference's size")
    ap.add_argument("--paste-zoom", type=float, default=1.0,
                    help="composite: multiplier on top of --paste-fit")
    ap.add_argument("--prompt", default="default", choices=sorted(PROMPT_STYLES),
                    help="which style clause to use (composite strategy)")
    ap.add_argument("--features", action="store_true",
                    help="measure the style crop (stroke weight, monoline-ness, ink colour) "
                         "and name it in the prompt instead of the generic effects list")
    ap.add_argument("--latent-mask", action="store_true",
                    help="also confine denoising to the box with SetLatentNoiseMask (default: pixel composite only)")
    ap.add_argument("--url", help="ComfyUI base url (default env COMFY_URL or http://127.0.0.1:8188)")
    a = ap.parse_args()
    run(a.style, a.target, a.text, a.out, a.strategy, a.bbox, a.style_bbox, a.where,
        a.seed, a.steps, a.cfg, a.mp, a.lora, a.cn_strength, a.url, latent_mask=a.latent_mask,
        backend=a.backend, unet=a.unet, gguf=(False if a.no_gguf else None),
        mask_mode=a.mask_mode, no_lora=a.no_lora, diff_threshold=a.diff_threshold,
        diff_grow=a.diff_grow, diff_feather=a.diff_feather,
        paste_fit_mode=a.paste_fit, paste_zoom=a.paste_zoom,
        features=a.features, template=a.prompt)


if __name__ == "__main__":
    main()
