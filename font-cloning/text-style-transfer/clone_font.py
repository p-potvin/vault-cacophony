"""End-to-end: a photo of handwriting -> a font in that hand.

    photo of a word  ->  pick one letter as the style reference
                     ->  generate the charset one glyph at a time
                     ->  normalise to a common cap height
                     ->  .ttf  (outline + optional OT-SVG colour)

Two things this does that the earlier probe did not.

**A gate after the first glyph.** A bad style reference wastes an hour, so glyph
one is generated alone and checked before the rest are queued: did the model draw
anything, is it a sane amount of ink, and - the failure that actually bit us - is
it merely a copy of the reference glyph? `--gate ask` stops for a human look,
`--gate auto` (default) proceeds unless a check fails, `--gate none` skips it.

**The whole batch is queued at once.** ComfyUI keeps the model resident across a
queue run, so the ~60 s GGUF dequantisation is paid once rather than once per
letter - roughly halving wall clock for a 52-glyph set.

    ..\\..\\.venv\\Scripts\\python.exe clone_font.py --photo my_handwriting.jpg \\
        --ref-bbox 0.31,0.22,0.09,0.30 --chars upper
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
import glyphs as G  # noqa: E402
import handwriting as H  # noqa: E402
import pipeline  # noqa: E402
from comfy_client import ComfyClient  # noqa: E402
from workflows import EditRequest, ModelSet, build_graph  # noqa: E402

UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
CHARSETS = {"upper": UPPER, "lower": LOWER, "digits": DIGITS,
            "letters": UPPER + LOWER, "all": UPPER + LOWER + DIGITS}


# ------------------------------------------------------------ reference glyph

def segment_letters(photo: Image.Image, min_frac: float = 0.004, dark_text: bool = True):
    """Candidate letter boxes in a photo of writing, left to right."""
    import cv2
    g = np.asarray(photo.convert("L"), np.uint8)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    mode = cv2.THRESH_BINARY_INV if dark_text else cv2.THRESH_BINARY
    _, bw = cv2.threshold(g, 0, 255, mode | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    area_min = min_frac * g.size
    boxes = [(stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= area_min]
    return sorted(boxes, key=lambda b: b[0])


def segmentation_preview(photo: Image.Image, boxes, path: str) -> str:
    im = photo.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for i, (x, y, w, h) in enumerate(boxes):
        d.rectangle((x, y, x + w, y + h), outline=(255, 40, 40), width=max(2, im.width // 400))
        d.text((x + 3, max(0, y - 14)), str(i), fill=(255, 40, 40))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path


def cutout_reference(photo: Image.Image, box, pad_frac: float = 0.18,
                     dark_text: bool = True) -> Image.Image:
    """Crop one letter and matte it to RGBA so only the ink is pasted later."""
    import cv2
    x, y, w, h = box
    px, py = int(w * pad_frac), int(h * pad_frac)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(photo.width, x + w + px), min(photo.height, y + h + py)
    crop = photo.convert("RGB").crop((x0, y0, x1, y1))

    g = np.asarray(crop.convert("L"), np.uint8)
    mode = cv2.THRESH_BINARY_INV if dark_text else cv2.THRESH_BINARY
    _, bw = cv2.threshold(cv2.GaussianBlur(g, (0, 0), 1.0), 0, 255, mode | cv2.THRESH_OTSU)
    alpha = cv2.GaussianBlur(bw, (0, 0), 0.8)
    out = crop.convert("RGBA")
    out.putalpha(Image.fromarray(alpha))
    return out


def place_reference(plate: Image.Image, ref: Image.Image, box, where: str):
    """Build image1: the plate with the reference letter placed for the model to see.

    `where` is "box" (the original behaviour - paste into the target box, which is
    what bleeds), "left" or "right" (paste beside it, at the same height, so the
    box itself stays empty).
    """
    if where == "box":
        return pipeline.paste_fit(plate, ref, box, "height")
    x0, y0, x1, y1 = box
    w, gap = x1 - x0, int((x1 - x0) * 0.25)
    if where == "left":
        sx0 = max(0, x0 - gap - w)
    else:
        sx0 = min(plate.width - w, x1 + gap)
    return pipeline.paste_fit(plate, ref, (sx0, y0, sx0 + w, y1), "height")


def settle(plate: Image.Image, result: Image.Image, box, shadow: bool = False):
    """Put the generated glyph on a clean plate, with or without its cast shadow.

    `shadow=False` composites through the glyph's own structure rather than
    through everything that changed, which drops the drop shadow the model paints
    behind the letter. The font never had it (build_font measures the face), so
    this only makes the saved PNG match what the font will actually contain.
    """
    import cv2
    if shadow:
        return Image.composite(result, plate, pipeline.diff_mask(plate, result, box))
    m = G.body_mask(result, plate)
    if int(m.sum()) < 500:                       # nothing structural found; keep the old path
        return Image.composite(result, plate, pipeline.diff_mask(plate, result, box))
    a = cv2.GaussianBlur((m * 255).astype("uint8"), (0, 0), 1.0)
    return Image.composite(result, plate, Image.fromarray(a).convert("L"))


# ------------------------------------------------------------------- the gate

def inspect(plate: Image.Image, result: Image.Image, ref_mask: np.ndarray | None):
    """Cheap sanity checks on the first glyph. Returns (ok, notes)."""
    import cv2
    m = G.face_mask(result, plate)
    notes, ok = [], True
    ink = int(m.sum())
    frac = ink / m.size
    if ink < 800:
        return False, [f"almost nothing drawn ({ink}px) - reference or box is probably wrong"]
    notes.append(f"ink {ink}px ({100*frac:.2f}% of frame)")
    if frac > 0.25:
        ok = False
        notes.append("suspiciously large - the model may have filled the frame")

    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    big = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
    if len(big) > 1 and big[1] > 0.35 * big[0]:
        ok = False
        notes.append(f"fragmented into {len(big)} pieces ({big[:3]}) - not one clean glyph")

    if ref_mask is not None:
        a = G.normalise(m).astype(bool)
        b = G.normalise(ref_mask).astype(bool)
        h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
        iou = (a & b).sum() / max(1, (a | b).sum())
        notes.append(f"IoU vs reference glyph {iou:.2f}")
        if iou > 0.80:
            ok = False
            notes.append("nearly identical to the reference - this is the reference-bleed failure; "
                         "use --ref-char with a less similar shape")
    return ok, notes


# ------------------------------------------------------------------ main flow

def build_request(client, plate, ref_rgba, ch, box, seed, tag, style=None, forbid=None,
                  template="default", extra_refs=(), paste="box"):
    tgt_name = client.upload_image(plate, f"plate_{tag}.png")
    # paste=False leaves the box empty. The model then has no reference letter in
    # the target area to leave behind, which is the bleed: it reads "replace the
    # text" as "add text" and draws the new letter over the pasted one.
    comp = place_reference(plate, ref_rgba, box, paste) if paste else plate
    img1 = client.upload_image(comp, f"comp_{tag}_{ord(ch):04x}.png")
    # Matte the reference onto the plate's own colour: on a dark ground the model
    # reads dark pencil as light-ink-on-dark-paper and returns an inverted glyph.
    import numpy as _np
    ground = tuple(int(v) for v in _np.asarray(plate.convert("RGB")).reshape(-1, 3).mean(0))
    ref = client.upload_image(pipeline.flatten_ref(ref_rgba, ground), f"ref_{tag}.png")
    # image3, when supplied: a second letter in the same hand, so the model has a
    # style to average rather than one letter to copy. build_graph takes refs[:2].
    ref_names = [ref] + [client.upload_image(pipeline.flatten_ref(x, ground),
                                             f"ref{i}_{tag}.png")
                         for i, x in enumerate(extra_refs, 2)]
    mask = client.upload_image(pipeline.bbox_mask(plate.size, box).convert("RGB"), f"mask_{tag}.png")
    req = EditRequest(image1=img1,
                      prompt=pipeline.prompt_composite(ch, style, forbid, template),
                      refs=ref_names,
                      mask=mask, original=tgt_name, seed=seed,
                      filename_prefix=f"tst/clone_{tag}_{ord(ch):04x}")
    return build_graph(req, ModelSet()), tgt_name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", required=True, help="picture containing the handwriting")
    ap.add_argument("--ref-bbox", help="x,y,w,h of the reference letter (fractions or px)")
    ap.add_argument("--ref-index", type=int, help="index from the segmentation preview instead")
    ap.add_argument("--ref-image", nargs="+", metavar="PNG",
                    help="RGBA cut-outs to use as the reference instead of cutting a fresh "
                         "one (e.g. Real-ESRGAN upscales). The first is pasted into the box; "
                         "any others are passed as extra style references, which stops every "
                         "glyph being conditioned on one letter")
    ap.add_argument("--light-text", action="store_true", help="writing is lighter than the paper")
    ap.add_argument("--target", help="background to draw onto (default: flat neutral)")
    ap.add_argument("--chars", default="upper",
                    help="upper|lower|letters|digits|all, or a literal string")
    ap.add_argument("--bbox", default="0.38,0.26,0.24,0.48", help="where each glyph is drawn")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--prompt", default="default", choices=sorted(pipeline.PROMPT_STYLES),
                    help="which style clause to use; head and tail are identical across "
                         "all of them so they can be compared directly")
    ap.add_argument("--style-desc", help="hand-written style phrase, instead of the measured one")
    ap.add_argument("--features", action="store_true",
                    help="describe the measured stroke weight in the prompt. Off by default: "
                         "Qwen-Image-Edit is instruction-tuned and wants an edit operation, "
                         "not a caption of the wanted result, and the measured wording scored "
                         "worse than the plain instruction (see README section 8)")
    ap.add_argument("--gate", choices=["auto", "ask", "none"], default="auto")
    ap.add_argument("--outdir", default="out/clone")
    ap.add_argument("--font-out", default="out/clone/font/Cloned-Regular.ttf")
    ap.add_argument("--ref-place", choices=["box", "left", "right", "none"], default="box",
                    help="where to put the reference in image 1. 'box' pastes it into the "
                         "target box (the original behaviour, and what bleeds); 'left'/'right' "
                         "put it beside the box so the box stays empty and any leftover falls "
                         "outside the composited region; 'none' shows no reference at all, "
                         "which measurably stops the model drawing")
    ap.add_argument("--colour", action="store_true")
    ap.add_argument("--shadow", action="store_true",
                    help="keep the cast shadow the model paints behind each glyph. Off by "
                         "default: the font never contains it, so keeping it in the PNG only "
                         "makes the preview disagree with the font")
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    a = ap.parse_args()

    chars = CHARSETS.get(a.chars, a.chars)
    os.makedirs(a.outdir, exist_ok=True)
    client = ComfyClient(a.url)
    if not client.is_alive():
        raise SystemExit(f"ComfyUI not reachable at {a.url}")

    prep_dir = os.path.join(a.outdir, "prep")
    page, flat, ink, _ = H.prepare(a.photo, prep_dir)
    print(f"page {page.size}, ink {int(ink.sum())}px -> {prep_dir}")

    # Measure the hand across the whole line, not just the reference letter: a
    # single glyph can be atypical, and the model needs the weight named because
    # the reference is pasted at box scale and so carries no scale of its own.
    feats = F.measure_page(ink, H.segment_page(ink), page)
    if a.style_desc:
        style, forbid = a.style_desc, ""
    elif feats and a.features:
        style, forbid = feats.describe(), feats.constraints()
    else:
        style, forbid = None, ""
    if feats:
        print(f"measured hand: {feats}   (scoring only unless --features)")
    if style:
        print(f"prompt style : {style}")
        if forbid:
            print(f"prompt guard : {forbid}")

    if a.ref_bbox:
        b = pipeline.parse_box(a.ref_bbox, page.size)
        rbox = (b[0], b[1], b[2] - b[0], b[3] - b[1])
    else:
        boxes = H.segment_page(ink)
        prev = segmentation_preview(page, boxes, os.path.join(a.outdir, "_segmentation.png"))
        print(f"{len(boxes)} letter candidates -> {prev}")
        if a.ref_index is None:
            for i, bx in enumerate(boxes):
                print(f"  [{i}] x={bx[0]:4d} y={bx[1]:4d} {bx[2]:3d}x{bx[3]:3d}")
            raise SystemExit("pick one with --ref-index N (or pass --ref-bbox)")
        rbox = boxes[a.ref_index]

    # matte straight from the ink mask - it is already lighting-corrected, so it
    # beats re-thresholding the colour crop
    x, y, w, h = rbox
    pad = int(max(w, h) * 0.18)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(page.width, x + w + pad), min(page.height, y + h + pad)
    # Do not let the padding reach into the neighbouring letters: in joined print
    # the gaps are narrower than the padding, and a sliver of the next letter in
    # the reference is exactly the contamination that causes glyph bleed.
    try:
        _all = H.segment_page(ink)
        left = [b for b in _all if b[0] + b[2] <= x]
        right = [b for b in _all if b[0] >= x + w]
        if left:
            x0 = max(x0, max(b[0] + b[2] for b in left) + 2)
        if right:
            x1 = min(x1, min(b[0] for b in right) - 2)
    except Exception:
        pass
    x0, x1 = min(x0, x), max(x1, x + w)
    sub_ink = ink[y0:y1, x0:x1]
    ref_rgba = page.crop((x0, y0, x1, y1)).convert("RGBA")
    import cv2 as _cv2
    alpha = _cv2.GaussianBlur((sub_ink * 255).astype("uint8"), (0, 0), 0.8)
    ref_rgba.putalpha(Image.fromarray(alpha))
    extra_refs = []
    if a.ref_image:
        for path in a.ref_image:
            im = Image.open(path)
            if "A" not in im.getbands():
                raise SystemExit(f"{path} has no alpha channel - it must be a cut-out, not a "
                                 f"flat crop, or the paper is pasted along with the ink")
            extra_refs.append(im.convert("RGBA"))
        print(f"reference from {a.ref_image[0]} {extra_refs[0].size} "
              f"(replaces the {ref_rgba.size} cut-out)")
        ref_rgba = extra_refs.pop(0)
        if extra_refs:
            print(f"plus {len(extra_refs)} style-only reference(s): "
                  f"{', '.join(os.path.basename(x) for x in a.ref_image[1:])}")

    ref_path = os.path.join(a.outdir, "_reference.png")
    ref_rgba.save(ref_path)
    print(f"reference letter {rbox} -> {ref_path} {ref_rgba.size}")

    if a.target:
        plate = pipeline.fit_megapixels(Image.open(a.target).convert("RGB"))
    else:
        plate = pipeline.fit_megapixels(Image.new("RGB", (1328, 752), (238, 238, 242)))
    plate_path = os.path.join(a.outdir, "_plate.png")
    plate.save(plate_path)
    box = pipeline.parse_box(a.bbox, plate.size)
    tag = str(int(time.time()))

    # ---- gate on the first glyph
    first = chars[0]
    print(f"\n[gate] generating {first!r} alone before committing to {len(chars)} glyphs")
    print(f"prompt       : {pipeline.prompt_composite(first, style, forbid, a.prompt)}")
    graph, _ = build_request(client, plate, ref_rgba, first, box, a.seed, tag, style,
                             forbid, a.prompt, extra_refs,
                             False if a.ref_place == "none" else a.ref_place)
    t0 = time.time()
    res = client.run(graph)
    if not res.images:
        raise SystemExit("first glyph produced no image")
    img = settle(plate, res.images[0], box, a.shadow)
    p0 = os.path.join(a.outdir, f"{first}.png")
    img.save(p0)
    print(f"[gate] {first} rendered in {time.time()-t0:.0f}s -> {p0}")

    ref_face = None
    try:
        ref_on_plate = pipeline.paste_fit(plate, ref_rgba, box, "height")  # noqa: E501
        ref_face = G.face_mask(ref_on_plate, plate)
        if ref_face.sum() < 500:
            ref_face = None
    except Exception:
        ref_face = None

    ok, notes = inspect(plate, img, ref_face)
    for n in notes:
        print(f"       {n}")
    if a.gate == "ask":
        print(f"\nInspect {p0}. Continue with the other {len(chars)-1} glyphs? [y/N] ", end="")
        if input().strip().lower() not in ("y", "yes"):
            raise SystemExit("stopped at the gate")
    elif a.gate == "auto" and not ok:
        raise SystemExit("[gate] FAILED - stopping before the batch. "
                         "Re-run with a different --ref-index/--ref-bbox, or --gate none to force.")
    print("[gate] passed\n" if ok else "[gate] warnings above, continuing\n")

    # ---- queue everything else in one go so the model stays resident
    rest = chars[1:]
    if rest:
        ids = {}
        for ch in rest:
            g, _ = build_request(client, plate, ref_rgba, ch, box, a.seed, tag, style,
                                 forbid, a.prompt, extra_refs,
                                 False if a.ref_place == "none" else a.ref_place)
            ids[client.submit(g)] = ch
        print(f"queued {len(ids)} glyphs in one batch (model stays loaded)")
        t0 = time.time()

        def progress(pid, done, total):
            print(f"  {done}/{total}  {time.time()-t0:.0f}s elapsed", flush=True)

        out = client.collect(list(ids), on_done=progress)
        for pid, ch in ids.items():
            r = out.get(pid)
            if isinstance(r, Exception) or not getattr(r, "images", None):
                print(f"  {ch}: FAILED ({r})")
                continue
            im = settle(plate, r.images[0], box, a.shadow)
            im.save(os.path.join(a.outdir, f"{ch}.png"))
        print(f"batch done in {time.time()-t0:.0f}s "
              f"({(time.time()-t0)/max(1,len(ids)):.0f}s/glyph)")

    # ---- assemble
    import build_font as BF
    got = [c for c in chars if os.path.exists(os.path.join(a.outdir, f"{c}.png"))]
    print(f"\nassembling {len(got)}/{len(chars)} glyphs")
    if a.colour:
        masks, art = BF.load_masks(got, [a.outdir], plate, colour=True)
    else:
        masks, art = BF.load_masks(got, [a.outdir], plate), {}
    BF.specimen(masks, os.path.join(os.path.dirname(a.font_out), "_silhouettes.png"))
    ttf = G.build_font(masks, a.font_out, family="Cloned Hand")
    print("font ->", ttf)
    if art:
        cpath = a.font_out.replace(".ttf", "-Color.ttf")
        G.add_colour_layer(ttf, art, masks, cpath)
        print("colour font ->", cpath)


if __name__ == "__main__":
    main()
