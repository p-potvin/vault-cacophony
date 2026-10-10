"""Assemble the generated alphabet into a TTF and render a specimen.

    ..\\..\\.venv\\Scripts\\python.exe build_font.py

Reads out/alphabet/<CH>.png (with out/alphabet_fix/<CH>.png overriding, for the
letters that had to be regenerated), extracts each glyph's face against the clean
plate, normalises every glyph to a common cap height, vectorises and writes a TTF.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glyphs as G  # noqa: E402
import pipeline  # noqa: E402


def load_masks(chars, dirs, plate, verbose=True, colour=False):
    out, art, report = {}, {}, []
    for ch in chars:
        src = None
        for d in dirs:
            p = os.path.join(d, f"{ch}.png")
            if os.path.exists(p):
                src = p
        if src is None:
            report.append((ch, None, "missing"))
            continue
        img = Image.open(src)
        m = G.face_mask(img, plate)
        if m.sum() < 500:
            report.append((ch, src, "empty"))
            continue
        ys, xs = np.where(m > 0)
        report.append((ch, src, f"{xs.max()-xs.min()}x{ys.max()-ys.min()}  {int(m.sum())}px"))
        if colour:
            out[ch], art[ch] = G.normalise_pair(m, G.effect_layer(img, plate))
        else:
            out[ch] = G.normalise(m, cap_px=G.CAP_HEIGHT_PX)
    if verbose:
        for ch, src, note in report:
            tag = os.path.basename(os.path.dirname(src)) if src else "-"
            print(f"  {ch}  {note:24s} [{tag}]")
    return (out, art) if colour else out


def specimen(masks, path, cols=13, tile=110):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = (len(masks) + cols - 1) // cols
    sheet = Image.new("L", (cols * tile, rows * tile), 255)
    for i, (ch, m) in enumerate(masks.items()):
        im = Image.fromarray((1 - m) * 255).convert("L")
        im.thumbnail((tile - 8, tile - 8), Image.LANCZOS)
        x = (i % cols) * tile + (tile - im.width) // 2
        y = (i // cols) * tile + (tile - im.height) // 2
        sheet.paste(im, (x, y))
    sheet.save(path)
    return path


def render_with_font(ttf, text, path, size=150):
    """Draw text with the built font, proving it actually loads and shapes."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    from PIL import ImageFont
    f = ImageFont.truetype(ttf, size)
    img = Image.new("RGB", (40 + size * len(text), int(size * 1.6)), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 20), text, font=f, fill=(20, 20, 30))
    img.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ap.add_argument("--dirs", nargs="+", default=["out/alphabet", "out/alphabet_fix"])
    ap.add_argument("--plate", default="examples/target_feathers.jpg")
    ap.add_argument("--out", default="out/font/CacophonyCloned-Regular.ttf")
    ap.add_argument("--simplify", type=float, default=1.2)
    ap.add_argument("--specimen-text", default="HAMBURGEFONS")
    ap.add_argument("--colour", action="store_true",
                    help="also emit an OT-SVG colour font carrying every rendered effect")
    a = ap.parse_args()

    plate = pipeline.fit_megapixels(Image.open(a.plate).convert("RGB"))
    print(f"plate {plate.size}; extracting {len(a.chars)} glyphs")
    if a.colour:
        masks, art = load_masks(a.chars, a.dirs, plate, colour=True)
    else:
        masks, art = load_masks(a.chars, a.dirs, plate), {}
    print(f"\n{len(masks)}/{len(a.chars)} glyphs extracted")

    sp = specimen(masks, "out/font/_silhouettes.png")
    print("silhouettes ->", sp)

    advances = {}
    for ch, im in art.items():
        xs = np.where(np.asarray(im.getchannel("A")) > 12)[1]
        if len(xs):
            advances[ch] = int((xs.max() - xs.min() + 1) * (G.CAP_HEIGHT / G.CAP_HEIGHT_PX)) \
                           + 2 * G.SIDE_BEARING
    ttf = G.build_font(masks, a.out, simplify=a.simplify, advances=advances or None)
    print("font ->", ttf, f"({os.path.getsize(ttf)/1024:.0f} KB)")

    from fontTools.ttLib import TTFont
    tf = TTFont(ttf)
    cmap = tf.getBestCmap()
    print(f"verified: {len(tf.getGlyphOrder())} glyphs, cmap covers "
          f"{sum(1 for c in a.chars if ord(c) in cmap)}/{len(a.chars)} requested chars")

    out = render_with_font(ttf, a.specimen_text, "out/font/_specimen.png")
    print("specimen ->", out)

    if art:
        cpath = a.out.replace(".ttf", "-Color.ttf")
        G.add_colour_layer(ttf, art, masks, cpath)
        from fontTools.ttLib import TTFont as _T
        cf = _T(cpath)
        print(f"colour font -> {cpath} ({os.path.getsize(cpath)/1024:.0f} KB), "
              f"SVG table with {len(cf[chr(83)+chr(86)+chr(71)+chr(32)].docList)} glyph documents")
        list(art.values())[0].save("out/font/_art_sample.png")
        print("colour art sample -> out/font/_art_sample.png")

        bpath = a.out.replace(".ttf", "-Bitmap.ttf")
        try:
            G.add_bitmap_layer(ttf, art, bpath)
            bf = _T(bpath)
            print(f"bitmap font -> {bpath} ({os.path.getsize(bpath)/1024:.0f} KB), "
                  f"CBDT/CBLC present: {'CBDT' in bf and 'CBLC' in bf}")
        except Exception as e:
            print(f"bitmap font FAILED: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
