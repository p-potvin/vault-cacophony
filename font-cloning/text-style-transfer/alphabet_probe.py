"""Render one letter at a time in a fixed style, to eyeball shape variation.

Everything is held constant except the requested character: same isolated
single-glyph style reference, same target, same box, same seed, same sampler.
So any difference between outputs is the model's letterform decision, which is
the only thing this probe is meant to expose.

    ..\\..\\.venv\\Scripts\\python.exe alphabet_probe.py --chars ABCDEFG
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline  # noqa: E402
from styles import letter_reference  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def contact_sheet(paths_labels, cols=7, tile_w=300, out="out/_alphabet_sheet.png"):
    tiles = []
    for p, lab in paths_labels:
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB")
        im = im.crop(pipeline.parse_box("0.30,0.18,0.40,0.64", im.size))
        tiles.append((im.resize((tile_w, round(tile_w * im.height / im.width)), Image.LANCZOS), lab))
    if not tiles:
        return None
    th, bar = tiles[0][0].height, 22
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * (th + bar)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(tiles):
        x, y = (i % cols) * tile_w, (i // cols) * (th + bar)
        sheet.paste(im, (x, y))
        d.text((x + 5, y + th + 5), lab, fill=(0, 0, 0))
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ap.add_argument("--ref-char", default="D", help="glyph the style reference is rendered from")
    ap.add_argument("--target", default="examples/target_feathers.jpg")
    ap.add_argument("--bbox", default="0.38,0.26,0.24,0.48")
    ap.add_argument("--strategy", default="composite")
    ap.add_argument("--paste-fit", default="height")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--outdir", default="out/alphabet")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    ref_path = os.path.join(a.outdir, f"_style_ref_{a.ref_char}.png")
    letter_reference(a.ref_char, rgba=True).save(ref_path)
    print(f"style reference: isolated '{a.ref_char}' -> {ref_path}\n")

    done, t0 = [], time.time()
    for i, ch in enumerate(a.chars, 1):
        out = os.path.join(a.outdir, f"{ch}.png")
        if os.path.exists(out):
            print(f"[{i}/{len(a.chars)}] {ch}: exists, skipping")
            done.append((out, ch))
            continue
        t = time.time()
        try:
            pipeline.run(ref_path, a.target, ch, out, strategy=a.strategy,
                         bbox=a.bbox, seed=a.seed, url=a.url, verbose=False,
                         paste_fit_mode=a.paste_fit)
            done.append((out, ch))
            print(f"[{i}/{len(a.chars)}] {ch}: {time.time()-t:.0f}s", flush=True)
        except Exception as e:
            print(f"[{i}/{len(a.chars)}] {ch}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
        if i % 5 == 0 or i == len(a.chars):
            s = contact_sheet(done)
            print(f"    sheet -> {s}  ({time.time()-t0:.0f}s elapsed)", flush=True)

    print(f"\n{len(done)}/{len(a.chars)} rendered in {time.time()-t0:.0f}s")
    print("sheet:", contact_sheet(done))


if __name__ == "__main__":
    main()
