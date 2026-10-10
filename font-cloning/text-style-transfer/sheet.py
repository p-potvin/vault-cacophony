"""Your own handwriting as a font, with no model in the loop.

Deriving someone's hand from one letter is not possible (README section 9): an M
says nothing about the bowl of a B. If you want *your* handwriting as a .ttf, the
honest way is to write every glyph once and process the scan.

    make-sheet   ->  print it, write in the cells, photograph it
    read-sheet   ->  rectify, drop the printed guides, cut each cell
                 ->  glyphs.build_font  ->  .ttf

Two design choices carry most of the reliability.

**Registration marks, not grid lines.** Four filled squares at known positions let
the photo be mapped onto template coordinates with one homography, so every cell
is then located arithmetically. Detecting printed cell borders in a phone photo is
much less dependable, and borders drawn over by the writer are unrecoverable.

**Cyan guides, read through the blue channel.** Cyan reflects blue, so a cyan grid
is nearly invisible in the blue channel while black or graphite stays dark - the
old non-photo-blue trick. It removes the guides exactly rather than by threshold
luck. The cost is that the pen must not be blue; write in black, or in pencil.

    python sheet.py make --out sheet.png
    python sheet.py read --photo sheet_photo.jpg --out out/myhand
"""

from __future__ import annotations

import os
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

UPPER = string.ascii_uppercase
LOWER = string.ascii_lowercase
DIGITS = string.digits
DEFAULT_CHARS = UPPER + LOWER + DIGITS + ".,!?'-"

# A4 at 150 dpi. The marks sit at these fractions of the page, and read_sheet maps
# a photo back onto exactly this coordinate system.
PAGE_W, PAGE_H = 1240, 1754
MARK_POS = [(0.06, 0.05), (0.94, 0.05), (0.94, 0.95), (0.06, 0.95)]
MARK_R = 26
GRID = (0.08, 0.135, 0.92, 0.93)      # left, top, right, bottom, as page fractions
GUIDE = (150, 214, 240)              # light cyan: vanishes in the blue channel
LABEL = (120, 196, 228)
BASELINE = 0.78                      # where the baseline guide sits within a cell
CELL_INSET = 0.10                    # margin trimmed off each cell before reading

# Characters whose ink drops below the baseline, so their lowest row must not be
# taken for it. Q and J are included because many hands give them a tail even
# where a typeface would not.
DESCENDERS = set("gjpqyQJ,;")


def _cells(chars: str, cols: int):
    rows = (len(chars) + cols - 1) // cols
    gx0, gy0, gx1, gy1 = (GRID[0] * PAGE_W, GRID[1] * PAGE_H,
                          GRID[2] * PAGE_W, GRID[3] * PAGE_H)
    cw, ch = (gx1 - gx0) / cols, (gy1 - gy0) / rows
    for i, c in enumerate(chars):
        r, k = divmod(i, cols)
        yield c, (gx0 + k * cw, gy0 + r * ch, gx0 + (k + 1) * cw, gy0 + (r + 1) * ch)


def make_template(chars: str = DEFAULT_CHARS, cols: int = 7,
                  out_path: str = "sheet.png") -> str:
    """Render the sheet to print."""
    im = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(im)
    try:
        f_lab = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)
        f_hdr = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)
    except OSError:
        f_lab = f_hdr = ImageFont.load_default()

    for fx, fy in MARK_POS:
        cx, cy = fx * PAGE_W, fy * PAGE_H
        d.rectangle((cx - MARK_R, cy - MARK_R, cx + MARK_R, cy + MARK_R), fill="black")

    # The baseline instruction is not decoration. A writer who centres each glyph
    # in its box instead of sitting it on the line destroys the descenders: tested,
    # g/p/q/y came back 0.02 of cap height below the baseline instead of 0.19.
    d.text((GRID[0] * PAGE_W, 0.066 * PAGE_H),
           "One character per box, in BLACK pen or pencil. Do not touch the box edges.",
           fill="black", font=f_hdr)
    d.text((GRID[0] * PAGE_W, 0.093 * PAGE_H),
           "Sit each letter ON the blue line - let the tails of g p q y hang below it.",
           fill="black", font=f_hdr)

    for c, (x0, y0, x1, y1) in _cells(chars, cols):
        d.rectangle((x0, y0, x1, y1), outline=GUIDE, width=2)
        base = y0 + (y1 - y0) * BASELINE                  # baseline hint
        d.line((x0 + 8, base, x1 - 8, base), fill=GUIDE, width=1)
        d.text((x0 + 7, y0 + 4), c, fill=LABEL, font=f_lab)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    im.save(out_path)
    return out_path


# ------------------------------------------------------------------ reading it

def find_marks(img: Image.Image):
    """The four registration squares, as (tl, tr, br, bl) in photo pixels."""
    g = np.asarray(img.convert("L"), np.uint8)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    # Otsu on an illumination-flattened copy, NOT an adaptive threshold: a solid
    # 50px square is larger than any sensible adaptive window, so the window sees
    # only black, and the middle of the mark is classified as background. That
    # silently lost one of the four marks in testing.
    k = max(3, int(min(g.shape) * 0.10) | 1)
    flat = np.clip(g.astype(np.float32) / np.maximum(cv2.GaussianBlur(g, (k, k), 0), 1)
                   * 255, 0, 255).astype(np.uint8)
    _, bw = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    area_lo, area_hi = 0.00004 * g.size, 0.004 * g.size

    cands = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not (area_lo <= a <= area_hi) or min(w, h) < 6:
            continue
        if not 0.7 <= w / h <= 1.4:                    # square
            continue
        if a / (w * h) < 0.80:                         # and solid
            continue
        cands.append((cent[i], a))
    if len(cands) < 4:
        return None

    pts = np.array([c for c, _ in cands], np.float32)
    mid = pts.mean(axis=0)
    quad = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):   # tl, tr, br, bl
        d = (pts[:, 0] - mid[0]) * sx + (pts[:, 1] - mid[1]) * sy
        quad.append(pts[int(np.argmax(d))])
    out = np.array(quad, np.float32)
    if len({tuple(p) for p in out}) < 4:
        return None
    # sanity: the quad must be convex, wide and tall enough to be the sheet
    w = np.linalg.norm(out[1] - out[0]) + np.linalg.norm(out[2] - out[3])
    h = np.linalg.norm(out[3] - out[0]) + np.linalg.norm(out[2] - out[1])
    if w < g.shape[1] * 0.8 or h < g.shape[0] * 0.8:
        return None
    return out


def rectify_sheet(img: Image.Image) -> Image.Image | None:
    """Map a photo of the sheet onto template coordinates."""
    marks = find_marks(img)
    if marks is None:
        return None
    dst = np.array([[f[0] * PAGE_W, f[1] * PAGE_H] for f in MARK_POS], np.float32)
    M = cv2.getPerspectiveTransform(marks, dst)
    warped = cv2.warpPerspective(np.asarray(img.convert("RGB")), M, (PAGE_W, PAGE_H),
                                 flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return Image.fromarray(warped)


def cell_ink(page: Image.Image, rect, inset: float = CELL_INSET,
             min_area: int = 24) -> np.ndarray | None:
    """Ink inside one cell, with the printed guides removed.

    Reads the blue channel, where cyan is nearly white; the guides therefore drop
    out instead of being thresholded away, and the printed label in the corner
    goes with them.
    """
    x0, y0, x1, y1 = rect
    dx, dy = (x1 - x0) * inset, (y1 - y0) * inset
    crop = page.crop((int(x0 + dx), int(y0 + dy), int(x1 - dx), int(y1 - dy)))
    b = np.asarray(crop, np.uint8)[..., 2]

    bg = cv2.GaussianBlur(b, (0, 0), max(3, min(b.shape) * 0.25))
    flat = np.clip(b.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)
    m = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 41, 14)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    # Scale the floor to the cell: a flat 60px minimum silently dropped the full
    # stop, which is legitimately tiny.
    floor = max(min_area, int(0.0006 * m.size))
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= floor]
    if not keep:
        return None
    # Keep every real stroke, not just the largest: an i is a stem plus a dot, and
    # a hurried E can arrive in two pieces.
    big = max(stats[i, cv2.CC_STAT_AREA] for i in keep)
    keep = [i for i in keep if stats[i, cv2.CC_STAT_AREA] >= 0.06 * big]
    return np.isin(lab, keep).astype(np.uint8)


def read_sheet(photo: str, chars: str = DEFAULT_CHARS, cols: int = 7,
               outdir: str | None = None):
    """Photo of a filled sheet -> {char: normalised mask}."""
    import handwriting as H
    import glyphs as G

    img = H.load_photo(photo, max_side=2600)
    page = rectify_sheet(img)
    if page is None:
        raise SystemExit("could not find the four registration marks - make sure all "
                         "four black corner squares are in frame and the sheet is lit "
                         "evenly")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        page.save(os.path.join(outdir, "_rectified.jpg"), quality=92)

    # Pass 1: cut every cell, and record where the printed baseline falls inside
    # each crop. The baseline is known from the template, not inferred from ink.
    raw, empty = {}, []
    prev = page.convert("RGB").copy()
    d = ImageDraw.Draw(prev)
    for c, rect in _cells(chars, cols):
        m = cell_ink(page, rect)
        if m is None or m.sum() < 40:
            empty.append(c)
            d.rectangle(rect, outline=(230, 60, 60), width=3)
            continue
        x0, y0, x1, y1 = rect
        ch = y1 - y0
        base_in_crop = (BASELINE - CELL_INSET) * ch / (1 - 2 * CELL_INSET)
        raw[c] = (m, base_in_crop)
        d.rectangle(rect, outline=(60, 180, 90), width=2)
    if outdir:
        prev.save(os.path.join(outdir, "_cells.jpg"), quality=92)
    if not raw:
        return {}, empty

    # Pass 2: fix the baseline. For a character with no descender the bottom of
    # the ink IS the baseline, so measure it there; the median gap between that
    # and the printed line then corrects the descender characters, whose own ink
    # bottom would otherwise be mistaken for the baseline.
    gaps = [np.where(m > 0)[0].max() - printed
            for c, (m, printed) in raw.items() if c not in DESCENDERS]
    shift = float(np.median(gaps)) if gaps else 0.0

    bases = {}
    for c, (m, printed) in raw.items():
        if c in DESCENDERS:
            bases[c] = printed + shift
        else:
            bases[c] = float(np.where(m > 0)[0].max())

    # Pass 3: one scale for the whole set, taken from the capitals, so x-height
    # stays shorter than cap height and ascenders and descenders keep proportion.
    heights = {c: bases[c] - np.where(m > 0)[0].min() for c, (m, _) in raw.items()}
    scale = G.shared_scale(heights)
    masks = {c: G.place_on_baseline(m, bases[c], scale) for c, (m, _) in raw.items()}
    return masks, empty


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make", help="render the sheet to print")
    mk.add_argument("--out", default="out/sheet/sheet.png")
    mk.add_argument("--chars", default=DEFAULT_CHARS)
    mk.add_argument("--cols", type=int, default=7)

    rd = sub.add_parser("read", help="read a photo of a filled sheet into a font")
    rd.add_argument("--photo", required=True)
    rd.add_argument("--out", default="out/sheet/read")
    rd.add_argument("--chars", default=DEFAULT_CHARS)
    rd.add_argument("--cols", type=int, default=7)
    rd.add_argument("--font-out", default="out/sheet/read/MyHand-Regular.ttf")
    rd.add_argument("--family", default="My Hand")
    a = ap.parse_args()

    if a.cmd == "make":
        p = make_template(a.chars, a.cols, a.out)
        rows = (len(a.chars) + a.cols - 1) // a.cols
        print(f"sheet -> {p}  ({len(a.chars)} cells, {a.cols}x{rows})")
        print("Print at 100% (no 'fit to page'), write in black, photograph it flat.")
        return

    import glyphs as G
    import build_font as BF
    masks, empty = read_sheet(a.photo, a.chars, a.cols, a.out)
    print(f"read {len(masks)}/{len(a.chars)} cells from {a.photo}")
    if empty:
        print(f"empty or unreadable: {''.join(empty)}")
    if not masks:
        raise SystemExit("nothing to build")
    BF.specimen(masks, os.path.join(a.out, "_silhouettes.png"))
    ttf = G.build_font(masks, a.font_out, family=a.family)
    print("font ->", ttf)


if __name__ == "__main__":
    main()
