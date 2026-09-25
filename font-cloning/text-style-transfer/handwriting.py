"""Get a clean, evenly-lit page of writing out of a phone photo.

A photo of a card on a table is not something you can threshold directly: the
paper is a bright quad inside a dark frame, the lighting falls off across it, and
pencil is only slightly darker than the paper it sits on. Thresholding the whole
frame just finds the card's edge.

    photo -> locate the paper -> rectify it -> flatten the illumination
          -> binarise the ink -> letter boxes

Flattening is division by a heavily-blurred copy of the page, which is the
standard trick for this: the blur estimates the local paper brightness, so
dividing removes the lighting gradient while leaving the strokes, and it copes
with a shadow across one corner in a way a global threshold cannot.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_photo(path: str, max_side: int = 2000) -> Image.Image:
    """Open, apply EXIF rotation (phone photos need it), and bound the size."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def find_paper(img: Image.Image, min_area_frac: float = 0.10):
    """Corners of the largest bright quadrilateral, or None."""
    g = np.asarray(img.convert("L"), np.uint8)
    g = cv2.GaussianBlur(g, (0, 0), 2.0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_frac * g.size:
        return None
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


def rectify(img: Image.Image, corners, margin: float = 0.02) -> Image.Image:
    """Perspective-correct the paper to a flat rectangle."""
    pts = np.asarray(corners, np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    src = np.array([tl, tr, br, bl], np.float32)

    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    w, h = max(64, w), max(64, h)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    out = cv2.warpPerspective(np.asarray(img.convert("RGB")),
                              cv2.getPerspectiveTransform(src, dst), (w, h))
    m = int(min(w, h) * margin)
    if m:
        out = out[m:h - m, m:w - m]
    return Image.fromarray(out)


def flatten(page: Image.Image, blur_frac: float = 0.06) -> Image.Image:
    """Remove the lighting gradient by dividing out a blurred copy of the page."""
    g = np.asarray(page.convert("L"), np.float32)
    k = max(3, int(min(page.size) * blur_frac) | 1)
    bg = cv2.GaussianBlur(g, (k, k), 0)
    flat = np.clip(g / np.maximum(bg, 1) * 255.0, 0, 255)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(flat.astype(np.uint8))


def ink_mask(flat: Image.Image, block: int = 51, C: int = 12,
             min_area: int = 40) -> np.ndarray:
    """Binary ink mask from a flattened page, using an adaptive threshold.

    Adaptive rather than Otsu: even after flattening, pencil density varies along
    a stroke, and a single global cut drops the lightest parts of letters.
    """
    g = np.asarray(flat.convert("L"), np.uint8)
    b = block | 1
    m = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, b, C)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return np.isin(lab, keep).astype(np.uint8)


def letter_boxes(mask: np.ndarray, min_h_frac: float = 0.25,
                 merge_gap_frac: float = 0.12):
    """Letter boxes from an ink mask, filtered to the main text line.

    Components far shorter than the tallest are dropped (specks, paper texture),
    and components that overlap horizontally are merged so an accent or a broken
    stroke joins its letter rather than becoming its own box.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return []
    boxes = [(stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
             for i in range(1, n)]
    tall = max(b[3] for b in boxes)
    boxes = [b for b in boxes if b[3] >= min_h_frac * tall]
    if not boxes:
        return []

    boxes.sort(key=lambda b: b[0])
    gap = merge_gap_frac * tall
    merged = [list(boxes[0])]
    for x, y, w, h in boxes[1:]:
        px, py, pw, ph = merged[-1]
        if x <= px + pw + gap:
            nx0, ny0 = min(px, x), min(py, y)
            nx1, ny1 = max(px + pw, x + w), max(py + ph, y + h)
            merged[-1] = [nx0, ny0, nx1 - nx0, ny1 - ny0]
        else:
            merged.append([x, y, w, h])
    return [tuple(b) for b in merged]


def neutralise(page: Image.Image, mask: np.ndarray, ring: int = 9) -> Image.Image:
    """Scale each channel so the paper beside the strokes is neutral grey.

    The sample card photographs violet - paper (208,199,220) as well as ink - so
    a reference cut from it tells the model "write in violet", and the model
    obliges with far more saturation than was there. Anchoring on the paper right
    next to the ink (rather than on a blurred background, which near a thin stroke
    *is* the tinted paper) removes the cast while leaving the pen its own colour.
    """
    a = np.asarray(page.convert("RGB"), np.float32)
    near = (cv2.dilate(mask, np.ones((ring, ring), np.uint8)) > 0) & (mask == 0)
    if not near.any():
        return page
    paper = np.median(a[near], axis=0)
    gain = paper.mean() / np.maximum(paper, 1.0)
    return Image.fromarray(np.clip(a * gain, 0, 255).astype(np.uint8))


def prepare(path: str, outdir: str | None = None):
    """Full preprocessing. Returns (page_rgb, flat_gray, ink_mask, boxes)."""
    import os
    photo = load_photo(path)
    corners = find_paper(photo)
    page = rectify(photo, corners) if corners is not None else photo
    flat = flatten(page)
    mask = ink_mask(flat)
    boxes = letter_boxes(mask)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        photo.save(os.path.join(outdir, "01_photo.jpg"), quality=90)
        page.save(os.path.join(outdir, "02_page.jpg"), quality=92)
        flat.save(os.path.join(outdir, "03_flat.jpg"), quality=92)
        Image.fromarray((1 - mask) * 255).save(os.path.join(outdir, "04_ink.png"))
        prev = page.convert("RGB").copy()
        from PIL import ImageDraw
        d = ImageDraw.Draw(prev)
        for i, (x, y, w, h) in enumerate(boxes):
            d.rectangle((x, y, x + w, y + h), outline=(230, 40, 40),
                        width=max(2, prev.width // 500))
            d.text((x + 2, max(0, y - 12)), str(i), fill=(230, 40, 40))
        prev.save(os.path.join(outdir, "05_boxes.jpg"), quality=92)
    return page, flat, mask, boxes


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--outdir", default="out/handwriting/prep")
    a = ap.parse_args()
    page, flat, mask, boxes = prepare(a.photo, a.outdir)
    print(f"page {page.size}  ink {int(mask.sum())}px  {len(boxes)} letter boxes")
    for i, b in enumerate(boxes):
        print(f"  [{i}] x={b[0]:4d} y={b[1]:4d} {b[2]:3d}x{b[3]:3d}")
    print("steps written to", a.outdir)


def split_touching(mask: np.ndarray, box=None, min_width_frac: float = 0.30,
                   valley_frac: float = 0.55, order_frac: float = 0.09):
    """Split a run of touching letters using the vertical ink profile.

    Print handwriting frequently joins adjacent letters, so connected components
    return whole words. Columns where the ink thins to a local minimum are the
    natural cut points; `min_width_frac` (as a share of line height) stops a
    single wide letter being sliced in half.
    """
    if box is not None:
        x, y, w, h = box
        sub = mask[y:y + h, x:x + w]
        ox, oy = x, y
    else:
        ys, xs = np.where(mask > 0)
        oy, ox = int(ys.min()), int(xs.min())
        sub = mask[oy:ys.max() + 1, ox:xs.max() + 1]

    h = sub.shape[0]
    prof = cv2.GaussianBlur(sub.sum(axis=0).astype(np.float32).reshape(1, -1),
                            (0, 0), 2.0).ravel()
    med = float(np.median(prof[prof > 0])) if (prof > 0).any() else 0.0
    order = max(3, int(h * order_frac))
    min_w = max(4, int(h * min_width_frac))

    cuts = []
    for i in range(order, len(prof) - order):
        w = prof[i - order:i + order + 1]
        if prof[i] == w.min() and prof[i] < med * valley_frac:
            if not cuts or i - cuts[-1] >= min_w:
                cuts.append(i)

    edges = [0] + cuts + [sub.shape[1]]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = sub[:, a:b]
        if seg.sum() < 30:
            continue
        ys, xs = np.where(seg > 0)
        out.append((ox + a + int(xs.min()), oy + int(ys.min()),
                    int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
    return out


def segment_page(mask: np.ndarray, split: bool = True):
    """Letter boxes, splitting any component wide enough to be several letters."""
    boxes = letter_boxes(mask, merge_gap_frac=0.0)
    if not split:
        return boxes
    tall = max((b[3] for b in boxes), default=1)
    out = []
    for b in boxes:
        if b[2] > 1.4 * tall:          # too wide to be one letter
            out.extend(split_touching(mask, b))
        else:
            out.append(b)
    return sorted(out, key=lambda b: b[0])
