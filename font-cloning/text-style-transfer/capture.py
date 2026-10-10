"""Read the letters a photo actually contains, and only generate the rest.

The supported use case is lettering in the wild - a poster, a sign, a label - where
the goal is the style and invented letterforms are acceptable (README section 9).
But a title still *shows* you letters, and an extracted letter is ground truth
where a generated one is a guess. CANADA DRY gives C A N D R Y outright; Carmen's
card gives ten. Generating twenty from six real references is a far better
conditioned problem than generating twenty-five from one.

    photo + the text it spells
        -> find the lettering (by colour distance, not by assuming dark-on-light)
        -> split it into letters
        -> score every instance and keep the best of each character
        -> real glyphs out, plus the list of characters still missing

Scoring matters because a photograph is not a scan. On a curved bottle the letters
at the ends of a word are foreshortened and rotated, and copying one of those into
a font bakes the curvature in permanently. Every component of the score is
reported separately so a rejection can be argued with.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

import features as F
import glyphs as G
import handwriting as H


# ------------------------------------------------------------------- finding ink

def _flat_lab(crop: Image.Image, blur: float) -> np.ndarray:
    """CIELAB with the lighting divided out of L, a and b untouched.

    A shadow is a brightness gradient, so flattening L removes it; a and b carry
    the red-on-white or gold-on-green signal the whole method rests on.
    """
    lab = cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    if blur:
        lab = cv2.GaussianBlur(lab, (0, 0), blur)
    L = lab[..., 0]
    k = max(3, int(min(L.shape) * 0.25) | 1)
    lab[..., 0] = np.clip(L / np.maximum(cv2.GaussianBlur(L, (k, k), 0), 1e-3) * 128.0,
                          0, 255)
    return lab


def _clean(m: np.ndarray) -> np.ndarray:
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _letter_count(mask: np.ndarray) -> int:
    """How many letter-shaped boxes a mask segments into."""
    if mask.sum() < 50:
        return 0
    boxes = H.segment_page(mask, split=False)
    boxes, _ = drop_non_letters(boxes)
    return len(boxes)


def ink_by_colour(img: Image.Image, bbox=None, blur: float = 1.0,
                  n_letters: int | None = None, ink_rgb=None, k: int = 4):
    """Mask of the lettering, chosen as the colour cluster that reads as letters.

    "Background is the median colour" fails whenever the lettering covers a lot of
    the crop - on CANADA DRY the red and the lilac shield are near-equal in area,
    so the median falls between them and only the deepest-red edges survive.

    Instead: cluster the crop into k colours, segment each cluster's mask into
    letter boxes, and keep the cluster whose count lands closest to `n_letters`.
    Chroma breaks ties, because lettering on packaging is nearly always the
    saturated thing. `ink_rgb` overrides all of it with a hand-picked colour.
    """
    crop = img.crop(bbox) if bbox else img
    lab = _flat_lab(crop, blur)
    hgt, wid = lab.shape[:2]

    if ink_rgb is not None:
        target = cv2.cvtColor(np.uint8([[list(ink_rgb)]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        target[0] = 128.0                       # lightness was flattened
        dist = np.linalg.norm(lab - target, axis=2)
        return _clean(dist < 28.0), crop

    # k-means on a subsample: the crop can be a few million pixels, and 20k is
    # plenty to find four colours. Assign the full image to the found centres.
    px = lab.reshape(-1, 3)
    rng = np.random.RandomState(0)
    sample = px[rng.choice(len(px), size=min(20000, len(px)), replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centres = cv2.kmeans(sample, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    d = np.linalg.norm(px[:, None, :] - centres[None, :, :], axis=2)
    labels = d.argmin(axis=1).reshape(hgt, wid)

    best, best_key = None, None
    for c in range(k):
        m = _clean(labels == c)
        share = float(m.mean())
        if share < 0.01 or share > 0.70:
            continue
        chroma = float(np.hypot(centres[c][1] - 128, centres[c][2] - 128))
        count = _letter_count(m)
        miss = abs(count - n_letters) if n_letters else 0
        key = (miss, -chroma)                   # closest count first, then most saturated
        if best_key is None or key < best_key:
            best, best_key = m, key
    if best is None:                            # nothing plausible: fall back to minority
        counts = np.bincount(labels.ravel(), minlength=k)
        best = _clean(labels == int(np.argmin(np.where(counts > 0, counts, counts.max() + 1))))
    return best, crop


def drop_non_letters(boxes, min_h_frac: float = 0.35, max_aspect: float = 3.0):
    """Remove boxes that cannot be letters, and say which.

    A rule at a time, each with a reason: too short to be type at this size, or so
    much wider than it is tall that it is a rule, an underline or a bullet.
    """
    if not boxes:
        return [], []
    med_h = float(np.median([b[3] for b in boxes]))
    keep, dropped = [], []
    for b in boxes:
        x, y, w, h = b
        if h < min_h_frac * med_h:
            dropped.append((b, f"height {h} is under {min_h_frac:.0%} of the median {med_h:.0f}"))
        elif w > max_aspect * h:
            dropped.append((b, f"aspect {w / max(1, h):.1f}:1 is too wide for a letter"))
        else:
            keep.append(b)
    return keep, dropped


def split_to_count(mask: np.ndarray, boxes, n: int, inner=(0.25, 0.75)):
    """Split the widest boxes at their deepest ink valley until there are n.

    Used when the text says there are more letters than boxes: two letters are
    touching. The cut is confined to the middle of the box so a letter is never
    sliced at its own edge, and each split produces two tight boxes.
    """
    boxes = [tuple(int(v) for v in b) for b in boxes]
    while len(boxes) < n and boxes:
        i = max(range(len(boxes)), key=lambda j: boxes[j][2])
        x, y, w, h = boxes[i]
        sub = mask[y:y + h, x:x + w]
        prof = cv2.GaussianBlur(sub.sum(axis=0).astype(np.float32).reshape(1, -1),
                                (0, 0), 2.0).ravel()
        lo, hi = int(w * inner[0]), int(w * inner[1])
        if hi - lo < 4:
            break
        cut = lo + int(np.argmin(prof[lo:hi]))
        parts = []
        for a, b in ((0, cut), (cut, w)):
            seg = sub[:, a:b]
            if seg.sum() < 30:
                continue
            ys, xs = np.where(seg > 0)
            parts.append((x + a + int(xs.min()), y + int(ys.min()),
                          int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
        if len(parts) < 2:
            break
        boxes = boxes[:i] + parts + boxes[i + 1:]
        boxes.sort(key=lambda b: b[0])
    return boxes


def largest_text_row(mask: np.ndarray, min_frac: float = 0.20):
    """Keep the tallest run of components, i.e. the main line of type.

    A label carries several sizes at once - CANADA DRY also says DEPUIS 1904 and
    carries a health warning. The title is the tallest thing present.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask, []
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    tall = float(np.percentile(hs, 95))
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_HEIGHT] >= min_frac * tall]
    if not keep:
        return mask, []
    ys = [stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2 for i in keep]
    mid = float(np.median(ys))
    band = [i for i in keep
            if abs(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2 - mid)
            < 0.7 * tall and stats[i, cv2.CC_STAT_HEIGHT] > 0.45 * tall]
    if not band:
        band = keep
    return np.isin(lab, band).astype(np.uint8), band


# --------------------------------------------------------------------- scoring

@dataclass
class Candidate:
    char: str
    box: tuple
    mask: np.ndarray = field(repr=False)
    sharpness: float = 0.0      # edge energy per unit contrast; ~2.5 is crisp
    height_dev: float = 0.0     # |height - median| / median, among like letters
    baseline_dev: float = 0.0   # distance off the fitted baseline, in cap heights
    gap: float = 0.0            # distance to the nearest neighbour, in cap heights
    clipped: bool = False
    # How much height variation to forgive. Typeset lettering holds cap height to
    # a few percent, so a deviation there really is the camera. Handwriting does
    # not: Carmen's swash C and B stand 73% and 54% over her median.
    height_tol: float = 0.25

    @property
    def score(self) -> float:
        """0..1, as a product of independent penalties.

        Each term is 1.0 for a letter that is crisp, the right height, sitting on
        the line, clear of its neighbours and inside the frame.
        """
        s = 1.0
        s *= min(1.0, self.sharpness / 1.8)
        s *= max(0.0, 1.0 - self.height_dev / max(0.05, self.height_tol))
        s *= max(0.0, 1.0 - self.baseline_dev / 0.20)
        s *= 0.55 + 0.45 * min(1.0, self.gap / 0.06)
        if self.clipped:
            s *= 0.3
        return round(s, 3)

    def why(self) -> str:
        bits = []
        if self.sharpness < 1.2:
            bits.append(f"soft ({self.sharpness:.1f})")
        if self.height_dev > 0.4 * self.height_tol:
            bits.append(f"height off by {self.height_dev:.0%}")
        if self.baseline_dev > 0.08:
            bits.append(f"{self.baseline_dev:.2f} off the baseline")
        if self.gap < 0.03:
            bits.append("touching a neighbour")
        if self.clipped:
            bits.append("clipped by the frame")
        return ", ".join(bits) or "clean"


def _sharpness(page: Image.Image, mask: np.ndarray, box) -> float:
    """Local edge energy per unit of ink contrast: crisp letters score higher."""
    x, y, w, h = box
    g = np.asarray(page.convert("L"), np.float32)[y:y + h, x:x + w]
    sub = mask[y:y + h, x:x + w]
    if sub.sum() < 20:
        return 0.0
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    edge = cv2.dilate(sub, np.ones((3, 3), np.uint8)) - cv2.erode(sub, np.ones((3, 3), np.uint8))
    if edge.sum() < 10:
        return 0.0
    amp = float(g[sub > 0].std()) + 8.0
    return float(lap[edge > 0].mean() / amp)


def score_candidates(page: Image.Image, mask: np.ndarray, boxes, text: str,
                     height_tol: float = 0.25):
    """Pair each box with its character and measure how usable it is."""
    letters = [c for c in text if not c.isspace()]
    if len(boxes) != len(letters):
        raise SystemExit(f"{len(boxes)} letter boxes found but {len(letters)} characters "
                         f"in {text!r}. Check --text, or the segmentation preview, and "
                         f"use --bbox to narrow the crop.")

    cap = float(np.median([b[3] for b in boxes]))

    # Cap height is constant across capitals whatever the letter, so a deviation is
    # the camera's doing. Lowercase is judged against the lowercase median instead,
    # since an x-height letter is legitimately shorter than a capital.
    def class_of(c):
        return "upper" if c.isupper() else "lower" if c.islower() else "other"

    med_h = {}
    for cls in ("upper", "lower", "other"):
        hs = [b[3] for b, c in zip(boxes, letters) if class_of(c) == cls]
        if hs:
            med_h[cls] = float(np.median(hs))

    # A straight line through the letter bottoms. A flat surface keeps them on it;
    # a curved one does not.
    bx = np.array([b[0] + b[2] / 2 for b in boxes], np.float64)
    by = np.array([b[1] + b[3] for b in boxes], np.float64)
    if len(boxes) >= 3:
        slope, intercept = np.polyfit(bx, by, 1)
        fitted = slope * bx + intercept
    else:
        fitted = by

    out = []
    for i, ((x, y, w, h), ch) in enumerate(zip(boxes, letters)):
        left = boxes[i - 1][0] + boxes[i - 1][2] if i else None
        right = boxes[i + 1][0] if i + 1 < len(boxes) else None
        gaps = [g for g in ((x - left) if left is not None else None,
                            (right - (x + w)) if right is not None else None)
                if g is not None]
        gap = (min(gaps) / cap) if gaps else 1.0
        ref_h = med_h.get(class_of(ch), cap)
        out.append(Candidate(
            char=ch, box=(x, y, w, h), mask=mask[y:y + h, x:x + w].copy(),
            sharpness=_sharpness(page, mask, (x, y, w, h)),
            height_dev=abs(h - ref_h) / max(1.0, ref_h),
            baseline_dev=abs(by[i] - fitted[i]) / max(1.0, cap),
            gap=float(gap),
            clipped=(x <= 1 or y <= 1 or x + w >= mask.shape[1] - 1
                     or y + h >= mask.shape[0] - 1),
            height_tol=height_tol,
        ))
    return out


def best_per_char(cands):
    """Keep the highest-scoring instance of each character."""
    best = {}
    for c in cands:
        if c.char not in best or c.score > best[c.char].score:
            best[c.char] = c
    return best


# ----------------------------------------------------------------------- driver

def extract(photo: str, text: str, bbox: str | None = None, outdir: str | None = None,
            max_side: int = 2600, split: bool = True, source: str = "colour",
            drop: tuple = (), height_tol: float | None = None, ink_rgb=None):
    """Photo + the text it spells -> candidates, one per letter instance.

    `source` is "colour" for lettering in the wild, or "paper" for something
    written on a page; see the module note for why neither substitutes for the
    other.
    """
    if source == "paper":
        crop, _flat, mask, _ = H.prepare(photo, None)
        if bbox:
            import pipeline
            b = pipeline.parse_box(bbox, crop.size)
            keep = np.zeros_like(mask)
            keep[b[1]:b[3], b[0]:b[2]] = 1
            mask = mask * keep
    else:
        img = H.load_photo(photo, max_side=max_side)
        box = None
        if bbox:
            import pipeline
            box = pipeline.parse_box(bbox, img.size)
        n = len([c for c in text if not c.isspace()])
        mask, crop = ink_by_colour(img, box, n_letters=n, ink_rgb=ink_rgb)
        mask, _ = largest_text_row(mask)

    boxes = H.segment_page(mask, split=split)
    boxes, dropped = drop_non_letters(boxes)
    for b, why in dropped:
        print(f"  dropped box {b}: {why}")
    if drop:
        boxes = [b for i, b in enumerate(boxes) if i not in set(drop)]
    n_letters = len([c for c in text if not c.isspace()])
    if split and len(boxes) < n_letters:
        before = len(boxes)
        boxes = split_to_count(mask, boxes, n_letters)
        print(f"  {before} boxes for {n_letters} letters: split touching pairs -> {len(boxes)}")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        crop.save(os.path.join(outdir, "01_crop.jpg"), quality=92)
        Image.fromarray((1 - mask) * 255).save(os.path.join(outdir, "02_ink.png"))
        prev = crop.convert("RGB").copy()
        d = ImageDraw.Draw(prev)
        for i, (x, y, w, h) in enumerate(boxes):
            d.rectangle((x, y, x + w, y + h), outline=(40, 120, 240),
                        width=max(2, prev.width // 500))
            d.text((x + 3, max(0, y - 13)), str(i), fill=(40, 120, 240))
        prev.save(os.path.join(outdir, "02b_boxes.jpg"), quality=92)

    if height_tol is None:
        height_tol = 0.60 if source == "paper" else 0.25
    cands = score_candidates(crop, mask, boxes, text, height_tol)
    # A box that touches the crop edge is only clipped if that edge is also the
    # photo's edge. Cropping to a word puts letters at the crop edge by design.
    if source != "paper" and box is not None:
        ox, oy = box[0], box[1]
        pw, ph = img.size
        for c in cands:
            x, y, w, h = c.box
            c.clipped = (ox + x <= 1 or oy + y <= 1
                         or ox + x + w >= pw - 1 or oy + y + h >= ph - 1)
    if outdir:
        prev = crop.convert("RGB").copy()
        d = ImageDraw.Draw(prev)
        for c in cands:
            x, y, w, h = c.box
            col = (60, 190, 90) if c.score >= 0.5 else (235, 160, 40) if c.score >= 0.25 \
                else (230, 60, 60)
            d.rectangle((x, y, x + w, y + h), outline=col, width=max(2, prev.width // 500))
            d.text((x + 3, max(0, y - 13)), f"{c.char} {c.score:.2f}", fill=col)
        prev.save(os.path.join(outdir, "03_candidates.jpg"), quality=92)
    return cands, crop, mask


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", required=True)
    ap.add_argument("--text", required=True, help="the characters the photo spells, in order")
    ap.add_argument("--bbox", help="x,y,w,h to narrow the crop before segmenting")
    ap.add_argument("--outdir", default="out/capture")
    ap.add_argument("--min-score", type=float, default=0.25,
                    help="reject a letter below this; it goes on the missing list instead")
    ap.add_argument("--no-split", action="store_true", help="do not split touching letters")
    ap.add_argument("--source", choices=["colour", "paper"], default="colour",
                    help="'colour' finds lettering by colour distance (posters, labels, "
                         "signs); 'paper' runs the document pipeline first (find the page, "
                         "rectify, flatten the lighting). Pick the one that matches the "
                         "photo - neither is a fallback for the other")
    ap.add_argument("--drop", default="", help="comma-separated box indices to ignore, "
                                               "read off 02b_boxes.jpg")
    ap.add_argument("--ink-colour", default="", metavar="R,G,B",
                    help="hand-pick the lettering colour instead of clustering for it")
    ap.add_argument("--height-tol", type=float, default=None,
                    help="how much cap-height variation to forgive (default 0.25 for "
                         "--source colour, 0.60 for --source paper, where capitals really "
                         "do vary)")
    ap.add_argument("--charset", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                    help="what a complete font would need, to report what is missing")
    ap.add_argument("--font-out", default="")
    a = ap.parse_args()

    drop = tuple(int(x) for x in a.drop.split(",") if x.strip())
    cands, crop, mask = extract(a.photo, a.text, a.bbox, a.outdir, split=not a.no_split,
                                source=a.source, drop=drop, height_tol=a.height_tol,
                                ink_rgb=(tuple(int(v) for v in a.ink_colour.split(","))
                                         if a.ink_colour else None))
    print(f"{len(cands)} letter instances\n")
    print(f"{'ch':>3} {'score':>6}  {'sharp':>6}{'height':>8}{'baseline':>10}{'gap':>7}   why")
    for c in cands:
        print(f"{c.char:>3} {c.score:>6.2f}  {c.sharpness:>6.2f}{c.height_dev:>8.2f}"
              f"{c.baseline_dev:>10.2f}{c.gap:>7.2f}   {c.why()}")

    best = best_per_char(cands)
    good = {ch: c for ch, c in best.items() if c.score >= a.min_score}
    have = "".join(sorted(good))
    missing = "".join(c for c in a.charset if c not in good)
    print(f"\nusable from the photo ({len(good)}): {have}")
    print(f"still to generate ({len(missing)}): {missing}")

    if good:
        masks = {ch: G.normalise(c.mask) for ch, c in good.items()}
        import build_font as BF
        BF.specimen(masks, os.path.join(a.outdir, "_extracted.png"))
        print(f"extracted silhouettes -> {os.path.join(a.outdir, '_extracted.png')}")
        if a.font_out:
            print("partial font ->", G.build_font(masks, a.font_out, family="Captured"))

    f = F.measure_page(mask, [c.box for c in cands], None)
    if f:
        # measure_page's white balance divides by the local mean, and on a crop
        # where the lettering covers half the area that mean IS the ink colour -
        # red came back "pale grey". Read it off the raw crop instead.
        f.ink_rgb = F._median_ink(mask, crop)
        print(f"\nmeasured lettering: {f}")


if __name__ == "__main__":
    main()
