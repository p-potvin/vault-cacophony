"""Measure lettering features from a reference image, to put numbers in the prompt.

This measures a reference letter so runs can be *scored* against the hand they
were meant to copy. It also knows how to phrase what it measured for a prompt,
but that is off by default - see below.

Measured on Carmen's page (A/E/O, same seed and box):

    plain edit instruction   ratio 0.059   modulation 0.99   saturation 0.75
    + measured features      ratio 0.025   modulation 0.95   saturation 0.04
    her hand                 ratio 0.102   modulation 0.12   saturation 0.16

Colour responded; weight did not, and two different wordings landed on the same
ratio at a quarter of the target - further from her hand than the plain
instruction had been. Two reasons. Structurally, the sampler runs at cfg 1.0, so
there is no guidance scale to give text leverage over the composite image. More
importantly the wording had drifted off-model: Qwen-Image-Edit is instruction-
tuned and wants a direct edit operation, where describe() produces a caption of
the desired result, and a phrase like "about a tenth as thick" is unlikely to
survive BPE as a quantity cross-attention can act on.

So: measure for scoring, and leave prompt-writing to the plain instruction unless
--features is passed explicitly.

Weight is measured with the distance transform. For a stroke of uniform width
*w*, the distance-to-edge of its interior pixels is uniform on (0, w/2], so the
mean is w/4 and

    w = 4 * mean(distance transform over the ink)

with no skeletonisation needed (cv2 has no thinning outside contrib). Verified
against synthetic bars: 20px -> 21.5, 40px -> 39.8, and it survives corners and
curvature (an L and a ring of width 20 read 21.7 and 22.2).

What goes in the prompt is not the absolute width but the *ratio* of stroke width
to cap height, because that is what survives being rescaled into the box.

    ..\\..\\.venv\\Scripts\\python.exe features.py out/clone/_reference.png
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# Stroke-width-to-cap-height ratios. Calibrated so the Arial family lands where
# its name says it does - measured on a capital M at 400px: Arial 0.126, Arial
# Bold 0.193, Arial Black 0.245, Times 0.120. The words are ones an image model
# has actually seen in captions, so they steer it rather than confuse it.
WEIGHT_BANDS: list[tuple[float, str]] = [
    (0.045, "hairline"),
    (0.065, "very thin"),
    (0.085, "thin"),
    (0.110, "light"),
    (0.150, "regular weight"),
    (0.185, "medium weight"),
    (0.225, "bold"),
    (0.270, "very bold"),
    (1.000, "extremely heavy"),
]

# Ink colour, named for the prompt. Saturated ink gets a hue name; unsaturated
# ink is described by lightness, which is what pencil and pen actually are.
_HUES = [(15, "red"), (45, "orange"), (70, "yellow"), (160, "green"),
         (200, "teal"), (260, "blue"), (290, "violet"), (340, "magenta"), (361, "red")]

# Denominators as words: "one tenth as thick as the letter is tall" is a target the
# model can aim at, where the band word "light" only says which way to go.
_FRACTIONS = {2: "half", 3: "a third", 4: "a quarter", 5: "a fifth", 6: "a sixth",
              7: "a seventh", 8: "an eighth", 9: "a ninth", 10: "a tenth",
              11: "an eleventh", 12: "a twelfth", 14: "a fourteenth",
              15: "a fifteenth", 16: "a sixteenth", 18: "an eighteenth",
              20: "a twentieth", 25: "a twenty-fifth", 30: "a thirtieth"}


def _fraction_word(ratio: float) -> str:
    """0.102 -> 'a tenth'. Snapped to the nearest word we have."""
    n = max(2, int(round(1.0 / max(ratio, 1e-6))))
    key = min(_FRACTIONS, key=lambda k: abs(k - n))
    return _FRACTIONS[key]


def _rgb_hue(r: int, g: int, b: int) -> float:
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d == 0:
        return 0.0
    if mx == r:
        h = 60 * (((g - b) / d) % 6)
    elif mx == g:
        h = 60 * ((b - r) / d + 2)
    else:
        h = 60 * ((r - g) / d + 4)
    return h % 360


@dataclass
class StrokeFeatures:
    """Everything measurable from one binary letterform."""

    width_px: float          # stroke width in the reference's own pixels
    cap_px: int              # letter height in the same pixels
    ratio: float             # width / height - the scale-invariant number
    modulation: float        # spread of the width along the stroke (0 = monoline)
    ink_frac: float          # ink pixels / bounding-box area
    ink_rgb: tuple[int, int, int] | None = None   # median ink colour, if measured

    @property
    def weight(self) -> str:
        for hi, name in WEIGHT_BANDS:
            if self.ratio < hi:
                return name
        return WEIGHT_BANDS[-1][1]

    @property
    def monoline(self) -> bool:
        """True when the stroke barely changes width - pencil, pen, marker.

        The 0.28 cut separates Arial (0.09) from Times (0.43), which is the
        distinction being made: constant width versus a modulated nib.
        """
        return self.modulation < 0.28

    @property
    def colour(self) -> str | None:
        """A colour phrase for the ink, or None if it was never measured."""
        if self.ink_rgb is None:
            return None
        r, g, b = (int(v) for v in self.ink_rgb)
        mx, mn = max(r, g, b), min(r, g, b)
        sat = (mx - mn) / max(1, mx)
        if sat < 0.18:
            return ("black" if mx < 70 else "dark grey" if mx < 130
                    else "grey" if mx < 190 else "pale grey")
        name = next(n for hi, n in _HUES if _rgb_hue(r, g, b) < hi)
        return f"{'dark ' if mx < 110 else ''}{name}"

    def describe(self, colour: bool = True) -> str:
        """The phrase to drop into a prompt."""
        stroke = ("even strokes of constant width" if self.monoline
                  else "strokes that thicken and thin")
        bits = [f"{self.weight} handwriting in {stroke}",
                f"each stroke about {_fraction_word(self.ratio)} as thick as the "
                f"letter is tall"]
        if colour and self.colour:
            bits.append(f"drawn in {self.colour}")
        return ", ".join(bits)

    def constraints(self) -> str:
        """How the strokes must be drawn, given what was measured.

        Entirely positive. The graph samples at cfg 1.0 with the distilled
        Lightning LoRA and an empty negative, so there is no negative branch for
        the sampler to subtract - every word here goes into the *positive*
        prompt, where "no outline" is just the token "outline".

        Off by default all the same: see the module docstring. Qwen-Image-Edit
        wants an edit instruction, and this is a description.
        """
        if not self.monoline:
            return ""
        return ("Every stroke is one solid line of flat, even colour, filled in "
                "completely and drawn in a single pass, like ordinary handwriting "
                "on paper.")

    def __str__(self) -> str:
        c = f", ink {self.colour}" if self.colour else ""
        return (f"{self.weight} (stroke {self.width_px:.1f}px on a {self.cap_px}px letter, "
                f"ratio {self.ratio:.3f}), "
                f"{'monoline' if self.monoline else 'modulated'} "
                f"(modulation {self.modulation:.2f}){c}")


def white_balance(page: Image.Image, blur_frac: float = 0.06) -> Image.Image:
    """Put the paper back at neutral so the ink's remaining tint is its own.

    Per-channel division by a blurred copy, which is the colour counterpart of
    handwriting.flatten(). Without it a photo taken in blue shade reports blue
    ink no matter what was used.
    """
    a = np.asarray(page.convert("RGB"), np.float32)
    k = max(3, int(min(page.size) * blur_frac) | 1)
    out = np.empty_like(a)
    for c in range(3):
        bg = cv2.GaussianBlur(a[..., c], (k, k), 0)
        out[..., c] = a[..., c] / np.maximum(bg, 1.0) * 235.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _median_ink(mask: np.ndarray, rgb: Image.Image, core_frac: float = 0.35):
    """Median colour of the darkest `core_frac` of the ink.

    The whole mask includes the antialiased skirt, which is mostly paper and
    drags every measurement toward white; the core is the stroke itself.
    """
    a = np.asarray(rgb.convert("RGB"))
    if a.shape[:2] != mask.shape:
        return None
    px = a[mask > 0]
    if not len(px):
        return None
    lum = px.mean(axis=1)
    core = px[lum <= np.percentile(lum, core_frac * 100)]
    return tuple(int(v) for v in np.median(core if len(core) else px, axis=0))


def _binary(src) -> np.ndarray:
    """Ink mask from an RGBA cut-out, an L image, or an array."""
    if isinstance(src, np.ndarray):
        return (src > 0).astype(np.uint8)
    im = src if isinstance(src, Image.Image) else Image.open(src)
    if "A" in im.getbands():
        return (np.asarray(im.getchannel("A"), np.uint8) > 96).astype(np.uint8)
    g = np.asarray(im.convert("L"), np.uint8)
    # No alpha, so the polarity is unknown: dark pen on paper and light gold on a
    # dark plate are both normal here. Lettering is the minority class in either
    # case, so split with Otsu and keep whichever side is smaller.
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    m = (bw > 0).astype(np.uint8)
    return m if m.mean() < 0.5 else (1 - m).astype(np.uint8)


def stroke_features(src, min_ink: int = 40,
                    rgb: Image.Image | None = None) -> StrokeFeatures | None:
    """Measure one letterform. `src` is a path, PIL image or binary array.

    Pass `rgb` (an aligned colour image) to also measure the ink colour.
    """
    m = _binary(src)
    if int(m.sum()) < min_ink:
        return None

    ys, xs = np.where(m > 0)
    cap = int(ys.max() - ys.min() + 1)
    bbox_area = cap * int(xs.max() - xs.min() + 1)

    # Pad so the distance transform does not treat the image border as an ink edge.
    p = cv2.copyMakeBorder(m, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    dt = cv2.distanceTransform(p, cv2.DIST_L2, 5)
    width = 4.0 * float(dt[p > 0].mean())

    # Ridge pixels approximate the skeleton. Their *count* is not a usable stroke
    # length (an even width leaves a two-pixel plateau, which halves any
    # area-over-length estimate - that cross-check was wrong and is gone), but the
    # distances along them are the local stroke width, which is what modulation is.
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.float32)) - 1e-4) & (p > 0)
    rid = 2.0 * dt[ridge]
    modulation = float(rid.std() / max(1e-6, rid.mean()))

    ink_rgb = _median_ink(m, rgb) if rgb is not None else None

    return StrokeFeatures(width_px=width, cap_px=cap, ratio=width / max(1, cap),
                          modulation=modulation,
                          ink_frac=float(m.sum()) / max(1, bbox_area), ink_rgb=ink_rgb)


def measure_page(ink: np.ndarray, boxes,
                 page: Image.Image | None = None) -> StrokeFeatures | None:
    """Aggregate over every letter on a prepared page - steadier than one letter.

    A single glyph can be atypical (an I is all stem, an O all curve); the median
    over the line is what you want when the whole page is available.
    """
    feats = [f for f in (stroke_features(ink[y:y + h, x:x + w]) for x, y, w, h in boxes) if f]
    if not feats:
        return None

    def med(key):
        return float(np.median([getattr(f, key) for f in feats]))

    ink_rgb = _median_ink(ink, white_balance(page)) if page is not None else None

    return StrokeFeatures(width_px=med("width_px"), cap_px=int(med("cap_px")),
                          ratio=med("ratio"), modulation=med("modulation"),
                          ink_frac=med("ink_frac"), ink_rgb=ink_rgb)


def selftest() -> None:
    """Check the estimator against shapes of known width, and against real fonts."""
    from PIL import ImageDraw, ImageFont
    print("synthetic (measured should track the true width):")
    for w in (5, 12, 20, 40):
        m = np.zeros((300, 300), np.uint8)
        m[30:270, 100:100 + w] = 1
        print(f"  bar w={w:<3d} -> {stroke_features(m).width_px:5.1f}")
    m = np.zeros((300, 300), np.uint8)
    cv2.circle(m, (150, 150), 100, 1, 20)
    print(f"  ring w=20 -> {stroke_features(m).width_px:5.1f}")

    print("real faces (the band names should match the font names):")
    for name in ("arial.ttf", "arialbd.ttf", "ariblk.ttf", "times.ttf"):
        try:
            ft = ImageFont.truetype("C:/Windows/Fonts/" + name, 400)
        except OSError:
            continue
        im = Image.new("L", (600, 600), 0)
        ImageDraw.Draw(im).text((60, 20), "M", 255, font=ft)
        f = stroke_features(np.asarray(im) > 128)
        print(f"  {name:<12s} ratio {f.ratio:.3f} -> {f.weight:<16s} "
              f"modulation {f.modulation:.2f} -> "
              f"{'monoline' if f.monoline else 'modulated'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Measure stroke weight of a letter cut-out.")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    for path in a.images:
        f = stroke_features(path, rgb=white_balance(Image.open(path)))
        print(f"{path}\n  {f}\n  prompt phrase: {f.describe()!r}" if f else f"{path}\n  no ink")
