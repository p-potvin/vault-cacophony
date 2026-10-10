"""Turn generated styled lettering into a usable font.

    rendered letter on a photo  ->  silhouette  ->  normalised  ->  contours  ->  TTF

Extraction does not need a segmentation model here. Every alphabet output was
produced from the *same* untouched target, so that plate is a perfect clean
background: differencing against it says exactly which pixels the model drew, and
a warm/bright colour test then separates the glyph *face* from its extrusion,
contour and cast shadow. SAM3 is available in the local ComfyUI and can be
swapped in for cases where no clean plate exists, but with one it can only add
error, so it is not used by default.

The face is what becomes the glyph: the extrusion is a 3D lighting effect, not
part of the letterform.

Normalisation matters because the model does not preserve cap height - measured
across A-Z the drawn glyphs varied 1.43x in height and ~4x in inked area, so each
silhouette is rescaled to a common cap height and sat on a common baseline before
it becomes a glyph.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from PIL import Image

UPM = 1000            # font units per em
CAP_HEIGHT = 700      # target cap height in font units
CAP_HEIGHT_PX = 700   # cap height normalise() renders at, in pixels
SIDE_BEARING = 40     # left/right space in font units


# ---------------------------------------------------------------- extraction

def face_mask(img: Image.Image, plate: Image.Image,
              diff_blur: float = 6.0, diff_thresh: int = 45,
              sat_min: float = 0.30, lum_min: float = 110.0,
              min_area_frac: float = 0.02, palette: str = "auto") -> np.ndarray:
    """Binary mask of the glyph's front face.

    `plate` is the untouched target the letter was drawn onto.

    `palette` picks how the letterform is told apart from its own extrusion and
    shadow inside the changed region:

      warm   the gold/metal case - bright, saturated, warm face
      dark   dark strokes on a light ground (pencil, ink)
      light  light strokes on a dark ground
      auto   decide from the data (default)

    Auto matters because the pipeline is not gold-specific: a photo of pencil
    handwriting has no bright saturated face at all, and the warm test would
    return an empty mask.
    """
    a = np.asarray(img.convert("RGB"), np.float32)
    p = np.asarray(plate.convert("RGB"), np.int16)
    if a.shape != p.shape:
        raise ValueError(f"plate {p.shape} does not match image {a.shape}")

    d = np.abs(p - a.astype(np.int16)).max(axis=2).astype(np.uint8)
    changed = cv2.GaussianBlur(d, (0, 0), diff_blur) > diff_thresh

    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    lum_p = np.asarray(plate.convert("RGB"), np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)

    mode = palette
    if mode == "auto":
        if not changed.any():
            return np.zeros(changed.shape, np.uint8)
        warm_share = float(((r >= g) & (g >= b) & (sat > sat_min) & (lum > lum_min))[changed].mean())
        if warm_share > 0.25:
            mode = "warm"
        else:
            mode = "dark" if lum[changed].mean() < lum_p[changed].mean() else "light"

    if mode == "warm":
        m = changed & (r >= g) & (g >= b) & (sat > sat_min) & (lum > lum_min)
    elif mode == "dark":
        # the ink is the darker half of what changed
        thr = float(np.percentile(lum[changed], 55))
        m = changed & (lum <= thr)
    elif mode == "light":
        thr = float(np.percentile(lum[changed], 45))
        m = changed & (lum >= thr)
    else:
        raise ValueError(f"unknown palette {palette!r}")

    # keep only components that are a real part of the letter
    m8 = m.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m8, 8)
    if n <= 1:
        return m8
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = areas.max()
    keep = {i + 1 for i, a_ in enumerate(areas) if a_ >= max(30, min_area_frac * biggest)}
    return np.isin(lab, list(keep)).astype(np.uint8)


# ------------------------------------------------------------- normalisation

def body_rows(mask: np.ndarray, frac: float = 0.22):
    """Rows belonging to the letter body, excluding thin descenders/ascenders.

    Q's tail is ~5% of its bounding box and O/I/J have none, so scaling by the
    bounding box shrinks Q relative to every other capital. Scaling by the body
    and letting the tail hang below the baseline is what a font actually wants.
    """
    rw = mask.sum(axis=1)
    if rw.max() == 0:
        raise ValueError("empty mask")
    rows = np.where(rw > frac * rw.max())[0]
    return int(rows.min()), int(rows.max())


def normalise(mask: np.ndarray, cap_px: int = 700, pad: int = 60) -> np.ndarray:
    """Crop to the glyph and scale so the *body* is `cap_px` tall.

    Returns a mask whose baseline sits at `pad + cap_px` from the top, so any
    descender occupies the rows below it.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("empty mask")
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    top, bot = body_rows(crop)
    s = cap_px / max(1, bot - top + 1)

    nw, nh = max(1, int(round(crop.shape[1] * s))), max(1, int(round(crop.shape[0] * s)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    crop = (crop > 0).astype(np.uint8)

    top_s = int(round(top * s))
    descender = nh - int(round((bot + 1) * s))
    out = np.zeros((cap_px + 2 * pad + max(0, descender), nw + 2 * pad), np.uint8)
    y = pad - top_s
    y0d, y1d = max(0, y), min(out.shape[0], y + nh)
    out[y0d:y1d, pad:pad + nw] = crop[y0d - y:y1d - y]
    return out


# ------------------------------------------------------------- vectorisation

def place_on_baseline(mask: np.ndarray, baseline_y: float, scale: float,
                      pad: int = 60) -> np.ndarray:
    """Scale a glyph by a *shared* factor and sit it on the common baseline.

    `baseline_y` is where the baseline falls in `mask`'s own rows. Unlike
    normalise(), nothing here is measured per-glyph, so an alphabet keeps its
    relative proportions: x-height stays shorter than cap height, ascenders rise
    above it and descenders drop below.
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.zeros((CAP_HEIGHT_PX + 2 * pad, 2 * pad), np.uint8)

    sw = max(1, int(round(mask.shape[1] * scale)))
    sh = max(1, int(round(mask.shape[0] * scale)))
    m = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    base = baseline_y * scale

    ys, xs = np.where(m > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # Canvas: cap height above the baseline plus whatever this glyph needs, and
    # room below for a descender.
    above = max(CAP_HEIGHT_PX, int(round(base - y0)))
    below = max(0, int(round(y1 - base)))
    H = above + below + 2 * pad
    W = (x1 - x0 + 1) + 2 * pad
    out = np.zeros((H, W), np.uint8)

    oy = int(round(pad + above - base))
    ox = pad - x0
    ys0, xs0 = max(0, oy), max(0, ox)
    ys1, xs1 = min(H, oy + m.shape[0]), min(W, ox + m.shape[1])
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = m[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
    return out


def shared_scale(heights: dict, cap_px: int = CAP_HEIGHT_PX) -> float:
    """One scale for a whole set, taken from the capitals.

    `heights` maps a character to its measured baseline-to-top height in source
    pixels. Capitals define cap height; if none were supplied, fall back to the
    tallest thing present.
    """
    caps = [h for c, h in heights.items() if c.isupper() and h > 0]
    ref = float(np.median(caps)) if caps else max(list(heights.values()) + [1])
    return cap_px / max(1.0, ref)


def contours(mask: np.ndarray, simplify: float = 1.2):
    """Outer contours and their holes, as integer point lists in image space."""
    cs, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    for i, c in enumerate(cs):
        if cv2.contourArea(c) < 12:
            continue
        eps = simplify * cv2.arcLength(c, True) / 1000.0
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            out.append((approx, hier[i][3] != -1))   # (points, is_hole)
    return out


def glyph_from_mask(mask: np.ndarray, simplify: float = 1.2, pad: int = 60):
    """Convert a normalised mask to (TTGlyphPen glyph, advance width)."""
    pen = TTGlyphPen(None)
    # image space (y down) -> font units (y up); the body occupies exactly
    # CAP_HEIGHT px starting at `pad`, so the baseline is pad + CAP_HEIGHT.
    scale = CAP_HEIGHT / CAP_HEIGHT_PX
    baseline = pad + CAP_HEIGHT_PX

    for pts, is_hole in contours(mask, simplify):
        # TrueType fills by non-zero winding: outers one way, holes the other.
        area = cv2.contourArea(pts.reshape(-1, 1, 2).astype(np.int32), oriented=True)
        ordered = pts if ((area < 0) != is_hole) else pts[::-1]
        pen.moveTo((float((ordered[0][0] - pad) * scale),
                    float((baseline - ordered[0][1]) * scale)))
        for x, y in ordered[1:]:
            pen.lineTo((float((x - pad) * scale), float((baseline - y) * scale)))
        pen.closePath()

    xs = np.where(mask > 0)[1]
    width = int((xs.max() - xs.min() + 1) * scale) + 2 * SIDE_BEARING
    return pen.glyph(), width


def advance_for(body: np.ndarray, pad: int = 60) -> int:
    """Advance width measured on the body, which is what the artwork occupies."""
    xs = np.where(body > 0)[1]
    if len(xs) == 0:
        return SIDE_BEARING * 2
    scale = CAP_HEIGHT / CAP_HEIGHT_PX
    return int((xs.max() - xs.min() + 1) * scale) + 2 * SIDE_BEARING


# ------------------------------------------------------------- font assembly

def build_font(glyphs: dict[str, np.ndarray], out_path: str,
               family: str = "Cacophony Cloned", style: str = "Regular",
               simplify: float = 1.2, pad: int = 60,
               advances: dict[str, int] | None = None) -> str:
    """`glyphs` maps a character to its normalised mask.

    `advances` overrides the per-glyph advance width. Pass the *drawn* extent
    (face plus extrusion) rather than the letterform alone, or colour artwork
    overlaps its neighbours.
    """
    order = [".notdef", "space"] + [f"uni{ord(c):04X}" for c in glyphs]
    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({0x20: "space", **{ord(c): f"uni{ord(c):04X}" for c in glyphs}})

    pen = TTGlyphPen(None)
    glyf = {".notdef": pen.glyph(), "space": pen.glyph()}
    metrics = {".notdef": (SIDE_BEARING * 2, 0), "space": (UPM // 3, 0)}
    for ch, m in glyphs.items():
        g, w = glyph_from_mask(m, simplify, pad)
        name = f"uni{ord(ch):04X}"
        glyf[name] = g
        metrics[name] = (int((advances or {}).get(ch, w)), SIDE_BEARING)

    fb.setupGlyf(glyf)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=int(UPM * 0.8), descent=-int(UPM * 0.2))
    fb.setupNameTable({"familyName": family, "styleName": style,
                       "fullName": f"{family} {style}",
                       "psName": f"{family.replace(' ', '')}-{style}",
                       "version": "1.0"})
    fb.setupOS2(sTypoAscender=int(UPM * 0.8), sTypoDescender=-int(UPM * 0.2),
                usWinAscent=int(UPM * 0.8), usWinDescent=int(UPM * 0.2),
                sCapHeight=CAP_HEIGHT)
    fb.setupPost()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fb.save(out_path)
    return out_path


# ------------------------------------------------------- colour (OT-SVG) font

def body_mask(img: Image.Image, plate: Image.Image, face: np.ndarray | None = None,
              strong_thresh: int = 45) -> np.ndarray:
    """Face + extrusion + contour: the solid parts of the glyph, no glow.

    Taken as the strongly-changed region *connected to the gold face*. The soft
    glow the model paints around the lettering is a weak change and is excluded,
    which is what stops background bleeding into the colour font.
    """
    a = np.asarray(img.convert("RGB"), np.int16)
    p = np.asarray(plate.convert("RGB"), np.int16)
    d = np.abs(p - a).max(axis=2).astype(np.uint8)
    strong = (cv2.GaussianBlur(d, (0, 0), 1.5) > strong_thresh)

    # Colour does the discriminating, not difference magnitude. Where gold lies over
    # a light pink feather the max-channel difference is only ~76, so a threshold
    # high enough to reject the glow was also punching holes in the letter face.
    #
    # What separates them is the BLUE channel: gold (198,173,116) and the brown
    # extrusion and near-black contour all have b well under r, while the pastel
    # background - including pink (211,166,192), which passes a naive r>b test - does
    # not. Specular highlights blow out toward white, so they are accepted separately.
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(axis=2), a.min(axis=2)
    warm = b < (0.80 * r + 10)
    spec = (mx > 200) & ((mx - mn) < 50)
    strong = (strong & (warm | spec)).astype(np.uint8)
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    if face is None:
        face = face_mask(img, plate)
    n, lab, _, _ = cv2.connectedComponentsWithStats(strong, 8)
    keep = set(np.unique(lab[face > 0])) - {0}
    if not keep:
        return face
    return np.isin(lab, list(keep)).astype(np.uint8)


def effect_layer(img: Image.Image, plate: Image.Image,
                 shadow: bool = False, shadow_offset=(14, 18),
                 shadow_blur: float = 9.0, shadow_alpha: float = 0.45) -> Image.Image:
    """The glyph as rendered - face, extrusion, contour, shadow - as clean RGBA.

    Alpha follows the glyph's structure, not the raw difference, so no background
    fringe travels with it. The cast shadow is re-synthesised as black at partial
    alpha from how much darker the output is than the plate, which reproduces the
    shading without carrying any background colour.
    """
    a8 = np.asarray(img.convert("RGB"), np.uint8)

    face = face_mask(img, plate)
    body = body_mask(img, plate, face)

    alpha = (body * 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
    rgb = a8.copy()

    if shadow:
        # Synthesised from the silhouette, not measured off the image. Measuring it
        # picks up the background's own texture (the model brightens as well as
        # darkens around the glyph), which showed up as a grey halo. Off by default:
        # the extrusion already carries the depth, and a wide soft shadow collides
        # with neighbouring glyphs in running text.
        dx, dy = shadow_offset
        sh = np.zeros_like(body)
        h, w = body.shape
        ys0, ys1 = max(0, dy), min(h, h + dy)
        xs0, xs1 = max(0, dx), min(w, w + dx)
        sh[ys0:ys1, xs0:xs1] = body[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        sh = cv2.GaussianBlur((sh * 255).astype(np.uint8), (0, 0), shadow_blur)
        sh = (sh.astype(np.float32) * shadow_alpha).astype(np.uint8)
        sh = np.where(body > 0, 0, sh).astype(np.uint8)
        rgb = np.where((sh > 0)[..., None] & (alpha[..., None] == 0), 0, rgb).astype(np.uint8)
        alpha = np.maximum(alpha, sh)

    out = Image.fromarray(rgb).convert("RGBA")
    out.putalpha(Image.fromarray(alpha))
    return out


def normalise_like(mask: np.ndarray, other: np.ndarray,
                   cap_px: int = CAP_HEIGHT_PX, pad: int = 60) -> np.ndarray:
    """Apply the transform normalise(mask) uses, to a second aligned mask."""
    ys, xs = np.where(mask > 0)
    x0, y0 = int(xs.min()), int(ys.min())
    crop = mask[y0:ys.max() + 1, x0:xs.max() + 1]
    top, bot = body_rows(crop)
    s = cap_px / max(1, bot - top + 1)
    ref = normalise(mask, cap_px, pad)
    H, W = ref.shape
    o = cv2.resize(other, (max(1, int(round(other.shape[1] * s))),
                           max(1, int(round(other.shape[0] * s)))),
                   interpolation=cv2.INTER_NEAREST)
    out = np.zeros((H, W), np.uint8)
    ox, oy = int(round(-x0 * s)) + pad, int(round(-(y0 + top) * s)) + pad
    xs0, ys0 = max(0, ox), max(0, oy)
    xs1, ys1 = min(W, ox + o.shape[1]), min(H, oy + o.shape[0])
    if xs1 > xs0 and ys1 > ys0:
        out[ys0:ys1, xs0:xs1] = o[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
    return out


def normalise_pair(mask: np.ndarray, colour: Image.Image,
                   cap_px: int = CAP_HEIGHT_PX, pad: int = 60):
    """Normalise the face mask and the colour layer through one shared transform.

    The colour art must land on exactly the same baseline and cap height as the
    outline, or the two layers of the font disagree.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("empty mask")
    x0, y0 = int(xs.min()), int(ys.min())
    crop = mask[y0:ys.max() + 1, x0:xs.max() + 1]
    top, bot = body_rows(crop)
    s = cap_px / max(1, bot - top + 1)

    norm_mask = normalise(mask, cap_px, pad)
    H, W = norm_mask.shape

    # place the colour art using the same origin (x0, y0 + top) and scale
    art = colour.resize((max(1, int(round(colour.width * s))),
                         max(1, int(round(colour.height * s)))), Image.LANCZOS)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.alpha_composite(art, (int(round(-x0 * s)) + pad,
                                 int(round(-(y0 + top) * s)) + pad))
    return norm_mask, canvas


def _svg_doc(gid: int, art: Image.Image, mask_shape, pad: int = 60) -> str:
    import base64
    import io
    buf = io.BytesIO()
    art.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    H, W = mask_shape
    scale = CAP_HEIGHT / CAP_HEIGHT_PX
    baseline = pad + CAP_HEIGHT_PX
    # OT-SVG: origin at the glyph origin, y axis pointing DOWN
    x = -pad * scale
    y = -baseline * scale
    return (f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink">'
            f'<g id="glyph{gid}">'
            f'<image x="{x:.1f}" y="{y:.1f}" width="{W*scale:.1f}" height="{H*scale:.1f}" '
            f'xlink:href="data:image/png;base64,{b64}"/>'
            f'</g></svg>')


def add_colour_layer(ttf_path: str, art: dict[str, Image.Image],
                     masks: dict[str, np.ndarray], out_path: str, pad: int = 60) -> str:
    """Attach an OT-SVG table so the glyphs render as the full artwork.

    The monochrome `glyf` outlines stay as the fallback for renderers without
    OT-SVG support, so the file is useful either way.
    """
    from fontTools.ttLib import TTFont, newTable
    f = TTFont(ttf_path)
    order = f.getGlyphOrder()
    docs = []
    for ch, a in art.items():
        name = f"uni{ord(ch):04X}"
        if name not in order:
            continue
        gid = order.index(name)
        docs.append((_svg_doc(gid, a, masks[ch].shape, pad), gid, gid))
    docs.sort(key=lambda d: d[1])
    t = newTable("SVG ")
    t.docList = docs
    t.compressed = False
    f["SVG "] = t
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    f.save(out_path)
    return out_path


def add_bitmap_layer(ttf_path: str, art: dict[str, Image.Image], out_path: str,
                     ppem: int = 128) -> str:
    """Attach CBDT/CBLC colour bitmaps.

    Chrome and Android render colour bitmaps but not OT-SVG (verified in-browser:
    Chrome silently falls back to the monochrome outlines), so this is the format
    to ship when browser coverage matters. Resolution-limited by nature, so OT-SVG
    stays the better choice for print.

    Note the class layout: CBLC's Strike/BitmapSizeTable/SbitLineMetrics live in
    the EBLC module it derives from, only the bitmap-glyph format is CBDT's own.
    """
    import io
    from fontTools.ttLib import TTFont, newTable
    from fontTools.ttLib.tables import C_B_D_T_
    from fontTools.ttLib.tables.E_B_L_C_ import (BitmapSizeTable, SbitLineMetrics,
                                                 Strike, eblc_index_sub_table_1)

    f = TTFont(ttf_path)
    order = f.getGlyphOrder()
    upm = f["head"].unitsPerEm
    scale = (CAP_HEIGHT / CAP_HEIGHT_PX) * (ppem / upm)

    glyph_data = {}
    for ch, im in art.items():
        name = f"uni{ord(ch):04X}"
        if name not in order:
            continue
        w = max(1, int(im.width * scale))
        h = max(1, int(im.height * scale))
        small = im.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, format="PNG")

        g = C_B_D_T_.cbdt_bitmap_format_17(b"", f)
        m = C_B_D_T_.SmallGlyphMetrics()
        m.height, m.width = h, w
        m.BearingX = 0
        m.BearingY = max(-128, min(127, int(round(ppem * CAP_HEIGHT / upm))))
        m.Advance = w
        g.metrics = m
        g.imageData = buf.getvalue()
        glyph_data[name] = g
    if not glyph_data:
        raise ValueError("no glyphs to embed")

    names = sorted(glyph_data, key=order.index)
    sub = eblc_index_sub_table_1(b"", f)
    sub.indexFormat, sub.imageFormat = 1, 17
    sub.imageSize = 0
    sub.names = names
    sub.firstGlyphIndex = order.index(names[0])
    sub.lastGlyphIndex = order.index(names[-1])

    hori = SbitLineMetrics()
    for attr in ("ascender", "descender", "widthMax", "caretSlopeNumerator",
                 "caretSlopeDenominator", "caretOffset", "minOriginSB",
                 "minAdvanceSB", "maxBeforeBL", "minAfterBL", "pad1", "pad2"):
        setattr(hori, attr, 0)
    # SbitLineMetrics fields are signed bytes, so everything here clamps to +/-127.
    # That also caps the usable ppem: 0.8 * 160 = 128 already overflows.
    def i8(v):
        return max(-128, min(127, int(v)))
    hori.ascender = i8(ppem * 0.8)
    hori.descender = i8(-ppem * 0.2)
    hori.widthMax = i8(ppem)

    bst = BitmapSizeTable()
    bst.colorRef = bst.flags = 0
    bst.bitDepth = 32
    bst.ppemX = bst.ppemY = ppem
    bst.hori, bst.vert = hori, hori
    bst.startGlyphIndex, bst.endGlyphIndex = sub.firstGlyphIndex, sub.lastGlyphIndex

    strike = Strike()
    strike.bitmapSizeTable = bst
    strike.indexSubTables = [sub]

    cblc = newTable("CBLC"); cblc.version = 3.0; cblc.strikes = [strike]
    cbdt = newTable("CBDT"); cbdt.version = 3.0; cbdt.strikeData = [glyph_data]
    f["CBLC"], f["CBDT"] = cblc, cbdt
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    f.save(out_path)
    return out_path
