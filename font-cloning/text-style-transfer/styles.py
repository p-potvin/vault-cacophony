"""Procedural styled-lettering renderer.

Used for two things:

1. Building test material - style references with real depth cues (extrusion,
   bevel, specular, contour, cast shadow) and optional photographic face texture,
   so the pipeline can be evaluated against a known ground truth.
2. The deterministic lane. README 5d measured that the model re-renders material
   from its own prior rather than sampling the reference, which is a feature for
   photographic textures and a defect for simple parametric styles. For those,
   fitting these parameters and re-rendering is exact, seed-independent and takes
   milliseconds.

Coordinates follow PIL: y grows downward, so the default extrusion angle points
down-right.
"""

from __future__ import annotations

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

WIN_FONTS = "C:/Windows/Fonts"


def text_mask(size, txt, font, dx=0, dy=0, stroke=0) -> Image.Image:
    """Centred single-line text as an L-mode mask."""
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    bb = d.textbbox((0, 0), txt, font=font, stroke_width=stroke)
    x = (size[0] - (bb[2] - bb[0])) / 2 - bb[0] + dx
    y = (size[1] - (bb[3] - bb[1])) / 2 - bb[1] + dy
    d.text((x, y), txt, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    return m


def vgrad(size, top, bottom) -> Image.Image:
    g = Image.new("RGB", (1, size[1]))
    p = g.load()
    for y in range(size[1]):
        t = y / max(1, size[1] - 1)
        p[0, y] = tuple(round(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
    return g.resize(size, Image.BILINEAR)


def extruded(bg: Image.Image, txt: str, font: ImageFont.FreeTypeFont,
             depth: int = 26, ang=(0.62, 0.78),
             face_top=(255, 226, 130), face_bot=(196, 128, 26),
             side_dark=(74, 40, 6), outline=(28, 14, 2),
             face_texture: Image.Image | None = None,
             shadow_alpha: float = 0.62, shadow_blur: float = 14.0,
             specular: float = 0.95, spec_width: int = 9) -> Image.Image:
    """Render `txt` over `bg` as extruded lettering.

    The parameters are exactly the ones a simple style is made of - which is why
    fitting them from a reference crop is a tractable alternative to generation.
    """
    size = bg.size
    out = bg.convert("RGBA")

    sh = text_mask(size, txt, font, depth * ang[0] + 16, depth * ang[1] + 20, stroke=4)
    sh = sh.filter(ImageFilter.GaussianBlur(shadow_blur))
    out.alpha_composite(Image.merge(
        "RGBA", (Image.new("L", size, 0),) * 3 + (sh.point(lambda v: int(v * shadow_alpha)),)))

    for i in range(depth, 0, -1):
        t = i / depth
        col = tuple(round(side_dark[c] * (0.45 + 0.55 * (1 - t))) for c in range(3))
        layer = Image.new("RGBA", size, col + (255,))
        layer.putalpha(text_mask(size, txt, font, i * ang[0], i * ang[1], stroke=3))
        out.alpha_composite(layer)

    ol = Image.new("RGBA", size, outline + (255,))
    ol.putalpha(text_mask(size, txt, font, 0, 0, stroke=5))
    out.alpha_composite(ol)

    m_face = text_mask(size, txt, font, 0, 0, stroke=0)
    face = (face_texture.resize(size, Image.LANCZOS).convert("RGB")
            if face_texture is not None else vgrad(size, face_top, face_bot))
    fl = face.convert("RGBA")
    fl.putalpha(m_face)
    out.alpha_composite(fl)

    if specular:
        hi = ImageChops.subtract(m_face, text_mask(size, txt, font, 0, spec_width))
        hi = hi.filter(ImageFilter.GaussianBlur(2)).point(lambda v: min(255, int(v * specular)))
        hl = Image.new("RGBA", size, (255, 255, 255, 255))
        hl.putalpha(hi)
        out.alpha_composite(hl)

    return out.convert("RGB")


GOLD = dict(face_top=(255, 226, 130), face_bot=(196, 128, 26),
            side_dark=(74, 40, 6), outline=(28, 14, 2), depth=26)


def glyph_alpha(size, txt, font, depth=26, ang=(0.62, 0.78), shadow_blur=14.0) -> Image.Image:
    """Coverage of the whole styled glyph: face + extrusion + cast shadow.

    Needed so a reference can be pasted onto a target as just the lettering. A
    rectangular paste drags the reference's own background into the scene, and the
    edit model keeps it - that is what put a dark panel behind the composite runs.
    """
    a = text_mask(size, txt, font, 0, 0, stroke=5)
    for i in range(depth, 0, -1):
        a = ImageChops.lighter(a, text_mask(size, txt, font, i * ang[0], i * ang[1], stroke=3))
    sh = text_mask(size, txt, font, depth * ang[0] + 16, depth * ang[1] + 20, stroke=4)
    return ImageChops.lighter(a, sh.filter(ImageFilter.GaussianBlur(shadow_blur)))


def letter_reference(char: str, size=(320, 420), font_path=f"{WIN_FONTS}/impact.ttf",
                     font_size=300, bg_top=(26, 24, 44), bg_bot=(8, 8, 16),
                     rgba: bool = False, **style) -> Image.Image:
    """One isolated glyph on a dark ground - a clean single-letter style reference.

    Cropping a letter out of a rendered word does not work: adjacent glyphs'
    extrusions overlap, so every tight crop catches a sliver of its neighbour.
    Rendering the glyph on its own avoids that entirely.
    """
    font = ImageFont.truetype(font_path, font_size)
    params = dict(GOLD)
    params.update(style)
    img = extruded(vgrad(size, bg_top, bg_bot), char, font, **params)
    if rgba:
        img = img.convert("RGBA")
        img.putalpha(glyph_alpha(size, char, font,
                                 params.get("depth", 26), params.get("ang", (0.62, 0.78))))
    return img


if __name__ == "__main__":
    import sys
    ch = sys.argv[1] if len(sys.argv) > 1 else "D"
    out = sys.argv[2] if len(sys.argv) > 2 else f"examples/style_letter_{ch}.jpg"
    letter_reference(ch).save(out, quality=95)
    print("wrote", out)
