"""Build a control->target dataset for a style-transfer LoRA.

Shape follows Ref2Font's config (`has_control: true`), which is the same shape as
our task, but not its content. Ref2Font renders `Aa` from a font file, so its data
is necessarily strict black and white with no volume - it can only teach
letterform. We generate the lettering procedurally, so colour, gradient, outline,
extrusion, shadow and photographic texture are all in distribution.

Each pair holds the *style* fixed and changes the *text*:

    control = string A in style S, on background B
    target  = string C (C != A) in the same style S, on the same background B

Three defects measured in the current pipeline are addressed by construction
rather than by prompting:

* **reference bleed** (README 5a; `O`/`Q` came out as the reference `D`) - every
  pair has control text != target text, so copying the control's glyphs is
  exactly what the loss penalises;
* **scale drift** (cap height varied 1.43x across A-Z) - control and target are
  rendered at the same cap height, so consistent sizing is supervised;
* **material drift on simple styles** (5d) - control and target share style
  parameters exactly, so the correct answer is to copy the material, not
  reinterpret it.

    ..\\..\\.venv\\Scripts\\python.exe lora_dataset.py --n 600 --out data/lora_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

from PIL import Image, ImageFont

import styles as S

WIN_FONTS = "C:/Windows/Fonts"
WALLPAPERS = "D:/Wallpapers_Minimalism"

DISPLAY_FONTS = ["impact.ttf", "COOPBL.TTF", "BRUSHSCI.TTF", "ALGER.TTF", "BAUHS93.TTF",
                 "BROADW.TTF", "CHILLER.TTF", "COLONNA.TTF", "FORTE.TTF", "GIGI.TTF",
                 "HARLOWSI.TTF", "ITCKRIST.TTF", "JOKERMAN.TTF", "LATINWD.TTF",
                 "MAGNETOB.TTF", "MATURASC.TTF", "OLDENGL.TTF", "PAPYRUS.TTF",
                 "PLAYBILL.TTF", "RAVIE.TTF", "SHOWG.TTF", "SNAP____.TTF",
                 "STENCIL.TTF", "VINERITC.TTF", "arialbd.ttf", "timesbd.ttf",
                 "georgiab.ttf", "verdanab.ttf", "trebucbd.ttf", "CASTELAR.TTF"]

WORDS = ["SUMMER", "HELLO", "WELCOME", "GOLD", "NEON", "DREAM", "SHINE", "STORM",
         "CANDY", "ROCKET", "OCEAN", "FOREST", "MIDNIGHT", "SUNSET", "THUNDER",
         "VELVET", "COPPER", "MARBLE", "SIGNAL", "ORBIT", "PRISM", "ECHO",
         "AURORA", "CIPHER", "NOVA", "QUARTZ", "ZENITH", "JUNGLE", "FROST"]


def random_style(rng: random.Random, textures):
    """Sample a coherent lettering style: colours, depth, and sometimes a texture."""
    hue = rng.random()
    base = [int(255 * v) for v in __import__("colorsys").hsv_to_rgb(hue, rng.uniform(0.35, 0.95), 1.0)]
    dark = [max(0, int(c * rng.uniform(0.16, 0.34))) for c in base]
    st = dict(
        depth=rng.choice([0, 8, 14, 20, 26, 34]),
        ang=(rng.uniform(-0.9, 0.9), rng.uniform(0.35, 0.95)),
        face_top=tuple(min(255, int(c * rng.uniform(1.0, 1.35))) for c in base),
        face_bot=tuple(int(c * rng.uniform(0.45, 0.85)) for c in base),
        side_dark=tuple(dark),
        outline=tuple(max(0, int(c * 0.4)) for c in dark),
        shadow_alpha=rng.uniform(0.0, 0.75),
        shadow_blur=rng.uniform(4, 20),
        specular=rng.choice([0.0, 0.6, 0.95]),
        spec_width=rng.randint(5, 14),
    )
    if textures and rng.random() < 0.35:
        st["face_texture"] = Image.open(rng.choice(textures)).convert("RGB")
    return st


def render(text, font, style, size, bg):
    return S.extruded(bg.copy(), text, font, **style)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="data/lora_v1")
    ap.add_argument("--size", type=int, nargs=2, default=[1024, 576])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--single-letter-frac", type=float, default=0.5,
                    help="share of pairs that are single glyphs rather than words")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    ctl_dir = os.path.join(a.out, "controls")
    tgt_dir = os.path.join(a.out, "targets")
    os.makedirs(ctl_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)

    fonts = [os.path.join(WIN_FONTS, f) for f in DISPLAY_FONTS
             if os.path.exists(os.path.join(WIN_FONTS, f))]
    textures = sorted(glob.glob(os.path.join(WALLPAPERS, "*.jpg")))[:40]
    backgrounds = textures[:]
    print(f"{len(fonts)} display fonts, {len(textures)} textures/backgrounds")
    if not fonts:
        raise SystemExit("no display fonts found")

    size = tuple(a.size)
    made = 0
    for i in range(a.n):
        fp = rng.choice(fonts)
        style = random_style(rng, textures)
        single = rng.random() < a.single_letter_frac
        if single:
            pool = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
            c_txt, t_txt = rng.sample(pool, 2)
            fs = int(size[1] * rng.uniform(0.55, 0.80))
        else:
            c_txt, t_txt = rng.sample(WORDS, 2)
            fs = int(size[1] * rng.uniform(0.20, 0.34))
        try:
            font = ImageFont.truetype(fp, fs)
        except Exception:
            continue

        if backgrounds and rng.random() < 0.7:
            bg = Image.open(rng.choice(backgrounds)).convert("RGB").resize(size, Image.LANCZOS)
        else:
            g = rng.randint(10, 60)
            bg = S.vgrad(size, (g, g, g + rng.randint(0, 25)), (g // 3, g // 3, g // 2))

        try:
            ctl = render(c_txt, font, style, size, bg)
            tgt = render(t_txt, font, style, size, bg)
        except Exception as e:
            print(f"  skip {i}: {type(e).__name__}")
            continue

        name = f"{made:05d}"
        ctl.save(os.path.join(ctl_dir, f"{name}.png"))
        tgt.save(os.path.join(tgt_dir, f"{name}.png"))
        with open(os.path.join(tgt_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(f'Replace the text in the image with "{t_txt}", keeping exactly the same '
                    f'font, letter shapes, colors, gradient, outline, shadow and lettering '
                    f'style, at the same size and position.')
        made += 1
        if made % 50 == 0:
            print(f"  {made}/{a.n}")

    meta = {"pairs": made, "size": list(size), "seed": a.seed,
            "single_letter_frac": a.single_letter_frac,
            "layout": {"image_directory": "targets", "control_directory": "controls",
                       "has_control": True, "caption_extension": ".txt"},
            "note": "control text always differs from target text; style/background identical"}
    with open(os.path.join(a.out, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n{made} pairs -> {a.out}")
    print(json.dumps(meta["layout"], indent=2))


if __name__ == "__main__":
    main()
