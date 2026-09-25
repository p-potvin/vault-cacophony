"""Gradio front-end for reference-styled text writing.

Upload a picture whose text has the style you want, a picture to write on, type
the new text, choose where it goes, and run. Backed by a running ComfyUI server
(Comfy Desktop on :8188, or run-comfy-headless.ps1 on :8189).

    ..\\..\\.venv\\Scripts\\python.exe app.py            # http://127.0.0.1:7870
"""

from __future__ import annotations

import os
import sys
import tempfile

import gradio as gr
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline  # noqa: E402
from comfy_client import ComfyClient  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URLS = ["http://127.0.0.1:8188", "http://127.0.0.1:8189"]


def pick_server(url: str) -> str:
    for u in ([url] if url else []) + DEFAULT_URLS:
        if u and ComfyClient(u).is_alive():
            return u
    raise gr.Error("No ComfyUI server reachable on :8188 or :8189. Open Comfy Desktop "
                   "or run run-comfy-headless.ps1.")


def preview_box(img, x, y, w, h, color=(255, 255, 0)):
    """Draw a box on an image so the user can see the region it denotes."""
    if img is None:
        return None
    img = img.convert("RGB").copy()
    box = pipeline.parse_box(f"{x},{y},{w},{h}", img.size)
    ImageDraw.Draw(img).rectangle(box, outline=color, width=max(2, img.width // 300))
    return img


def preview_style(img, x, y, w, h):
    return preview_box(img, x, y, w, h, color=(0, 255, 255))


def generate(style, target, text, strategy, backend, x, y, w, h, sx, sy, sw, sh, where, seed,
             steps, cfg, mp, latent_mask, mask_mode, feather, no_lora, url,
             progress=gr.Progress()):
    if style is None or target is None:
        raise gr.Error("Need both a style image and a target image.")
    if not text.strip():
        raise gr.Error("Type the text to write.")
    server = pick_server(url)
    tmp = tempfile.mkdtemp(prefix="tst_")
    sp, tp = os.path.join(tmp, "style.png"), os.path.join(tmp, "target.png")
    style.convert("RGB").save(sp)
    target.convert("RGB").save(tp)
    out = os.path.join(HERE, "out", "ui_last.png")
    progress(0, desc="queued on ComfyUI")
    img = pipeline.run(sp, tp, text, out, strategy=strategy, bbox=f"{x},{y},{w},{h}",
                       style_bbox=f"{sx},{sy},{sw},{sh}", where=where, seed=int(seed) if seed >= 0 else None,
                       steps=int(steps) if steps else None, cfg=cfg if cfg >= 0 else None,
                       mp=mp, latent_mask=latent_mask, url=server, verbose=False, backend=backend,
                       mask_mode=mask_mode, diff_feather=feather, no_lora=no_lora)
    return img, (f"server {server} · backend {backend} · strategy {strategy} · "
                 f"mask {mask_mode} · seed {seed}")


with gr.Blocks(title="Text style transfer") as demo:
    gr.Markdown("## Write text on a picture in the exact style of text from another picture\n"
                "Qwen-Image-Edit-2511 via ComfyUI. **Cyan box** = the text whose style to copy "
                "(crop it tight - background inside the box contaminates the reference). "
                "**Yellow box** = a *hint* for where the new text goes \u2014 the model sizes the "
                "lettering itself and routinely draws ~50 px outside it, which the composite now "
                "follows instead of clipping.")
    with gr.Row():
        style = gr.Image(label="Style image (has the text style you want)", type="pil")
        style_prev = gr.Image(label="Style crop preview", type="pil", interactive=False)
    with gr.Row():
        sx = gr.Slider(0, 1, 0.0, step=0.01, label="style crop x")
        sy = gr.Slider(0, 1, 0.0, step=0.01, label="style crop y")
        sw = gr.Slider(0.05, 1, 1.0, step=0.01, label="style crop w")
        sh = gr.Slider(0.05, 1, 1.0, step=0.01, label="style crop h")
    with gr.Row():
        target = gr.Image(label="Target image (write onto this)", type="pil")
        preview = gr.Image(label="Placement preview", type="pil", interactive=False)
    with gr.Row():
        text = gr.Textbox(label="Text to write", value="HELLO", scale=2)
        strategy = gr.Radio(["multiref", "composite", "inpaint-cn"], value="multiref",
                            label="Strategy", scale=2)
        backend = gr.Radio(["qwen", "klein"], value="qwen", label="Backend", scale=1)
        where = gr.Textbox(label="Position phrase (multiref)", value="in the center of the image", scale=2)
    with gr.Row():
        x = gr.Slider(0, 1, 0.15, step=0.01, label="box x")
        y = gr.Slider(0, 1, 0.35, step=0.01, label="box y")
        w = gr.Slider(0.05, 1, 0.70, step=0.01, label="box w")
        h = gr.Slider(0.05, 1, 0.30, step=0.01, label="box h")
    with gr.Accordion("Sampler / server", open=False):
        with gr.Row():
            seed = gr.Number(value=-1, label="seed (-1 = random)", precision=0)
            steps = gr.Number(value=4, label="steps (4 with the Lightning LoRA / Klein)", precision=0)
            cfg = gr.Number(value=1.0, label="cfg (1.0 for Lightning)")
            mp = gr.Slider(0.25, 2.0, 1.0, step=0.05, label="working megapixels")
        with gr.Row():
            mask_mode = gr.Radio(["diff", "bbox", "none"], value="diff", label="Composite mask",
                                 info="diff: follows what the model drew, no clipping (recommended). "
                                      "bbox: old feathered box. none: raw output.")
            feather = gr.Slider(0, 16, 2.0, step=0.5, label="edge feather (px)",
                                info="2 = clean cut. The old 16 px feather is what faded the lettering.")
        latent_mask = gr.Checkbox(value=False, label="also constrain denoising with a latent mask (SetLatentNoiseMask)")
        no_lora = gr.Checkbox(value=False, label="drop the 4-step Lightning LoRA (slower; raise steps/cfg)")
        url = gr.Textbox(label="ComfyUI URL (blank = auto :8188 / :8189)", value="")
    run_btn = gr.Button("Write it", variant="primary")
    with gr.Row():
        result = gr.Image(label="Result", type="pil")
    status = gr.Markdown()

    for comp in (target, x, y, w, h):
        comp.change(preview_box, [target, x, y, w, h], preview)
    for comp in (style, sx, sy, sw, sh):
        comp.change(preview_style, [style, sx, sy, sw, sh], style_prev)
    run_btn.click(generate, [style, target, text, strategy, backend, x, y, w, h, sx, sy, sw, sh,
                             where, seed, steps, cfg, mp, latent_mask, mask_mode, feather, no_lora,
                             url], [result, status])
    gr.Examples([[os.path.join(HERE, "examples", "style_summer.jpg"),
                  os.path.join(HERE, "examples", "target_waves.jpg"), "HELLO",
                  0.17, 0.35, 0.63, 0.28]],
                inputs=[style, target, text, sx, sy, sw, sh])

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7870")))
