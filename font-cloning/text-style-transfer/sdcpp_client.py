"""Talk to a resident stable-diffusion.cpp server instead of ComfyUI.

Why bother, when ComfyUI already works: **the per-glyph reload**. Measured on one
klein 4B edit, the 127 s wall was 31 s loading the text encoder, 36 s encoding the
prompt, 38 s loading the diffusion model, 74 s sampling and 3 s decoding. Two
thirds of that is setup, and `sd-cli` pays it again on every invocation — exactly
the cost that made ComfyUI re-dequantise per glyph (README 5e). A 26-glyph set
pays it 26 times either way.

`sd-server` holds the model. It speaks two dialects, both found in the binary:

    /sdapi/v1/txt2img, /sdapi/v1/img2img      A1111-style, JSON, base64 images
    /v1/images/generations, /v1/images/edits  OpenAI-style, multipart

This client uses the A1111 dialect, because it is JSON end to end and takes the
sampler, step count, seed and size as named fields, where the OpenAI one hides
them behind a `size` string and a model name.

`start_server()` launches the binary and waits for it to answer, so a caller can
own the process lifetime; `ServerSpec` names the weights once.

    srv = SDCppServer(SPECS["klein4b"])
    with srv:
        img = srv.edit(composite, "Replace the text ...", refs=[ref_n])
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import time
from dataclasses import dataclass, field

import requests
from PIL import Image

SD_DIR = os.environ.get("SDCPP_DIR", r"D:\AI\sdcpp\sd")
MODELS = os.environ.get("SDCPP_MODELS", r"D:\comfyUI\resources\ComfyUI\models")


@dataclass
class ServerSpec:
    """One model stack, named. Paths are relative to MODELS."""

    diffusion: str
    vae: str
    llm: str
    llm_vision: str | None = None
    model_args: str | None = None       # e.g. qwen_image_zero_cond_t=true
    flow_shift: float | None = None
    lora_dir: str | None = "loras"
    steps: int = 4
    cfg: float = 1.0
    sampler: str = "euler"

    def argv(self, port: int, offload: bool = True) -> list[str]:
        a = [os.path.join(SD_DIR, "sd-server.exe"),
             "--listen-port", str(port),
             "--diffusion-model", os.path.join(MODELS, self.diffusion),
             "--vae", os.path.join(MODELS, self.vae),
             "--llm", os.path.join(MODELS, self.llm),
             "--diffusion-fa"]
        if self.llm_vision:
            a += ["--llm_vision", os.path.join(MODELS, self.llm_vision)]
        if self.model_args:
            a += ["--model-args", self.model_args]
        if self.flow_shift is not None:
            a += ["--flow-shift", str(self.flow_shift)]
        if self.lora_dir:
            a += ["--lora-model-dir", os.path.join(MODELS, self.lora_dir)]
        if offload:
            a += ["--offload-to-cpu"]
        return a


# Measured on one Canada Dry E, seed 77, 1328x752 (README section 11):
#   klein4b   127 s wall,  74 s sampling, 36 s encoding, no slab serifs
#   klein9b   144 s wall, 135 s sampling,  2 s encoding, slab serifs, best box fit
#   qwen2511  309 s wall, 256 s sampling, correct serifs, darker, overflows the box
#
# Note the encode column: a GGUF encoder runs the same prompt ~20x faster than a
# bf16 safetensors one, which is why the larger 9B stack has the shorter setup.
# klein4b would drop to ~93 s given a GGUF Qwen3-4B.
SPECS: dict[str, ServerSpec] = {
    "klein4b": ServerSpec(
        diffusion="diffusion_models/flux-2-klein-4b-fp8.safetensors",
        vae="vae/flux2-vae.safetensors",
        llm="text_encoders/qwen_3_4b.safetensors"),
    "klein9b": ServerSpec(
        diffusion="diffusion_models/flux-2-klein-9b-Q8_0.gguf",
        vae="vae/flux2-vae.safetensors",
        # NOT qwen_3_8b_fp8mixed.safetensors: its 311 weight_scale tensors are
        # 0-dimensional, and sd.cpp asserts scale_nelements == 1 || == out_features.
        llm="text_encoders/Qwen3-8B-Q5_K_M.gguf"),
    "qwen2511": ServerSpec(
        diffusion="diffusion_models/qwen-image-edit-2511-Q4_0.gguf",
        vae="vae/qwen_image_vae.safetensors",
        llm="text_encoders/Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_S.gguf",
        llm_vision="clip/Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf",
        model_args="qwen_image_zero_cond_t=true",   # the 2511 mode; quality collapses without it
        flow_shift=3.0),
}


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _unb64(s: str) -> Image.Image:
    if "," in s[:64]:                      # tolerate a data: URL prefix
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(s)))


class SDCppServer:
    """A resident sd-server process, plus the calls to drive it."""

    def __init__(self, spec: ServerSpec, port: int = 1234, url: str | None = None,
                 offload: bool = True, verbose: bool = True):
        self.spec = spec
        self.port = port
        self.base = (url or f"http://127.0.0.1:{port}").rstrip("/")
        self.offload = offload
        self.verbose = verbose
        self.proc: subprocess.Popen | None = None
        self.log_path = os.path.join(SD_DIR, f"server_{port}.log")

    # -- lifetime ------------------------------------------------------------

    def is_alive(self, timeout: float = 3.0) -> bool:
        try:
            r = requests.get(f"{self.base}/sdapi/v1/samplers", timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def start(self, wait_s: float = 900.0) -> "SDCppServer":
        """Launch the server and block until it answers.

        The wait is generous because it covers reading the weights - 69 s for
        klein 4B, and a 20B GGUF is a good deal slower off a cold cache.
        """
        if self.is_alive():
            if self.verbose:
                print(f"[sdcpp] reusing the server already on {self.base}")
            return self
        argv = self.spec.argv(self.port, self.offload)
        if self.verbose:
            print(f"[sdcpp] starting {os.path.basename(self.spec.diffusion)} on :{self.port}")
        log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)

        t0 = time.time()
        while time.time() - t0 < wait_s:
            if self.proc.poll() is not None:
                raise SystemExit(f"sd-server exited with {self.proc.returncode}; "
                                 f"see {self.log_path}")
            if self.is_alive():
                if self.verbose:
                    print(f"[sdcpp] ready in {time.time() - t0:.0f}s")
                return self
            time.sleep(2.0)
        self.stop()
        raise SystemExit(f"sd-server did not answer within {wait_s:.0f}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- work ----------------------------------------------------------------

    def edit(self, image: Image.Image, prompt: str, refs: list[Image.Image] | None = None,
             seed: int = 77, steps: int | None = None, cfg: float | None = None,
             denoise: float = 1.0, timeout: float = 1800.0) -> Image.Image:
        """One edit. `image` is image1; `refs` become image2, image3, ...

        Reference images go in `ref_images`, matching sd-cli's repeatable `-r`.
        A server that does not know the field ignores it rather than failing, so
        the result is checked against the input: an edit that returns the input
        unchanged means the references never arrived.
        """
        body = {
            "init_images": [_b64(image)],
            "prompt": prompt,
            "negative_prompt": "",
            "steps": steps if steps is not None else self.spec.steps,
            "cfg_scale": cfg if cfg is not None else self.spec.cfg,
            "sampler_name": self.spec.sampler,
            "width": image.width,
            "height": image.height,
            "seed": seed,
            "denoising_strength": denoise,
        }
        if refs:
            body["ref_images"] = [_b64(r) for r in refs]

        r = requests.post(f"{self.base}/sdapi/v1/img2img", json=body, timeout=timeout)
        if r.status_code != 200:
            raise SystemExit(f"sd-server {r.status_code}: {r.text[:400]}")
        data = r.json()
        imgs = data.get("images") or []
        if not imgs:
            raise SystemExit(f"sd-server returned no image: {str(data)[:400]}")
        return _unb64(imgs[0])


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="klein4b", choices=sorted(SPECS))
    ap.add_argument("--image", required=True, help="image1: the composite")
    ap.add_argument("--ref", action="append", default=[], help="extra reference (repeatable)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="out/sdcpp_edit.png")
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the edit N times, to show the second one skips the load")
    ap.add_argument("--keep-up", action="store_true", help="leave the server running")
    a = ap.parse_args()

    srv = SDCppServer(SPECS[a.spec], port=a.port)
    srv.start()
    try:
        img1 = Image.open(a.image)
        refs = [Image.open(p) for p in a.ref]
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        for i in range(a.repeat):
            t0 = time.time()
            out = srv.edit(img1, a.prompt, refs, seed=a.seed + i)
            p = a.out if a.repeat == 1 else a.out.replace(".png", f"_{i}.png")
            out.save(p)
            print(f"  edit {i + 1}/{a.repeat}: {time.time() - t0:.0f}s -> {p}")
    finally:
        if not a.keep_up:
            srv.stop()


if __name__ == "__main__":
    main()
