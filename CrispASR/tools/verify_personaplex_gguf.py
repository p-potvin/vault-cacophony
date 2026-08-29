#!/usr/bin/env python3
"""Verify a converted PersonaPlex GGUF against the reference safetensors.

This is the phase-1 acceptance gate for models/convert-personaplex-to-gguf.py.
It is deliberately stricter than a cosine check: every tensor is stored either
as F32 (norms, biases, 1-D) or F16 (everything else), so applying the *same*
dtype rounding to the reference must reproduce the GGUF payload **bit for
bit**. Anything less is a real defect — a transposed matrix, an off-by-one
slice, a name collision silently overwriting a tensor — not rounding noise.

Cosine is reported anyway, because when a tensor does mismatch the cosine is
what tells you whether it is a transpose (cos near 0), a scale (cos 1.0 with
a large max-abs delta), or garbage.

Usage:
  python tools/verify_personaplex_gguf.py \\
      --gguf   D:/vault-cacophony/gguf/personaplex-7b-f16.gguf \\
      --source <snapshot dir containing model.safetensors>
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf

MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"


def load_converter():
    """Reuse the converter's own name mapping — a second copy would drift."""
    path = os.path.join(os.path.dirname(__file__), "..", "models",
                        "convert-personaplex-to-gguf.py")
    spec = importlib.util.spec_from_file_location("pplx_convert", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gguf_tensors(path: str) -> dict:
    """name -> ndarray in torch (row-major) order.

    Note the asymmetry: `t.shape` is ggml's own order, fastest-varying
    dimension first, so it reads reversed relative to torch — but `t.data`
    is already handed back as a numpy view in torch order. Verified on a
    probe file: a torch [3, 4] round-trips with t.shape == (4, 3) and
    t.data.shape == (3, 4), comparing equal without any transpose. The
    reshape below is therefore a no-op guard, kept only so a future gguf
    release that changes this fails loudly instead of silently.
    """
    reader = gguf.GGUFReader(path)
    out = {}
    for t in reader.tensors:
        arr = np.asarray(t.data)
        shape = tuple(int(d) for d in reversed(t.shape))
        if arr.size == int(np.prod(shape)):
            arr = arr.reshape(shape)
        out[str(t.name)] = arr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--source", required=True, help="dir with model.safetensors + Mimi")
    ap.add_argument("--max-report", type=int, default=15)
    args = ap.parse_args()

    import torch
    from safetensors import safe_open

    conv = load_converter()
    print(f"Reading {args.gguf} ...")
    got = gguf_tensors(args.gguf)
    print(f"  {len(got)} tensors in the GGUF")

    n_ok = n_bad = n_skip = 0
    failures = []

    def compare(gguf_name, ref, label):
        nonlocal n_ok, n_bad, n_skip
        if gguf_name not in got:
            failures.append((gguf_name, label, "ABSENT from GGUF", None, None))
            n_bad += 1
            return
        mine = got[gguf_name]
        # Reproduce the converter's dtype decision exactly.
        want = ref.astype(np.float32) if mine.dtype == np.float32 \
            else ref.astype(np.float16)
        if mine.shape != want.shape:
            # A reshape mismatch is still worth a numeric verdict if the
            # element count matches — that distinguishes transpose from junk.
            if mine.size != want.size:
                failures.append((gguf_name, label,
                                 f"shape {mine.shape} vs {want.shape}", None, None))
                n_bad += 1
                return
            a, b = mine.reshape(-1).astype(np.float32), want.reshape(-1).astype(np.float32)
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
            failures.append((gguf_name, label,
                             f"shape {mine.shape} vs {want.shape}", cos, None))
            n_bad += 1
            return
        if np.array_equal(mine, want):
            n_ok += 1
            return
        a, b = mine.reshape(-1).astype(np.float32), want.reshape(-1).astype(np.float32)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        mad = float(np.max(np.abs(a - b)))
        failures.append((gguf_name, label, "value mismatch", cos, mad))
        n_bad += 1

    lm_path = os.path.join(args.source, "model.safetensors")
    with safe_open(lm_path, framework="pt") as f:
        for name in sorted(f.keys()):
            t = f.get_tensor(name)
            ref = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
            compare(conv.shorten_name("lm." + name), ref, "lm")
            del t, ref

    mimi_path = os.path.join(args.source, MIMI_NAME)
    with safe_open(mimi_path, framework="pt") as f:
        keys = list(f.keys())
        # Codebooks are synthesized by the converter, not passed through, so
        # recompute them here the same way rather than skipping the check.
        for name in sorted(keys):
            if name.endswith("._codebook.embedding_sum"):
                prefix = name[: -len("embedding_sum")]
                usage = f.get_tensor(prefix + "cluster_usage").numpy()
                emb = (f.get_tensor(name).numpy()
                       / np.maximum(usage, 1e-5)[:, None]).astype(np.float32)
                compare(conv.shorten_name("mimi." + prefix + "embedding"),
                        emb, "mimi-codebook")
                continue
            if name.endswith("cluster_usage") or name.endswith("_initialized"):
                n_skip += 1
                continue
            t = f.get_tensor(name)
            ref = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
            compare(conv.shorten_name("mimi." + name), ref, "mimi")
            del t, ref

    print(f"\n  bit-identical : {n_ok}")
    print(f"  mismatched    : {n_bad}")
    print(f"  skipped       : {n_skip} (training accumulators, intentionally dropped)")

    if failures:
        print(f"\nFirst {min(len(failures), args.max_report)} failures:")
        for name, label, why, cos, mad in failures[: args.max_report]:
            extra = ""
            if cos is not None:
                extra = f"  cos={cos:.6f}"
                if mad is not None:
                    extra += f" max|d|={mad:.3e}"
            print(f"  [{label}] {name}: {why}{extra}")
        sys.exit(1)

    print("\nPASS — every tensor is bit-identical to the reference after the "
          "converter's own dtype rounding.")


if __name__ == "__main__":
    main()
