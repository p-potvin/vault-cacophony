#!/usr/bin/env python3
"""Benchmark the NeMo-Speech.cpp ASR models against each other, and against AF3.

One long English recording, every model, same machine, same audio. The question
this answers is whether NeMo-Speech.cpp is fast and good enough to replace the
audio.cpp + Audio Flamingo path, so the numbers that matter are wall time,
realtime factor, and peak VRAM -- accuracy is compared separately by reading the
transcripts, since there is no reference for this recording.

Realtime factor is reported both ways because both conventions are in use:

    rtf      = wall / audio    lower is better, < 1.0 is faster than realtime
    speedup  = audio / wall    higher is better, "N times realtime"

Audio Flamingo is not a peer of these models and the numbers should not be read
as if it were. It is a 8.2B audio-language model doing transcription as one of
many tasks, invoked through llama-mtmd-cli, which reloads the whole model for
every chunk -- and AF3 must be chunked at 29 s (see docs/audio-flamingo.md), so
a 45 minute file is ~94 process launches. `load_s_est` reports that overhead so
the compute cost can be read separately from the harness cost.

CUDA note: nemo-speech.exe links against CUDA 13 and needs `bin\\x64` on PATH,
not just `bin`. Without it every invocation exits 53 with no output at all.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NEMO_BIN = REPO / "NeMo-Speech.cpp" / "build-cuda" / "bin" / "nemo-speech.exe"
CUDA_X64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"

# AF3 reloads the model per chunk; measured load time from docs/audio-flamingo.md.
AF3_LOAD_S = 1.87
AF3_CHUNK_S = 29.0


def cuda_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{CUDA_X64};{CUDA_BIN};" + env.get("PATH", "")
    return env


class GpuSampler(threading.Thread):
    """Poll nvidia-smi for peak used memory while a run is in flight."""

    def __init__(self, period: float = 1.0):
        super().__init__(daemon=True)
        self.period = period
        self.peak_mib = 0
        self._done = threading.Event()

    def run(self) -> None:
        while not self._done.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                v = int(out.stdout.strip().splitlines()[0])
                self.peak_mib = max(self.peak_mib, v)
            except Exception:
                pass
            self._done.wait(self.period)

    def stop(self) -> int:
        self._done.set()
        self.join(timeout=5)
        return self.peak_mib


def baseline_vram() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def timed(cmd: list[str], env: dict, out_path: Path | None) -> dict:
    """Run cmd, sampling VRAM, and return timing plus captured stdout."""
    sampler = GpuSampler()
    base = baseline_vram()
    sampler.start()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)
    wall = time.perf_counter() - t0
    peak = sampler.stop()
    if out_path is not None and proc.stdout:
        out_path.write_text(proc.stdout, encoding="utf-8")
    return {
        "wall_s": round(wall, 2),
        "returncode": proc.returncode,
        "peak_vram_mib": peak,
        "baseline_vram_mib": base,
        "model_vram_mib": max(0, peak - base),
        "stdout": proc.stdout or "",
        "stderr": (proc.stderr or "")[-4000:],
    }


def run_nemo(model: str, audio: Path, outdir: Path, args: argparse.Namespace,
             extra: list[str] | None = None, tag: str | None = None) -> dict:
    tag = tag or model
    out_txt = outdir / f"{tag}.txt"
    cmd = [str(NEMO_BIN), "transcribe", str(audio), "--model", model,
           "--backend", f"cuda:{args.gpu}"]
    if args.language:
        cmd += ["--language", args.language]
    cmd += extra or []
    res = timed(cmd, cuda_env(), out_txt)
    text = res.pop("stdout", "")
    res.update({
        "model": model,
        "tag": tag,
        "extra_args": extra or [],
        "chars": len(text.strip()),
        "words": len(text.split()),
        "transcript_path": str(out_txt),
    })
    return res


def run_af3(audio: Path, outdir: Path, args: argparse.Namespace) -> dict:
    out_txt = outdir / "audio-flamingo.txt"
    cmd = [sys.executable, str(REPO / "scripts" / "audio_flamingo.py"),
           "--audio", str(audio), "--task", "asr"]
    res = timed(cmd, os.environ.copy(), out_txt)
    text = res.pop("stdout", "")
    n_chunks = max(1, int(args.audio_s // AF3_CHUNK_S) + 1)
    res.update({
        "model": "audio-flamingo-3 (Q4_K_M)",
        "tag": "audio-flamingo",
        "chunks": n_chunks,
        "load_s_est": round(n_chunks * AF3_LOAD_S, 1),
        "chars": len(text.strip()),
        "words": len(text.split()),
        "transcript_path": str(out_txt),
    })
    return res


def finalize(rows: list[dict], audio_s: float) -> list[dict]:
    for r in rows:
        w = r.get("wall_s") or 0.0
        if w > 0:
            r["rtf"] = round(w / audio_s, 4)
            r["speedup_x_realtime"] = round(audio_s / w, 1)
        if "load_s_est" in r and w > r["load_s_est"]:
            compute = w - r["load_s_est"]
            r["rtf_excl_load"] = round(compute / audio_s, 4)
            r["speedup_excl_load"] = round(audio_s / compute, 1)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--audio-s", type=float, default=None, help="duration; probed if omitted")
    ap.add_argument("--outdir", type=Path, default=Path("D:/HuggingFace/bench/results"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--models", default="nemotron-3.5,nemotron-en,parakeet-tdt,parakeet-ctc",
                    help="comma-separated indexed names")
    ap.add_argument("--stream-models", default="nemotron-3.5,nemotron-en",
                    help="also benchmark these with --stream")
    ap.add_argument("--with-af3", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if args.audio_s is None:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(args.audio)],
            capture_output=True, text=True)
        args.audio_s = float(probe.stdout.strip())

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for m in [s for s in args.models.split(",") if s]:
        print(f"[offline] {m} ...", file=sys.stderr, flush=True)
        rows.append(run_nemo(m, args.audio, args.outdir, args))
        print(f"    {rows[-1]['wall_s']}s rc={rows[-1]['returncode']}", file=sys.stderr, flush=True)

    for m in [s for s in args.stream_models.split(",") if s]:
        print(f"[stream]  {m} ...", file=sys.stderr, flush=True)
        rows.append(run_nemo(m, args.audio, args.outdir, args,
                             extra=["--stream"], tag=f"{m}-stream"))
        print(f"    {rows[-1]['wall_s']}s rc={rows[-1]['returncode']}", file=sys.stderr, flush=True)

    if args.with_af3:
        print("[af3]     audio-flamingo ...", file=sys.stderr, flush=True)
        rows.append(run_af3(args.audio, args.outdir, args))
        print(f"    {rows[-1]['wall_s']}s rc={rows[-1]['returncode']}", file=sys.stderr, flush=True)

    rows = finalize(rows, args.audio_s)
    payload = {"audio": str(args.audio), "audio_s": args.audio_s, "results": rows}
    out = args.json_out or (args.outdir / "asr_bench.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    hdr = f"{'model':<28}{'wall_s':>9}{'rtf':>9}{'xRT':>8}{'VRAM MiB':>10}{'words':>8}{'rc':>4}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['tag']:<28}{r.get('wall_s',0):>9.1f}{r.get('rtf',0):>9.4f}"
              f"{r.get('speedup_x_realtime',0):>8.1f}{r.get('model_vram_mib',0):>10}"
              f"{r.get('words',0):>8}{r.get('returncode',-1):>4}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
