#!/usr/bin/env python3
"""
6-way benchmark comparing NeMo-Speech.cpp (old build vs new build):
- parakeet-tdt (offline)
- nemotron-3.5 (offline)
- nemotron-3.5 (streaming: --stream)
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

AUDIO_PATH = Path(r"C:\Users\Administrator\Desktop\Github Repos\vault-cacophony\samples\benchmark_20min_44k.wav")
OUT_DIR = Path(r"C:\Users\Administrator\Desktop\Github Repos\vault-cacophony\samples\bench_6way_results")

OLD_BIN = Path(r"C:\Users\Administrator\Desktop\Github Repos\vault-cacophony\NeMo-Speech.cpp.old\build-cuda\bin\nemo-speech.exe")
NEW_BIN = Path(r"C:\Users\Administrator\Desktop\Github Repos\NeMo-Speech.cpp-upstream\build-cuda-server\bin\nemo-speech.exe")

CUDA_X64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"


def get_cuda_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{CUDA_X64};{CUDA_BIN};" + env.get("PATH", "")
    return env


class GpuSampler(threading.Thread):
    def __init__(self, gpu_id: int = 0, interval: float = 0.1):
        super().__init__(daemon=True)
        self.gpu_id = gpu_id
        self.interval = interval
        self.peak_mib = 0
        self._done = threading.Event()

    def run(self) -> None:
        while not self._done.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self.gpu_id}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                v = int(out.stdout.strip().splitlines()[0])
                if v > self.peak_mib:
                    self.peak_mib = v
            except Exception:
                pass
            self._done.wait(self.interval)

    def stop(self) -> int:
        self._done.set()
        self.join(timeout=3)
        return self.peak_mib


def get_baseline_vram(gpu_id: int = 0) -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def get_audio_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def run_benchmark(tag: str, binary: Path, model: str, extra_args: list[str], audio_path: Path, audio_s: float, gpu_id: int = 0) -> dict:
    print(f"[*] Starting run: {tag} (binary={binary.name}, model={model}, extra={extra_args})...", flush=True)
    out_txt = OUT_DIR / f"{tag}.txt"
    cmd = [
        str(binary), "transcribe", str(audio_path),
        "--model", model,
        "--backend", f"cuda:{gpu_id}"
    ] + extra_args

    sampler = GpuSampler(gpu_id=gpu_id, interval=0.1)
    base_vram = get_baseline_vram(gpu_id=gpu_id)
    sampler.start()
    
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=get_cuda_env()
    )
    wall_s = time.perf_counter() - t0
    peak_vram = sampler.stop()

    if proc.stdout:
        out_txt.write_text(proc.stdout, encoding="utf-8")

    words = len(proc.stdout.strip().split()) if proc.stdout else 0
    rtf = wall_s / audio_s if audio_s > 0 else 0
    rtfx = audio_s / wall_s if wall_s > 0 else 0

    result = {
        "tag": tag,
        "binary_version": "old" if "old" in str(binary).lower() else "new",
        "binary_path": str(binary),
        "model": model,
        "mode": "streaming" if "--stream" in extra_args else "offline",
        "extra_args": extra_args,
        "returncode": proc.returncode,
        "wall_s": round(wall_s, 2),
        "rtf": round(rtf, 4),
        "rtfx": round(rtfx, 2),
        "baseline_vram_mib": base_vram,
        "peak_vram_mib": peak_vram,
        "model_vram_mib": max(0, peak_vram - base_vram),
        "words": words,
        "transcript_path": str(out_txt),
    }

    if proc.returncode != 0:
        print(f"    FAILED with code {proc.returncode}!", flush=True)
        print(f"    stderr: {proc.stderr[:500]}", flush=True)
    else:
        print(f"    COMPLETED in {result['wall_s']}s | RTFx: {result['rtfx']}x | Peak VRAM: {result['peak_vram_mib']} MiB (Delta: {result['model_vram_mib']} MiB) | Words: {result['words']}", flush=True)

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Probing audio duration for {AUDIO_PATH}...")
    audio_s = get_audio_duration(AUDIO_PATH)
    print(f"Audio duration: {audio_s:.2f} seconds ({audio_s/60:.2f} min)")

    configs = [
        # (tag, binary, model, extra_args)
        ("old_parakeet_tdt_offline", OLD_BIN, "parakeet-tdt", []),
        ("old_nemotron_35_offline", OLD_BIN, "nemotron-3.5", []),
        ("old_nemotron_35_stream", OLD_BIN, "nemotron-3.5", ["--stream"]),
        ("new_parakeet_tdt_offline", NEW_BIN, "parakeet-tdt", []),
        ("new_nemotron_35_offline", NEW_BIN, "nemotron-3.5", []),
        ("new_nemotron_35_stream", NEW_BIN, "nemotron-3.5", ["--stream"]),
    ]

    results = []
    for tag, binary, model, extra_args in configs:
        res = run_benchmark(tag, binary, model, extra_args, AUDIO_PATH, audio_s, gpu_id=0)
        results.append(res)
        time.sleep(1.0)  # cooldown between runs

    json_path = OUT_DIR / "benchmark_6way_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"audio_path": str(AUDIO_PATH), "audio_duration_s": audio_s, "runs": results}, f, indent=2)

    print("\n" + "="*85)
    print(f"{'Build':<6} | {'Model':<14} | {'Mode':<9} | {'Wall Time':<10} | {'RTFx (xRT)':<11} | {'Peak VRAM':<10} | {'Words':<6} | {'RC':<3}")
    print("="*85)
    for r in results:
        print(f"{r['binary_version']:<6} | {r['model']:<14} | {r['mode']:<9} | {r['wall_s']:>8.2f} s | {r['rtfx']:>9.2f}x | {r['peak_vram_mib']:>6} MiB | {r['words']:>6} | {r['returncode']:>3}")
    print("="*85)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
