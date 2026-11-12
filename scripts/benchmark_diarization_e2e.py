#!/usr/bin/env python3
"""
End-to-End Diarization Benchmark (ASR + DIA, no separation)
Tests NeMo-Speech.cpp with the new nemotron-3-diarization model vs legacy sortformer.
Audio: 5-speaker conversation from cacophony_output.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import soundfile as sf

AUDIO_PATH = Path(r"C:\Users\Administrator\Desktop\Github Repos\vault-cacophony\cacophony_output\benchmark_12turns\cacophony_master.wav")
ALT_AUDIO_PATH = Path(r"C:\Users\Administrator\Desktop\Github Repos\vault-cacophony\cacophony_output\ui_run_hats\cacophony_master.wav")
OUT_DIR = Path(r"C:\Users\Administrator\Desktop\Github Repos\vault-cacophony\samples\bench_diarization_results")

NEW_BIN = Path(r"C:\Users\Administrator\Desktop\Github Repos\vault-cacophony\NeMo-Speech.cpp\build-cuda\bin\nemo-speech.exe")
OLD_BIN = Path(r"C:\Users\Administrator\Desktop\Github Repos\vault-cacophony\NeMo-Speech.cpp.old\build-cuda\bin\nemo-speech.exe")

CUDA_X64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"


def get_cuda_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{CUDA_X64};{CUDA_BIN};" + env.get("PATH", "")
    return env


class GpuSampler(threading.Thread):
    def __init__(self, gpu_id: int = 0, interval: float = 0.05):
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


def analyze_transcript(json_path: Path):
    if not json_path.exists():
        return 0, 0, 0
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        words = data.get("words", [])
        total_words = len(words)
        speakers = set()
        turns = 0
        curr_spk = None
        for w in words:
            spk = w.get("speaker")
            if spk is not None:
                speakers.add(spk)
                if spk != curr_spk:
                    turns += 1
                    curr_spk = spk
        return total_words, len(speakers), turns
    except Exception:
        return 0, 0, 0


def run_benchmark(config_name: str, exe: Path, asr_model: str, diar_model: str | None, audio_path: Path) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{config_name}.json"
    out_srt = OUT_DIR / f"{config_name}.srt"
    if out_json.exists():
        out_json.unlink()
    if out_srt.exists():
        out_srt.unlink()

    audio_info = sf.info(str(audio_path))
    audio_dur = audio_info.duration

    cmd = [
        str(exe),
        "transcribe",
        str(audio_path),
        "--model", asr_model,
        "--backend", "cuda:0",
        "--format", "json",
        "-o", str(out_json),
        "--force",
    ]
    if diar_model:
        cmd.append("--diarize")
        if diar_model != "default":
            cmd.extend(["--diar-model", diar_model])

    baseline_vram = get_baseline_vram()
    sampler = GpuSampler(gpu_id=0, interval=0.05)
    sampler.start()

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=get_cuda_env(), capture_output=True, text=True)
    t1 = time.perf_counter()
    peak_vram = sampler.stop()

    wall_time = t1 - t0
    rtfx = audio_dur / wall_time if wall_time > 0 else 0
    words, n_spk, n_turns = analyze_transcript(out_json)

    res = {
        "config": config_name,
        "exe": str(exe),
        "asr_model": asr_model,
        "diar_model": diar_model or "none",
        "audio_dur_s": round(audio_dur, 2),
        "wall_time_s": round(wall_time, 2),
        "rtfx": round(rtfx, 2),
        "baseline_vram_mib": baseline_vram,
        "peak_vram_mib": peak_vram,
        "vram_delta_mib": peak_vram - baseline_vram,
        "words": words,
        "speakers_detected": n_spk,
        "turns_detected": n_turns,
        "exit_code": proc.returncode,
        "stderr_head": proc.stderr[:300] if proc.stderr else "",
    }
    print(f"[{config_name}] Wall: {wall_time:.2f}s | RTFx: {rtfx:.2f}x | Peak VRAM: {peak_vram} MiB | Spk: {n_spk} | Turns: {n_turns} | RC: {proc.returncode}")
    return res


def main():
    print(f"Testing Audio: {AUDIO_PATH} ({sf.info(str(AUDIO_PATH)).duration:.2f}s, 5 speakers ground truth)")
    
    configs = [
        ("new_nemotron35_nemotron_dia", NEW_BIN, "nemotron-3.5", "nemotron-3-diarization"),
        ("new_nemotron35_sortformer", NEW_BIN, "nemotron-3.5", "sortformer"),
        ("new_nemotron35_pure_asr", NEW_BIN, "nemotron-3.5", None),
        ("new_parakeet_nemotron_dia", NEW_BIN, "parakeet-tdt", "nemotron-3-diarization"),
        ("new_parakeet_pure_asr", NEW_BIN, "parakeet-tdt", None),
        ("old_nemotron35_sortformer", OLD_BIN, "nemotron-3.5", "sortformer"),
    ]

    results = []
    for cfg_name, exe, asr_m, dia_m in configs:
        res = run_benchmark(cfg_name, exe, asr_m, dia_m, AUDIO_PATH)
        results.append(res)
        time.sleep(1.0)

    # Save summary
    with open(OUT_DIR / "diarization_bench_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 95)
    print(f"{'Config':<30} | {'ASR':<13} | {'Diarizer':<22} | {'Wall (s)':<8} | {'RTFx':<8} | {'VRAM':<9} | {'Spk':<4} | {'Turns':<5}")
    print("=" * 95)
    for r in results:
        print(f"{r['config']:<30} | {r['asr_model']:<13} | {r['diar_model']:<22} | {r['wall_time_s']:>7.2f}s | {r['rtfx']:>7.2f}x | {r['peak_vram_mib']:>5} MiB | {r['speakers_detected']:>3} | {r['turns_detected']:>4}")
    print("=" * 95)


if __name__ == "__main__":
    main()
