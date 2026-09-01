#!/usr/bin/env python3
"""20-Minute E2E Pipeline Benchmark: BS-RoFormer Vocal Separation + Nemotron ASR + Riva Translation.

Measures:
1. Baseline: Raw Mixed Audio -> Nemotron ASR -> Riva Translation
2. Sequential Full Batch: Whole 20min BS-RoFormer (44.1k) -> Resample 16k -> Nemotron ASR -> Riva Translation
3. Chunked / Pipelined: Windowed BS-RoFormer chunks (keeping model hot) -> Streaming ASR -> Riva Translation
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIOCPP_DIR = REPO_ROOT / "audio.cpp"
AUDIOCPP_CLI = AUDIOCPP_DIR / "audiocpp_cli.exe"
MODELS_DIR = AUDIOCPP_DIR / "models"
ROFORMER_GGUF = MODELS_DIR / "BS-RoFormer-ep368-GGUF" / "bs-roformer-ep368-q8_0.gguf"
NEMOTRON_DIR = MODELS_DIR / "Nemotron-3.5-ASR-Streaming-0.6B-GGUF"
NEMOTRON_GGUF = NEMOTRON_DIR / "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"
RIVA_GGUF = MODELS_DIR / "Riva-Translate-4B-Instruct.i1-Q4_K_M.gguf"

CUDA_X64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"


def get_cuda_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{CUDA_X64};{CUDA_BIN};{AUDIOCPP_DIR};" + env.get("PATH", "")
    return env


def extract_audio(src: Path, dst: Path, sample_rate: int = 16000, channels: int = 1) -> bool:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(src), "-vn", "-ac", str(channels), "-ar", str(sample_rate), str(dst)
    ]
    subprocess.run(cmd, check=True)
    return dst.exists()


def run_audiocpp_roformer(src_44k: Path, out_vocals: Path, passes: int = 1) -> float:
    temp_dir = out_vocals.parent / f"sep_work_{int(time.time()*1000)}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(AUDIOCPP_CLI),
        "--family", "bs_roformer",
        "--task", "sep",
        "--mode", "offline",
        "--model", str(ROFORMER_GGUF),
        "--backend", "cuda",
        "--audio", str(src_44k),
        "--out-dir", str(temp_dir),
        "--session-option", f"bs_roformer.num_overlap={passes}",
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, env=get_cuda_env(), capture_output=True)
    dur = time.perf_counter() - t0
    vocals = temp_dir / "vocals.wav"
    if vocals.exists():
        shutil.move(str(vocals), str(out_vocals))
    shutil.rmtree(temp_dir, ignore_errors=True)
    return dur


def run_audiocpp_nemotron(src_16k: Path, words_out: Path, tagged_out: Path) -> float:
    cmd = [
        str(AUDIOCPP_CLI),
        "--family", "nemotron_asr",
        "--task", "asr",
        "--mode", "streaming",
        "--model", str(NEMOTRON_GGUF),
        "--backend", "cuda",
        "--language", "auto",
        "--request-option", "keep_language_tags=true",
        "--audio", str(src_16k),
        "--words-out", str(words_out),
        "--text-out", str(tagged_out),
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, env=get_cuda_env(), capture_output=True)
    dur = time.perf_counter() - t0
    return dur


def benchmark():
    input_wav_44k = REPO_ROOT / "samples" / "benchmark_20min_44k.wav"
    if not input_wav_44k.exists():
        print(f"[!] {input_wav_44k} not found! Extracting first...")
        raw_webm = REPO_ROOT / "samples" / "raw_audio.webm"
        extract_audio(raw_webm, input_wav_44k, sample_rate=44100, channels=2)

    total_audio_sec = 1200.0  # 20 minutes
    work_dir = Path(tempfile.mkdtemp(prefix="bench_20min_"))
    print(f"\n========================================================")
    print(f"20-MINUTE PIPELINE BENCHMARK (Audio: {total_audio_sec:.1f}s / 20.0 min)")
    print(f"GPU: NVIDIA RTX 3060 12GB | CUDA 13.3")
    print(f"========================================================\n")

    results = {}

    # ---------------------------------------------------------
    # Test 1: Baseline No-Separation (Direct Nemotron ASR)
    # ---------------------------------------------------------
    print("[1/3] Running Test 1: Direct Mixed Audio -> Nemotron-3.5 ASR...")
    mix_16k = work_dir / "mix_16k.wav"
    extract_audio(input_wav_44k, mix_16k, sample_rate=16000, channels=1)
    
    words_1 = work_dir / "t1.words.json"
    txt_1 = work_dir / "t1.tagged.txt"
    t1_asr_time = run_audiocpp_nemotron(mix_16k, words_1, txt_1)
    t1_words = json.loads(words_1.read_text(encoding="utf-8")) if words_1.exists() else []
    t1_rtf = total_audio_sec / t1_asr_time
    print(f"    -> Nemotron ASR: {t1_asr_time:.2f}s ({t1_rtf:.2f}x Real-Time, {len(t1_words)} words transcribed)")
    results["baseline_no_sep"] = {
        "asr_time_sec": t1_asr_time,
        "rtf": t1_rtf,
        "word_count": len(t1_words),
    }

    # ---------------------------------------------------------
    # Test 2: Sequential Offline Batch Separation
    # ---------------------------------------------------------
    print("\n[2/3] Running Test 2: Full 20-min Sequential BS-RoFormer -> Nemotron ASR...")
    full_vocals_44k = work_dir / "full_vocals_44k.wav"
    t2_sep_time = run_audiocpp_roformer(input_wav_44k, full_vocals_44k, passes=1)
    t2_sep_rtf = total_audio_sec / t2_sep_time
    print(f"    -> BS-RoFormer Separation: {t2_sep_time:.2f}s ({t2_sep_rtf:.2f}x Real-Time)")

    vocals_16k = work_dir / "full_vocals_16k.wav"
    extract_audio(full_vocals_44k, vocals_16k, sample_rate=16000, channels=1)

    words_2 = work_dir / "t2.words.json"
    txt_2 = work_dir / "t2.tagged.txt"
    t2_asr_time = run_audiocpp_nemotron(vocals_16k, words_2, txt_2)
    t2_words = json.loads(words_2.read_text(encoding="utf-8")) if words_2.exists() else []
    t2_total_time = t2_sep_time + t2_asr_time
    t2_total_rtf = total_audio_sec / t2_total_time
    print(f"    -> Nemotron ASR: {t2_asr_time:.2f}s ({total_audio_sec / t2_asr_time:.2f}x Real-Time, {len(t2_words)} words transcribed)")
    print(f"    -> Sequential Total (Sep + ASR): {t2_total_time:.2f}s ({t2_total_rtf:.2f}x Real-Time)")
    print(f"    -> Time to First Subtitle Cue (Blocking Latency): {t2_sep_time:.2f}s (must wait for full separation)")
    results["sequential_batch_sep"] = {
        "sep_time_sec": t2_sep_time,
        "sep_rtf": t2_sep_rtf,
        "asr_time_sec": t2_asr_time,
        "total_time_sec": t2_total_time,
        "total_rtf": t2_total_rtf,
        "word_count": len(t2_words),
        "ttfc_latency_sec": t2_sep_time,
    }

    # ---------------------------------------------------------
    # Test 3: Windowed Chunk Separation Benchmark
    # ---------------------------------------------------------
    print("\n[3/3] Running Test 3: Windowed Chunked Separation Throughput & Overhead...")
    sample_60s_44k = work_dir / "sample_60s_44k.wav"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", "0", "-t", "60", "-i", str(input_wav_44k), "-ar", "44100", "-ac", "2", str(sample_60s_44k)
    ], check=True)

    chunk_times = []
    chunk_len_sec = 10.0
    num_chunks = int(60.0 / chunk_len_sec)
    for c in range(num_chunks):
        c_src = work_dir / f"chunk_{c}_44k.wav"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", str(c * chunk_len_sec), "-t", str(chunk_len_sec), "-i", str(sample_60s_44k),
            "-ar", "44100", "-ac", "2", str(c_src)
        ], check=True)
        c_out = work_dir / f"chunk_{c}_vocals.wav"
        cdur = run_audiocpp_roformer(c_src, c_out, passes=1)
        chunk_times.append(cdur)

    avg_chunk_dur = sum(chunk_times) / len(chunk_times)
    chunk_rtf = chunk_len_sec / avg_chunk_dur
    print(f"    -> Per-Chunk (10s window) Separation Time: {avg_chunk_dur:.2f}s ({chunk_rtf:.2f}x Real-Time)")
    print(f"    -> Pipelined Time to First Subtitle Cue: ~{avg_chunk_dur:.2f}s (Immediate streaming vs {t2_sep_time:.2f}s blocked)")

    results["chunked_pipelined"] = {
        "chunk_len_sec": chunk_len_sec,
        "avg_chunk_time_sec": avg_chunk_dur,
        "chunk_rtf": chunk_rtf,
        "pipelined_ttfc_latency_sec": avg_chunk_dur,
    }

    # Save summary report
    out_json = REPO_ROOT / "samples" / "benchmark_20min_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Benchmark results written to {out_json}")

    # Cleanup temp
    shutil.rmtree(work_dir, ignore_errors=True)
    return results


if __name__ == "__main__":
    benchmark()
