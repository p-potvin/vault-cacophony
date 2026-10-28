"""Comprehensive benchmarking harness for Vault-Cacophony multi-speaker pipeline.

Measures:
- Wall time, Phase breakdown (HF LLM inference, Raggae tone scoring, Dual-GPU TTS synthesis, Room mixing)
- System RAM, Process RSS, CPU utilization
- GPU 0 (RTX 3060) and GPU 1 (RTX 2060) VRAM and utilization
- Audio duration, End-to-End RTFX, TTS Engine RTFX
- Bottleneck and dependency analysis ("Where is the bottleneck? Who is waiting for what?")
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import psutil

# Ensure repo root and scripts are in sys.path
REPO_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cacophony.orchestrator import CacophonyOrchestrator, parse_args  # noqa: E402
from cacophony.turn_taker import TransitionType  # noqa: E402


class ResourceSampler:
    """Samples CPU, RAM, and NVIDIA GPU stats in a background thread."""

    def __init__(self, interval_sec: float = 0.25):
        self.interval = interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.cpu_percents: List[float] = []
        self.ram_used_gb: List[float] = []
        self.process_rss_mb: List[float] = []
        self.gpu0_vram_mb: List[float] = []
        self.gpu0_util: List[float] = []
        self.gpu1_vram_mb: List[float] = []
        self.gpu1_util: List[float] = []
        self.proc = psutil.Process(os.getpid())

    def _query_gpus(self) -> Dict[int, Dict[str, float]]:
        res = {0: {"vram": 0.0, "util": 0.0}, 1: {"vram": 0.0, "util": 0.0}}
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                encoding="utf-8",
                errors="replace",
                timeout=1.0,
            )
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    idx = int(parts[0])
                    vram = float(parts[1])
                    util = float(parts[2])
                    if idx in res:
                        res[idx] = {"vram": vram, "util": util}
        except Exception:
            pass
        return res

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                self.cpu_percents.append(psutil.cpu_percent(interval=None))
                mem = psutil.virtual_memory()
                self.ram_used_gb.append((mem.total - mem.available) / (1024**3))
                self.process_rss_mb.append(self.proc.memory_info().rss / (1024**2))

                gpu_data = self._query_gpus()
                self.gpu0_vram_mb.append(gpu_data[0]["vram"])
                self.gpu0_util.append(gpu_data[0]["util"])
                self.gpu1_vram_mb.append(gpu_data[1]["vram"])
                self.gpu1_util.append(gpu_data[1]["util"])
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, object]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        def _stats(arr):
            return {
                "max": round(float(np.max(arr)), 2) if arr else 0.0,
                "avg": round(float(np.mean(arr)), 2) if arr else 0.0,
                "min": round(float(np.min(arr)), 2) if arr else 0.0,
            }

        return {
            "samples_collected": len(self.cpu_percents),
            "cpu_percent": _stats(self.cpu_percents),
            "ram_used_gb": _stats(self.ram_used_gb),
            "process_rss_mb": _stats(self.process_rss_mb),
            "gpu0_vram_mb": _stats(self.gpu0_vram_mb),
            "gpu0_util_percent": _stats(self.gpu0_util),
            "gpu1_vram_mb": _stats(self.gpu1_vram_mb),
            "gpu1_util_percent": _stats(self.gpu1_util),
        }


def run_benchmark(
    turns: int = 12,
    batch_size: int = 6,
    engine_type: str = "omnivoice",
    script_name: str = "testbench",
    out_dir_name: str = "cacophony_output/benchmark_12turns",
    seed_topic: Optional[str] = None,
) -> Dict[str, object]:
    """Execute a fully timed, resource-profiled multi-speaker dialogue run."""
    out_dir = REPO_DIR / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"  VAULT-CACOPHONY PROFILING & BENCHMARK HARNESS")
    print(f"  Turns: {turns} | Batch Size: {batch_size} | Engine: {engine_type}")
    print(f"  Script: {script_name} | Out Dir: {out_dir}")
    print(f"=======================================================\n")

    sampler = ResourceSampler(interval_sec=0.20)
    sampler.start()

    wall_t0 = time.perf_counter()

    # Step 1: Initialize Orchestrator
    init_t0 = time.perf_counter()
    sys_argv_backup = list(sys.argv)
    cli_args = [
        "orchestrator.py",
        "--turns", str(turns),
        "--batch-size", str(batch_size),
        "--engine-type", engine_type,
        "--script", script_name,
        "--out-dir", str(out_dir),
        "--dual-gpu",
    ]
    if seed_topic:
        cli_args.extend(["--seed-topic", seed_topic])
    sys.argv = cli_args
    args = parse_args()
    sys.argv = sys_argv_backup

    orch = CacophonyOrchestrator(args)
    init_time = time.perf_counter() - init_t0

    # Step 2: Warmup GPU TTS Pool
    warm_t0 = time.perf_counter()
    warm_urls = orch.tts.warm_pool()
    warmup_time = time.perf_counter() - warm_t0

    # Step 3: Run Orchestrator with phase hooks
    batch_stats: List[Dict[str, object]] = []

    # Patch brain.generate_turns_batch, smart_picker.annotate_batch, and tts.synthesize_batch_parallel
    original_generate = orch.brain.generate_turns_batch
    original_annotate = orch.smart_picker.annotate_batch
    original_synthesize = orch.tts.synthesize_batch_parallel

    timings = {
        "llm_times": [],
        "scoring_times": [],
        "tts_times": [],
        "batch_durations": [],
    }

    def timed_generate(*g_args, **g_kwargs):
        t0 = time.perf_counter()
        res = original_generate(*g_args, **g_kwargs)
        elapsed = time.perf_counter() - t0
        timings["llm_times"].append(elapsed)
        return res

    def timed_annotate(*a_args, **a_kwargs):
        t0 = time.perf_counter()
        res = original_annotate(*a_args, **a_kwargs)
        elapsed = time.perf_counter() - t0
        timings["scoring_times"].append(elapsed)
        return res

    def timed_synthesize(*s_args, **s_kwargs):
        t0 = time.perf_counter()
        res = original_synthesize(*s_args, **s_kwargs)
        elapsed = time.perf_counter() - t0
        timings["tts_times"].append(elapsed)
        return res

    orch.brain.generate_turns_batch = timed_generate
    orch.smart_picker.annotate_batch = timed_annotate
    orch.tts.synthesize_batch_parallel = timed_synthesize

    pipeline_t0 = time.perf_counter()
    master_wav = orch.run()
    total_pipeline_time = time.perf_counter() - pipeline_t0

    wall_time = time.perf_counter() - wall_t0
    resource_stats = sampler.stop()

    # Measure output audio
    import soundfile as sf
    audio_data, sr = sf.read(str(master_wav))
    audio_duration_sec = len(audio_data) / sr

    total_llm_time = sum(timings["llm_times"])
    total_scoring_time = sum(timings["scoring_times"])
    total_tts_time = sum(timings["tts_times"])
    mixing_time = max(0.0, total_pipeline_time - (total_llm_time + total_scoring_time + total_tts_time))

    # Real-Time Factor (RTFX = audio seconds generated per wall clock second)
    e2e_rtfx = audio_duration_sec / wall_time if wall_time > 0 else 0.0
    tts_rtfx = audio_duration_sec / total_tts_time if total_tts_time > 0 else 0.0

    # Bottleneck percentages
    pipeline_sum = total_llm_time + total_scoring_time + total_tts_time + mixing_time
    llm_pct = (total_llm_time / pipeline_sum * 100) if pipeline_sum > 0 else 0.0
    tts_pct = (total_tts_time / pipeline_sum * 100) if pipeline_sum > 0 else 0.0
    scoring_pct = (total_scoring_time / pipeline_sum * 100) if pipeline_sum > 0 else 0.0
    mixing_pct = (mixing_time / pipeline_sum * 100) if pipeline_sum > 0 else 0.0

    primary_bottleneck = "TTS Engine" if total_tts_time >= total_llm_time else "HF LLM Inference"

    status_file = out_dir / "run_status.json"
    actual_completed_turns = turns
    if status_file.exists():
        try:
            status_data = json.loads(status_file.read_text(encoding="utf-8"))
            actual_completed_turns = int(status_data.get("completed_turns", turns))
        except Exception:
            pass

    report = {
        "turns_completed": actual_completed_turns,
        "turns_requested": turns,
        "batch_size": batch_size,
        "batches_run": len(timings["llm_times"]),
        "engine_type": engine_type,
        "script": script_name,
        "seed_topic": orch.seed_topic,
        "audio": {
            "master_wav": str(master_wav),
            "sample_rate": sr,
            "duration_seconds": round(audio_duration_sec, 2),
            "channels": audio_data.shape[1] if audio_data.ndim > 1 else 1,
        },
        "timings_seconds": {
            "total_wall_time": round(wall_time, 2),
            "orchestrator_init": round(init_time, 2),
            "warmup_tts_servers": round(warmup_time, 2),
            "total_pipeline_time": round(total_pipeline_time, 2),
            "llm_inference_total": round(total_llm_time, 2),
            "llm_inference_per_batch": [round(t, 2) for t in timings["llm_times"]],
            "tone_scoring_total": round(total_scoring_time, 2),
            "tts_synthesis_total": round(total_tts_time, 2),
            "tts_synthesis_per_batch": [round(t, 2) for t in timings["tts_times"]],
            "room_acoustic_mix": round(mixing_time, 2),
            "avg_time_per_turn": round(wall_time / turns, 2),
        },
        "performance_rtfx": {
            "end_to_end_rtfx": round(e2e_rtfx, 2),
            "tts_engine_rtfx": round(tts_rtfx, 2),
            "note": "RTFX > 1.0 means faster than real-time playback.",
        },
        "bottleneck_breakdown": {
            "primary_bottleneck": primary_bottleneck,
            "llm_inference_percent": round(llm_pct, 1),
            "tts_synthesis_percent": round(tts_pct, 1),
            "tone_scoring_percent": round(scoring_pct, 1),
            "mixing_percent": round(mixing_pct, 1),
            "dependency_flow": (
                "HF LLM generates turns -> Raggae Bridge / Smart Picker annotates fillers/tones -> "
                "Dual-GPU Pool synthesizes audio buffers -> Turn Taking Scheduler calculates barge-in offsets -> "
                "Acoustic Room Mixer convolves 3D room reverb."
            ),
            "who_waits_for_what": (
                "1. Smart Picker waits for LLM batch generation to finish.\n"
                "2. Dual-GPU TTS synthesis waits for Smart Picker annotations.\n"
                "3. Turn Scheduler waits for synthesized audio lengths to stagger barge-in offsets.\n"
                "4. Batch N+1 waits for Batch N history to be committed."
            ),
        },
        "hardware_resources": resource_stats,
    }

    report_path = out_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[+] Benchmark Report written to: {report_path}")

    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Vault-Cacophony Benchmark Harness")
    p.add_argument("--turns", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--engine-type", type=str, default="omnivoice")
    p.add_argument("--script", type=str, default="testbench")
    p.add_argument("--out-dir", type=str, default="cacophony_output/benchmark_12turns")
    args = p.parse_args()

    run_benchmark(
        turns=args.turns,
        batch_size=args.batch_size,
        engine_type=args.engine_type,
        script_name=args.script,
        out_dir_name=args.out_dir,
    )
