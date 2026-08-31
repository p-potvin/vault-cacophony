#!/usr/bin/env python3
"""Run a bounded Nemotron-3.5 ASR benchmark on Open ASR LibriSpeech clean parquet.

Methodology matches Audio Flamingo benchmark:
- Reads records from Open ASR parquet (G:/OpenASR/open-asr-leaderboard/librispeech/test.clean-00000-of-00001.parquet).
- Writes temporary WAV file for each row.
- Transcribes using audiocpp_cli with nemotron-3.5-asr-streaming-0.6b-q8_0.gguf on CUDA.
- Computes per-utterance elapsed time and real-time factor.
- Emits JSONL manifest compatible with evaluate_openasr.py and summarize_openasr_results.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLI = REPO_ROOT / "audio.cpp" / "audiocpp_cli.exe"
DEFAULT_MODEL = REPO_ROOT / "audio.cpp" / "models" / "Nemotron-3.5-ASR-Streaming-0.6B-GGUF" / "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"
DEFAULT_PARQUET = Path(r"G:\OpenASR\open-asr-leaderboard\librispeech\test.clean-00000-of-00001.parquet")


def records(parquet_path: Path, limit: int | None = 250, offset: int = 0) -> Iterator[dict[str, Any]]:
    """Yield records from parquet file."""
    emitted = 0
    skipped = 0
    source = pq.ParquetFile(parquet_path)
    for batch in source.iter_batches(batch_size=min(limit or 32, 32)):
        for record in batch.to_pylist():
            if skipped < offset:
                skipped += 1
                continue
            yield record
            emitted += 1
            if limit is not None and emitted >= limit:
                return


import io
import soundfile as sf

def transcribe_nemotron(
    record: dict[str, Any],
    cli_path: Path,
    model_path: Path,
    directory: Path,
    timeout_s: int = 120,
) -> str:
    audio = record.get("audio")
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
        raise ValueError(f"{record.get('id', '<unknown>')}: embedded audio bytes are missing")

    rec_id = record.get("id") or "audio"
    audio_path = directory / f"{rec_id}.wav"
    
    # Decode audio bytes (FLAC/WAV/etc) and save as 16kHz mono PCM WAV
    data, sr = sf.read(io.BytesIO(audio["bytes"]))
    sf.write(str(audio_path), data, sr, format="WAV", subtype="PCM_16")

    tagged_path = directory / f"{audio_path.stem}.tagged.txt"
    words_path = directory / f"{audio_path.stem}.words.json"

    env = os.environ.copy()
    cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
    if os.path.exists(cuda_bin):
        env["PATH"] = f"{cuda_x64};{cuda_bin};" + env.get("PATH", "")

    cmd = [
        str(cli_path),
        "--task", "asr",
        "--family", "nemotron_asr",
        "--model", str(model_path),
        "--backend", "cuda",
        "--mode", "streaming",
        "--language", "en-US",
        "--audio", str(audio_path),
        "--words-out", str(words_path),
        "--text-out", str(tagged_path),
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(f"{record.get('id', '<unknown>')}: Nemotron timed out after {timeout_s}s") from err

    if completed.returncode != 0:
        raise RuntimeError(
            f"{record.get('id', '<unknown>')}: Nemotron ASR exited with code {completed.returncode}\n{completed.stderr}"
        )

    # Read output transcript from tagged_path or parse text_output
    if tagged_path.exists():
        raw_text = tagged_path.read_text(encoding="utf-8").strip()
    else:
        # Fallback to parsing stdout text_output=...
        lines = [l for l in completed.stdout.splitlines() if l.startswith("text_output=")]
        raw_text = lines[0].split("text_output=", 1)[1].strip() if lines else ""

    # Remove any <en-US> tags from nemotron output
    clean_text = raw_text.replace("<en-US>", "").replace("</en-US>", "").strip()
    return clean_text


def main():
    parser = argparse.ArgumentParser(description="Benchmark Nemotron-3.5 ASR against OpenASR dataset.")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output", type=Path, default=Path(r"G:\OpenASR\results\nemotron-librispeech-clean-250.jsonl"))
    parser.add_argument("--limit", type=int, default=250, help="Number of rows to benchmark (default: 250)")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f"[!] Parquet file not found: {args.parquet}", file=sys.stderr)
        sys.exit(1)

    if not args.cli.exists():
        print(f"[!] audiocpp_cli not found: {args.cli}", file=sys.stderr)
        sys.exit(1)

    if not args.model.exists():
        print(f"[!] Nemotron model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = set()
    if args.resume and args.output.exists():
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    completed_ids.add(item.get("id"))

    mode = "a" if args.resume else "w"
    completed = 0
    total_audio_sec = 0.0
    total_proc_sec = 0.0

    print(f"[*] Starting Nemotron-3.5 ASR benchmark: {args.limit} rows from {args.parquet.name}...")

    with open(args.output, mode, encoding="utf-8", newline="\n") as out_f, tempfile.TemporaryDirectory(
        prefix="openasr-nemo-"
    ) as temp_dir:
        temp_path = Path(temp_dir)
        for record in records(args.parquet, limit=args.limit, offset=args.offset):
            rec_id = record.get("id")
            if rec_id in completed_ids:
                continue

            t0 = time.perf_counter()
            prediction = transcribe_nemotron(record, args.cli, args.model, temp_path)
            elapsed = time.perf_counter() - t0

            audio_len = float(record.get("audio_length_s", 0.0))
            total_audio_sec += audio_len
            total_proc_sec += elapsed

            rtfx = audio_len / elapsed if elapsed > 0 else 0.0

            result = {
                "id": rec_id,
                "dataset": record.get("dataset", "librispeech"),
                "reference": record["text"],
                "prediction": prediction,
                "audio_length_s": audio_len,
                "transcription_seconds": elapsed,
                "model": "nemotron-3.5-asr-streaming-0.6b-q8_0",
            }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            completed += 1

            print(f"[{completed}/{args.limit}] {rec_id} ({audio_len:.1f}s audio in {elapsed:.2f}s, {rtfx:.1f}x RTFX)")
            print(f"    Ref:  {record['text']}")
            print(f"    Pred: {prediction}\n")

    overall_rtfx = total_audio_sec / total_proc_sec if total_proc_sec > 0 else 0.0
    print(f"[*] Completed {completed} utterances ({total_audio_sec:.1f}s audio in {total_proc_sec:.1f}s -> {overall_rtfx:.1f}x Real-time factor).")
    print(f"[*] Results saved to {args.output}")


if __name__ == "__main__":
    main()
