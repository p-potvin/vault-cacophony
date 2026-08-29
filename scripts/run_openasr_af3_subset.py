#!/usr/bin/env python3
"""Run a bounded local Audio Flamingo ASR smoke subset from Open ASR parquet.

Rows are processed serially and each audio payload is placed in a temporary file
for the existing 29-second-chunking AF3 wrapper.  The command stops immediately
on a wrapper failure or empty prediction; it never continues silently after a
bad model result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq


def records(parquet_path: Path, limit: int, offset: int = 0) -> Iterator[dict[str, Any]]:
    """Yield one bounded row range without loading an entire benchmark split."""
    emitted = 0
    skipped = 0
    source = pq.ParquetFile(parquet_path)
    for batch in source.iter_batches(batch_size=min(limit, 32)):
        for record in batch.to_pylist():
            if skipped < offset:
                skipped += 1
                continue
            yield record
            emitted += 1
            if emitted >= limit:
                return


def transcribe(record: dict[str, Any], args: argparse.Namespace, directory: Path) -> str:
    audio = record.get("audio")
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
        raise ValueError(f"{record.get('id', '<unknown>')}: embedded audio bytes are missing")
    source_name = Path(audio.get("path") or "audio.wav").name
    audio_path = directory / source_name
    audio_path.write_bytes(audio["bytes"])
    command = [
        str(args.runner_python),
        str(args.af3_wrapper),
        "--audio",
        str(audio_path),
        "--task",
        "asr",
        "--model",
        str(args.model),
        "--mmproj",
        str(args.mmproj),
        "--bin",
        str(args.binary),
        "--ctx",
        "8192",
        "--temp",
        "0",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError(
            f"{record.get('id', '<unknown>')}: AF3 failed with {completed.returncode}\n{completed.stderr}"
        )
    prediction = completed.stdout.strip()
    if not prediction:
        raise RuntimeError(f"{record.get('id', '<unknown>')}: AF3 returned an empty transcript")
    return prediction


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0, help="Rows to skip before the bounded slice")
    parser.add_argument("--runner-python", type=Path, default=root / ".venv" / "Scripts" / "python.exe")
    parser.add_argument("--af3-wrapper", type=Path, default=root / "scripts" / "audio_flamingo.py")
    parser.add_argument("--model", type=Path, default=Path("D:/HuggingFace/gguf/af3-Q4_K_M.gguf"))
    parser.add_argument("--mmproj", type=Path, default=Path("D:/HuggingFace/gguf/mmproj-af3-f16.gguf"))
    parser.add_argument("--binary", type=Path, default=Path("D:/HuggingFace/llama-bin/llama-mtmd-cli.exe"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.offset < 0:
        parser.error("--offset must not be negative")
    return args


def main() -> int:
    args = parse_args()
    if not args.parquet.is_file():
        raise SystemExit(f"Parquet file not found: {args.parquet}")
    for path in (args.runner_python, args.af3_wrapper, args.model, args.mmproj, args.binary):
        if not path.is_file():
            raise SystemExit(f"Required local path not found: {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output, tempfile.TemporaryDirectory(
        prefix="openasr-af3-"
    ) as temporary:
        directory = Path(temporary)
        for record in records(args.parquet, args.limit, args.offset):
            prediction = transcribe(record, args, directory)
            result = {
                "id": record.get("id"),
                "dataset": record.get("dataset", "librispeech"),
                "reference": record["text"],
                "prediction": prediction,
                "audio_length_s": record.get("audio_length_s"),
                "model": "audio-flamingo-3-q4_k_m",
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            completed += 1
            print(f"[{completed}/{args.limit}] {result['id']}: {prediction}")
    if completed != args.limit:
        raise SystemExit(f"Only {completed} rows were available; expected {args.limit}")
    print(f"Wrote {completed} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
