#!/usr/bin/env python3
"""Transcribe audio using audio.cpp's audiocpp_cli and VAD speech segments.

Accepts an input audio file (e.g. vocals.wav) and a speech_timestamps.json file
(as emitted by `audiocpp_cli --task vad ... --segments-out speech_timestamps.json`),
runs Nemotron ASR via `audiocpp_cli` on CUDA, and formats the output as an .srt subtitle file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIOCPP_DIR = REPO_ROOT / "audio.cpp"
DEFAULT_CLI = AUDIOCPP_DIR / "audiocpp_cli.exe"
DEFAULT_MODEL = AUDIOCPP_DIR / "models" / "nemotron-3.5-asr-streaming-0.6b"
WORDS_TO_SRT = REPO_ROOT / "scripts" / "words_to_srt.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with VAD timestamps via audio.cpp")
    parser.add_argument("--audio", required=True, type=Path, help="Path to input audio WAV")
    parser.add_argument("--segments", type=Path, default=None, help="Optional speech_timestamps.json from VAD")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to ASR model directory")
    parser.add_argument("--family", default="nemotron_asr", help="Model family (default: nemotron_asr)")
    parser.add_argument("--backend", default="cuda", help="Backend device (default: cuda)")
    parser.add_argument("--language", default="es-US", help="ASR language code (e.g. es-US, en-US, auto)")
    parser.add_argument("--out", required=True, type=Path, help="Output .srt file path")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI, help="Path to audiocpp_cli executable")
    return parser.parse_args()


def ensure_16k_mono(audio_path: Path, temp_dir: Path) -> Path:
    """Ensure audio is 16 kHz mono WAV for Nemotron ASR frontend."""
    out_path = temp_dir / f"{audio_path.stem}_16k.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
            "-ac", "1", "-ar", "16000", str(out_path)
        ],
        check=True
    )
    return out_path


def run_asr_streaming(
    cli: Path,
    model: Path,
    family: str,
    backend: str,
    language: str,
    audio_16k: Path,
    words_out: Path,
    text_out: Path
) -> None:
    """Run audiocpp_cli in streaming mode to extract word timestamps."""
    cmd = [
        str(cli),
        "--task", "asr",
        "--family", family,
        "--model", str(model),
        "--backend", backend,
        "--mode", "streaming",
        "--language", language,
        "--request-option", "keep_language_tags=true",
        "--audio", str(audio_16k),
        "--words-out", str(words_out),
        "--text-out", str(text_out),
    ]
    print(f"Running ASR: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cli.parent))


def filter_words_by_vad(words_path: Path, segments_path: Path, sample_rate_vad: int = 44100) -> None:
    """Optionally filter or tag word timestamps against VAD segments."""
    if not segments_path or not segments_path.exists():
        return
    with open(segments_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    if not segments:
        return
    with open(words_path, "r", encoding="utf-8") as f:
        words = json.load(f)
    
    # Convert VAD sample boundaries (at sample_rate_vad) to 16 kHz samples
    vad_spans_16k = [
        (int(s["start_sample"] * 16000 / sample_rate_vad),
         int(s["end_sample"] * 16000 / sample_rate_vad))
        for s in segments
    ]
    
    # Filter words to only those overlapping a VAD segment (with 0.3s tolerance)
    tol = int(0.3 * 16000)
    filtered = []
    for w in words:
        ws, we = w["start_sample"], w["end_sample"]
        overlaps = any(ws <= end + tol and we >= start - tol for start, end in vad_spans_16k)
        if overlaps:
            filtered.append(w)
    
    if filtered:
        with open(words_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f)


def build_srt(words_path: Path, srt_out: Path) -> None:
    """Convert words.json to .srt using words_to_srt.py."""
    srt_out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, str(WORDS_TO_SRT),
            "--words", str(words_path),
            "--out", str(srt_out),
            "--merge-tokens"
        ],
        check=True
    )


def main() -> None:
    args = parse_args()
    if not args.cli.exists():
        raise FileNotFoundError(f"audiocpp_cli not found at: {args.cli}")
    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found at: {args.audio}")

    with tempfile.TemporaryDirectory(prefix="cacophony_asr_") as tmp:
        tmp_dir = Path(tmp)
        audio_16k = ensure_16k_mono(args.audio, tmp_dir)
        words_json = tmp_dir / "words.json"
        text_out = tmp_dir / "tagged.txt"

        run_asr_streaming(
            cli=args.cli,
            model=args.model,
            family=args.family,
            backend=args.backend,
            language=args.language,
            audio_16k=audio_16k,
            words_out=words_json,
            text_out=text_out
        )

        if args.segments:
            filter_words_by_vad(words_json, args.segments)

        build_srt(words_json, args.out)
        print(f"Subtitles written successfully to: {args.out}")


if __name__ == "__main__":
    main()
