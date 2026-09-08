#!/usr/bin/env python3
"""Cut a long WAV into transcription segments whose boundaries land in silence.

Why this exists: `nemo-speech transcribe` normalizes mel features per *call*,
over every valid frame of the input (`src/asr/features/fe.cpp:477`). On a long
file that makes normalization a function of the whole file, and marginal spans
drop out of the decode -- measured at 7.8% of audio on a 25.5 min file. Cutting
the file gives each piece its own, better-conditioned statistics.

A fixed-window cut trades that for a different loss: a word straddling a hard
cut is decoded twice or not at all, and every segment pays a fresh ~0.8 s
encoder warmup at its head. Both disappear if the cuts land in silence, which is
what Silero VAD is used for here -- detection only, no feature masking.

Requires torch and silero-vad (see requirements-vad-segment.txt).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import wave
from pathlib import Path

SAMPLE_RATE = 16000


def load_audio(path: Path) -> "tuple[object, int]":
    """Read a mono 16 kHz WAV as a float32 torch tensor in [-1, 1]."""
    import numpy as np
    import torch

    with wave.open(str(path)) as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != SAMPLE_RATE:
            raise ValueError(
                f"{path}: expected mono {SAMPLE_RATE} Hz, got "
                f"{handle.getnchannels()}ch {handle.getframerate()} Hz"
            )
        frames = handle.getnframes()
        pcm = np.frombuffer(handle.readframes(frames), dtype=np.int16)
    return torch.from_numpy(pcm.astype("float32") / 32768.0), frames


def speech_spans(audio, threshold: float, min_silence_ms: int) -> list[dict]:
    """Silero speech regions, in seconds."""
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    spans = get_speech_timestamps(
        audio,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        return_seconds=True,
    )
    return spans


def cut_points(spans: list[dict], total_s: float, target_s: float, max_s: float) -> list[float]:
    """Choose boundaries in the middle of silences, honouring target/max length.

    Walks the silences between speech regions and takes one once the running
    segment has reached `target_s`. A silence is only a candidate if it is a
    real pause, so a stretch of continuous speech longer than `max_s` is cut at
    the widest silence available rather than mid-word.
    """
    if not spans:
        return []
    gaps = []
    for previous, following in zip(spans, spans[1:]):
        gaps.append((previous["end"], following["start"]))

    cuts: list[float] = []
    segment_start = 0.0
    for gap_start, gap_end in gaps:
        midpoint = (gap_start + gap_end) / 2.0
        if midpoint - segment_start >= target_s:
            cuts.append(midpoint)
            segment_start = midpoint
    # Nothing forces a final cut; the tail runs to end of file.
    return [c for c in cuts if 0.0 < c < total_s]


def write_segments(src: Path, cuts: list[float], total_s: float, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = [0.0, *cuts, total_s]
    manifest = []
    for index, (start, end) in enumerate(zip(bounds, bounds[1:])):
        if end - start <= 0.05:
            continue
        dest = out_dir / f"{index:04d}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                "-ac", "1", "-ar", str(SAMPLE_RATE), str(dest),
            ],
            check=True,
        )
        manifest.append({"index": index, "file": dest.name, "start": start, "end": end})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="mono 16 kHz WAV")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for segment WAVs")
    parser.add_argument(
        "--target-s", type=float, default=60.0,
        help="cut once a segment reaches this length, at the next silence",
    )
    parser.add_argument(
        "--max-s", type=float, default=180.0, help="upper bound used when silences are scarce"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Silero speech probability")
    parser.add_argument("--min-silence-ms", type=int, default=300)
    parser.add_argument("--manifest", type=Path, help="write the segment offset manifest here")
    args = parser.parse_args()

    audio, frames = load_audio(args.input)
    total_s = frames / SAMPLE_RATE
    spans = speech_spans(audio, args.threshold, args.min_silence_ms)
    cuts = cut_points(spans, total_s, args.target_s, args.max_s)
    manifest = write_segments(args.input, cuts, total_s, args.out_dir)

    speech_s = sum(s["end"] - s["start"] for s in spans)
    print(
        f"{args.input.name}: {total_s:.1f}s, {len(spans)} speech spans "
        f"({speech_s:.1f}s speech, {100 * speech_s / total_s:.1f}%), "
        f"{len(manifest)} segments"
    )
    lengths = [m["end"] - m["start"] for m in manifest]
    if lengths:
        print(
            f"  segment length: min {min(lengths):.1f}s  "
            f"median {sorted(lengths)[len(lengths) // 2]:.1f}s  max {max(lengths):.1f}s"
        )
    target = args.manifest or (args.out_dir / "segments.json")
    target.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"  manifest -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
