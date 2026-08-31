#!/usr/bin/env python3
"""End-to-End Speech-to-Text Translation & Multi-Language .SRT Pipeline.

Workflow:
1. Extract 16 kHz mono WAV via ffmpeg from input audio/video.
2. Run ASR (Nemotron-3.5 / SenseVoice / Parakeet) to obtain word-level timestamps.
3. Build base .srt with optimal pause-based cue segmentation.
4. Translate sentences via local Riva-Translate-4B GGUF.
5. Proportionally redistribute translated text to produce synchronized <name>.<lang>.srt files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from riva_engine import RivaEngine, DEFAULT_MODEL_PATH
from translate_srt_riva import parse_srt, render_srt, translate_cues

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIOCPP_CLI = REPO_ROOT / "audio.cpp" / "audiocpp_cli.exe"
DEFAULT_NEMO_CLI = REPO_ROOT / "NeMo-Speech.cpp" / "build-cuda" / "bin" / "nemo-speech.exe"
WORDS_TO_SRT_SCRIPT = REPO_ROOT / "scripts" / "words_to_srt.py"


def extract_audio(input_path: Path, output_wav: Path, sample_rate: int = 16000, channels: int = 1) -> bool:
    """Extract audio track as PCM WAV with specified sample rate and channel count using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        str(output_wav),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_wav.exists()
    except Exception as e:
        print(f"[!] ffmpeg extraction failed: {e}", file=sys.stderr)
        return False


def run_asr_audiocpp(
    wav_path: Path,
    words_json_path: Path,
    tagged_txt_path: Path,
    engine: str = "nemotron",
    cli_path: Path = DEFAULT_AUDIOCPP_CLI,
) -> bool:
    """Run ASR using audiocpp_cli."""
    if not cli_path.exists():
        print(f"[!] audiocpp_cli not found at {cli_path}", file=sys.stderr)
        return False

    models_dir = REPO_ROOT / "audio.cpp" / "models"
    if engine == "nemotron":
        candidates = [
            models_dir / "Nemotron-3.5-ASR-Streaming-0.6B-GGUF" / "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf",
            models_dir / "Nemotron-3.5-ASR-Streaming-0.6B-GGUF" / "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
            models_dir / "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
        ]
        model_gguf = next((c for c in candidates if c.exists()), candidates[0])

        cmd = [
            str(cli_path),
            "--task", "asr",
            "--family", "nemotron_asr",
            "--model", str(model_gguf),
            "--backend", "cuda",
            "--mode", "streaming",
            "--language", "auto",
            "--request-option", "keep_language_tags=true",
            "--audio", str(wav_path),
            "--words-out", str(words_json_path),
            "--text-out", str(tagged_txt_path),
        ]
    else:  # parakeet
        candidates = [
            models_dir / "Parakeet-TDT-0.6B-v3-GGUF" / "parakeet-tdt-0.6b-v3-q8_0.gguf",
            models_dir / "Parakeet-TDT-0.6B-v3-GGUF" / "parakeet-tdt-0.6b-v3.q8_0.gguf",
        ]
        model_gguf = next((c for c in candidates if c.exists()), candidates[0])

        cmd = [
            str(cli_path),
            "--task", "asr",
            "--family", "parakeet_tdt",
            "--model", str(model_gguf),
            "--backend", "cuda",
            "--audio", str(wav_path),
            "--words-out", str(words_json_path),
        ]

    try:
        subprocess.run(cmd, check=True)
        return words_json_path.exists()
    except subprocess.CalledProcessError as e:
        print(f"[!] ASR failed with exit code {e.returncode}", file=sys.stderr)
        return False


def run_vocal_separation(
    wav_path: Path,
    output_vocals_path: Path,
    separator: str = "bs_roformer",
    passes: int = 1,
    cli_path: Path = DEFAULT_AUDIOCPP_CLI,
) -> bool:
    """Run vocal separation (BS-RoFormer or HTDemucs) using audiocpp_cli."""
    if not cli_path.exists():
        print(f"[!] audiocpp_cli not found at {cli_path}", file=sys.stderr)
        return False

    models_dir = REPO_ROOT / "audio.cpp" / "models"
    if separator == "bs_roformer":
        model_path = models_dir / "BS-RoFormer-ep368-GGUF" / "bs-roformer-ep368-q8_0.gguf"
        family = "bs_roformer"
    else:
        model_path = models_dir / "htdemucs-f16.gguf"
        family = "htdemucs"

    if not model_path.exists():
        print(f"[!] Separation model not found: {model_path}", file=sys.stderr)
        return False

    sep_out_dir = output_vocals_path.parent / "sep_temp"
    sep_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(cli_path),
        "--family", family,
        "--task", "sep",
        "--mode", "offline",
        "--model", str(model_path),
        "--backend", "cuda",
        "--audio", str(wav_path),
        "--out-dir", str(sep_out_dir),
    ]
    if family == "bs_roformer":
        cmd.extend(["--session-option", f"bs_roformer.num_overlap={passes}"])

    try:
        env = os.environ.copy()
        cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
        cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
        if os.path.exists(cuda_bin):
            env["PATH"] = f"{cuda_x64};{cuda_bin};" + env.get("PATH", "")

        subprocess.run(cmd, check=True, env=env)
        vocals_file = sep_out_dir / "vocals.wav"
        if vocals_file.exists():
            shutil.move(str(vocals_file), str(output_vocals_path))
            shutil.rmtree(str(sep_out_dir), ignore_errors=True)
            return True
        else:
            print(f"[!] Separation produced no vocals stem", file=sys.stderr)
            return False
    except subprocess.CalledProcessError as e:
        print(f"[!] Separation failed with exit code {e.returncode}", file=sys.stderr)
        return False


def build_source_srt(
    words_json: Path,
    tagged_txt: Optional[Path],
    output_srt: Path,
    engine: str = "nemotron",
    gap: float = 0.6,
    max_chars: int = 42,
    max_dur: float = 5.0,
    width: int = 36,
) -> bool:
    """Convert words JSON to .srt subtitle file."""
    cmd = [
        sys.executable,
        str(WORDS_TO_SRT_SCRIPT),
        "--words", str(words_json),
        "--out", str(output_srt),
        "--gap", str(gap),
        "--max-chars", str(max_chars),
        "--max-dur", str(max_dur),
        "--width", str(width),
    ]
    if engine == "nemotron":
        cmd.append("--merge-tokens")
        if tagged_txt and tagged_txt.exists():
            cmd.extend(["--tagged-text", str(tagged_txt)])

    try:
        subprocess.run(cmd, check=True)
        return output_srt.exists()
    except subprocess.CalledProcessError as e:
        print(f"[!] words_to_srt failed with exit code {e.returncode}", file=sys.stderr)
        return False


def process_media_file(
    media_path: Path,
    target_langs: List[str],
    source_lang: str = "en",
    asr_engine: str = "nemotron",
    output_dir: Optional[Path] = None,
    riva_model_path: str = DEFAULT_MODEL_PATH,
    keep_intermediates: bool = False,
    gap: float = 0.6,
    separate: bool = True,
    separator: str = "bs_roformer",
    sep_passes: int = 1,
    overwrite: bool = False,
    skip_completed: bool = False,
) -> bool:
    """Process a single audio/video file through the translation pipeline."""
    out_dir = output_dir or media_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = media_path.stem

    base_srt = out_dir / f"{stem}.srt"
    temp_dir = Path(tempfile.mkdtemp(prefix="trans_pipe_"))

    try:
        # Step 1: ASR if base SRT does not already exist or overwrite is requested
        if not base_srt.exists() or overwrite:
            if separate:
                print(f"[*] Extracting 44.1kHz audio from {media_path.name} for {separator} separation...")
                mix_44k = temp_dir / f"{stem}.44k.wav"
                if not extract_audio(media_path, mix_44k, sample_rate=44100, channels=2):
                    return False

                print(f"[*] Isolating vocal stem via {separator} (passes={sep_passes})...")
                vocals_44k = temp_dir / f"{stem}.vocals_44k.wav"
                if run_vocal_separation(mix_44k, vocals_44k, separator=separator, passes=sep_passes):
                    print(f"    -> Vocal separation successful, resampling vocals to 16kHz mono for ASR...")
                    asr_input_wav = temp_dir / f"{stem}.vocals_16k.wav"
                    if not extract_audio(vocals_44k, asr_input_wav, sample_rate=16000, channels=1):
                        asr_input_wav = vocals_44k
                else:
                    print(f"[!] Vocal separation failed; falling back to mixed audio", file=sys.stderr)
                    asr_input_wav = temp_dir / f"{stem}.16k.wav"
                    if not extract_audio(media_path, asr_input_wav, sample_rate=16000, channels=1):
                        return False
            else:
                print(f"[*] Extracting 16kHz audio from {media_path.name}...")
                asr_input_wav = temp_dir / f"{stem}.16k.wav"
                if not extract_audio(media_path, asr_input_wav, sample_rate=16000, channels=1):
                    return False

            print(f"[*] Running ASR ({asr_engine})...")
            words_json = temp_dir / f"{stem}.words.json"
            tagged_txt = temp_dir / f"{stem}.tagged.txt"
            if not run_asr_audiocpp(asr_input_wav, words_json, tagged_txt, engine=asr_engine):
                return False

            print(f"[*] Building primary .srt subtitles...")
            if not build_source_srt(words_json, tagged_txt, base_srt, engine=asr_engine, gap=gap):
                return False
            print(f"    -> Wrote base transcript: {base_srt}")
        else:
            print(f"[*] Using existing base SRT: {base_srt}")

        # Step 2: Parse Cues
        with open(base_srt, encoding="utf-8") as f:
            cues = parse_srt(f.read())

        if not cues:
            print(f"[!] No cues found in {base_srt}", file=sys.stderr)
            return False

        # Filter target languages that need translation
        needed_langs = []
        for lang in target_langs:
            dest_file = out_dir / f"{stem}.{lang}.srt"
            if dest_file.exists() and not overwrite and skip_completed:
                print(f"[*] Subtitles already exist for '{lang}': {dest_file} (skipped)")
            else:
                needed_langs.append(lang)

        # Step 3: Translate via Riva-Translate-4B
        if needed_langs:
            print(f"[*] Initializing Riva-Translate-4B engine ({len(cues)} cues)...")
            engine = RivaEngine(model_path=riva_model_path)

            for lang in needed_langs:
                print(f"[*] Translating to '{lang}'...")
                translated_cues = translate_cues(
                    cues=cues,
                    engine=engine,
                    target_lang=lang,
                    source_lang=source_lang,
                )
                dest_file = out_dir / f"{stem}.{lang}.srt"
                with open(dest_file, "w", encoding="utf-8") as f:
                    f.write(render_srt(translated_cues))
                print(f"    -> Generated: {dest_file}")

        return True

    finally:
        if not keep_intermediates and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Speech to Translated SRT Pipeline.")
    parser.add_argument("input", help="Path to input audio/video file or directory")
    parser.add_argument("--langs", "-t", required=True, help="Comma-separated target languages (e.g. es,fr,de)")
    parser.add_argument("--source", "-s", default="en", help="Source language (default: en)")
    parser.add_argument("--asr-engine", choices=["nemotron", "parakeet"], default="nemotron", help="ASR model")
    parser.add_argument("--out-dir", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to Riva-Translate GGUF")
    parser.add_argument("--gap", type=float, default=0.6, help="Silence gap threshold in seconds for cues")
    parser.add_argument("--separate", action="store_true", default=True, help="Isolate vocal stem before ASR transcription (on by default)")
    parser.add_argument("--no-separate", action="store_true", help="Disable vocal stem separation before ASR")
    parser.add_argument("--separator", choices=["bs_roformer", "htdemucs"], default="bs_roformer", help="Separation model")
    parser.add_argument("--sep-passes", type=int, default=1, help="Inference passes for BS-RoFormer (default: 1)")
    parser.add_argument("--overwrite", "-w", action="store_true", help="Force re-running ASR and re-generating target SRTs, ignoring existing .srt files")
    parser.add_argument("--skip-completed", "--skip-existing", "--skip-if-translated", action="store_true", help="Skip processing if target translations already exist")
    parser.add_argument("--keep-intermediates", action="store_true", help="Retain intermediate WAV and words JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    target_langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    do_separate = not args.no_separate
    skip_completed = args.skip_completed

    if input_path.is_file():
        success = process_media_file(
            media_path=input_path,
            target_langs=target_langs,
            source_lang=args.source,
            asr_engine=args.asr_engine,
            output_dir=Path(args.out_dir) if args.out_dir else None,
            riva_model_path=args.model,
            keep_intermediates=args.keep_intermediates,
            gap=args.gap,
            separate=do_separate,
            separator=args.separator,
            sep_passes=args.sep_passes,
            overwrite=args.overwrite,
            skip_completed=skip_completed,
        )
        sys.exit(0 if success else 1)
    elif input_path.is_dir():
        media_extensions = {".wav", ".mp3", ".m4a", ".mp4", ".mkv", ".flac", ".ogg", ".webm"}
        files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in media_extensions]
        print(f"[*] Found {len(files)} media files in {input_path}")
        failures = 0
        for f in files:
            ok = process_media_file(
                media_path=f,
                target_langs=target_langs,
                source_lang=args.source,
                asr_engine=args.asr_engine,
                output_dir=Path(args.out_dir) if args.out_dir else None,
                riva_model_path=args.model,
                keep_intermediates=args.keep_intermediates,
                gap=args.gap,
                separate=do_separate,
                separator=args.separator,
                sep_passes=args.sep_passes,
                overwrite=args.overwrite,
                skip_completed=skip_completed,
            )
            if not ok:
                failures += 1
        print(f"[*] Batch completed: {len(files) - failures} succeeded, {failures} failed.")
        sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
