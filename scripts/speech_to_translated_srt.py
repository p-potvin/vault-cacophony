#!/usr/bin/env python3
"""End-to-End Speech-to-Text Translation & Multi-Language .SRT Pipeline.

Workflow:
1. Extract 44.1 kHz stereo audio for BS-RoFormer vocal isolation (single CUDA session, 0 process reloads).
2. Resample isolated vocal stem to 16 kHz mono for ASR.
3. Run official Nemotron-3.5-ASR (or Parakeet) via NeMo-Speech.cpp on CUDA to generate clean base .srt.
4. Translate base .srt via local Riva-Translate-4B V2 on CUDA using the official language pair schema.
5. Proportionally redistribute translated sentences to produce synchronized <name>.<lang>.srt files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from riva_engine import RivaEngine, DEFAULT_MODEL_PATH, clean_console_text, ensure_cuda_dlls
from translate_srt_riva import parse_srt, render_srt, translate_cues

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIOCPP_CLI = REPO_ROOT / "audio.cpp" / "audiocpp_cli.exe"
NEMO_SPEECH_CLI = REPO_ROOT / "NeMo-Speech.cpp" / "build-cuda" / "bin" / "nemo-speech.exe"


def get_cuda_env() -> Dict[str, str]:
    """Get environment with CUDA 13.3, PyTorch libs, NeMo-Speech, and audio.cpp in PATH."""
    env = os.environ.copy()
    cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
    nemo_bin = str(REPO_ROOT / "NeMo-Speech.cpp" / "build-cuda" / "bin")
    audiocpp_dir = str(REPO_ROOT / "audio.cpp")
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")

    paths = [cuda_x64, cuda_bin, nemo_bin, audiocpp_dir]
    if os.path.exists(torch_lib):
        paths.append(torch_lib)

    env["PATH"] = ";".join(paths) + ";" + env.get("PATH", "")
    return env


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
        print(clean_console_text(f"[!] ffmpeg extraction failed: {e}"), file=sys.stderr)
        return False


def run_vocal_separation(
    wav_path: Path,
    output_vocals_path: Path,
    separator: str = "bs_roformer",
    passes: int = 1,
    cli_path: Path = DEFAULT_AUDIOCPP_CLI,
) -> bool:
    """Run continuous vocal separation (BS-RoFormer or HTDemucs) in a single persistent CUDA session."""
    if not cli_path.exists():
        print(clean_console_text(f"[!] audiocpp_cli not found at {cli_path}"), file=sys.stderr)
        return False

    models_dir = REPO_ROOT / "audio.cpp" / "models"
    if separator == "bs_roformer":
        model_path = models_dir / "BS-RoFormer-ep368-GGUF" / "bs-roformer-ep368-q8_0.gguf"
        family = "bs_roformer"
    else:
        model_path = models_dir / "htdemucs-f16.gguf"
        family = "htdemucs"

    if not model_path.exists():
        print(clean_console_text(f"[!] Separation model not found: {model_path}"), file=sys.stderr)
        return False

    cuda_env = get_cuda_env()
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
        subprocess.run(cmd, check=True, env=cuda_env)
        vocals_file = sep_out_dir / "vocals.wav"
        if vocals_file.exists():
            shutil.move(str(vocals_file), str(output_vocals_path))
            shutil.rmtree(str(sep_out_dir), ignore_errors=True)
            return True
        else:
            print(clean_console_text("[!] Separation produced no vocals stem"), file=sys.stderr)
            return False
    except subprocess.CalledProcessError as e:
        print(clean_console_text(f"[!] Separation failed with exit code {e.returncode}"), file=sys.stderr)
        return False


def run_asr_nemo_speech(
    wav_path: Path,
    output_srt_path: Path,
    engine: str = "nemotron",
    cli_path: Path = NEMO_SPEECH_CLI,
    device: str = "cuda:0",
) -> bool:
    """Run ASR using NeMo-Speech.cpp nemo-speech CLI to generate clean .srt subtitles."""
    if not cli_path.exists():
        print(clean_console_text(f"[!] nemo-speech CLI not found at {cli_path}"), file=sys.stderr)
        return False

    model_name = "nemotron-3.5" if engine == "nemotron" else "parakeet-tdt"

    cmd = [
        str(cli_path),
        "transcribe",
        str(wav_path),
        "--model", model_name,
        "--device", device,
        "--format", "srt",
        "-o", str(output_srt_path),
        "--force",
    ]

    try:
        subprocess.run(cmd, check=True, env=get_cuda_env())
        return output_srt_path.exists()
    except subprocess.CalledProcessError as e:
        print(clean_console_text(f"[!] ASR failed with exit code {e.returncode}"), file=sys.stderr)
        return False


def process_media_file(
    media_path: Path,
    target_langs: List[str],
    source_lang: str = "auto",
    asr_engine: str = "nemotron",
    output_dir: Optional[Path] = None,
    riva_model_path: str = DEFAULT_MODEL_PATH,
    keep_intermediates: bool = False,
    separate: bool = True,
    separator: str = "bs_roformer",
    sep_passes: int = 1,
    overwrite: bool = False,
    skip_completed: bool = False,
    per_cue: bool = False,
    use_punct: bool = True,
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
                print(clean_console_text(f"[*] Extracting 44.1kHz audio from {media_path.name} for {separator} separation..."))
                mix_44k = temp_dir / f"{stem}.44k.wav"
                if not extract_audio(media_path, mix_44k, sample_rate=44100, channels=2):
                    return False

                print(clean_console_text(f"[*] Isolating vocal stem via {separator} on CUDA (passes={sep_passes})..."))
                vocals_44k = temp_dir / f"{stem}.vocals_44k.wav"
                if run_vocal_separation(mix_44k, vocals_44k, separator=separator, passes=sep_passes):
                    print(clean_console_text("    -> Vocal separation successful, resampling vocals to 16kHz mono for ASR..."))
                    asr_input_wav = temp_dir / f"{stem}.vocals_16k.wav"
                    if not extract_audio(vocals_44k, asr_input_wav, sample_rate=16000, channels=1):
                        asr_input_wav = vocals_44k
                else:
                    print(clean_console_text("[!] Vocal separation failed; falling back to mixed audio"), file=sys.stderr)
                    asr_input_wav = temp_dir / f"{stem}.16k.wav"
                    if not extract_audio(media_path, asr_input_wav, sample_rate=16000, channels=1):
                        return False
            else:
                print(clean_console_text(f"[*] Extracting 16kHz audio from {media_path.name}..."))
                asr_input_wav = temp_dir / f"{stem}.16k.wav"
                if not extract_audio(media_path, asr_input_wav, sample_rate=16000, channels=1):
                    return False

            print(clean_console_text(f"[*] Running ASR ({asr_engine}) via NeMo-Speech.cpp on CUDA..."))
            if not run_asr_nemo_speech(asr_input_wav, base_srt, engine=asr_engine):
                return False
            print(clean_console_text(f"    -> Generated base transcript: {base_srt}"))
        else:
            print(clean_console_text(f"[*] Using existing base SRT: {base_srt}"))

        # Step 2: Parse Cues
        with open(base_srt, encoding="utf-8", errors="replace") as f:
            cues = parse_srt(f.read())

        if not cues:
            print(clean_console_text(f"[!] No cues found in {base_srt}"), file=sys.stderr)
            return False

        # Filter target languages that need translation
        needed_langs = []
        for lang in target_langs:
            dest_file = out_dir / f"{stem}.{lang}.srt"
            if dest_file.exists() and not overwrite and skip_completed:
                print(clean_console_text(f"[*] Subtitles already exist for '{lang}': {dest_file.name} (skipped)"))
            else:
                needed_langs.append(lang)

        # Step 3: Translate via Riva-Translate-4B V2 on CUDA
        if needed_langs:
            print(clean_console_text(f"[*] Initializing Riva-Translate-4B V2 engine ({len(cues)} cues)..."))
            engine = RivaEngine(model_path=riva_model_path)

            for lang in needed_langs:
                print(clean_console_text(f"[*] Translating to '{lang}'..."))
                translated_cues = translate_cues(
                    cues=cues,
                    engine=engine,
                    target_lang=lang,
                    source_lang=source_lang,
                    per_cue=per_cue,
                    use_punct=use_punct,
                )
                dest_file = out_dir / f"{stem}.{lang}.srt"
                with open(dest_file, "w", encoding="utf-8") as f:
                    f.write(render_srt(translated_cues))
                print(clean_console_text(f"    -> Generated: {dest_file}"))

        return True

    finally:
        if not keep_intermediates and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Speech to Translated SRT Pipeline.")
    parser.add_argument("input", help="Path to input audio/video file or directory")
    parser.add_argument("--langs", "-t", required=True, help="Comma-separated target languages (e.g. es,fr,de)")
    parser.add_argument("--source", "-s", default="auto", help="Source language (default: auto)")
    parser.add_argument("--asr-engine", choices=["nemotron", "parakeet"], default="nemotron", help="ASR model")
    parser.add_argument("--out-dir", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to Riva-Translate GGUF")
    parser.add_argument("--separate", action="store_true", default=True, help="Isolate vocal stem before ASR (on by default)")
    parser.add_argument("--no-separate", action="store_true", help="Disable vocal stem separation before ASR")
    parser.add_argument("--separator", choices=["bs_roformer", "htdemucs"], default="bs_roformer", help="Separation model")
    parser.add_argument("--sep-passes", type=int, default=1, help="Inference passes for BS-RoFormer (default: 1)")
    parser.add_argument("--per-cue", action="store_true", help="Translate cue-by-cue instead of chunked sentences")
    parser.add_argument("--no-punct", action="store_true", help="Disable Silero TE punctuation restoration")
    parser.add_argument("--overwrite", "-w", action="store_true", help="Force re-running ASR and re-generating target SRTs")
    parser.add_argument("--skip-completed", "--skip-existing", "--skip-if-translated", action="store_true", help="Skip processing if target translations already exist")
    parser.add_argument("--keep-intermediates", action="store_true", help="Retain intermediate WAV files")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(clean_console_text(f"[!] Input not found: {input_path}"), file=sys.stderr)
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
            separate=do_separate,
            separator=args.separator,
            sep_passes=args.sep_passes,
            overwrite=args.overwrite,
            skip_completed=skip_completed,
            per_cue=args.per_cue,
            use_punct=not args.no_punct,
        )
        sys.exit(0 if success else 1)
    elif input_path.is_dir():
        media_extensions = {".wav", ".mp3", ".m4a", ".mp4", ".mkv", ".flac", ".ogg", ".webm"}
        files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in media_extensions]
        print(clean_console_text(f"[*] Found {len(files)} media files in {input_path}"))
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
                separate=do_separate,
                separator=args.separator,
                sep_passes=args.sep_passes,
                overwrite=args.overwrite,
                skip_completed=skip_completed,
                per_cue=args.per_cue,
                use_punct=not args.no_punct,
            )
            if not ok:
                failures += 1
        print(clean_console_text(f"[*] Batch completed: {len(files) - failures} succeeded, {failures} failed."))
        sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
