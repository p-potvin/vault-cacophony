#!/usr/bin/env python3
"""Two NeMo-Speech.cpp pipelines, timed stage by stage against a long recording.

The question both modes answer is the same one: with every stage loaded, does
the whole thing still run faster than realtime on one 12 GB card?

    stack       ASR + Sortformer diarization, one pass, speaker-tagged JSON.
                Adds the NanoCodec + Magpie tokenizer load to prove the whole
                model set is co-resident rather than merely sequentially
                loadable -- on a 12 GB card that is the part in doubt.

    translate   ASR -> Riva-Translate 4B -> Magpie TTS. Three models, three
                stages, each timed separately so the bottleneck is visible
                rather than averaged away.

RTF here is wall/audio: below 1.0 is faster than realtime. Each stage also gets
its own RTF, because a pipeline that is 0.9 overall built from one stage at 0.05
and another at 0.85 is a very different engineering problem from two at 0.45.

TTS is the stage that cannot be run whole by default. Synthesizing a 45 minute
transcript produces 45 minutes of audio, so `--tts-max-chars` caps it at a
representative sample and the script reports measured TTS RTF plus the
extrapolated full-transcript cost, clearly labelled as an extrapolation.

CUDA note: nemo-speech.exe needs CUDA 13's `bin\\x64` on PATH or every call
exits 53 silently.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NEMO_BIN = REPO / "NeMo-Speech.cpp" / "build-cuda" / "bin" / "nemo-speech.exe"
RIVA_GGUF = REPO / "NeMo-Speech.cpp" / "models" / "Riva-Translate-4B-Instruct-v2-Q4_K_M.gguf"
CUDA_X64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"


def cuda_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{CUDA_X64};{CUDA_BIN};" + env.get("PATH", "")
    return env


class GpuSampler(threading.Thread):
    def __init__(self, period: float = 1.0):
        super().__init__(daemon=True)
        self.period = period
        self.peak_mib = 0
        self._done = threading.Event()

    def run(self) -> None:
        while not self._done.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                self.peak_mib = max(self.peak_mib, int(out.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            self._done.wait(self.period)

    def stop(self) -> int:
        self._done.set()
        self.join(timeout=5)
        return self.peak_mib


def gpu_now() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def stage(name: str, cmd: list[str], audio_s: float | None = None) -> dict:
    """Run one pipeline stage, timed, with peak VRAM."""
    print(f"[{name}] {' '.join(str(c) for c in cmd[:6])} ...", file=sys.stderr, flush=True)
    sampler = GpuSampler()
    base = gpu_now()
    sampler.start()
    t0 = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=cuda_env())
    wall = time.perf_counter() - t0
    peak = sampler.stop()
    rec = {
        "stage": name,
        "wall_s": round(wall, 2),
        "returncode": proc.returncode,
        "peak_vram_mib": peak,
        "vram_delta_mib": max(0, peak - base),
        "stdout": proc.stdout or "",
        "stderr_tail": (proc.stderr or "")[-3000:],
    }
    if audio_s:
        rec["rtf"] = round(wall / audio_s, 4)
        rec["speedup_x_realtime"] = round(audio_s / wall, 1)
    if proc.returncode != 0:
        print(f"    FAILED rc={proc.returncode}: {rec['stderr_tail'][-500:]}",
              file=sys.stderr, flush=True)
    else:
        print(f"    {rec['wall_s']}s  rtf={rec.get('rtf', '-')}  vram={rec['vram_delta_mib']} MiB",
              file=sys.stderr, flush=True)
    return rec


def transcript_from_json(raw: str) -> tuple[str, dict]:
    """Pull the plain transcript out of nemo-speech --json output."""
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip(), {}
    for key in ("text", "transcript"):
        if isinstance(doc.get(key), str):
            return doc[key].strip(), doc
    return raw.strip(), doc


def segment_for_nmt(text: str, max_words: int) -> list[str]:
    """Split a transcript into NMT-sized pieces, punctuation first, word cap second.

    Sentence boundaries are preferred because they translate better, but they
    cannot be depended on: unpunctuated conversational ASR can run for hundreds
    of words without a full stop, which would hand Riva a prompt past its
    context limit. Every emitted piece is therefore at most max_words long.
    """
    pieces: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        words = sent.split()
        if not words:
            continue
        for i in range(0, len(words), max_words):
            pieces.append(" ".join(words[i:i + max_words]))
    return pieces


def speaker_stats(doc: dict) -> dict:
    """Count speakers and turns in a diarized JSON result."""
    words = doc.get("words") or []
    spk = [w.get("speaker") for w in words if isinstance(w, dict) and w.get("speaker")]
    turns = 1 + sum(1 for a, b in zip(spk, spk[1:]) if a != b) if spk else 0
    return {"words_with_speaker": len(spk), "distinct_speakers": len(set(spk)), "turns": turns}


def run_stack(args: argparse.Namespace, out: Path) -> list[dict]:
    """ASR + diarization in one pass, then prove the TTS stack co-loads."""
    rows: list[dict] = []
    js = out / "stack.json"
    cmd = [NEMO_BIN, "transcribe", args.audio, "--model", args.asr_model,
           "--backend", f"cuda:{args.gpu}", "--diarize", "--format", "json",
           "--output", js, "--force"]
    if args.language:
        cmd += ["--language", args.language]
    r = stage("asr+diarize", cmd, args.audio_s)
    if js.exists():
        text, doc = transcript_from_json(js.read_text(encoding="utf-8"))
        r.update({"words": len(text.split()), **speaker_stats(doc)})
    rows.append(r)

    # Co-residency check: load the TTS stack (Magpie + NanoCodec + tokenizer)
    # while the ASR result is still on disk, and record what the GPU held.
    r2 = stage("tts-stack-load", [
        NEMO_BIN, "synthesize", "NeMo Speech pipeline co-residency check.",
        "--backend", f"cuda:{args.gpu}", "--output", out / "coresidency.wav", "--force"])
    rows.append(r2)
    return rows


def run_translate(args: argparse.Namespace, out: Path) -> list[dict]:
    """ASR -> Riva translation -> Magpie TTS, each stage timed on its own."""
    rows: list[dict] = []

    js = out / "translate_asr.json"
    cmd = [NEMO_BIN, "transcribe", args.audio, "--model", args.asr_model,
           "--backend", f"cuda:{args.gpu}", "--format", "json",
           "--output", js, "--force"]
    if args.language:
        cmd += ["--language", args.language]
    r = stage("asr", cmd, args.audio_s)
    text = ""
    if js.exists():
        text, _ = transcript_from_json(js.read_text(encoding="utf-8"))
        r["words"] = len(text.split())
    rows.append(r)
    if not text:
        print("no transcript; stopping", file=sys.stderr)
        return rows

    # Riva translates a sentence or short paragraph well and degrades on much
    # longer input, and rejects anything past nmt.model.n_ctx (default 1024
    # tokens) outright. Sentence punctuation alone is not enough to rely on:
    # conversational ASR output runs for hundreds of words between full stops --
    # a 60 s sample of this podcast split into exactly ONE "sentence" -- so the
    # split falls back to a word cap whenever punctuation does not arrive.
    sents = segment_for_nmt(text, args.nmt_max_words)
    src_txt = out / "asr_sentences.txt"
    src_txt.write_text("\n".join(sents), encoding="utf-8")
    tgt_txt = out / f"translated_{args.target_lang}.txt"
    r = stage("translate", [
        NEMO_BIN, "translate", "--model", args.nmt_model, "--from", args.source_lang,
        "--to", args.target_lang, "--input", src_txt, "--output", tgt_txt,
        "--backend", f"cuda:{args.gpu}", "--force"], args.audio_s)
    r["sentences"] = len(sents)
    rows.append(r)

    translated = tgt_txt.read_text(encoding="utf-8").strip() if tgt_txt.exists() else ""
    if not translated:
        print("no translation; stopping", file=sys.stderr)
        return rows

    # TTS on a bounded sample; synthesizing the whole transcript would produce
    # another 45 minutes of audio.
    sample = translated[: args.tts_max_chars]
    tts_in = out / "tts_input.txt"
    tts_in.write_text(sample, encoding="utf-8")
    wav = out / f"tts_{args.target_lang}.wav"
    r = stage("tts", [
        NEMO_BIN, "synthesize", "--input", tts_in, "--output", wav,
        "--language", args.tts_language, "--backend", f"cuda:{args.gpu}", "--force"])
    r["input_chars"] = len(sample)
    r["total_translated_chars"] = len(translated)
    if wav.exists():
        try:
            import wave
            with wave.open(str(wav)) as w:
                spoken = w.getnframes() / w.getframerate()
            r["audio_out_s"] = round(spoken, 2)
            if spoken > 0:
                r["tts_rtf"] = round(r["wall_s"] / spoken, 4)
                r["tts_speedup_x_realtime"] = round(spoken / r["wall_s"], 1)
            scale = len(translated) / max(1, len(sample))
            r["extrapolated_full_tts_s"] = round(r["wall_s"] * scale, 1)
        except Exception as e:
            r["wav_error"] = str(e)
    rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--audio-s", type=float, default=None)
    ap.add_argument("--mode", choices=["stack", "translate"], required=True)
    ap.add_argument("--asr-model", default="nemotron-3.5")
    ap.add_argument("--nmt-model", default=str(RIVA_GGUF))
    ap.add_argument("--language", default="en-US", help="ASR language prompt; '' to omit")
    ap.add_argument("--source-lang", default="en")
    ap.add_argument("--target-lang", default="es")
    ap.add_argument("--tts-language", default="es-ES")
    ap.add_argument("--tts-max-chars", type=int, default=1200)
    ap.add_argument("--nmt-max-words", type=int, default=40,
                    help="hard cap per translation request; Riva rejects prompts past n_ctx")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("D:/HuggingFace/bench/pipeline"))
    args = ap.parse_args()

    if args.audio_s is None:
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", str(args.audio)],
                           capture_output=True, text=True)
        args.audio_s = float(p.stdout.strip())

    out = args.outdir / args.mode
    out.mkdir(parents=True, exist_ok=True)

    rows = run_stack(args, out) if args.mode == "stack" else run_translate(args, out)

    total = sum(r["wall_s"] for r in rows if r.get("returncode") == 0)
    payload = {
        "mode": args.mode,
        "audio": str(args.audio),
        "audio_s": args.audio_s,
        "total_wall_s": round(total, 2),
        "total_rtf": round(total / args.audio_s, 4),
        "total_speedup_x_realtime": round(args.audio_s / total, 1) if total else None,
        "peak_vram_mib": max((r.get("peak_vram_mib", 0) for r in rows), default=0),
        "stages": [{k: v for k, v in r.items() if k != "stdout"} for r in rows],
    }
    (out / "pipeline.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    hdr = f"{'stage':<18}{'wall_s':>9}{'rtf':>9}{'xRT':>8}{'VRAM MiB':>10}{'rc':>4}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['stage']:<18}{r['wall_s']:>9.1f}{r.get('rtf', 0):>9.4f}"
              f"{r.get('speedup_x_realtime', 0):>8.1f}{r.get('vram_delta_mib', 0):>10}"
              f"{r['returncode']:>4}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<18}{total:>9.1f}{payload['total_rtf']:>9.4f}"
          f"{payload['total_speedup_x_realtime'] or 0:>8.1f}"
          f"{payload['peak_vram_mib']:>10}")
    print(f"\nwrote {out / 'pipeline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
