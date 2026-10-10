#!/usr/bin/env python3
"""Run NeMo-Speech.cpp ASR models over an Open ASR leaderboard parquet split.

Produces JSONL manifests that `scripts/evaluate_openasr.py` scores with the
leaderboard's own English path: Whisper's `EnglishTextNormalizer` followed by
`evaluate.load('wer')`. This script only runs inference and records timing; it
deliberately computes no WER of its own, so the scoring stays in one place and
stays comparable to the Audio Flamingo runs already in `G:/OpenASR/results`.

Why directory mode rather than one process per utterance: `nemo-speech
transcribe DIR --output-dir` loads the model once and shares it across every
file, and `--concurrency` lets compatible work batch on the GPU. The existing
per-utterance runner pays a model load for each of 2620 rows, which measures the
loader rather than the model. RTFx here is therefore total audio seconds over
total wall seconds, which is what the leaderboard reports.

Note LibriSpeech test.clean contains utterances longer than 29 s (the first row
is 34.955 s), so anything with a fixed window -- Audio Flamingo especially --
must chunk rather than truncate. The NeMo models have no such limit.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
NEMO_BIN = REPO / "NeMo-Speech.cpp" / "build-cuda" / "bin" / "nemo-speech.exe"
CUDA_X64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
DEFAULT_PARQUET = Path(r"G:\OpenASR\open-asr-leaderboard\librispeech\test.clean-00000-of-00001.parquet")


def cuda_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{CUDA_X64};{CUDA_BIN};" + env.get("PATH", "")
    return env


def records(parquet_path: Path, limit: int | None, offset: int) -> Iterator[dict[str, Any]]:
    emitted = 0
    skipped = 0
    src = pq.ParquetFile(parquet_path)
    for batch in src.iter_batches(batch_size=32):
        for rec in batch.to_pylist():
            if skipped < offset:
                skipped += 1
                continue
            yield rec
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def stage_wavs(parquet_path: Path, limit: int | None, offset: int,
               wav_dir: Path) -> tuple[list[dict], float]:
    """Write each utterance to <id>.wav, keeping only metadata in memory.

    The audio bytes are deliberately dropped as soon as each WAV is on disk.
    Holding all 2620 blobs costs ~0.9 GiB of RSS in this process, and that is
    enough to make the *child* abort: nemo-speech's offline RNNT path asks for a
    very large contiguous host allocation and, when malloc returns NULL, dies on

        GGML_ASSERT(ctx->mem_buffer != NULL) failed   ggml/src/ggml.c:1610

    which reads like a GPU or model problem and is neither. Measured on this box
    (31 GiB total, heavily loaded): the run succeeds with ~8.4 GiB available and
    fails with ~7.6 GiB. The CTC and TDT heads are unaffected, which is why the
    failure looked model-specific and, when free memory drifted between runs,
    looked like a file-count threshold. It is neither -- it is host RAM.
    """
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta: list[dict] = []
    total = 0.0
    for rec in records(parquet_path, limit, offset):
        data, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32")
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        sf.write(str(wav_dir / f"{rec['id']}.wav"), data, sr, subtype="PCM_16")
        total += len(data) / sr
        meta.append({"id": rec["id"], "text": rec["text"],
                     "dataset": rec.get("dataset", ""),
                     "audio_length_s": rec.get("audio_length_s")})
    return meta, total


def metadata_only(parquet_path: Path, limit: int | None, offset: int) -> list[dict]:
    """Row metadata without audio bytes, for when the WAVs are already staged."""
    src = pq.ParquetFile(parquet_path)
    out: list[dict] = []
    skipped = 0
    for batch in src.iter_batches(batch_size=256,
                                  columns=["id", "text", "dataset", "audio_length_s"]):
        for rec in batch.to_pylist():
            if skipped < offset:
                skipped += 1
                continue
            out.append(rec)
            if limit is not None and len(out) >= limit:
                return out
    return out


def avail_gib() -> float:
    """Available host RAM. Logged because it, not the GPU, is what fails here."""
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return -1.0


def _run_one_dir(model: str, in_dir: Path, out_dir: Path, args: argparse.Namespace,
                 extra: list[str] | None) -> tuple[float, int, str]:
    cmd = [str(NEMO_BIN), "transcribe", str(in_dir), "--output-dir", str(out_dir),
           "--model", model, "--backend", f"cuda:{args.gpu}",
           "--format", "text", "--concurrency", str(args.concurrency), "--force"]
    if args.language:
        cmd += ["--language", args.language]
    cmd += extra or []
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=cuda_env())
    return time.perf_counter() - t0, proc.returncode, (proc.stderr or "")[-3000:]


def transcribe_dir(model: str, wav_dir: Path, out_dir: Path, args: argparse.Namespace,
                   extra: list[str] | None = None) -> tuple[float, int, str]:
    """Transcribe a directory, optionally in sub-batches of --batch-files.

    Batching exists to work around a crash, not for speed. The cache-aware RNNT
    models (nemotron-en, nemotron-3.5) abort with

        GGML_ASSERT(ctx->mem_buffer != NULL) failed   ggml/src/ggml.c:1610

    once a single `transcribe DIR` call is given more than ~128 files. Measured:
    128 files succeed, 150 fail, independent of --concurrency (1, 2 and 4 all
    behave the same) and independent of utterance length (the longest 34.95 s
    utterances sit inside a working 20-file subset). The CTC and TDT heads --
    parakeet-ctc, parakeet-tdt -- are unaffected and run all 2620 in one call,
    so this is specific to the cache-aware RNNT path rather than to the loader.

    Sub-batch directories are hardlinked rather than copied; the WAVs are the
    same bytes and a 5.4 h split would otherwise cost 622 MB of pointless I/O.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.batch_files:
        return _run_one_dir(model, wav_dir, out_dir, args, extra)

    wavs = sorted(wav_dir.glob("*.wav"))
    batch_root = wav_dir.parent / f"_batches_{wav_dir.name}"
    if batch_root.exists():
        shutil.rmtree(batch_root)
    total_wall = 0.0
    for start in range(0, len(wavs), args.batch_files):
        chunk = wavs[start:start + args.batch_files]
        bdir = batch_root / f"b{start:06d}"
        bdir.mkdir(parents=True, exist_ok=True)
        for w in chunk:
            target = bdir / w.name
            try:
                os.link(w, target)
            except OSError:
                shutil.copy2(w, target)
        wall, rc, err = _run_one_dir(model, bdir, out_dir, args, extra)
        total_wall += wall
        if rc != 0:
            shutil.rmtree(batch_root, ignore_errors=True)
            return total_wall, rc, err
    shutil.rmtree(batch_root, ignore_errors=True)
    return total_wall, 0, ""


def collect(rows: list[dict], out_dir: Path, jsonl: Path) -> int:
    """Pair each reference with its transcript file; returns rows written."""
    written = 0
    with jsonl.open("w", encoding="utf-8") as fh:
        for rec in rows:
            hyp_path = out_dir / f"{rec['id']}.txt"
            if not hyp_path.exists():
                continue
            prediction = hyp_path.read_text(encoding="utf-8", errors="replace").strip()
            fh.write(json.dumps({
                "id": rec["id"],
                "dataset": rec.get("dataset", ""),
                "reference": rec["text"],
                "prediction": prediction,
                "audio_length_s": rec.get("audio_length_s"),
            }, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--models", default="parakeet-tdt,nemotron-en,nemotron-3.5,parakeet-ctc")
    ap.add_argument("--limit", type=int, default=None, help="utterances; default all")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--language", default="", help="ASR language prompt; '' to omit")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--batch-files", type=int, default=0,
                    help="split the directory into sub-runs of N files; use 100 for the "
                         "cache-aware RNNT models, which abort above ~128 (0 = one call)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workdir", type=Path, default=Path("D:/HuggingFace/openasr"))
    ap.add_argument("--results", type=Path, default=Path("G:/OpenASR/results"))
    ap.add_argument("--tag", default="librispeech-clean")
    ap.add_argument("--extra", default="", help="extra CLI args, space separated")
    args = ap.parse_args()

    rows = metadata_only(args.parquet, args.limit, args.offset)
    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    wav_dir = args.workdir / f"{args.tag}_wav_{args.offset}_{len(rows)}"
    if not wav_dir.exists() or len(list(wav_dir.glob("*.wav"))) != len(rows):
        print(f"staging {len(rows)} wavs -> {wav_dir}", file=sys.stderr, flush=True)
        rows, audio_s = stage_wavs(args.parquet, args.limit, args.offset, wav_dir)
        (wav_dir / "_audio_seconds.txt").write_text(str(audio_s), encoding="utf-8")
    else:
        audio_s = float((wav_dir / "_audio_seconds.txt").read_text(encoding="utf-8"))
    print(f"{len(rows)} utterances, {audio_s / 3600:.2f} h of audio", file=sys.stderr, flush=True)

    args.results.mkdir(parents=True, exist_ok=True)
    extra = args.extra.split() if args.extra else None
    summary: list[dict] = []

    for model in [m for m in args.models.split(",") if m]:
        safe = model.replace("/", "-")
        out_dir = args.workdir / f"{args.tag}_out_{safe}"
        print(f"[{model}] transcribing ... (host RAM available {avail_gib():.2f} GiB)",
              file=sys.stderr, flush=True)
        wall, rc, err = transcribe_dir(model, wav_dir, out_dir, args, extra)
        jsonl = args.results / f"{safe}-{args.tag}-{len(rows)}.jsonl"
        n = collect(rows, out_dir, jsonl)
        rtfx = round(audio_s / wall, 1) if wall else 0.0
        row = {"model": model, "wall_s": round(wall, 2), "rtfx": rtfx,
               "utterances_expected": len(rows), "utterances_written": n,
               "returncode": rc, "jsonl": str(jsonl)}
        if rc != 0:
            row["stderr_tail"] = err[-800:]
            print(f"    FAILED rc={rc}: {err[-400:]}", file=sys.stderr, flush=True)
        else:
            print(f"    {wall:.1f}s  RTFx {rtfx}  wrote {n}/{len(rows)}",
                  file=sys.stderr, flush=True)
        summary.append(row)

    out = args.results / f"nemo-{args.tag}-{len(rows)}.runs.json"
    out.write_text(json.dumps({"audio_seconds": audio_s, "runs": summary}, indent=2),
                   encoding="utf-8")

    print(f"\n{'model':<20}{'wall_s':>10}{'RTFx':>9}{'rows':>8}{'rc':>4}")
    print("-" * 51)
    for r in summary:
        print(f"{r['model']:<20}{r['wall_s']:>10.1f}{r['rtfx']:>9.1f}"
              f"{r['utterances_written']:>8}{r['returncode']:>4}")
    print(f"\nwrote {out}")
    print("score with: python scripts/evaluate_openasr.py --input <jsonl> --output <json>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
