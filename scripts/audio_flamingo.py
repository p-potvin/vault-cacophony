#!/usr/bin/env python3
"""Run Audio Flamingo 3 on ggml, through llama.cpp's mtmd audio path.

AF3 is NVIDIA's audio-language model: a whisper-large-v3-shaped encoder feeding
a two-layer projector into Qwen2.5-7B. That shape is why it runs here at all --
llama.cpp already carried the encoder for Ultravox and Voxtral, so upstream
registers AF3 in `conversion/ultravox.py` and maps it onto the MUSIC_FLAMINGO
projector. No new C++ was needed; see docs/audio-flamingo.md for the build.

Quantized to Q4_K_M the model is 4.68 GB on disk and roughly 6.4 GB resident at
`-c 8192` (7.7 GB if you let n_ctx default to 32768, which is 1.9 GB of KV cache
for a model that caps audio at ten minutes). It fits a 12 GB card with room to
spare, which the bf16 original -- 16.5 GB -- does not.

    prompt eval   1239 tok/s
    generation      61 tok/s
    audio encode   ~210 ms per 30 s window

THE THIRTY-SECOND RULE, which is the whole reason this file exists
------------------------------------------------------------------
AF3 processes audio in 30 s windows. **Transcription is only reliable inside a
single window.** The moment a clip needs a second one, ASR collapses -- and it
collapses quietly, returning fluent text rather than an error:

    29.0 s   1 window    "they've got the best equipment that money can buy..."   correct
    29.9 s   1 window    correct
    29.99 s  2 windows   "the"
    30.0 s   2 windows   "the"
    60.0 s   3 windows   "to get the money to get the money to get the money..."

The split happens just under 30 s, not at it -- 29.9 s is one window and 29.99 s
is two -- so a naive 30 s chunker lands exactly on the broken case. CHUNK_S is
29.0 to keep a real margin.

This is not a quantization artifact. The unquantized f16 fails the same 60 s
clip, hallucinating "they're going to be able to get the money to get the
equipment they need to get the job done" against speech that says nothing of the
kind. It is the mtmd audio path, which upstream flags as experimental on every
run. So the fix is ours to make: chunk before the model sees it.

Captioning and audio understanding are *not* affected -- a 90 s clip spanning
four windows is described correctly. Only transcription needs chunking, so only
--task asr chunks. Anything else is handed the whole file.

Chunk boundaries are placed at the quietest frame near the deadline rather than
at a fixed offset, because cutting mid-word costs a word in both chunks and the
search is nearly free.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR = 16000
# 29.0 s, not 30. The window split fires between 29.9 and 29.99 s and a clip
# that trips it returns one word, so the margin is deliberate.
CHUNK_S = 29.0
# Look this far back from the deadline for a quiet frame to cut on.
SEEK_BACK_S = 2.0
FRAME_S = 0.02

# llama.cpp log lines: "0.03.129.743 I sched_reserve: ..."
LOG_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+\s+[IWED]\s")
# AF3 wraps transcripts in a sentence rather than emitting them bare.
ASR_RE = re.compile(r"^the spoken content of the audio is\s*'(.*)'\.?\s*$", re.I | re.S)

ASR_PROMPT = "Transcribe the input speech."


def load_16k_mono(path: Path) -> np.ndarray:
    """Decode anything soundfile can read down to 16 kHz mono float32."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    return np.ascontiguousarray(x, dtype=np.float32)


def split_points(x: np.ndarray, chunk_s: float = CHUNK_S) -> list[tuple[int, int]]:
    """Cut x into <= chunk_s pieces, preferring quiet frames as boundaries.

    Returns (start, end) sample offsets. The last piece is whatever is left and
    may be much shorter; that is fine, short clips transcribe correctly.
    """
    limit = int(chunk_s * SR)
    seek = int(SEEK_BACK_S * SR)
    frame = int(FRAME_S * SR)

    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(x):
        end = start + limit
        if end >= len(x):
            spans.append((start, len(x)))
            break
        # Quietest frame in the last SEEK_BACK_S before the deadline.
        lo = max(start + frame, end - seek)
        window = x[lo:end]
        n_frames = len(window) // frame
        if n_frames > 0:
            energy = np.abs(window[: n_frames * frame].reshape(n_frames, frame)).mean(axis=1)
            end = lo + int(energy.argmin()) * frame
        spans.append((start, end))
        start = end
    return spans


def strip_logs(stdout: str) -> str:
    """Keep the model's answer, drop llama.cpp's running commentary."""
    lines = [ln for ln in stdout.splitlines() if not LOG_RE.match(ln)]
    return "\n".join(lines).strip()


def run_once(wav: Path, prompt: str, args: argparse.Namespace) -> str:
    cmd = [
        str(args.bin),
        "-m", str(args.model),
        "--mmproj", str(args.mmproj),
        "--audio", str(wav),
        "-p", prompt,
        "-ngl", str(args.ngl),
        "-c", str(args.ctx),
        "--temp", str(args.temp),
        "-n", str(args.n_predict),
    ]
    if args.chat_template:
        cmd += ["--chat-template", args.chat_template]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        raise SystemExit(f"llama-mtmd-cli exited {proc.returncode}")
    # llama-mtmd-cli puts every log line on stderr and only the model's answer on
    # stdout, so the two streams must stay separate -- merging them drags in the
    # startup banner and the chat-template example, which look like model output.
    return strip_logs(proc.stdout or "")


def unwrap_asr(text: str) -> str:
    m = ASR_RE.match(text.strip())
    return m.group(1).strip() if m else text.strip()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument(
        "--task",
        choices=["asr", "ask"],
        default="asr",
        help="asr chunks at 29 s and transcribes; ask sends the whole file",
    )
    ap.add_argument("--prompt", default=None, help="prompt for --task ask")
    ap.add_argument("--model", type=Path, default=Path("D:/HuggingFace/gguf/af3-Q4_K_M.gguf"))
    ap.add_argument("--mmproj", type=Path, default=Path("D:/HuggingFace/gguf/mmproj-af3-f16.gguf"))
    ap.add_argument("--bin", type=Path, default=Path("D:/HuggingFace/llama-bin/llama-mtmd-cli.exe"))
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument(
        "--ctx", type=int, default=8192, help="32768 costs 1.9 GB of KV cache for no gain here"
    )
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument(
        "--chat-template",
        default=None,
        help="pass 'chatml' for a GGUF built without the embedded template",
    )
    ap.add_argument("--chunk-s", type=float, default=CHUNK_S)
    ap.add_argument(
        "--timestamps", action="store_true", help="prefix each ASR chunk with its start time"
    )
    args = ap.parse_args()

    if args.task == "ask" and not args.prompt:
        ap.error("--task ask needs --prompt")

    x = load_16k_mono(args.audio)
    dur = len(x) / SR

    if args.task == "ask":
        # Multi-window understanding is fine; hand it the whole file.
        print(run_once(args.audio, args.prompt, args))
        return 0

    spans = split_points(x, args.chunk_s)
    if len(spans) > 1:
        sys.stderr.write(
            f"{args.audio.name}: {dur:.1f} s -> {len(spans)} chunks "
            f"(AF3 transcribes reliably only within one 30 s window)\n"
        )

    out: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        for i, (a, b) in enumerate(spans):
            piece = Path(td) / f"chunk{i:04d}.wav"
            sf.write(str(piece), x[a:b], SR, subtype="PCM_16")
            text = unwrap_asr(run_once(piece, ASR_PROMPT, args))
            if args.timestamps:
                t = a / SR
                out.append(f"[{int(t) // 60:02d}:{t % 60:05.2f}] {text}")
            else:
                out.append(text)

    print(("\n" if args.timestamps else " ").join(s for s in out if s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
