#!/usr/bin/env python3
"""Build an .srt from audio.cpp's --words-out JSON.

Cue boundaries come from gaps between recognised words, so there is no VAD or
silence-RMS threshold to tune. This is what makes the cues usable: asking the
ASR for subtitles directly gives one cue for the whole file, and splitting
purely on character count cuts mid-sentence at arbitrary points.

audio.cpp emits [{start_sample, end_sample, word, confidence}, ...] at the
model's 16 kHz feature rate.
"""

from __future__ import annotations

import argparse
import json
import sys

SR = 16000


def ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap(text: str, width: int, max_lines: int = 2) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if cur and len(cand) > width:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        # Rebalance rather than drop: join the overflow onto the last line.
        head, tail = lines[: max_lines - 1], " ".join(lines[max_lines - 1:])
        lines = head + [tail]
    return "\n".join(lines)


def build(words, gap, max_chars, max_dur, width):
    cues, cur = [], []

    def flush():
        if not cur:
            return
        start = cur[0]["start_sample"] / SR
        end = cur[-1]["end_sample"] / SR
        text = " ".join(w["word"] for w in cur).strip()
        if text:
            cues.append((start, end, wrap(text, width)))
        cur.clear()

    for w in words:
        if cur:
            prev_end = cur[-1]["end_sample"] / SR
            this_start = w["start_sample"] / SR
            # Project the cue length *including* this word: splitting only once
            # the cap is already exceeded overflows the last wrapped line.
            length = len(" ".join(x["word"] for x in cur)) + 1 + len(w["word"])
            dur = w["end_sample"] / SR - cur[0]["start_sample"] / SR
            # A pause is the strongest signal of a natural break; the char and
            # duration caps only exist to stop a run-on stretch of speech from
            # becoming an unreadable wall.
            if (this_start - prev_end) >= gap or length > max_chars or dur >= max_dur:
                flush()
        cur.append(w)
    flush()
    return cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=float, default=0.6, help="seconds of silence that starts a new cue")
    ap.add_argument("--max-chars", type=int, default=76,
                    help="soft cap before forcing a split; 76 is the largest value that still "
                         "wraps into two lines of --width without overflow (84 leaves 8 lines long)")
    ap.add_argument("--max-dur", type=float, default=6.0, help="seconds before forcing a split")
    ap.add_argument("--width", type=int, default=42, help="characters per displayed line")
    args = ap.parse_args()

    with open(args.words, encoding="utf-8") as f:
        data = json.load(f)
    words = data["words"] if isinstance(data, dict) and "words" in data else data
    words = [w for w in words if w.get("word", "").strip()]
    if not words:
        print(f"[!] no words in {args.words}", file=sys.stderr)
        return 1

    cues = build(words, args.gap, args.max_chars, args.max_dur, args.width)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(cues, 1):
            f.write(f"{i}\n{ts(start)} --> {ts(end)}\n{text}\n\n")
    print(f"    {len(cues)} cues -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
