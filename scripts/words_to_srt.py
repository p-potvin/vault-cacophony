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
import os
import re
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


def merge_tokens(tokens):
    """Nemotron times sub-word pieces, not words: 'Mi' 'st' 'er' ' ' 'Q' 'u' 'il' 'ter'.

    A piece that begins with a space opens a new word, everything else extends
    the current one, and a bare space token closes it. Pieces sometimes share a
    timestamp, so a word spans the earliest start and the latest end it saw.
    Parakeet already emits whole words and needs none of this.
    """
    words, cur = [], None
    for t in tokens:
        piece = t.get("word", "")
        if not piece.strip():
            if cur:
                words.append(cur)
                cur = None
            continue
        if piece.startswith(" ") or cur is None:
            if cur:
                words.append(cur)
            cur = {"word": piece.strip(),
                   "start_sample": t["start_sample"],
                   "end_sample": t["end_sample"]}
        else:
            cur["word"] += piece
            cur["start_sample"] = min(cur["start_sample"], t["start_sample"])
            cur["end_sample"] = max(cur["end_sample"], t["end_sample"])
    if cur:
        words.append(cur)
    return words


TAG_RE = re.compile(r"<([A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?)>")


def languages_by_word(tagged_text, word_count):
    """Map each language tag to the word index where it takes effect.

    `--language auto` makes the model announce the language it decoded each
    segment in, but the tag lands in the transcript only -- the token stream has
    it stripped -- so the position has to come from counting words in the text.
    Both come out of the same decode, so the counts line up; when they do not,
    the caller falls back to one language for the file rather than tagging cues
    with a guess.
    """
    marks, seen = [], 0
    pos = 0
    for m in TAG_RE.finditer(tagged_text):
        seen += len(tagged_text[pos:m.start()].split())
        marks.append((seen, m.group(1)))
        pos = m.end()
    seen += len(tagged_text[pos:].split())
    drift = abs(seen - word_count) / max(word_count, 1)
    return marks, drift


def build(words, gap, max_chars, max_dur, width):
    cues, cur = [], []
    index = [0]

    def flush():
        if not cur:
            return
        start = cur[0]["start_sample"] / SR
        end = cur[-1]["end_sample"] / SR
        text = " ".join(w["word"] for w in cur).strip()
        if text:
            # The first word's index is what a language mark is matched against.
            cues.append((start, end, wrap(text, width), index[0] - len(cur)))
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
        index[0] += 1
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
    ap.add_argument("--merge-tokens", action="store_true",
                    help="input times sub-word pieces rather than words (nemotron)")
    ap.add_argument("--tagged-text", help="transcript with <lang> tags, from --language auto")
    ap.add_argument("--tags-out", help="write per-cue language to this JSON")
    args = ap.parse_args()

    with open(args.words, encoding="utf-8") as f:
        data = json.load(f)
    words = data["words"] if isinstance(data, dict) and "words" in data else data
    # Merge before filtering: the bare space token is what separates one word
    # from the next, so dropping blanks first glues the transcript into
    # "theapostle of themiddle classes".
    if args.merge_tokens:
        words = merge_tokens(words)
    words = [w for w in words if w.get("word", "").strip()]
    if not words:
        print(f"[!] no words in {args.words}", file=sys.stderr)
        return 1

    cues = build(words, args.gap, args.max_chars, args.max_dur, args.width)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, (start, end, text, _) in enumerate(cues, 1):
            f.write(f"{i}\n{ts(start)} --> {ts(end)}\n{text}\n\n")

    langs = None
    if args.tagged_text and os.path.exists(args.tagged_text):
        with open(args.tagged_text, encoding="utf-8") as f:
            marks, drift = languages_by_word(f.read(), len(words))
        if marks and drift <= 0.05:
            langs = []
            for start, end, text, first in cues:
                lang = next((code for at, code in reversed(marks) if at <= first), marks[0][1])
                langs.append({"start": round(start, 3), "end": round(end, 3), "language": lang})
        elif marks:
            # Counting words in the transcript and counting merged tokens
            # disagreed, so per-cue placement would be fiction. The file-level
            # answer is still worth keeping.
            langs = [{"start": 0.0, "end": cues[-1][1], "language": marks[0][1], "whole_file": True}]
            print(f"    [!] language tags drifted {drift:.0%} from the word stream; "
                  f"tagging the file, not the cues", file=sys.stderr)
    if langs and args.tags_out:
        with open(args.tags_out, "w", encoding="utf-8") as f:
            json.dump(langs, f, ensure_ascii=False, indent=1)

    spoken = sorted({l["language"] for l in langs}) if langs else []
    extra = f", {'/'.join(spoken)}" if spoken else ""
    print(f"    {len(cues)} cues{extra} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
