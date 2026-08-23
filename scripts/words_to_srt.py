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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tagstore

SR = 16000
# Sub-word pieces further apart than this belong to different words.
PIECE_GAP_S = 0.5


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
        # A piece that lands far from the one before it usually belongs to the
        # next word: the space that would have separated them went missing at a
        # streaming seam. Letting it extend the current word instead stretches
        # that word -- and the cue holding it -- across the whole gap; one file
        # had "mother." on screen for nineteen seconds.
        stranded = cur is not None and (
            t["start_sample"] - cur["end_sample"]) / SR > PIECE_GAP_S
        if stranded and not any(c.isalnum() for c in piece):
            # Trailing punctuation is the exception: it belongs to the word it
            # follows, so take the character and leave the timing alone, or the
            # transcript reads "welcome his gospel" / ". He had written".
            cur["word"] += piece
            continue
        if piece.startswith(" ") or cur is None or stranded:
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


def readable(cues, fill, min_dur, cps, cue_gap):
    """Hold a cue long enough to read it, and never past the next one.

    Nemotron reports one 80 ms frame per token, so a cue ending on a one-token
    word "ends" 80 ms after that word began however long it was really spoken:
    "No" flashes for 80 ms and is gone. Two corrections, in order. `fill` gives
    back the time the frame grid truncated. Then the cue is held for at least
    `min_dur`, and for however long its characters take to read at `cps` -- the
    subtitle convention rather than a guess.

    The next cue's start always wins, so nothing overlaps and no time is
    invented inside real speech. This runs after cue boundaries are decided:
    growing an end before that would shrink the measured pause and silently
    merge a cue with its neighbour.
    """
    out = []
    for i, (start, end, text, first) in enumerate(cues):
        want = max(end + fill, start + max(min_dur, len(text.replace("\n", " ")) / cps))
        if i + 1 < len(cues):
            want = min(want, max(end, cues[i + 1][0] - cue_gap))
        out.append((start, want, text, first))
    return out


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
    ap.add_argument("--fill", type=float, default=0.25,
                    help="seconds given back to a cue whose end the model's frame grid "
                         "truncated; 0 disables")
    ap.add_argument("--min-dur", type=float, default=0.9,
                    help="shortest time a cue stays on screen, unless the next cue starts sooner")
    ap.add_argument("--cps", type=float, default=17.0,
                    help="reading speed in characters per second; a cue is held at least this long")
    ap.add_argument("--cue-gap", type=float, default=0.08,
                    help="gap kept between one cue and the next when a cue is extended")
    ap.add_argument("--tagged-text", help="transcript with <lang> tags, from --language auto")
    ap.add_argument("--tags-out", help="tag store to write the language track and cues into")
    ap.add_argument("--media", help="media path recorded in the tag store")
    ap.add_argument("--model-name", help="model credited for the language track")
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
    cues = readable(cues, args.fill, args.min_dur, args.cps, args.cue_gap)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, (start, end, text, _) in enumerate(cues, 1):
            f.write(f"{i}\n{ts(start)} --> {ts(end)}\n{text}\n\n")

    spans = []
    if args.tagged_text and os.path.exists(args.tagged_text):
        with open(args.tagged_text, encoding="utf-8") as f:
            marks, drift = languages_by_word(f.read(), len(words))
        if marks and drift <= 0.05:
            for start, end, text, first in cues:
                code = next((c for at, c in reversed(marks) if at <= first), marks[0][1])
                spans.append({"start": round(start, 3), "end": round(end, 3), "value": code})
        elif marks:
            # Counting words in the transcript and counting merged tokens
            # disagreed, so per-cue placement would be fiction. The file-level
            # answer is still worth keeping.
            spans = [{"start": 0.0, "end": round(cues[-1][1], 3), "value": marks[0][1]}]
            print(f"    [!] language tags drifted {drift:.0%} from the word stream; "
                  f"tagging the file, not the cues", file=sys.stderr)

    if args.tags_out:
        # Merge into whatever is already there: this pass owns the language
        # track and the cues, and must not disturb a speaker track written
        # before it or after it.
        store = tagstore.load(args.tags_out)
        tagstore.set_media(store, args.media, cues[-1][1] if cues else None)
        # The store keeps cue text on one line; the wrapping is a display
        # decision that belongs to the .srt, not to the annotations.
        tagstore.put_cues(store, [(s, e, " ".join(t.split())) for s, e, t, _ in cues])
        tagstore.put_track(store, "language", tagstore.collapse(spans), "asr",
                           model=args.model_name)
        tagstore.save(store, args.tags_out)

    spoken = sorted({s["value"] for s in spans})
    extra = f", {'/'.join(spoken)}" if spoken else ""
    print(f"    {len(cues)} cues{extra} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
