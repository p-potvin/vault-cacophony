#!/usr/bin/env python3
"""Translate an .srt to one or more languages via deep_translator.

Reads <name>.srt and writes <name>.<lang>.srt per target. Timings are copied
verbatim; only text is translated.

Two things matter for quality:

1. SENTENCES, NOT CUES. A cue is a display unit, not a unit of meaning — cue
   boundaries fall wherever the speaker paused. Translating each cue alone
   feeds the translator fragments, and "ses tours dans la rue" split across a
   boundary comes back as "towers in the street" because *tours* without its
   sentence is ambiguous. So cues are stitched into sentences, the sentences
   are translated, and the result is redistributed over the original cues in
   proportion to how much of the sentence each cue carried.

2. BATCHING. deep_translator's translate_batch() is a for-loop over
   translate() (base.py _translate_batch), one HTTP request per item. Joining
   sentences into ~3000-char groups cuts request count by orders of magnitude
   and gives the translator surrounding context. A group whose reply has the
   wrong number of parts is retried one sentence at a time, because a mangled
   delimiter would otherwise desynchronise everything after it.
"""

from __future__ import annotations

import argparse
import re
import sys

MAX_GROUP_CHARS = 3000
SEP = "\n@@@\n"

# The endpoint can answer HTTP 200 with an HTML error page, which deep_translator
# hands back as though it were a translation — observed dropping "Error 500
# (Server Error)!!1500.That's an error." verbatim into a subtitle cue. Nothing
# raises, so the only defence is to recognise the shape of the reply.
_ERROR_MARKERS = (
    "That’s an error", "That's an error", "Server Error",
    "Please try again later", "<!DOCTYPE", "<html",
)


def looks_like_error(reply: str, source: str) -> bool:
    if not reply:
        return True
    if any(mark in reply for mark in _ERROR_MARKERS):
        return True
    # A translation stays within an order of magnitude of its source length.
    return len(reply) > 200 and len(reply) > 5 * max(len(source), 1)

CUE_RE = re.compile(
    r"(?P<idx>\d+)\s*\n(?P<time>[\d:,]+\s*-->\s*[\d:,]+)\s*\n(?P<text>.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)
# Split after . ! ? or … when followed by space+capital, or at end of text.
SENT_END = re.compile(r"(?<=[.!?…])\s+")


def parse_srt(text):
    return [(m.group("idx"), m.group("time"), " ".join(m.group("text").split()))
            for m in CUE_RE.finditer(text)]


def render_srt(cues):
    return "".join(f"{i}\n{t}\n{x}\n\n" for i, t, x in cues)


def group_texts(texts, max_chars=MAX_GROUP_CHARS):
    groups, cur, n = [], [], 0
    for t in texts:
        add = len(t) + len(SEP)
        if cur and n + add > max_chars:
            groups.append(cur)
            cur, n = [], 0
        cur.append(t)
        n += add
    if cur:
        groups.append(cur)
    return groups


def translate_many(texts, target, source="auto"):
    """texts -> translations, cached and batched. Order preserved."""
    from deep_translator import GoogleTranslator

    tr = GoogleTranslator(source=source, target=target)
    cache = {}

    def one(s):
        if s not in cache:
            try:
                reply = tr.translate(s) or ""
            except Exception as e:
                print(f"    [!] translation failed ({e}); keeping source", file=sys.stderr)
                reply = ""
            if looks_like_error(reply, s):
                print("    [!] endpoint returned an error page; keeping source", file=sys.stderr)
                reply = s
            cache[s] = reply
        return cache[s]

    uniq = [t for t in dict.fromkeys(texts) if t.strip()]
    for group in group_texts(uniq):
        if len(group) == 1:
            one(group[0])
            continue
        try:
            reply = tr.translate(SEP.join(group)) or ""
        except Exception as e:
            print(f"    [!] batch failed ({e}); falling back per sentence", file=sys.stderr)
            reply = ""
        if looks_like_error(reply, "".join(group)):
            print("    [!] batch returned an error page; retrying per sentence", file=sys.stderr)
            reply = ""
        parts = [p.strip() for p in reply.split("@@@")] if reply else []
        if len(parts) == len(group):
            for src, dst in zip(group, parts):
                cache.setdefault(src, dst if dst and not looks_like_error(dst, src) else src)
        else:
            for src in group:
                one(src)
    return [cache.get(t, t) for t in texts]


def split_proportional(text, weights):
    """Split `text` into len(weights) parts on word boundaries, sized by weights.

    Used to lay a translated sentence back over the cues it spanned. Word
    counts rarely divide as cleanly as character counts, so this walks words
    and closes each part once it has met its share.
    """
    words = text.split()
    total_w = sum(weights) or 1
    n = len(weights)
    if n == 1 or not words:
        return [text] + [""] * (n - 1)

    out, i = [], 0
    consumed = 0.0
    for k, w in enumerate(weights):
        if k == n - 1:
            out.append(" ".join(words[i:]))
            break
        consumed += w / total_w
        target = max(i + 1, min(len(words) - (n - k - 1), round(consumed * len(words))))
        out.append(" ".join(words[i:target]))
        i = target
    return out


def build_sentences(cues):
    """-> (sentences, spans) where spans[i] = [(cue_index, char_len), ...]."""
    joined, marks = [], []
    pos = 0
    for ci, (_, _, text) in enumerate(cues):
        if joined:
            joined.append(" ")
            pos += 1
        joined.append(text)
        marks.append((pos, pos + len(text), ci))
        pos += len(text)
    full = "".join(joined)

    sentences, spans = [], []
    start = 0
    for m in list(SENT_END.finditer(full)) + [None]:
        end = m.start() if m else len(full)
        s = full[start:end].strip()
        if s:
            overlaps = []
            for (cs, ce, ci) in marks:
                lo, hi = max(cs, start), min(ce, end)
                if hi > lo:
                    overlaps.append((ci, hi - lo))
            if overlaps:
                sentences.append(s)
                spans.append(overlaps)
        start = m.end() if m else end
    return sentences, spans


def translate_srt(cues, target, source, per_cue):
    if per_cue:
        texts = [c[2] for c in cues]
        return list(zip([c[0] for c in cues], [c[1] for c in cues],
                        translate_many(texts, target, source)))

    sentences, spans = build_sentences(cues)
    translated = translate_many(sentences, target, source)

    parts = {i: [] for i in range(len(cues))}
    for sent, span in zip(translated, spans):
        weights = [w for _, w in span]
        for (ci, _), piece in zip(span, split_proportional(sent, weights)):
            if piece:
                parts[ci].append(piece)
    return [(c[0], c[1], " ".join(parts[i]).strip() or c[2])
            for i, c in enumerate(cues)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", required=True)
    ap.add_argument("--langs", required=True, help="comma-separated target codes")
    ap.add_argument("--source", default="auto")
    ap.add_argument("--per-cue", action="store_true",
                    help="translate each cue in isolation (old behaviour; loses context)")
    args = ap.parse_args()

    with open(args.srt, encoding="utf-8") as f:
        cues = parse_srt(f.read())
    if not cues:
        print(f"[!] no cues parsed from {args.srt}", file=sys.stderr)
        return 1

    base = args.srt[:-4] if args.srt.lower().endswith(".srt") else args.srt
    for lang in [l.strip() for l in args.langs.split(",") if l.strip()]:
        out = translate_srt(cues, lang, args.source, args.per_cue)
        dest = f"{base}.{lang}.srt"
        with open(dest, "w", encoding="utf-8") as f:
            f.write(render_srt(out))
        print(f"    wrote {dest} ({len(out)} cues)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
