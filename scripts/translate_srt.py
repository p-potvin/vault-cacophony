#!/usr/bin/env python3
"""Translate an .srt in place to one or more languages, via deep_translator.

Reads <name>.srt and writes <name>.<lang>.srt per target. Cue timings are
copied verbatim; only the text is translated.

deep_translator's own translate_batch() is not a batch — it is a for-loop over
translate() (base.py _translate_batch), so it issues one HTTP request per cue.
Joining cues into groups with a delimiter cuts request count by orders of
magnitude and lets the translator see surrounding lines for context. When a
group comes back with the wrong number of parts the whole group is retried one
cue at a time, because a mangled batch would otherwise desynchronise every
subsequent cue against its timing.
"""

from __future__ import annotations

import argparse
import re
import sys

# Google's endpoint accepts ~5000 chars; stay well under so a multi-byte
# expansion during translation cannot push a request over the limit.
MAX_GROUP_CHARS = 3000
# A delimiter the translator will not touch, reword, or translate away.
SEP = "\n@@@\n"

CUE_RE = re.compile(
    r"(?P<idx>\d+)\s*\n(?P<time>[\d:,]+\s*-->\s*[\d:,]+)\s*\n(?P<text>.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


def parse_srt(text: str):
    return [(m.group("idx"), m.group("time"), m.group("text").strip())
            for m in CUE_RE.finditer(text)]


def render_srt(cues) -> str:
    return "".join(f"{i}\n{t}\n{x}\n\n" for i, t, x in cues)


def group_cues(texts, max_chars=MAX_GROUP_CHARS):
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


def translate_all(texts, target, source="auto"):
    from deep_translator import GoogleTranslator

    tr = GoogleTranslator(source=source, target=target)
    cache, out = {}, []

    def one(s):
        if s not in cache:
            try:
                cache[s] = tr.translate(s) or s
            except Exception as e:
                print(f"    [!] cue translation failed ({e}); keeping source", file=sys.stderr)
                cache[s] = s
        return cache[s]

    # Subtitles repeat a lot; only translate each distinct string once.
    for group in group_cues([t for t in texts if t]):
        if len(group) == 1:
            one(group[0])
            continue
        try:
            reply = tr.translate(SEP.join(group)) or ""
        except Exception as e:
            print(f"    [!] batch failed ({e}); falling back per cue", file=sys.stderr)
            reply = ""
        parts = [p.strip() for p in reply.split("@@@")] if reply else []
        if len(parts) == len(group):
            for src, dst in zip(group, parts):
                cache.setdefault(src, dst or src)
        else:
            # Wrong part count means the delimiter was mangled. Retrying the
            # group per cue is the only way to keep text aligned with timings.
            for src in group:
                one(src)

    for t in texts:
        out.append(cache.get(t, t) if t else t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", required=True)
    ap.add_argument("--langs", required=True, help="comma-separated target codes")
    ap.add_argument("--source", default="auto")
    args = ap.parse_args()

    with open(args.srt, encoding="utf-8") as f:
        cues = parse_srt(f.read())
    if not cues:
        print(f"[!] no cues parsed from {args.srt}", file=sys.stderr)
        return 1

    base = args.srt[:-4] if args.srt.lower().endswith(".srt") else args.srt
    texts = [c[2] for c in cues]

    for lang in [l.strip() for l in args.langs.split(",") if l.strip()]:
        translated = translate_all(texts, lang, args.source)
        out = [(i, t, new) for (i, t, _), new in zip(cues, translated)]
        dest = f"{base}.{lang}.srt"
        with open(dest, "w", encoding="utf-8") as f:
            f.write(render_srt(out))
        print(f"    wrote {dest} ({len(out)} cues)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
