#!/usr/bin/env python3
"""Translate an .srt subtitle file using local Riva-Translate-4B GGUF.

Reads <name>.srt and generates <name>.<lang>.srt for one or more target languages.
Timings and cue synchronisation are preserved exactly.

Key Features:
1. Sentence-level Context: Cues are merged into full grammatical sentences for
   high translation accuracy, then redistributed proportionally across original cues.
2. 100% Offline: Runs locally with zero web requests or API rate limits.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Split after . ! ? or … when followed by space/end of text
SENT_END = re.compile(r"(?<=[.!?…])\s+")
CUE_RE = re.compile(
    r"(?P<idx>\d+)\s*\n(?P<time>[\d:,]+\s*-->\s*[\d:,]+)\s*\n(?P<text>.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


def parse_srt(text: str) -> List[Tuple[str, str, str]]:
    """Parse SRT content into a list of (index, timestamp, text) tuples."""
    return [
        (m.group("idx"), m.group("time"), " ".join(m.group("text").split()))
        for m in CUE_RE.finditer(text)
    ]


def render_srt(cues: List[Tuple[str, str, str]]) -> str:
    """Render list of (index, timestamp, text) tuples into SRT format."""
    return "".join(f"{i}\n{t}\n{x}\n\n" for i, t, x in cues)


def split_proportional(text: str, weights: List[int]) -> List[str]:
    """Split translated text into len(weights) segments on word boundaries."""
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


def build_sentences(cues: List[Tuple[str, str, str]]) -> Tuple[List[str], List[List[Tuple[int, int]]]]:
    """Build sentences from cues while tracking cue span coverage."""
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
            for cs, ce, ci in marks:
                lo, hi = max(cs, start), min(ce, end)
                if hi > lo:
                    overlaps.append((ci, hi - lo))
            if overlaps:
                sentences.append(s)
                spans.append(overlaps)
        start = m.end() if m else end
    return sentences, spans


def translate_cues(
    cues: List[Tuple[str, str, str]],
    engine,
    target_lang: str,
    source_lang: str = "en",
    per_cue: bool = False,
) -> List[Tuple[str, str, str]]:
    """Translate cues to the target language."""
    if per_cue:
        texts = [c[2] for c in cues]
        translated = engine.translate_batch(texts, target_lang=target_lang, source_lang=source_lang)
        return [(c[0], c[1], trans) for c, trans in zip(cues, translated)]

    sentences, spans = build_sentences(cues)
    if not sentences:
        return cues

    translated_sents = engine.translate_batch(sentences, target_lang=target_lang, source_lang=source_lang)

    parts = {i: [] for i in range(len(cues))}
    for sent, span in zip(translated_sents, spans):
        weights = [w for _, w in span]
        for (ci, _), piece in zip(span, split_proportional(sent, weights)):
            if piece:
                parts[ci].append(piece)

    return [
        (c[0], c[1], " ".join(parts[i]).strip() or c[2])
        for i, c in enumerate(cues)
    ]


def main():
    parser = argparse.ArgumentParser(description="Translate .srt files with Riva-Translate-4B GGUF.")
    parser.add_argument("--srt", required=True, help="Input .srt file path")
    parser.add_argument("--langs", required=True, help="Comma-separated target language codes (e.g. es,fr,de)")
    parser.add_argument("--source", default="en", help="Source language code (default: en)")
    parser.add_argument("--model", help="Path to Riva GGUF model")
    parser.add_argument("--per-cue", action="store_true", help="Translate cue-by-cue instead of sentences")
    parser.add_argument("--out-dir", help="Optional output directory")
    parser.add_argument("--overwrite", "-w", action="store_true", help="Overwrite existing translated files")
    parser.add_argument("--skip-completed", "--skip-existing", "--skip-if-translated", action="store_true", help="Skip translation if output already exists")
    args = parser.parse_args()

    srt_path = Path(args.srt)
    if not srt_path.exists():
        print(f"[!] Error: SRT file not found: {srt_path}", file=sys.stderr)
        sys.exit(1)

    with open(srt_path, encoding="utf-8") as f:
        cues = parse_srt(f.read())

    if not cues:
        print(f"[!] Error: No cues parsed from {srt_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else srt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_stem = srt_path.stem

    target_langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    needed_langs = []
    for lang in target_langs:
        dest_file = out_dir / f"{base_stem}.{lang}.srt"
        if dest_file.exists() and not args.overwrite and args.skip_completed:
            print(f"[*] Subtitles already exist for '{lang}': {dest_file} (skipped)")
        else:
            needed_langs.append(lang)

    if not needed_langs:
        print(f"[*] All requested translations already exist for {srt_path.name}")
        return

    from riva_engine import RivaEngine, DEFAULT_MODEL_PATH

    model_path = args.model or DEFAULT_MODEL_PATH
    engine = RivaEngine(model_path=model_path)

    for lang in needed_langs:
        print(f"[*] Translating to '{lang}' ({len(cues)} cues)...")
        translated_cues = translate_cues(
            cues,
            engine=engine,
            target_lang=lang,
            source_lang=args.source,
            per_cue=args.per_cue,
        )
        dest_file = out_dir / f"{base_stem}.{lang}.srt"
        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(render_srt(translated_cues))
        print(f"    -> Wrote {dest_file}")


if __name__ == "__main__":
    main()
