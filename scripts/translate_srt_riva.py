#!/usr/bin/env python3
"""Translate an .srt subtitle file using local Riva-Translate-4B GGUF.

Reads <name>.srt and generates <name>.<lang>.srt for one or more target languages.
Timings and cue synchronisation are preserved exactly.

Key Features:
1. Sentence-level Context: Cues are merged into bounded grammatical sentences using
   pause gaps and optional Silero TE punctuation restoration for high translation accuracy,
   then redistributed proportionally across original cues.
2. 100% Offline: Runs locally on CUDA with zero web requests or API rate limits.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from riva_engine import RivaEngine, DEFAULT_MODEL_PATH, clean_console_text

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Split after . ! ? or … when followed by space/end of text
SENT_END = re.compile(r"(?<=[.!?…])\s+")
CUE_RE = re.compile(
    r"(?P<idx>\d+)\s*\n(?P<time>[\d:,]+\s*-->\s*[\d:,]+)\s*\n(?P<text>.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


def parse_timestamp(ts: str) -> float:
    """Convert SRT timestamp string (00:01:23,456) to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(ts)


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

    # If fewer words than cues, distribute words across cues without leaving cues empty if possible
    if len(words) <= n:
        out = []
        for idx in range(n):
            if idx < len(words):
                out.append(words[idx])
            else:
                out.append("")
        return out

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


_SILERO_TE_CACHE = None


def get_silero_te():
    """Lazily load Silero TE (Text Enhancer) punctuation restoration model."""
    global _SILERO_TE_CACHE
    if _SILERO_TE_CACHE is False:
        return None
    if _SILERO_TE_CACHE is not None:
        return _SILERO_TE_CACHE
    try:
        import torch
        model, _, languages, _, apply_te = torch.hub.load(
            "snakers4/silero-models", "silero_te", trust_repo=True, verbose=False
        )
        _SILERO_TE_CACHE = (model, languages, apply_te)
        return _SILERO_TE_CACHE
    except Exception:
        _SILERO_TE_CACHE = False
        return None


def restore_punctuation(text: str, lang: str = "en") -> str:
    """Apply Silero punctuation and capitalization if supported."""
    te = get_silero_te()
    if not te or not text.strip():
        return text
    _, languages, apply_te = te
    lang_code = lang.lower().split("-")[0]
    if lang_code in languages:
        try:
            return apply_te(text, lan=lang_code)
        except Exception:
            return text
    return text


def build_sentences(
    cues: List[Tuple[str, str, str]],
    source_lang: str = "en",
    max_cues_per_chunk: int = 3,
    max_chars_per_chunk: int = 140,
    gap_threshold: float = 0.5,
    use_punct: bool = True,
) -> Tuple[List[str], List[List[Tuple[int, int]]]]:
    """Build bounded sentences from cues with pause-gap & punctuation segmentation."""
    if not cues:
        return [], []

    # Parse timestamps for pause-gap detection
    cue_times = []
    for _, time_str, text in cues:
        times = time_str.split("-->")
        start_sec = parse_timestamp(times[0]) if len(times) >= 1 else 0.0
        end_sec = parse_timestamp(times[1]) if len(times) >= 2 else 0.0
        cue_times.append((start_sec, end_sec))

    # Segment cues into chunks based on pause gaps and max cues/chars
    chunks = []
    curr_chunk = [0]
    curr_chars = len(cues[0][2])

    for i in range(1, len(cues)):
        prev_end = cue_times[i - 1][1]
        curr_start = cue_times[i][0]
        gap = curr_start - prev_end
        text_len = len(cues[i][2])

        # Break chunk if pause gap >= threshold, or max cues reached, or character limit reached
        if (
            gap >= gap_threshold
            or len(curr_chunk) >= max_cues_per_chunk
            or (curr_chars + text_len) > max_chars_per_chunk
        ):
            chunks.append(curr_chunk)
            curr_chunk = [i]
            curr_chars = text_len
        else:
            curr_chunk.append(i)
            curr_chars += text_len
    if curr_chunk:
        chunks.append(curr_chunk)

    sentences = []
    spans = []

    for chunk_indices in chunks:
        chunk_cues = [cues[ci] for ci in chunk_indices]
        joined = " ".join(c[2] for c in chunk_cues)

        if use_punct:
            joined = restore_punctuation(joined, lang=source_lang)

        # Map character weights to each cue
        weights = [max(1, len(cues[ci][2])) for ci in chunk_indices]
        span_entries = [(ci, w) for ci, w in zip(chunk_indices, weights)]

        sentences.append(joined)
        spans.append(span_entries)

    return sentences, spans


def translate_cues(
    cues: List[Tuple[str, str, str]],
    engine: RivaEngine,
    target_lang: str,
    source_lang: str = "en",
    per_cue: bool = False,
    use_punct: bool = True,
) -> List[Tuple[str, str, str]]:
    """Translate cues to the target language preserving original cue boundaries."""
    if not cues:
        return []

    if per_cue:
        texts = [c[2] for c in cues]
        translated = engine.translate_batch(texts, target_lang=target_lang, source_lang=source_lang)
        return [(c[0], c[1], trans) for c, trans in zip(cues, translated)]

    sentences, spans = build_sentences(cues, source_lang=source_lang, use_punct=use_punct)
    if not sentences:
        return cues

    translated_sents = engine.translate_batch(sentences, target_lang=target_lang, source_lang=source_lang)

    parts = {i: [] for i in range(len(cues))}
    for sent, span in zip(translated_sents, spans):
        weights = [w for _, w in span]
        for (ci, _), piece in zip(span, split_proportional(sent, weights)):
            if piece:
                parts[ci].append(piece)

    out_cues = []
    for i, c in enumerate(cues):
        cue_text = " ".join(parts[i]).strip()
        # Fallback to direct translation if proportional split resulted in empty text
        if not cue_text:
            cue_text = engine.translate(c[2], target_lang=target_lang, source_lang=source_lang)
        out_cues.append((c[0], c[1], cue_text))

    return out_cues


def main():
    parser = argparse.ArgumentParser(description="Translate .srt files with Riva-Translate-4B GGUF.")
    parser.add_argument("--srt", required=True, help="Input .srt file path")
    parser.add_argument("--langs", required=True, help="Comma-separated target language codes (e.g. es,fr,de)")
    parser.add_argument("--source", default="en", help="Source language code (default: en)")
    parser.add_argument("--model", help="Path to Riva GGUF model")
    parser.add_argument("--per-cue", action="store_true", help="Translate cue-by-cue instead of chunked sentences")
    parser.add_argument("--no-punct", action="store_true", help="Disable Silero TE punctuation restoration")
    parser.add_argument("--out-dir", help="Optional output directory")
    parser.add_argument("--overwrite", "-w", action="store_true", help="Overwrite existing translated files")
    parser.add_argument("--skip-completed", "--skip-existing", "--skip-if-translated", action="store_true", help="Skip translation if output already exists")
    args = parser.parse_args()

    srt_path = Path(args.srt)
    if not srt_path.exists():
        print(f"[!] Error: SRT file not found: {srt_path}", file=sys.stderr)
        sys.exit(1)

    with open(srt_path, encoding="utf-8", errors="replace") as f:
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
            print(clean_console_text(f"[*] Subtitles already exist for '{lang}': {dest_file.name} (skipped)"))
        else:
            needed_langs.append(lang)

    if not needed_langs:
        print(clean_console_text(f"[*] All requested translations already exist for {srt_path.name}"))
        return

    model_path = args.model or DEFAULT_MODEL_PATH
    engine = RivaEngine(model_path=model_path)

    for lang in needed_langs:
        print(clean_console_text(f"[*] Translating to '{lang}' ({len(cues)} cues)..."))
        translated_cues = translate_cues(
            cues,
            engine=engine,
            target_lang=lang,
            source_lang=args.source,
            per_cue=args.per_cue,
            use_punct=not args.no_punct,
        )
        dest_file = out_dir / f"{base_stem}.{lang}.srt"
        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(render_srt(translated_cues))
        print(clean_console_text(f"    -> Wrote {dest_file}"))


if __name__ == "__main__":
    main()
