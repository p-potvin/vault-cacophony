#!/usr/bin/env python3
"""Translate plain text, markdown, or text-based documents with Riva-Translate-4B GGUF.

Reads <file>.<ext> and generates <file>.<lang>.<ext> for one or more target languages.
Handles paragraph grouping, line-preserving formats, and local offline inference.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List

from riva_engine import RivaEngine, DEFAULT_MODEL_PATH

SENT_SPLIT = re.compile(r"(?<=[.!?…\n])\s+")


def translate_text_content(text: str, engine: RivaEngine, target_lang: str, source_lang: str = "en") -> str:
    """Translate full text while preserving paragraph structure."""
    paragraphs = text.split("\n\n")
    translated_paras = []

    for para in paragraphs:
        if not para.strip():
            translated_paras.append(para)
            continue

        # If short paragraph, translate directly
        if len(para) < 500:
            tr = engine.translate(para.strip(), target_lang=target_lang, source_lang=source_lang)
            translated_paras.append(tr)
        else:
            # Split into sentences for longer paragraphs
            sentences = [s.strip() for s in SENT_SPLIT.split(para) if s.strip()]
            tr_sents = engine.translate_batch(sentences, target_lang=target_lang, source_lang=source_lang)
            translated_paras.append(" ".join(tr_sents))

    return "\n\n".join(translated_paras)


def main():
    parser = argparse.ArgumentParser(description="Translate text files with Riva-Translate-4B GGUF.")
    parser.add_argument("--input", "-i", required=True, help="Input text file path")
    parser.add_argument("--langs", "-t", required=True, help="Comma-separated target languages (e.g. es,fr,de)")
    parser.add_argument("--source", "-s", default="en", help="Source language (default: en)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to Riva GGUF model")
    parser.add_argument("--out-dir", "-o", help="Optional output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        content = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = input_path.read_text(encoding="latin-1", errors="replace")

    engine = RivaEngine(model_path=args.model)
    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    ext = input_path.suffix

    target_langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    for lang in target_langs:
        print(f"[*] Translating {input_path.name} to '{lang}'...")
        translated = translate_text_content(content, engine, target_lang=lang, source_lang=args.source)
        dest = out_dir / f"{stem}.{lang}{ext}"
        dest.write_text(translated, encoding="utf-8")
        print(f"    -> Wrote {dest}")


if __name__ == "__main__":
    main()
