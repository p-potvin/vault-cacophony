#!/usr/bin/env python3
"""Riva-Translate-4B translation engine.

Wraps the quantized Riva-Translate-4B-Instruct GGUF model for fast, local,
offline neural machine translation. Implements the official NVIDIA prompt
specification and language pair mappings across 50+ language pairs.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Official Riva Language Pair Mappings (Source -> Target Language Display Names)
RIVA_LANGUAGE_PAIRS: Dict[str, Tuple[str, str]] = {
    # English to World Languages
    "en-zh-cn": ("English", "Simplified Chinese"),
    "en-zh": ("English", "Simplified Chinese"),
    "en-zh-tw": ("English", "Traditional Chinese"),
    "en-ar": ("English", "Arabic"),
    "en-de": ("English", "German"),
    "en-es": ("English", "European Spanish"),
    "en-es-es": ("English", "European Spanish"),
    "en-es-us": ("English", "Latin American Spanish"),
    "en-fr": ("English", "French"),
    "en-ja": ("English", "Japanese"),
    "en-ko": ("English", "Korean"),
    "en-ru": ("English", "Russian"),
    "en-pt": ("English", "Brazilian Portuguese"),
    "en-pt-br": ("English", "Brazilian Portuguese"),
    "en-pt-pt": ("English", "European Portuguese"),
    "en-it": ("English", "Italian"),
    "en-nl": ("English", "Dutch"),
    "en-pl": ("English", "Polish"),
    "en-cs": ("English", "Czech"),
    "en-sv": ("English", "Swedish"),
    "en-da": ("English", "Danish"),
    "en-fi": ("English", "Finnish"),
    "en-no": ("English", "Norwegian"),
    "en-hu": ("English", "Hungarian"),
    "en-ro": ("English", "Romanian"),
    "en-bg": ("English", "Bulgarian"),
    "en-uk": ("English", "Ukrainian"),
    "en-sk": ("English", "Slovak"),
    "en-hr": ("English", "Croatian"),
    "en-sl": ("English", "Slovenian"),
    "en-et": ("English", "Estonian"),
    "en-lv": ("English", "Latvian"),
    "en-lt": ("English", "Lithuanian"),
    "en-el": ("English", "Greek"),
    "en-tr": ("English", "Turkish"),
    "en-id": ("English", "Indonesian"),
    "en-vi": ("English", "Vietnamese"),
    "en-th": ("English", "Thai"),
    "en-hi": ("English", "Hindi"),

    # World Languages to English
    "zh-en": ("Simplified Chinese", "English"),
    "zh-cn-en": ("Simplified Chinese", "English"),
    "zh-tw-en": ("Traditional Chinese", "English"),
    "ar-en": ("Arabic", "English"),
    "de-en": ("German", "English"),
    "es-en": ("European Spanish", "English"),
    "es-es-en": ("European Spanish", "English"),
    "es-us-en": ("Latin American Spanish", "English"),
    "fr-en": ("French", "English"),
    "ja-en": ("Japanese", "English"),
    "ko-en": ("Korean", "English"),
    "ru-en": ("Russian", "English"),
    "pt-en": ("Brazilian Portuguese", "English"),
    "pt-br-en": ("Brazilian Portuguese", "English"),
    "it-en": ("Italian", "English"),
    "nl-en": ("Dutch", "English"),
    "pl-en": ("Polish", "English"),
    "cs-en": ("Czech", "English"),
    "sv-en": ("Swedish", "English"),
    "da-en": ("Danish", "English"),
    "fi-en": ("Finnish", "English"),
    "no-en": ("Norwegian", "English"),
    "hu-en": ("Hungarian", "English"),
    "ro-en": ("Romanian", "English"),
    "bg-en": ("Bulgarian", "English"),
    "uk-en": ("Ukrainian", "English"),
    "sk-en": ("Slovak", "English"),
    "hr-en": ("Croatian", "English"),
    "sl-en": ("Slovenian", "English"),
    "et-en": ("Estonian", "English"),
    "lv-en": ("Latvian", "English"),
    "lt-en": ("Lithuanian", "English"),
    "el-en": ("Greek", "English"),
    "tr-en": ("Turkish", "English"),
    "id-en": ("Indonesian", "English"),
    "vi-en": ("Vietnamese", "English"),
    "th-en": ("Thai", "English"),
    "hi-en": ("Hindi", "English"),
}

DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent
    / "audio.cpp"
    / "models"
    / "Riva-Translate-4B-Instruct.i1-Q4_K_M.gguf"
)


def normalize_code(code: str) -> str:
    """Normalize language tags (e.g., 'es_ES' -> 'es-es', 'spa' -> 'es')."""
    c = code.strip().lower().replace("_", "-")
    alias_map = {
        "spa": "es",
        "fra": "fr",
        "fre": "fr",
        "deu": "de",
        "ger": "de",
        "zho": "zh",
        "chi": "zh",
        "jpn": "ja",
        "kor": "ko",
        "rus": "ru",
        "por": "pt",
        "ita": "it",
        "nld": "nl",
        "dut": "nl",
        "pol": "pl",
        "ces": "cs",
        "cze": "cs",
        "swe": "sv",
        "dan": "da",
        "fin": "fi",
        "nor": "no",
        "hun": "hu",
        "ron": "ro",
        "rum": "ro",
        "bul": "bg",
        "ukr": "uk",
        "slk": "sk",
        "slo": "sk",
        "hrv": "hr",
        "slv": "sl",
        "est": "et",
        "lav": "lv",
        "lit": "lt",
        "ell": "el",
        "gre": "el",
        "tur": "tr",
        "ind": "id",
        "vie": "vi",
        "tha": "th",
        "hin": "hi",
        "eng": "en",
    }
    return alias_map.get(c, c)


def resolve_language_pair(source: str, target: str) -> Tuple[str, str, str]:
    """Resolve source and target language to (tag, source_name, target_name)."""
    s = normalize_code(source)
    t = normalize_code(target)

    # If already a combined tag e.g. 'en-es'
    if "-" in s and not t:
        tag = s
    elif "-" in t and not s:
        tag = t
    else:
        tag = f"{s}-{t}"

    if tag in RIVA_LANGUAGE_PAIRS:
        src_name, tgt_name = RIVA_LANGUAGE_PAIRS[tag]
        return tag, src_name, tgt_name

    # Check base code (e.g. 'es-es' -> 'es')
    s_base = s.split("-")[0]
    t_base = t.split("-")[0]
    base_tag = f"{s_base}-{t_base}"
    if base_tag in RIVA_LANGUAGE_PAIRS:
        src_name, tgt_name = RIVA_LANGUAGE_PAIRS[base_tag]
        return base_tag, src_name, tgt_name

    # Check reverse or default English pair
    if s_base != "en" and t_base != "en":
        raise ValueError(
            f"Unsupported language pair: '{source}' -> '{target}'. "
            "Riva-Translate-4B requires English on either the source or target side."
        )
    raise ValueError(f"Language pair not found in Riva registry: '{source}' -> '{target}'")


def build_riva_prompt(source_lang_name: str, target_lang_name: str, text: str) -> str:
    """Build the official NVIDIA Riva translation prompt."""
    return (
        f"System\n"
        f"You are an expert at translating text from {source_lang_name} to {target_lang_name}.</s>\n"
        f"<s>User\n"
        f"What is the {target_lang_name} translation of the sentence: {text.strip()}</s>\n"
        f"<s>Assistant\n"
    )


class RivaEngine:
    """Inference engine for Riva-Translate-4B GGUF."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Riva GGUF model not found at: {self.model_path}")

        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._llm = None
        self._cache: Dict[Tuple[str, str, str], str] = {}

    def _init_llm(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
            )
        except ImportError:
            raise RuntimeError(
                "llama_cpp package is required. Install via: uv pip install llama-cpp-python"
            )

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "en",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Translate a single string."""
        if not text.strip():
            return text

        tag, src_name, tgt_name = resolve_language_pair(source_lang, target_lang)
        cache_key = (src_name, tgt_name, text.strip())
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._init_llm()
        prompt = build_riva_prompt(src_name, tgt_name, text)

        res = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "<|endoftext|>", "\n<s>"],
            echo=False,
        )

        translated = res["choices"][0]["text"].strip()
        # Clean up possible formatting artifacts
        if translated.startswith('"') and translated.endswith('"') and len(translated) > 1:
            translated = translated[1:-1].strip()

        self._cache[cache_key] = translated
        return translated

    def translate_batch(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: str = "en",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> List[str]:
        """Translate a list of strings preserving original ordering."""
        results = []
        for t in texts:
            if not t.strip():
                results.append(t)
            else:
                results.append(
                    self.translate(
                        t,
                        target_lang=target_lang,
                        source_lang=source_lang,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                )
        return results


def main():
    parser = argparse.ArgumentParser(description="Translate text using Riva-Translate-4B GGUF.")
    parser.add_argument("text", nargs="?", help="Text to translate (or pipe via stdin)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to Riva GGUF model")
    parser.add_argument("--from-lang", "-s", default="en", help="Source language (default: en)")
    parser.add_argument("--to-lang", "-t", required=True, help="Target language (e.g. es, fr, de)")
    parser.add_argument("--ctx", type=int, default=4096, help="Context length")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers to offload (-1=all)")
    args = parser.parse_args()

    input_text = args.text
    if not input_text:
        if not sys.stdin.isatty():
            input_text = sys.stdin.read()
        else:
            parser.print_help()
            sys.exit(1)

    engine = RivaEngine(
        model_path=args.model,
        n_ctx=args.ctx,
        n_gpu_layers=args.gpu_layers,
    )

    result = engine.translate(
        text=input_text,
        target_lang=args.to_lang,
        source_lang=args.from_lang,
    )
    print(result)


if __name__ == "__main__":
    main()
