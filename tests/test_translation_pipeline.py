#!/usr/bin/env python3
"""Tests for the Riva translation and SRT generation pipeline."""

import os
import sys
import unittest
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from riva_engine import (
    normalize_code,
    resolve_language_pair,
    build_riva_prompt,
    RIVA_LANGUAGE_PAIRS,
)
from translate_srt_riva import (
    parse_srt,
    render_srt,
    split_proportional,
    build_sentences,
)


class TestRivaEngine(unittest.TestCase):
    def test_normalize_code(self):
        self.assertEqual(normalize_code("spa"), "es")
        self.assertEqual(normalize_code("fre"), "fr")
        self.assertEqual(normalize_code("es_ES"), "es-es")
        self.assertEqual(normalize_code("EN"), "en")

    def test_resolve_language_pair_valid(self):
        tag, src, tgt = resolve_language_pair("en", "es")
        self.assertEqual(src, "English")
        self.assertEqual(tgt, "European Spanish")

        tag, src, tgt = resolve_language_pair("en", "fr")
        self.assertEqual(src, "English")
        self.assertEqual(tgt, "French")

        tag, src, tgt = resolve_language_pair("de", "en")
        self.assertEqual(src, "German")
        self.assertEqual(tgt, "English")

        tag, src, tgt = resolve_language_pair("en", "ja")
        self.assertEqual(src, "English")
        self.assertEqual(tgt, "Japanese")

    def test_resolve_language_pair_invalid(self):
        # Riva requires English on either side
        with self.assertRaises(ValueError):
            resolve_language_pair("fr", "de")

    def test_build_riva_prompt(self):
        prompt = build_riva_prompt("English", "French", "Hello world.")
        expected = (
            "System\n"
            "You are an expert at translating text from English to French.</s>\n"
            "<s>User\n"
            "What is the French translation of the sentence: Hello world.</s>\n"
            "<s>Assistant\n"
        )
        self.assertEqual(prompt, expected)


class TestSrtProcessing(unittest.TestCase):
    def setUp(self):
        self.sample_srt = (
            "1\n"
            "00:00:01,000 --> 00:00:03,500\n"
            "Welcome to the speech translation\n\n"
            "2\n"
            "00:00:03,600 --> 00:00:05,800\n"
            "pipeline demo.\n\n"
            "3\n"
            "00:00:06,000 --> 00:00:08,000\n"
            "It runs entirely offline.\n\n"
        )

    def test_parse_and_render_srt(self):
        cues = parse_srt(self.sample_srt)
        self.assertEqual(len(cues), 3)
        self.assertEqual(cues[0][0], "1")
        self.assertEqual(cues[0][1], "00:00:01,000 --> 00:00:03,500")
        self.assertEqual(cues[0][2], "Welcome to the speech translation")

        rendered = render_srt(cues)
        cues_reparsed = parse_srt(rendered)
        self.assertEqual(cues, cues_reparsed)

    def test_build_sentences(self):
        cues = parse_srt(self.sample_srt)
        sentences, spans = build_sentences(cues)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0], "Welcome to the speech translation pipeline demo.")
        self.assertEqual(sentences[1], "It runs entirely offline.")
        # Check that sentence 0 spans cues 0 and 1
        cue_indices = [ci for ci, _ in spans[0]]
        self.assertEqual(cue_indices, [0, 1])
        # Sentence 1 spans cue 2
        cue_indices_1 = [ci for ci, _ in spans[1]]
        self.assertEqual(cue_indices_1, [2])

    def test_split_proportional(self):
        text = "Bienvenue dans la démonstration du pipeline de traduction vocale."
        weights = [33, 15]  # roughly 2:1 ratio
        parts = split_proportional(text, weights)
        self.assertEqual(len(parts), 2)
        # Verify joined words equal original text
        self.assertEqual(" ".join(parts), text)
        self.assertTrue(len(parts[0]) > len(parts[1]))


from words_to_srt import merge_tokens, build, readable


class TestTokenMerging(unittest.TestCase):
    def test_nemotron_subword_merging(self):
        # Simulates raw tokens from Nemotron ASR
        raw_tokens = [
            {"word": " Wel", "start_sample": 0, "end_sample": 1600},
            {"word": "come", "start_sample": 1600, "end_sample": 3200},
            {"word": " ", "start_sample": 3200, "end_sample": 3300},
            {"word": "to", "start_sample": 3400, "end_sample": 4800},
            {"word": " the", "start_sample": 5000, "end_sample": 6400},
            {"word": " show", "start_sample": 6500, "end_sample": 8000},
            {"word": ".", "start_sample": 8000, "end_sample": 8100},
        ]
        merged = merge_tokens(raw_tokens)
        words = [w["word"] for w in merged]
        self.assertEqual(words, ["Welcome", "to", "the", "show."])
        self.assertEqual(merged[0]["start_sample"], 0)
        self.assertEqual(merged[0]["end_sample"], 3200)

    def test_sentencepiece_underscore_tokens(self):
        # Simulates SentencePiece \u2581 tokens
        raw_tokens = [
            {"word": "\u2581Hel", "start_sample": 0, "end_sample": 1600},
            {"word": "lo", "start_sample": 1600, "end_sample": 3200},
            {"word": "\u2581world", "start_sample": 4000, "end_sample": 6000},
            {"word": "!", "start_sample": 6000, "end_sample": 6100},
        ]
        merged = merge_tokens(raw_tokens)
        words = [w["word"] for w in merged]
        self.assertEqual(words, ["Hello", "world!"])


if __name__ == "__main__":
    unittest.main()
