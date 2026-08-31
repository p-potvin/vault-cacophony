#!/usr/bin/env python3
"""Calculate accuracy, coverage, and audio statistics for an Open ASR JSONL manifest."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-utterances", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_utterances < 1:
        raise SystemExit("--expected-utterances must be positive")
    try:
        import evaluate
        import jiwer
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError as error:
        raise SystemExit("Install requirements-openasr-eval.txt before calculating metrics.") from error

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"No rows in {args.input}")
    normalizer = EnglishTextNormalizer()
    raw_references = [row["reference"] for row in rows]
    raw_predictions = [row["prediction"] for row in rows]
    references = [normalizer(row["reference"]) for row in rows]
    predictions = [normalizer(row["prediction"]) for row in rows]
    errors = jiwer.process_words(references, predictions)
    durations = [float(row["audio_length_s"]) for row in rows if row.get("audio_length_s") is not None]
    unique_ids = {row.get("id") for row in rows}
    normalized_wer = evaluate.load("wer").compute(predictions=predictions, references=references)
    reference_words = errors.hits + errors.substitutions + errors.deletions

    report = {
        "scope": {
            "expected_utterances": args.expected_utterances,
            "completed_utterances": len(rows),
            "coverage_percent": len(rows) / args.expected_utterances * 100,
            "complete": len(rows) == args.expected_utterances,
            "last_completed_id": rows[-1].get("id"),
        },
        "methodology": {
            "normalizer": "whisper_normalizer.english.EnglishTextNormalizer",
            "normalized_wer": "evaluate.load('wer')",
            "raw_wer_and_error_counts": "jiwer",
        },
        "accuracy": {
            "normalized_wer": normalized_wer,
            "normalized_wer_percent": normalized_wer * 100,
            "raw_wer": jiwer.wer(raw_references, raw_predictions),
            "raw_wer_percent": jiwer.wer(raw_references, raw_predictions) * 100,
            "cer": jiwer.cer(raw_references, raw_predictions),
            "cer_percent": jiwer.cer(raw_references, raw_predictions) * 100,
            "exact_match_percent": sum(ref == pred for ref, pred in zip(references, predictions)) / len(rows) * 100,
            "hits": errors.hits,
            "substitutions": errors.substitutions,
            "deletions": errors.deletions,
            "insertions": errors.insertions,
            "reference_words": reference_words,
        },
        "audio": {
            "seconds": sum(durations),
            "hours": sum(durations) / 3600,
            "mean_seconds": statistics.mean(durations),
            "median_seconds": statistics.median(durations),
            "min_seconds": min(durations),
            "max_seconds": max(durations),
        },
        "integrity": {
            "unique_ids": len(unique_ids),
            "duplicate_ids": len(rows) - len(unique_ids),
        },
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
