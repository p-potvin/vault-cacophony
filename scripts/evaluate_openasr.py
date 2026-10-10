#!/usr/bin/env python3
"""Score prepared English ASR transcripts with Open ASR-style normalized WER.

The scorer deliberately does not run inference or download benchmark data.  It
compares an input JSONL manifest containing model predictions to its references,
using the same two operations used by the Open ASR leaderboard's English WER
path: Whisper's EnglishTextNormalizer followed by the `evaluate` WER metric.

Non-English and code-switched material must select `--normalizer basic`.  The
English normalizer applies English-only number, contraction and filler rules, so
on mixed-language text it rewrites the English half of a sentence and leaves the
other half raw, which biases any comparison between differently pinned
languages.  Neither normalizer is strictly correct for code-switched references;
whichever is chosen must be applied to every condition being compared, and the
report records which one ran.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol


class TextNormalizer(Protocol):
    def __call__(self, text: str) -> str: ...


class WerMetric(Protocol):
    def compute(self, *, predictions: list[str], references: list[str]) -> float: ...


REQUIRED_FIELDS = ("reference", "prediction")

NORMALIZER_PATHS = {
    "english": "whisper_normalizer.english.EnglishTextNormalizer",
    "basic": "whisper_normalizer.basic.BasicTextNormalizer",
}
DEFAULT_NORMALIZER = "english"

APOSTROPHES = re.compile("['’ʼ՚]")


class ApostropheStripper:
    """Drop apostrophes before delegating, keeping contractions one token.

    The basic normalizer splits `what's` into `what s`, which charges two token
    errors for one word on English text.  Removing the apostrophe first yields
    `whats` on both sides of the comparison instead.
    """

    def __init__(self, inner: TextNormalizer) -> None:
        self._inner = inner

    def __call__(self, text: str) -> str:
        return self._inner(APOSTROPHES.sub("", text))


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL manifest and reject incomplete or non-string rows."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
            for field in REQUIRED_FIELDS:
                if not isinstance(record.get(field), str):
                    raise ValueError(f"{path}:{line_number}: {field!r} must be a string")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no evaluation rows found")
    return records


def normalized_pairs(
    records: Iterable[dict[str, Any]], normalizer: TextNormalizer
) -> list[tuple[dict[str, Any], str, str]]:
    """Normalize references and predictions before WER, without mutating input."""
    pairs: list[tuple[dict[str, Any], str, str]] = []
    for position, record in enumerate(records, start=1):
        for field in REQUIRED_FIELDS:
            if not isinstance(record.get(field), str):
                raise ValueError(f"record {position}: {field!r} must be a string")
        pairs.append((record, normalizer(record["reference"]), normalizer(record["prediction"])))
    return pairs


def score_records(
    records: Iterable[dict[str, Any]],
    normalizer: TextNormalizer,
    metric: WerMetric,
    *,
    normalizer_path: str = NORMALIZER_PATHS[DEFAULT_NORMALIZER],
    strip_apostrophes: bool = False,
) -> dict[str, Any]:
    """Return micro-WER overall and independently for each optional dataset label."""
    pairs = normalized_pairs(records, normalizer)
    references = [reference for _, reference, _ in pairs]
    predictions = [prediction for _, _, prediction in pairs]
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record, reference, prediction in pairs:
        dataset = record.get("dataset")
        if isinstance(dataset, str) and dataset.strip():
            groups[dataset].append((reference, prediction))

    overall = metric.compute(predictions=predictions, references=references)
    result: dict[str, Any] = {
        "methodology": {
            "normalizer": normalizer_path,
            "strip_apostrophes": strip_apostrophes,
            "metric": "evaluate.load('wer')",
            "aggregation": "micro WER over all normalized utterances",
        },
        "utterances": len(pairs),
        "wer": overall,
        "wer_percent": overall * 100,
        "by_dataset": {},
    }
    for dataset, values in sorted(groups.items()):
        refs, preds = zip(*values)
        value = metric.compute(predictions=list(preds), references=list(refs))
        result["by_dataset"][dataset] = {
            "utterances": len(values),
            "wer": value,
            "wer_percent": value * 100,
        }
    return result


def load_openasr_components(
    normalizer_name: str = DEFAULT_NORMALIZER,
    strip_apostrophes: bool = False,
) -> tuple[TextNormalizer, WerMetric, str]:
    """Import optional evaluation dependencies only when the scorer is executed."""
    try:
        import evaluate

        if normalizer_name == "basic":
            from whisper_normalizer.basic import BasicTextNormalizer as Normalizer
        else:
            from whisper_normalizer.english import EnglishTextNormalizer as Normalizer
    except ImportError as error:
        raise SystemExit(
            "Missing evaluation dependencies. Install requirements-openasr-eval.txt "
            "in an isolated virtual environment before scoring."
        ) from error
    normalizer: TextNormalizer = Normalizer()
    if strip_apostrophes:
        normalizer = ApostropheStripper(normalizer)
    return normalizer, evaluate.load("wer"), NORMALIZER_PATHS[normalizer_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL rows with reference and prediction")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; stdout is always written")
    parser.add_argument(
        "--normalizer",
        choices=sorted(NORMALIZER_PATHS),
        default=DEFAULT_NORMALIZER,
        help="Text normalizer to apply before WER; use 'basic' for non-English or code-switched references",
    )
    parser.add_argument(
        "--strip-apostrophes",
        action="store_true",
        help="Remove apostrophes before normalizing so contractions stay one token",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_records(args.input)
    normalizer, metric, normalizer_path = load_openasr_components(
        args.normalizer, args.strip_apostrophes
    )
    result = score_records(
        records,
        normalizer,
        metric,
        normalizer_path=normalizer_path,
        strip_apostrophes=args.strip_apostrophes,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
