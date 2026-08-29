from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_openasr.py"
SPEC = importlib.util.spec_from_file_location("evaluate_openasr", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LowercaseNormalizer:
    def __call__(self, text: str) -> str:
        return text.lower().replace(".", "")


class ExactMetric:
    def compute(self, *, predictions: list[str], references: list[str]) -> float:
        return sum(prediction != reference for prediction, reference in zip(predictions, references)) / len(references)


class OpenAsrEvaluationTests(unittest.TestCase):
    def test_normalizes_before_overall_and_dataset_scoring(self) -> None:
        report = MODULE.score_records(
            [
                {"dataset": "clean", "reference": "Hello World.", "prediction": "hello world"},
                {"dataset": "other", "reference": "One.", "prediction": "two"},
            ],
            LowercaseNormalizer(),
            ExactMetric(),
        )

        self.assertEqual(report["utterances"], 2)
        self.assertEqual(report["wer"], 0.5)
        self.assertEqual(report["wer_percent"], 50.0)
        self.assertEqual(report["by_dataset"]["clean"]["wer"], 0.0)
        self.assertEqual(report["by_dataset"]["other"]["wer"], 1.0)

    def test_rejects_missing_required_fields(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.normalized_pairs([{"reference": "present"}], LowercaseNormalizer())


if __name__ == "__main__":
    unittest.main()
