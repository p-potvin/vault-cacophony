# Open ASR-compatible English WER baseline

This scorer evaluates already-produced English transcripts. It does not invoke
an ASR model, download a benchmark dataset, send requests, or submit anything
to the leaderboard.

It follows the English scoring pattern used by the Open ASR leaderboard:

```python
import evaluate
from whisper_normalizer.english import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()
wer = evaluate.load("wer")
score = wer.compute(
    predictions=[normalizer(prediction)],
    references=[normalizer(reference)],
)
```

## Install

Create an isolated environment, then install the scorer dependencies:

```powershell
uv venv --python 3.12 .venv-openasr-eval
.\.venv-openasr-eval\Scripts\python.exe -m pip install -r requirements-openasr-eval.txt
```

The scorer uses the no-argument Whisper English normalizer API also used by the
Open ASR repository's English scoring path. The public Transformers class now
requires a model-provided spelling map and is therefore not interchangeable.

## Official test data

The official `hf-audio/open-asr-leaderboard` parquet dataset can be evaluated
without keeping a duplicate cache on the system drive. Set `HF_HOME` on the
data drive and either stream a split with `datasets` or make one bounded local
copy. The current repository metadata reports 47.944 GB, so the local-copy
command is permitted only when the target has a 50 GB budget available.

```powershell
$env:HF_HOME = 'G:\OpenASR\hf-cache'
hf download hf-audio/open-asr-leaderboard `
  --repo-type dataset `
  --local-dir 'G:\OpenASR\open-asr-leaderboard' `
  --max-workers 1
```

For streaming, do not start a benchmark-wide inference loop without separately
approving its model, request count, rate, duration, and stop condition:

```python
from datasets import load_dataset

dataset = load_dataset("hf-audio/open-asr-leaderboard", "librispeech", split="test", streaming=True)
```

## Input manifest

Provide UTF-8 JSONL. Every non-empty row requires `reference` and `prediction`.
`dataset` is optional; when present, the report also contains a separate
micro-WER for that dataset. Keep original text in the manifest: normalization
happens inside the scorer and the originals are not rewritten.

```json
{"id":"ls-clean-0001","dataset":"librispeech_clean","reference":"The price is $100.","prediction":"the price is one hundred dollars"}
```

## Score

```powershell
.\.venv-openasr-eval\Scripts\python.exe scripts\evaluate_openasr.py `
  --input D:\benchmarks\openasr-predictions.jsonl `
  --output D:\benchmarks\openasr-wer.json
```

The top-level `wer` is one micro-WER over all normalized utterances. `by_dataset`
uses the same calculation independently per supplied dataset label. The report
includes the normalizer and metric names so a result remains attributable to
this baseline.

## Scope boundary

Matching the normalizer and WER calculation makes a local comparison compatible
with the leaderboard's English scoring convention. It does not reproduce the
leaderboard unless the same test splits, model decoding settings, audio
preprocessing, and runtime/RTFx measurement procedure are also used.
