# Handoff — plurilingual pipeline, Tue 08 Sep 2026

For whoever picks this up next. The reconciliation research is done and written
up in [pipeline-reconciliation.md](pipeline-reconciliation.md); this file is
what to do next and what not to waste time on.

The short version: **language detection and routing are designed and measured on
synthetic audio, but nothing is built.** The `-EnglishOnly` flag does not exist,
per-segment routing does not exist, and the configs in `config/pipelines/` are
untested drafts. Real test material is now available and is the first job.

---

## Corrections to carry forward

These override earlier conclusions in this repo's docs. They came from the
project owner and are not re-derivable from the code.

**Ollama is not a viable translator here — mark it unreliable.** 1B models
cannot translate at all, 7-8B models do not fit alongside the ASR stack on a
12 GB card. That leaves the 3-4B band, which is Riva. The live overlay still
defaults to `-TranslateEngine local` (Ollama); that default is wrong and should
change to `riva`. `Start-LiveSubtitles.ps1` now accepts `riva` (it did not
until this session — the wrapper's ValidateSet omitted it).

**deep-translator was always intended for offline audio only.** The earlier
question "was it ever used live?" was malformed. It is an offline fallback. What
*is* worth keeping from the measurements: `translate_batch` is not a batch
(`base.py:149` loops `translate()` per item), it trips Google's rate limit
reliably, and `auto→es` returned Google's HTML `Error 500` page **as the
translation**. Whatever else changes, it needs a guard rejecting HTML/error
bodies before its output is written to an .srt.

**The English pivot is a fallback, not the plan.** fr→en→es through Riva works
and measured 0.362 s/cue for both legs, but the owner's concern is correct and
should be treated as the working assumption: **it compounds errors**. One
mis-recognised word on dirty audio becomes a wrong English word becomes a wrong
Spanish sentence. Use it where the source is clean; do not assume it on EDA/MDA
material.

**Non-English live translation may simply be out of reach** on this hardware by
any route considered. The honest answer is a purpose-built NMT stack
(OpenNMT-class), which is explicitly out of scope. Do not spend cycles trying to
make Riva or Google hit live latency for non-English pairs — measure it, report
it, and let the owner decide.

**`config/server.example.yaml` is not a baseline.** The asymmetric CTC geometry
in it (`ctc_left_padding: 3.84`, `ctc_right_padding: 0.16`) was written quickly
without tests. The two configs in `config/pipelines/` inherited those numbers
and carry the same caveat. They are a starting point, not a measured default.

**nemotron-3.5 is the default for everything.** It is the only model that
reports a detected language, and it is near the top on both speed and accuracy.
Prefer it unless there is a specific reason not to.

---

## Test material (verified present)

| path | content | duration |
|---|---|---|
| `G:\OpenASR\Multilingual\enspa_dev\clips` | 155 clips, en/es **intra-sentence** code-switching | 35.9 min total, 5.8-14.7 s each |
| `G:\OpenASR\Multilingual\enspa_dev\metadata.tsv` | references for all 155 | 156 rows incl. header |
| `G:\OpenASR\Multilingual\French Girl Reacts to Quebecois Canadian French.mp3` | long-form en + fr | 11.6 min |
| `G:\OpenASR\Multilingual\quebecois vs francais france.mp3` | fr only, two accents | 25.5 min |
| `G:\OpenASR\Multilingual\fr-CA\` | Common Voice-style fr-CA, `dev/test/train.tsv` | not measured |

`metadata.tsv` is the valuable one. Columns:
`audio_filename, speaker, transcript, language, convo_id`. That gives **WER on
code-switched audio** and speaker labels to score diarization against.

**Corrected:** `convo_id` is **not** a splicing key. It is `1` on all 155 rows —
there is one conversation, not several, so splicing yields exactly one 35.9-min
file. The ordering key is **TSV row order**, which the owner confirmed by ear is
the true turn order. Two speakers, 77/78 rows, alternating on 78 of 154
transitions. 151 rows are `enspa`; 4 are `spa` (rows 26, 29, 97, 130), a small
monolingual control. A sample transcript, to
show what "code-switched" means here:

> Okay entonces descansa take your time okay I hope you feel better espero que
> ya no te duela tanto la cabeza I love you.

This switches mid-sentence with no pause. My earlier per-segment detection test
used **synthesized clips with hard cuts between languages** and is not
representative of this. Assume the 6/8 detection result does not transfer until
re-measured here.

---

## Status — steps 1 and 2 are done

Measured Tue, 08 Sep 2026. Results in
[plurilingual-step1-results.md](plurilingual-step1-results.md).

**Step 1 failed and step 3 should not be built as designed.** Detection returns
an empty `languages` on 58% of 10 s segments, and the rate is flat (50-58%)
from 5 s to 30 s, so there is no sweet spot to tune. Step 2 then removed the
motive: `--language auto` scores 23.99% WER against 27.40% pinned `es-US` and
45.61% pinned `en-US`, so `auto` is not a compromise, it is the best option.

Detection is fine on inter-sentential material (12.9% empty on the French file),
so the failure is specific to mid-clause switching. Steps 4 and 5 stand.

## Next steps, in order

### 1. Measure detection on real code-switched audio (blocks everything else)

Splice the enspa clips in TSV row order into one long file, run
`nemotron-3.5 --language auto` in segments, and answer:

- how often does `languages` come back empty, and for how many consecutive segments?
- does it ever report the *wrong* language rather than none?
- does the sticky carry-forward rule survive mid-sentence switching, or does it
  smear one language across the other's segments?
- what segment length is the sweet spot? (5 s was arbitrary)

**Decision this unblocks:** whether per-segment routing is viable at all, or
whether plurilingual needs a different approach entirely.

### 2. WER with and without routing

`metadata.tsv` makes this measurable. Score the transcript against the reference
for the whole enspa set:

- nemotron-3.5 `--language auto` (routing candidate)
- nemotron-3.5 pinned to `en-US`, and pinned to `es-US`
- parakeet-tdt (multilingual, no language field)

Use the existing scorer — `scripts/evaluate_openasr.py` takes JSONL with
`reference` and `prediction`. `scripts/run_openasr_nemo.py` has the
directory-mode runner to copy from.

**Pass `--normalizer basic`.** The scorer defaulted to Whisper's
`EnglishTextNormalizer` with no override; it now takes `--normalizer
english|basic` (default `english`, so existing English runs are unchanged). The
English normalizer applies English-only number, contraction and filler rules —
on this material 137 of 155 references normalize differently between the two,
and it rewrites the English half of a code-switched sentence while leaving the
Spanish half raw (`twelve`→`12` but `cuatro` and `ciento cincuenta` untouched).
That biases exactly the en-pinned vs es-pinned comparison this step exists to
make. Neither normalizer is *correct* for code-switched text — `basic` splits
`what's` into `what s` — so pick one, use it for every condition, and read
`methodology.normalizer` back from the report.

**Decision this unblocks:** whether `auto` costs accuracy against a pinned
language, which is the whole premise of routing.

### 3. Build `-EnglishOnly` and per-segment routing

Only after 1 and 2. The design is in
[pipeline-reconciliation.md](pipeline-reconciliation.md#what-still-needs-a-decision-from-you);
the routing rule is: detected == target → skip NMT; one side English → single
Riva call; neither side English → pivot, flagged as lower confidence.

`-EnglishOnly` supersedes the language flags and selects
`config/pipelines/english-only.yaml`; its absence selects `multilingual.yaml`
and brings Riva up in the same resident server.

### 4. Long-form and accent checks

The two mp3s are for after routing works. The Québécois/France file is an
accent-robustness check within one language — a different question from
switching, and worth keeping separate so a failure is attributable.

### 5. CTC geometry sweep

Still unmeasured, still carrying the owner's caveat. Needs WER-vs-latency on
parakeet-ctc before any of these numbers become a default.

---

## Gotchas that will cost time

**`nemo-speech.exe` exits 53 with no output** unless CUDA 13's `bin\x64` is on
PATH — not just `bin`. No error, no message. Every script in this repo sets it;
anything new must too.

**The resident server is reused even when it lacks what you need.** A server
started without a diarizer is `ready` but answers **HTTP 400** to a diarized
request. `subtitles_server.py` now compares required capabilities and the loaded
ASR model and restarts when short — do not bypass that.

**`--translate-to` and `--format srt` are incompatible** ("currently supports
text and json output"). Translation runs as a second pass over finished cues.

**Riva rejects non-English pairs cleanly**:
`nmt: unsupported language pair: fr -> es`. Confirmed fr→es, es→de, de→fr.

**`languages` is whole-file and unmapped.** On a 39 s clip containing four
languages it returned `["en-US","de-DE"]` — two of four, with no mapping from
language to text span. Per-segment is the only usable granularity.

**Empty source language breaks translation silently.** `-TranslateFrom ""`
produced HTTP 400 on the server path and `Invalid source or target language!` on
the CLI, then fell through to Google. It defaults to `en` now; do not let an
empty value reach either path.

---

## State of the tree

Nothing here is committed. Modified this session:

- `vault-commander/cli/Start-BetterSubtitles.ps1` — Riva GGUF resolution, CTC geometry params, language defaults, capability guard
- `vault-commander/cli/Start-LiveSubtitles.ps1` — `riva` added to `-TranslateEngine`, engine status line fixed
- `vault-commander/cli/utils/subtitles_server.py` — `--access-log` fix, log capture, capability-aware restart, CTC flags
- `vault-commander/cli/utils/subtitles_translator.py` — `/v1/translations` server path, speaker tags dropped
- `vault-commander/cli/utils/subtitles_separator.py` — `win_long_path()` for >260-char paths
- `vault-commander/cli/utils/nemo_asr.py` — delegates server lifecycle to `subtitles_server`
- `vault-cacophony/config/pipelines/*.yaml` — new, untested
- `vault-cacophony/docs/pipeline-reconciliation.md` — new
- `vault-cacophony/scripts/evaluate_openasr.py` — `--normalizer english|basic`,
  methodology field now reports the selected path
- `vault-cacophony/tests/test_openasr_evaluation.py` — two tests covering the default
  and the selected normalizer
- `vault-cacophony/docs/openasr-evaluation.md` — "Normalizer selection" section

`NeMo-Speech.cpp/src/asr/encoder/rel_pos_attention.cpp` also carries the
query-length gate on the fused attention op (`NEMO_SPEECH_RELPOS_MAX_Q`,
default 512) — that one is measured and worth keeping: 2.15× on offline ASR with
no effect on streaming.
