# Pipeline reconciliation: SEP + ASR + DIA + NMT + PNC

Tue, 08 Sep 2026

Five named audio types, reconciled against what the stack can actually do. Every
number and every failure below is measured on this box, not inferred.

## The headline: five types collapse to two server profiles

The five cases differ along three axes, and only one of them is a *server*
concern:

| axis | where it lives | flag |
|---|---|---|
| Standard vs Dirty | client (BS-RoFormer, audio.cpp) | `-Separate` |
| English vs Multilingual | **server** (ASR model, NMT present) | `-EnglishOnly` |
| Monolingual vs Plurilingual | client (routing per segment) | automatic |

So there are exactly **two** resident server configs, not five:

- [`config/pipelines/english-only.yaml`](../config/pipelines/english-only.yaml) — nemotron-en, no NMT
- [`config/pipelines/multilingual.yaml`](../config/pipelines/multilingual.yaml) — nemotron-3.5 + Riva

`-EnglishOnly` picks the first and supersedes the language flags; its absence
picks the second and brings Riva up with it. Everything else is a client flag.

---

## Finding 1: language detection works, per segment

`nemotron-3.5 --language auto` returns a **`languages`** field. It is the only
model in the set that does — parakeet-tdt transcribes the same 25 languages
correctly but reports nothing, so it cannot drive routing.

Whole-file, synthesized clips of known language:

| clip | reported |
|---|---|
| fr | `fr-FR` |
| es | `es-US` |
| de | `de-DE` |
| en | `en-US` |

Four for four. **But on a mixed-language file it is not usable whole-file**: a
39 s clip containing fr → en → es → de returned `["en-US","de-DE"]` — two of the
four, with **no mapping from language to text span**. The transcript itself was
correct in all four languages; only the attribution is missing.

Per 5 s segment it works:

| window | detected | expected |
|---|---|---|
| 0-5 s | *(none)* | fr |
| 5-10 s | `fr-FR` | fr |
| 10-15 s | `en-US` | en |
| 15-20 s | `en-US` | en |
| 20-25 s | `es-ES` | es |
| 25-30 s | `es-ES` | es |
| 30-35 s | *(none)* | de |
| 35-40 s | `de-DE` | de |

Six of eight, and **both misses are the first segment after a language change** —
a warm-up effect, not a wrong answer. The mitigation is a sticky carry-forward:
when a segment reports nothing, reuse the previous segment's language. Real
content changes language rarely, so an empty read is far more likely to be
warm-up than an actual switch.

**Verdict: dynamic routing is feasible**, on segments, with sticky fallback.
Route per segment: detected == target → skip NMT entirely; otherwise translate.

## Finding 2: Riva is English-centric, and this changes the plan

| pair | result |
|---|---|
| fr → en | OK |
| en → fr | OK |
| fr → es | `nmt: unsupported language pair: fr -> es` |
| es → de | `nmt: unsupported language pair: es -> de` |
| de → fr | `nmt: unsupported language pair: de -> fr` |

It fails cleanly and immediately, which is the good case — no silent degradation.

## Finding 3: use an English pivot, not Google

The plan was to send non-English↔non-English to deep-translator. Measured, that
is the worse option on every axis.

**deep-translator 1.9.1 against Google, as installed:**

| call | result |
|---|---|
| `translate()` fr→es | 1.81 s — but failed with `TranslationNotFound` on an earlier identical attempt |
| `translate()` fr→en | 0.35 s — same intermittency |
| `translate()` auto→es | returned **`Error 500 (Server Error)!!1500...`** *as the translation* |
| `translate_batch()` 30 cues | **fails every time** |

Three things matter here. `translate_batch` is not a batch — `base.py:149` loops
calling `translate()` once per item, so a "batch" of 30 cues is 30 sequential
HTTP requests and reliably trips rate limiting. The calls are **intermittent**:
the same input failed and then succeeded minutes apart, so this is throttling
rather than breakage. And the `auto→es` case is the dangerous one — it wrote
Google's HTML error page into the output *as if it were a translation*, which a
subtitle pipeline would happily burn into an .srt.

**The English pivot through Riva beats it outright.** Both legs are supported
pairs, so fr→es becomes fr→en→es as two local calls:

```
pivot fr->en->es : 3 cues in 1.09 s (0.362 s/cue, both legs)
  fr: Le systeme fonctionne tres bien aujourd hui.
  en: The system is working very well today.
  es: El sistema funciona muy bien hoy en día.
```

Local, deterministic, no rate limit, no network round-trip, and nothing leaves
the machine. Against Google's 0.35-1.8 s per cue for a *single* leg, with
failures.

**Recommendation: pivot through English via Riva for non-English pairs.** Keep
Google as a last-resort fallback only, and if it stays, it needs a hard 5000-char
cap, sequential pacing with backoff, and a guard that rejects any response
containing `Error 500` or HTML — without that guard it silently poisons output.

### Was Google ever used successfully on live audio?

No evidence that it was, and good reason to think it never worked well.
`translate_batch` fails outright, and single calls at 0.35-1.8 s each are
already at the edge for a live overlay that finalizes a cue ~1.2 s after the
speaker stops. The live overlay's default engine is `local` (Ollama), not
Google; Google is its fallback. Treat Google as **offline, last-resort, and
guarded** — not a live path.

---

## The commands

Start the resident server once, then run the client. Both commands share it.

**English pipelines (ESA, EDA)**

```bash
nemo-speech serve --config config/pipelines/english-only.yaml
```

```bash
vw better-subtitles -Input "D:\Media\Season 1" -Recurse -SkipExisting -EnglishOnly            # ESA
vw better-subtitles -Input "D:\Media\Season 1" -Recurse -EnglishOnly -Separate                # EDA
```

**Multilingual pipelines (MSA, MDA)**

```bash
nemo-speech serve --config config/pipelines/multilingual.yaml
```

```bash
vw better-subtitles -Input "D:\Media\Film.mkv" -TranslateTo en                                # MSA
vw better-subtitles -Input "D:\Media\Film.mkv" -TranslateTo en -Separate                      # MDA
```

**gRPC surface**, same configs, when a Riva-compatible client is wanted:

```bash
riva_server --config config/pipelines/multilingual.yaml --bind 0.0.0.0:50051
```

Override anything per run without editing the file — precedence is
**defaults < YAML < env < CLI**:

```bash
nemo-speech serve --config config/pipelines/multilingual.yaml --asr.backend.gpu 1 --nmt.model.n_ctx 8192
```

---

## What still needs a decision from you

**Plurilingual routing** is feasible but unbuilt. The design that the
measurements support:

1. segment the audio (the streaming path already does)
2. read `languages` per segment, sticky carry-forward on empty
3. detected == target → skip NMT for that segment
4. detected ≠ target, one side English → single Riva call
5. neither side English → Riva twice, pivoting through English

Step 5 doubles NMT cost for those segments — 0.362 s/cue measured against
0.2 s/cue for a single leg. On a fully non-English → non-English film that is
roughly +80% on the translation stage, which is still a small share of the total
because ASR dominates.

**The open question is step 2's reliability on real content.** My test used
synthesized clips with hard cuts between languages. Real plurilingual audio
switches mid-sentence, over music, with accents — and I have no measurement for
that. Before building it I would want a real bilingual sample from your library
to check whether the warm-up miss is one segment or many.

**Also unmeasured:** the CTC geometry in these configs (3.84 / 0.16) is
reasoned from the redundancy arithmetic, not benchmarked. It wants a WER-vs-latency
sweep on parakeet-ctc before being trusted as a default.
