# Silero VAD, KenLM, ITN — what works here and what does not

Tue, 08 Sep 2026. Answers the integration questions raised after the
[segmentation audit](plurilingual-segmentation-audit.md).

## Summary

| goal | verdict |
|---|---|
| Silero VAD masking | **works**, converted and wired into `config/pipelines/*.yaml` (off by default) |
| VAD as a fix for the long-form dropouts | **no effect at all** — identical gaps |
| Flashlight + KenLM for es/fr conditioning | **blocked** — CTC-only path, and the only CTC model is English |
| Sparrowhawk ITN (`.far`) | **blocked on Windows** — cannot be built |
| extra 21 GB `arpa.bin` for Spanish | **not needed** |

## Silero VAD

The GGUF in `models/silero-vad-v6.2.1.gguf` is a genuine silero-vad v6.2.1
export (30 tensors, both 8 kHz and 16 kHz heads, provenance `snakers4/silero-vad`)
but it was written by `vokra-core 0.1.0-alpha.0`, which uses its own metadata
namespace and tensor naming:

| | this repo's loader expects | the vokra file has |
|---|---|---|
| arch key | `general.architecture = "vad"` | `vokra.model.arch = "silero-vad"` |
| hparams | `vad.*` namespace | absent |
| tensors | `encoder.0.conv.weight` | `sr16k.encoder.0.reparam_conv.weight` |

`silero_vad.cpp:285` reads `general.architecture` and refuses anything but
`vad`, so it fails at startup. Fixing the metadata key alone is not enough —
the tensor names and hparams differ too.

Produced a loadable one with the repo's own converter:

```bash
uv pip install --no-deps "silero-vad==6.2.0"
python NeMo-Speech.cpp/convert_model.py silero \
  --outfile NeMo-Speech.cpp/models/silero-v6.2.0-nemo.gguf
```

It emits the 16 kHz head only (15 tensors, 1.24 MB), which is all the ASR path
uses. Verified loading: `[asr] ... vad=on`.

**Two things to know before enabling it.**

Masking is off until `asr.vad.masker.mask_enable` is set — loading the model
alone does nothing. And enabling it moves the RNNT path from `mode=offline` to
`mode=streaming` (`step=160ms`), which is a larger behavioural change than the
masking itself and is not what you want when measuring offline accuracy. It is
therefore wired in with `mask_enable: false`.

## VAD does not fix the long-form dropouts

Whole-file French, 11.6 min, offline vs streaming+masking:

| | words | gaps > 2 s | total gap |
|---|---|---|---|
| offline, no VAD | 2332 | 5 | 14.0 s |
| streaming + masking | 2336 | 5 | 14.0 s |

The gap positions and durations are **identical to the centisecond** — 4.96 s at
6:52, 2.40 s at 7:24, 2.32 s at 10:33, 2.24 s at 10:46. Masking changes nothing
about them.

### The dropped audio is ordinary speech

Cutting 6:51–6:58 out and transcribing it alone returns a full sentence:

> Residency à quoi on s'attendre ses étudiants là, les étudiants qui finissent
> le cours deux vont être capables de raconter des

So the model can read that span perfectly well. It simply does not, when the
span sits inside the longer file — and the whole-file transcript stops right
after `residency.`, which is the first word of it. This is a **long-form
decoding defect**, not silence, not music, not segmentation, and not something
VAD can reach. It is the largest unexplained loss remaining.

## Flashlight + KenLM is blocked

The plan was to condition toward a language with a KenLM rather than retraining.
It cannot work with any model currently in `index.json`:

- `recognizer.cpp:112` gates the flashlight path on
  `decoder.kind == Flashlight` **and** casts the model to `CtcModel`;
  `decoder.h:40` states RNNT boosting is "independent of the flashlight (CTC)
  path". Flashlight beam search is CTC-only.
- The only CTC entry in `index.json` is `nvidia/parakeet-ctc-1.1b`, which is
  English. Measured on enspa: **55.58% WER overall, 100% on the monolingual
  Spanish clips.**

So a Spanish or French KenLM has no model to attach to. `flashlight_decoder.cpp`
also warns loudly when lexicon token pieces are missing from the model's CTC
vocab, which is exactly what an `es.sp.model` lexicon against an English CTC
vocab would produce.

The RNNT-side lever that *does* exist is word boosting — `--speech-context`,
`asr.decoder.boosting_tree_alpha` (1.0), `boosting_depth_scaling` (2.0),
`boosting_max_boost` (5.0). That biases toward specific phrases, so it is useful
for names and domain vocabulary. It is not a language model and will not
substitute for one.

### The extra 21 GB is not needed

`es.arpa.trie.bin` (9.7 GB) begins with

```
mmap lm http://kheafield.com/code format version 5
```

so it *is* a KenLM binary — "trie" names KenLM's TRIE binary structure, not a
flashlight lexicon trie. `flashlight_decoder.cpp:187` passes `lm_path` straight
to `KenLM(...)`, and the flashlight `Trie` is **built in memory at load** from
the lexicon plus the LM (`build_lexicon_trie`, line 192), never read from disk.
A separate `arpa.bin` would add nothing.

What is missing is not the LM but `asr.decoder.lexicon_path`, a flashlight
lexicon TSV. Moot while the CTC constraint stands.

## Sparrowhawk ITN cannot be built on Windows

Per-request language-specific ITN would work the way the docs describe — point
`asr.postproc.itn_model_dir` at a parent whose children are `en`, `es`, `de`,
each holding `tokenize_and_classify.far` and `verbalize.far`, and the request
language picks the child. But it needs `-DNEMO_SPEECH_WITH_NORM=ON`, and
`CMakeLists.txt:136` reads:

```cmake
if(WIN32 AND NEMO_SPEECH_WITH_NORM)
    "NEMO_SPEECH_WITH_NORM is not supported on Windows; disable normalization"
```

`build-cuda/CMakeCache.txt` confirms `NEMO_SPEECH_WITH_NORM:BOOL=OFF`. There is
no point sourcing `.far` files for this host; it would need WSL or a Linux box.

This is also the likely explanation for the `twenty twenty seven` /
`two thousand twenty seven` split seen between `auto` and `es-US` — with no ITN
stage compiled in, number rendering falls to whatever the model emits.

## Language flag policy

Per the owner's decision: **pass `--language` whenever the language is known,
and use `auto` only for genuinely unknown audio.** The measured WER favours
`auto` in aggregate (23.87% vs 27.19%), but the more useful reading of that
table is how little the language flag moves the result and how strong the
tokenizer's English bias is — telling it "this is entirely Spanish" helps
against that bias rather than hurting.

---

# Follow-up: the dropouts are not language switches

Tue, 08 Sep 2026, second pass.

## Language switching is not the cause

The switch hypothesis fit the bilingual file well — all four largest gaps land
exactly on an en/fr boundary (`residency.`|`Vont`, `intimidating`|`d'accepter`,
`critique.`|`French`, `facilement`|`I`), 7 of 11 overall. But the monolingual
control refutes it:

| file | length | gaps > 1.5 s | dropped | % of audio | per minute |
|---|---|---|---|---|---|
| French/English (bilingual) | 11.6 min | 11 | 24.1 s | 3.46% | 0.95 |
| Quebecois vs France (**monolingual fr**) | 25.5 min | 44 | 119.2 s | **7.80%** | **1.73** |

The monolingual file drops nearly **twice as often**. And those gaps are not
silence — they measure 68–134% of the file's mean RMS — nor are they empty:
transcribed in isolation, 4 of the 5 largest return ordinary French.

> Et dans la prononciation, mais sinon c'est pas mal le même vocabulaire que
> qu'au Québec et par

So switches correlate on the bilingual file, but they do not cause the effect.
Whatever this is, it gets worse with length, not with language mixing.

## rnnt_right_context is the best lever found

`asr.streaming.rnnt_right_context` (`-1` = the model's max, 3 for nemotron-3.5):

| file | right=1 | right=2 | right=3 (max) |
|---|---|---|---|
| French 11.6 min | 11 gaps / 24.1 s | 13 gaps / 43.0 s | **7 gaps / 15.1 s** |
| Quebecois 25.5 min | 44 gaps / 119.2 s | — | **31 gaps / 77.7 s** |

Roughly **-37%** dropped time on the French file and **-35%** on the Quebecois,
which also recovers 83 words. Both configs now set `-1`.

Two caveats. The sweep is **not monotonic** — `right=2` is worse than `right=1`
on the French file, so intermediate values cannot be assumed to interpolate. And
it costs latency: the encoder step goes 160 ms to 320 ms, which matters for the
live overlay and is free for offline subtitles.

It reduces the problem; it does not solve it. The Quebecois file still loses
77.7 s, 5.09% of its audio, at the model's maximum right context.

Also worth recording: offline and streaming `right=1` produce **byte-identical**
transcripts, so the earlier "offline vs streaming" distinction was not a real
variable.

## VAD masking was correctly enabled, and does not help

Verified by A/B rather than by the log line, all at `right=3`:

| condition | words | gaps | dropped | text md5 |
|---|---|---|---|---|
| no VAD | 2324 | 7 | 15.1 s | `8cfae76e` |
| VAD loaded, `mask_enable` off | 2324 | 7 | 15.1 s | `8cfae76e` |
| VAD + masking on | 2322 | 7 | 15.1 s | `1318fc9b` |
| VAD + `--endpointing --vad-based-eou` | 2324 | 7 | 15.1 s | `8cfae76e` |

Loading the model alone is byte-identical to no VAD, confirming the documented
behaviour that `mask_enable` is the real switch. Masking on changes the text, so
it was genuinely active — it moves 2 words and **zero** gaps.

VAD-driven endpointing is byte-identical too. That is expected: it emits one
final per utterance mid-stream, which is a server streaming behaviour and has
nothing to do with single-file transcription.

## Where this leaves the dropouts

Not silence, not music, not segmentation, not language switching, not reachable
by VAD. It scales with file length, it is partly mitigated by more right
context, and the audio involved transcribes perfectly when cut out and fed in
alone. The next thing worth doing is a length bisection — transcribe the first
2, 4, 8, 16 minutes of the Quebecois file and find where a given span starts
being dropped.

---

# Root cause: whole-file feature normalization

Tue, 08 Sep 2026, third pass. This explains the length dependence.

## Later audio changes earlier transcription

Prefixes of the 25.5 min Quebecois file, transcribed independently, comparing
only the words in the **first 3 minutes** — audio that is byte-identical in
every run:

| prefix | words in 0–3 min | identical to the 5 min run |
|---|---|---|
| 5 min | 513 | — |
| 10 min | 507 | no (similarity 0.937) |
| 15 min | 507 | no (0.937) |
| 20 min | 507 | no (0.937) |
| 26 min | 507 | no (0.937) |

Appending audio at 20:00 changes what is transcribed at 2:00, and the changes
are losses:

```
 5 min: invitation avec grand plaisir. Ça
26 min: invitation, ça

 5 min: fait sur le français québécois.
26 min: fait.
```

It stabilises from 10 minutes on — 10, 15, 20 and 26 are identical to each
other.

## Why

`src/asr/features/fe.cpp:477`:

> Per-feature normalization over the valid frames **in this call**. Streaming
> mode skips it and maintains incremental statistics in the runner.

The offline path computes, per mel bin, a mean and an unbiased variance across
**every valid frame of the whole file**, then applies them to all frames
(`fe.cpp:484-501`). So the normalization applied to the frame at 3:35 is a
function of audio at 20:00. A longer file moves those statistics, every frame
shifts, and marginal spans fall out of the decode. Once the file is long enough
for the statistics to converge — about 10 minutes here — the effect saturates,
which is exactly the stabilisation in the table.

The same mechanism explains the earlier bisection: the 3:35 span is dropped for
2.96 s at a 5 min prefix and 8.32 s at every prefix of 10 min or more.

## What follows

This reframes the dropouts. They are not a decoder defect and not about
language: whole-file normalization means long-form accuracy is inherently
length-dependent, and the marginal spans it costs are the "gaps".

It also partly rehabilitates segmenting. Cutting a long file into pieces gives
each piece its own, better-conditioned statistics — that is likely why the
isolated cut-outs transcribe perfectly. The earlier finding stands that naive
fixed-window cuts damage boundaries, so the shape of a fix is: **segment on
silence (VAD), not on a fixed grid**, so each segment is separately normalized
and no cut lands mid-word.

Worth testing next, in order:

1. VAD-bounded segmentation vs whole-file on both long files, scored on gap time
   and word count.
2. Whether the streaming incremental-statistics path (which skips this
   normalization) shows the same length dependence — if not, that is the
   cheaper fix.
3. A windowed normalization option in `fe.cpp` — statistics over a rolling
   window rather than the whole call.
