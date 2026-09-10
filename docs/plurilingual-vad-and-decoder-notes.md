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

# Root cause: whole-file feature normalization — RETRACTED

Tue, 08 Sep 2026, third pass, **corrected in the fifth pass below**. The
measurements in this section are sound and reproducible; the *explanation* is
wrong. `nemotron-3.5` sets `asr.preprocessor.normalize = "NA"`, so the
whole-file normalization described here never executes for the model every one
of these tests used. Read this section for the evidence of length dependence,
not for its cause. See "Correction: normalization is not the mechanism".

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

---

# VAD-bounded segmentation — measured

Tue, 08 Sep 2026, fourth pass. `scripts/vad_segment.py` cuts a file at the
midpoint of Silero-detected silences, targeting a segment length, so no cut
lands mid-word and each segment is normalized on its own statistics.

## Right context stops mattering once the file is short

`rnnt_right_context` 1, 2 and 3 produce **byte-identical output** on the 5 min
prefix of the bilingual file — same md5, 1010 words — and identical merged
results at *every* VAD segment length tested on both files. It only diverges on
long whole-file input (11.6 min: RC1/RC2/RC3 agree only 0.90–0.92 pairwise).

So right context is not really a quality knob. It is a mitigation for long-input
degradation, and it is redundant once the input is segmented.

## French/English, 11.6 min

| method | words | gaps | dropped | % audio |
|---|---|---|---|---|
| whole-file RC1 | 2332 | 11 | 24.1 s | 3.46% |
| whole-file RC2 | 2262 | 13 | 43.0 s | 6.17% |
| whole-file RC3 | 2324 | 7 | 15.1 s | 2.17% |
| **VAD-segmented (7 seg, any RC)** | **2363** | **4** | **12.2 s** | **1.75%** |

Segmentation wins on every metric — most words, fewest gaps, least dropped time.

## Quebecois, 25.5 min — segment length has an optimum

Total gap time conflates two different things, so both are broken out: `warmup`
is the per-segment encoder head gap, `real loss` is dropped content.

| method | segs | words | gaps | gap total | warmup | real loss |
|---|---|---|---|---|---|---|
| whole-file RC1 | 1 | 4048 | 44 | 119.2 s | 0.6 s | 119.2 s |
| whole-file RC3 | 1 | 4131 | 31 | 77.7 s | 0.6 s | 77.7 s |
| VAD target 60 s | 25 | 4069 | 34 | 96.1 s | 32.2 s | 63.9 s |
| **VAD target 150 s** | 11 | **4147** | 35 | 81.9 s | 17.8 s | **64.1 s** |
| VAD target 240 s | 7 | 4099 | 31 | 82.0 s | 5.9 s | 76.1 s |

**150 s is the optimum.** Real loss is flat at ~64 s for 60 s and 150 s segments
and climbs back to 76 s at 240 s — approaching whole-file behaviour, exactly as
the normalization account predicts. Meanwhile warmup falls monotonically with
fewer segments. 150 s takes the low real loss without paying 25 segments' worth
of warmup, and yields the highest word count of any configuration tested.

Note that on raw "dropped seconds" whole-file RC3 (77.7 s) looks competitive with
VAD 150 s (81.9 s). It is not: 17.8 s of the latter is warmup at known
boundaries, so it recovers ~14 s more real speech and 16 more words.

## Recommended configuration

- segment with `scripts/vad_segment.py --target-s 150 --min-silence-ms 100
  --threshold 0.7`
- transcribe segments, any right context (it makes no difference once segmented)
- the two files needed different VAD sensitivity to find silences at all: the
  bilingual file is 98.8% speech and yields only 5 silences at defaults, hence
  `--min-silence-ms 100 --threshold 0.7`

Remaining: ~64 s of real loss on the Quebecois file is still unexplained by
normalization alone, and warmup is a genuine per-segment cost that a proper
overlap-and-merge would avoid. Those are the next two things worth attacking.

---

# Correction: normalization is not the mechanism

Tue, 08 Sep 2026, fifth pass.

## What was wrong

The third pass attributed the long-form dropouts to `fe.cpp:477`, which
normalizes mel features over every frame of the call. That code is real and it
would have that effect — but it is gated on `cfg_.normalize_per_feature`, which
comes from the model:

| model | `asr.preprocessor.normalize` |
|---|---|
| nemotron-3.5-asr-streaming-0.6b | **`NA`** |
| nemotron-speech-streaming-en-0.6b | **`NA`** |
| parakeet-ctc-1.1b | `per_feature` |
| parakeet-tdt-0.6b-v3 | `per_feature` |

Every long-form test in these notes used nemotron-3.5, so **no per-feature
normalization ran at all**. The mechanism cannot be the one claimed.

Proved directly: a windowed-normalization option was added and swept at 10, 30,
60 and 150 s on the 25.5 min file. All four are byte-identical to the
unmodified whole-call path — 4131 words, 31 gaps, 77.7 s — because the code is
never reached for this model.

## What still stands

The length dependence itself is measured and reproducible, and it is not run
noise: transcribing the same file twice gives byte-identical output
(md5 `342A796A…` both runs). So appending audio really does change earlier
transcription, deterministically. The mechanism is still unknown.

Everything else in these notes is unaffected — the monolingual control, the
right-context sweep, the VAD A/B, and the VAD-segmentation results are all
measurements, not inferences from the retracted cause. Segmentation still helps;
the reason it helps is what is now open again.

## The windowed-normalization knob

Implemented anyway, since it is correct for the models that do use per-feature
normalization. `NEMO_SPEECH_NORM_WINDOW_S` (seconds, 0 = off = whole call)
switches `fe.cpp` to statistics gathered over a centered window around each
frame, via prefix sums so the cost stays O(n_mels x n_frames).

- Unset, it is **bit-identical** to the previous build: the 25.5 min file
  reproduces 4131 words / 31 gaps / 77.7 s and the same text md5 across the
  rebuild.
- It measurably changes output on `parakeet-tdt`, which does use per-feature
  normalization, so the implementation works.
- It does not help there either: on the Quebecois file, off = 1922 words /
  644.5 s dropped, 30 s = 1831 / 676.2 s, 60 s = 1886 / 648.2 s. parakeet-tdt is
  a poor fit for this material regardless (42% of audio dropped against
  nemotron's 5%).

Keep it as a sweepable knob for the parakeet models. It is not a fix for the
nemotron long-form problem and must not be described as one.

## fr-CA

Accepted without error, and it does change output, but it does not help:

| file | flag | words | gaps | dropped |
|---|---|---|---|---|
| Quebecois 25.5 min | `fr-FR` | 4131 | 31 | 77.7 s |
| Quebecois 25.5 min | `fr-CA` | 4102 | 36 | 90.4 s |
| bilingual 11.6 min | `auto` | 2324 | 7 | 15.1 s |
| bilingual 11.6 min | `fr-CA` | 1570 | 26 | 201.3 s |

0.951 word-sequence similarity to `fr-FR` on the monolingual file, and the model
still reports `fr-FR` back in `languages` either way. On the half-English file
it is catastrophic, as expected from pinning a single language.

## Build note

`build-cuda` was rebuilt with `-Profile server -Flashlight`
(ASR + diarization + TTS + NMT + HTTP + flashlight). An earlier run of
`build.ps1` with script defaults had reconfigured it to NMT/HTTP/flashlight OFF;
that is undone. `NEMO_SPEECH_WITH_NORM` remains OFF — it cannot be enabled on
Windows.

---

# Root cause, found: the automatic offline-to-streaming switch

Tue, 08 Sep 2026, sixth pass. This is the actual mechanism, and it supersedes
both the retracted normalization account and the "unknown" verdict after it.

## The switch

`recognizer.cpp:559-568`:

```cpp
const bool exceeds_offline_limit = exceeds_offline_position_limit(*model_, n, sample_rate);
const bool supports_streaming = head == Ctc || rnnt->supports_cache_streaming();
const bool use_streaming = (exceeds_offline_limit || (vulkan && ...)) && supports_streaming;
```

`exceeds_offline_position_limit` projects the input through the frontend and
subsampling and compares the result against the encoder's positional-embedding
table. For nemotron-3.5 — `hop 160`, `subsampling_factor 8`,
`conv_context causal`, `pos_emb_max_len 5000` — that puts the boundary at

**399.9 s = 6 min 40 s.**

Below it the file is decoded by the offline full-context runner. Above it, the
recognizer silently swaps in the buffered cache-aware streaming runner. Nothing
in the output says which ran; only the `[asr] mode=` line on stderr does.

Confirmed by straddling it. Two prefixes of the same file:

```
395 s  ->  [asr] mode=offline   head=rnnt attention-left=56 attention-right=3
405 s  ->  [asr] mode=streaming head=rnnt left=56 center=1 right=1 step=160ms
```

## What it costs

Comparing only the shared 0–395 s — byte-identical audio, the sole difference
being 10 s appended past the threshold:

| | words | gaps > 1.5 s | dropped | % of audio |
|---|---|---|---|---|
| 395 s file — **offline** | 1074 | 6 | 18.3 s | 4.64% |
| 405 s file — **streaming** | 1060 | 7 | 25.0 s | 6.32% |

Word-sequence similarity 0.924, and the losses are the familiar ones:

```
offline  : invitation avec grand plaisir. Ça      streaming: invitation, ça
offline  : fait sur le français québécois.        streaming: fait.
```

Ten seconds of extra audio at the end of a file deletes words six minutes
earlier, because those ten seconds change which decoder runs.

## Everything else follows from this

- **The prefix ladder.** 5 min (300 s) is offline; 10, 15, 20 and 26 min are all
  streaming. That is why 0–3 min gave 513 words at 5 min and 507 at every longer
  prefix, identical to each other. The "saturation at 10 minutes" was not
  statistics converging — every prefix ≥ 10 min simply ran the same path.
- **Right context.** `asr.streaming.*` is inert in the offline path, which is
  why RC1/RC2/RC3 are byte-identical on the 5 min file and on every VAD segment,
  and only diverge above 400 s.
- **Why segmentation helps.** VAD segments (60, 150, 240 s targets) all sit
  under 399.9 s, so each is decoded offline. Segmentation was never fixing
  normalization; it was keeping the input in the better runner.
- **Why isolated cut-outs read perfectly.** They are seconds long.
- **The enspa WER table is unaffected** — clips are 5.8–14.7 s, far below the
  threshold, all offline.

## A correction to earlier notes

Runs of the 11.6 min bilingual file were labelled an "offline baseline". At
696 s that file is **above** the threshold, so those runs were streaming. The
`mode=offline` line quoted earlier came from a probe over four 10 s segments.
The claim that "offline and streaming right=1 are byte-identical" was therefore
comparing streaming with streaming — true but vacuous.

## The offline runner already solves this, unreachably

`OfflineRunner::offline_segments_` splits long audio at `max_offline_samples_`
(a binary search for the largest length under the position limit) and snaps each
cut to the centre of the quietest 100 ms window via `snap_to_quiet_` — the same
design as `scripts/vad_segment.py`. But `use_streaming` diverts long input to
the streaming runner before `OfflineRunner` is ever constructed, and it is
gated on `supports_cache_streaming()`. So the offline-only models
(parakeet-tdt) get quiet-snapped offline segmentation for free on long files,
while the cache-aware ones (nemotron-3.5, nemotron-en) never do.

## Recommendation

Keep every ASR call under **399.9 s** for nemotron models. `vad_segment.py
--target-s 150` already does; anything up to ~300 s is safe with margin. The
measured 240 s optimum stands and is comfortably inside the limit.

The alternative, if upstream behaviour can be changed, is to let cache-aware
models fall through to `OfflineRunner`'s existing segmentation instead of the
streaming runner, which would fix this for every caller without a wrapper.

---

# The fix, built and measured

Tue, 08 Sep 2026, seventh pass.

## What changed

`Recognizer::recognize` — the complete-buffer/file path — no longer picks the
streaming runner on length alone:

```cpp
const bool vulkan_requires_streaming = vulkan && head != HeadKind::Ctc;
const bool use_streaming =
    ((longform_streaming && exceeds_offline_limit) || vulkan_requires_streaming)
    && supports_streaming;
```

Over-limit input now falls through to `OfflineRunner`, which already splits at
`max_offline_samples_()` and snaps every cut to the quietest 100 ms window.

**Streaming is not closed off.** Three things were kept deliberately:

1. `streaming_recognize()` — live capture and server streams — builds its runner
   unconditionally and was not touched. Verified: `transcribe --stream` on the
   25.5 min file still logs `mode=streaming`.
2. The Vulkan RNNT compatibility route is preserved as-is.
3. `NEMO_SPEECH_LONGFORM_STREAMING=1` restores the old length-triggered
   behaviour. Verified: it reproduces the previous default **exactly** —
   4048 words, 44 gaps, 119.2 s.

## Measured

Quebecois 25.5 min, `--language fr-FR`:

| condition | words | gaps | dropped | % audio |
|---|---|---|---|---|
| old default (streaming) | 4048 | 44 | 119.2 s | 7.80% |
| old streaming + RC3 | 4131 | 31 | 77.7 s | 5.09% |
| old VAD-seg target 150 s | 4147 | 35 | 81.9 s | 5.36% |
| **new default (offline)** | **4114** | **32** | **81.1 s** | **5.31%** |
| new + escape hatch | 4048 | 44 | 119.2 s | 7.80% |

Bilingual 11.6 min, `--language auto`:

| condition | words | gaps | dropped |
|---|---|---|---|
| old streaming RC1 | 2332 | 11 | 24.1 s |
| old streaming RC3 | 2324 | 7 | 15.1 s |
| **new default (offline)** | **2348** | 8 | 23.3 s |

Against the **default** — which is what callers actually got — this is a clear
win: +66 words and 38 s less dropped on the Quebecois file, and the highest word
count of any configuration on the bilingual file. It is not a clean sweep:
hand-tuned streaming RC3 still edges it on dropped seconds for the Quebecois
file (77.7 s against 81.1 s), while recovering fewer words. VAD segmentation
remains marginally ahead on word count and is now largely redundant.

## Regression

155 enspa clips (mostly 5.8–14.7 s, one at 24.6 s) are all far below the limit
and were offline before and after. 154 of 155 byte-identical.

The one that differed is **not** a regression: running the *same* new binary
twice gives 3 differing clips out of 155, and the second run matches the old
build exactly. Directory transcription with `--concurrency 4` batches utterances
together and the batch composition varies between runs, which flips marginal
clips.

**This means per-clip results carry roughly 2% run-to-run noise whenever
`--concurrency > 1`.** Single-file transcription is deterministic (verified
earlier: two runs of the 25.5 min file byte-identical). Differences of one or
two clips in any batch comparison are noise. The auto-vs-es head-to-head
(71 against 26 with 58 ties) is far outside that band and stands.

## Config consequence

`asr.streaming.rnnt_right_context` is inert on the file path now, because the
offline encoder ignores `asr.streaming.*`. Both pipeline configs are reverted
from `-1` to the model default `1`: the measurement that justified `-1` came
from file runs that were silently streaming, and on the live path it only buys
latency (encoder step 160 ms to 320 ms).

---

# Full re-run on the fixed build

Tue, 08 Sep 2026, eighth pass. Everything below was re-measured after the
dispatch fix. Pre-fix outputs are kept under `scratchpad/marked/`.

## Concurrency 1 is deterministic

155 enspa clips transcribed twice at `--concurrency 1`: **155/155
byte-identical**. That confirms `--concurrency 4` was the sole source of the
per-clip variation, and all comparison runs below use concurrency 1.

The aggregates barely moved, so the earlier concurrency-4 numbers were sound —
the noise flipped individual marginal clips without shifting the totals.

## enspa WER, re-run

| condition | as-is | apostrophes stripped | enspa (151) | spa (4) | empty |
|---|---|---|---|---|---|
| nemotron `auto` | **23.87%** | 23.96% | 23.89% | 22.03% | 0 |
| nemotron `es-US` | 27.17% | 27.08% | 27.23% | 22.03% | 1 |
| parakeet-tdt | 26.19% | 26.22% | 26.22% | 23.73% | 2 |
| nemotron `en-US` | 46.17% | 46.94% | 45.55% | 100.00% | 10 |
| parakeet-ctc | 55.58% | 56.38% | 55.06% | 100.00% | 0 |

Unchanged conclusions: `auto` leads, `en-US` is destructive on Spanish, and
apostrophe handling is irrelevant (≤0.8 points, unsigned).

## Long files — VAD segmentation still earns its keep

Quebecois 25.5 min, everything now offline:

| condition | words | gaps | dropped | warmup | real loss |
|---|---|---|---|---|---|
| whole-file `fr-FR` | 4114 | 32 | 81.1 s | — | 81.1 s |
| whole-file `fr-CA` | 4089 | 40 | 97.0 s | — | 97.0 s |
| VAD 60 s (25 seg) | 4069 | 34 | 96.1 s | 32.2 s | 63.9 s |
| **VAD 150 s (11 seg)** | **4147** | 35 | 81.9 s | 17.8 s | **64.1 s** |
| VAD 240 s (7 seg) | 4099 | 31 | 82.0 s | 5.9 s | 76.1 s |

Bilingual 11.6 min:

| condition | words | gaps | dropped |
|---|---|---|---|
| whole-file `auto` | 2348 | 8 | 23.3 s |
| **VAD-segmented (7 seg), `auto`** | **2363** | **4** | **12.2 s** (5.7 s warmup) |
| whole-file `fr-FR` | 1484 | 22 | 186.2 s |
| whole-file `fr-CA` | 1621 | 23 | 185.5 s |

**Segmentation is not redundant after the fix.** On the Quebecois file it still
recovers ~17 s more real speech than whole-file (64.1 s against 81.1 s of loss)
and 33 more words; on the bilingual file it halves the dropped time. The
`--target-s 150` optimum holds, and the VAD-segmented numbers are byte-identical
to the pre-fix run — expected, since segments were always under the threshold
and always decoded offline.

## fr-CA

On the metrics, `fr-FR` wins the monolingual file: 4114 words / 81.1 s dropped
against fr-CA's 4089 / 97.0 s, at 0.949 word-sequence similarity. On the
bilingual file `fr-CA` beats `fr-FR` (1621 against 1484 words) but both are far
behind `auto`, which is expected when half the audio is English.

This contradicts the owner's reading that fr-CA is the better Quebecois
transcript. The two are not measuring the same thing: gap-and-word-count metrics
reward coverage, not correctness of the words recovered, and there is no
reference transcript for this file so no WER is available. The side-by-side is
in
[docs/transcripts/v2-quebecois-frFR-vs-frCA-vs-vad.md](transcripts/v2-quebecois-frFR-vs-frCA-vs-vad.md)
— judge it by reading, and treat the table above as coverage only.

Note also that the model reports `languages: ["fr-FR"]` whichever of the two is
requested.

## Recommended pipeline

1. `scripts/vad_segment.py --target-s 150 --min-silence-ms 100 --threshold 0.7`
2. transcribe segments with `--concurrency 1` for reproducibility
3. `--language auto` for mixed audio, an explicit code when the language is
   known and single
4. right context: leave at the default; it is inert on this path

---

# The "black hole": a language-switch dropout, and separation fixes it

Thu, 11 Sep 2026. Traced with the owner's reference transcript
(`French Girl Reacts... [ZFc3-CdK1vg].en.vtt`).

## What is missing

At **63.20 s** the narrator switches from French back to English. The reference
reads:

> First of all, I want to say that I understand everyone very well. Like,
> there's over 8 million people in the province of Quebec...

The pipeline emits `...je vais penser en anglais. There's over eight million...`
— **3.76 s of English silently swallowed** at the switch, with the timeline
still correct on both sides, which is why it never desyncs.

## It is the switch, not the length

The 45 s test clip reproduces it, far below any positional limit, so the
offline/streaming threshold is not involved here. Cutting the span out and
transcribing it alone returns it perfectly.

Sweeping *only* the amount of French preceding the switch, every clip ending at
the same point:

| French before the switch | result | `languages` |
|---|---|---|
| 0.0 s | **kept** | `en-US` |
| 0.5 s | **kept** | `en-US` |
| 1.0 s | dropped (partial) | `en-US` |
| 1.5 s | dropped | `[]` |
| 2.0 s | dropped (hard) | `fr-FR` |
| 2.5 s | dropped | `fr-FR` |
| 3.0 s | dropped (hard) | `fr-FR` |
| 4.0 s | **kept** | `fr-FR, en-US` |
| 6.0 s | **kept** | `fr-FR, vi-VN` |

**The failure is a band, not a slope.** With almost no French before it, the
model reads the English fine. With plenty of French, it detects both languages
and reads it fine. In between — roughly **1 to 3 seconds of preceding French** —
it commits to `fr-FR` and swallows the English until it re-locks.

This partially reverses the earlier "the dropouts are not language switches"
conclusion. That verdict came from the monolingual Québécois file having *more*
gaps than the bilingual one, which is still true — so there is more than one
mechanism. But this particular hole is unambiguously a switch artifact, and it is
reproducible on demand.

## Vocal separation recovers it

Of the four configurations tried on this file, exactly one keeps the sentence:

| run | result |
|---|---|
| whole-file, offline | dropped |
| whole-file + **BS-RoFormer separation** | **kept** |
| VAD-segmented | dropped |
| 45 s clip, no separation | dropped |

```
20
00:01:03,760 --> 00:01:07,120
[Speaker 1] First of all, I want to say that I
understand everyone very well like
```

That also explains the +17 words measured earlier for the separated run (2365
against 2348): separation is not just cleaning music, it is removing whatever
was pushing the decoder to stay locked on French through the switch.

**Practical consequence: run separation for any code-switched material**, not
only for noisy or musical sources. It was previously treated as optional
cleanup; on switching audio it recovers content nothing else recovers.

VAD segmentation does **not** help here, which is worth noting given it helped
with the long-form dropouts — further evidence these are two different faults.
