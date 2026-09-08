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
