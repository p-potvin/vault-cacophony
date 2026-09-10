# Magpie TTS — first pass, Wed, 09 Sep 2026

`nvidia/magpie_tts_multilingual_357m` + `nemo-nano-codec-22khz-1.89kbps-21.5fps`,
both already in the local model cache. Verdict: it works, it is fast, and the
English conversation path is very good.

## Rates and voices, from the GGUF metadata

- `nano_codec.sample_rate = 22050` — 22 kHz is the codec's **native** rate, not a
  choice. `--sample-rate` only downsamples ("8 kHz through model rate").
- `magpietts.baked_speakers = 5`, `magpietts.speaker_names = ['John', 'Sofia',
  'Aria', 'Jason', 'Leo']`. Pretrained voices, selected with `--voice NAME` or
  `--speaker N`.
- Synthesis measured at **2.0–2.8× realtime** on cuda:0.

## Always pass non-ASCII text via `--input`

Passing accented text as a command-line argument corrupts it before the model
ever sees it. Same sentence, two ways:

| input method | round-trip transcript |
|---|---|
| command-line argument | *Niekiomia Pin, el Haré, su abuela.* |
| `--input` (UTF-8 file) | *El niño pequeño comió una piña en el jardín de su abuel* |

and in French:

| command-line argument | `--input` |
|---|---|
| *Et tout comme hiver, je profore les crops, lachete* | *Comme hiver, je préfère les crêpes à la châtaigne, n'est-ce pas?* |

This also explains an apparent voice/accent bug. `fr2.wav` and `fr3.wav` were
generated with the **same** `--language fr-FR`, but fr2 sounds like an English
speaker reading French while fr3 is clean native French. The difference is
entirely the corrupted input text: mangled bytes produce non-French phoneme
sequences, which the model renders with a foreign accent. Language control
itself is fine — `--language fr-CA` produces a recognisably Québécois delivery,
distinct from `fr-FR`.

## English conversation with diarization — the best case

Eight-turn dialogue, two baked voices (John / Sofia), synthesised per turn and
spliced with 350 ms gaps into a 50.6 s conversation, then transcribed with
`--diarize --language auto`.

| metric | result |
|---|---|
| WER (English normalizer) | **1.54%** |
| WER (basic normalizer) | 1.53% |
| reference / hypothesis words | 131 / 131 |
| diarizer labels emitted | exactly 2 |
| word-level speaker accuracy | **130/131 = 99.2%** |
| turns with correct majority speaker | **8/8** |

Both speaker errors are single words at turn boundaries. The remaining ASR
errors are trivial: `Sofia` → `Sophia`, `support costs` → `support cost`.

**Read this as a ceiling, not an expectation.** The audio is TTS-generated:
clean, no overlap, no crosstalk, no background, no room, and produced by a model
in the same family as the recogniser. Real conversational audio will be worse.
What it establishes is that the diarization and ASR wiring is sound — when the
audio is easy, the pipeline does not get in its own way.

One artifact worth noting: sentence boundaries merge across turns. The
transcript reads "...none of them read the documentation, right? So the
question..." — John's last word and Sofia's first end up in one sentence. The
diarizer still labels them correctly, so this is punctuation/segmentation, not a
speaker error. Anything that consumes sentences rather than words needs to cut
on the speaker change, not on the punctuation.

## Pronunciation: ASR boosting does not transfer to TTS

`Montréal` is mispronounced (the silent `t` is voiced). Word boosting fixes the
*recognition* side — `--speech-context PHRASE` with
`asr.decoder.boosting_tree_alpha` / `boosting_max_boost` — and that is an **ASR**
feature: it biases the decoder toward expected words. There is no equivalent
boost on the synthesis side.

The TTS levers for pronunciation are `--tn-model-dir` (text-normalization
grammars) and, failing that, respelling the word phonetically in the input text.
Worth testing before assuming a shared dictionary can serve both directions.

## Open

- voice cloning is reported poor; pretrained voices preferred for now
- `--voice` across all five speakers, and whether voice identity holds across
  languages
- whether `--tn-model-dir` can fix `Montréal`, and whether those grammars build
  on Windows (the ASR-side Sparrowhawk ITN does not — see
  [plurilingual-vad-and-decoder-notes.md](plurilingual-vad-and-decoder-notes.md))
- multi-speaker conversation beyond two voices, and overlapping speech
