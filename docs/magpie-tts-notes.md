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

---

# Multi-speaker and multilingual, Wed, 10 Sep 2026

Four more conversations, all Magpie TTS with baked voices spliced at 350 ms
gaps, transcribed with `--language auto --diarize`.

| conversation | speakers | labels emitted | speaker accuracy | turns | WER |
|---|---|---|---|---|---|
| **4 speakers, no Aria** (control) | 4 | **4** | **96.2%** | **10/10** | 1.48% |
| 4 speakers incl. Aria | 4 | 3 | 66.9% | 7/10 | 2.96% |
| 5 speakers | 5 | 4 | 67.0% | 7/10 | **0.00%** |
| 3 speakers, en/es/fr | 3 | 2 | 71.3% | 6/9 | 12.75% |

## The diarizer is fine. Two of the voices are not distinguishable.

Every failing run collapsed **exactly one** speaker pair, and it was always
**Sofia and Aria** — the two female baked voices. In the 4-speaker case Aria's
turns went to Sofia; in the 5-speaker case Sofia's went to Aria; in the
multilingual case all three of Aria's French turns went to Sofia.

The control settles it. Same script, same four speakers, Aria swapped for
another voice: the diarizer emits all **4** labels and gets **10/10 turns at
96.2%**, against 3 labels and 7/10 at 66.9% with Aria present. This is not the
4-speaker capacity limit and not a speaker-count problem — `sortformer_4spk`
handles four distinct voices well. Sofia and Aria simply are not separable.

**Practical rule: never put Sofia and Aria in the same conversation.**
John / Sofia / Jason / Leo is a clean four-way set. Aria is fine alone or
alongside the male voices.

Note that speaker accuracy and WER are independent here: the 5-speaker run
transcribed **perfectly** (0.00% WER) while still merging two speakers. Word
recognition and speaker attribution fail separately, so a good transcript is no
evidence that the labels are right.

## Multilingual: languages detected, errors sit on the switches

The en/es/fr conversation returned `['en-US', 'es-US', 'fr-FR']` — all three
correctly identified in one file. Spanish and French *within* a turn are
near-perfect; every notable error is at a language boundary:

```
ref : Bonjour à tous. De notre côté, le rapport sera prêt jeudi matin.
asr : Bonjour a tus côté, le rapport sera prêt jeudi matin
ref : Sí, los ingresos subieron un doce por ciento...
asr : See, los ingresos subieron un doce por ciento...
ref : C'est la même tendance chez nous...
asr : será la même tendance chez nous...
```

`Sí` becomes English `See`, and Spanish `será` bleeds into the start of the
French turn. That 12.75% WER is not spread across the file, it is concentrated
at the seams — the same intra-file switching weakness measured on the enspa set,
reproduced here in clean synthetic audio with no accent or noise to blame.

Voice identity does hold across languages: Sofia and Aria remain recognisably
themselves in Spanish and French, which is why the diarizer merged them there
too.

---

# Gap sweep, voice-embedding check, and an end-to-end run

Wed, 10 Sep 2026.

## Inter-turn gap: 100-350 ms, and long pauses hurt

The same two conversations re-spliced at 0 / 100 / 350 / 1000 ms:

| conversation | 0 ms | 100 ms | 350 ms | 1000 ms |
|---|---|---|---|---|
| conv4 (English) WER | 3.70% | **2.22%** | 2.96% | 3.70% |
| convml (en/es/fr) WER | 17.65% | **14.71%** | 14.71% | **26.47%** |
| labels emitted (conv4 / convml) | 3 / 2 | 3 / 2 | 3 / 2 | 3 / 2 |

Two things fall out. **The gap does not touch the speaker merge** — three labels
for four speakers at every gap tested, so the Sofia/Aria collapse is not a
boundary or timing artifact.

And **a long pause is actively harmful on multilingual audio**. At 1000 ms the
en/es/fr file lost language detection entirely, returning `['en-US']` instead of
all three, and WER nearly doubled to 26.47%. Silence apparently lets the
language state fall back to English. Keep inter-turn silence in the
**100-350 ms** band; a full second is worse than none at all.

## The two merged voices really are the closest pair

The owner's objection was that Sofia and Aria sound nothing alike. They don't —
but the diarizer is not listening the way a person does. Same sentence in all
five voices, CAM++ (WeSpeaker VoxCeleb) embeddings, cosine similarity:

```
  0.422  Sofia / Aria     <- merged in all three failing runs
  0.345  Jason / Leo      <- the single Leo->Jason miss in conv5
  0.327  John / Leo
  0.226  John / Jason
  0.205  Aria / Jason
  0.193  John / Sofia
  0.165  Aria / Leo
  0.127  John / Aria
  0.111  Sofia / Jason
  0.052  Sofia / Leo
```

Sofia/Aria is the top pair by a clear margin, and Jason/Leo — the second-closest
— is exactly where the one remaining speaker error landed. The merges track
embedding distance, not perceived similarity. Speaker embeddings key on pitch
range, formant structure and timbre statistics rather than the qualities a
listener attends to, which is why the ear and the model disagree.

This also gives a cheap pre-flight check: embed any candidate voice set with
`scripts/voiceprint.py` and reject pairs above roughly 0.4 before generating
anything.

## End to end from a real audio file

`Start-BetterSubtitles.ps1 -Speakers -LowMemory` on the real 11.6 min
French/English recording, with the new defaults (nemotron-3.5, `--language auto`)
and BS-RoFormer separation:

- 696.1 s of audio in **217.3 s wall** — about 3.2x realtime for the whole
  pipeline, separation at RTFx 4.35 and ASR at RTFx 12.24
- **2365 words**, 223 cues, against 2348 from the raw file, so separation is
  worth roughly 17 words here
- diarization found **4 speakers**: the narrator at 169 cues plus three voices
  from the clips she plays

The structure is right where it matters:

```
[Speaker 1] ...and to understand that, let's watch our first video.
[Speaker 2] Tous les jeunes adultes que nous a interpellons ici à Montréal...
[Speaker 3] Moi, j'ai que
[Speaker 2] toutes mes séries en anglais, même dans...
```

The narrator/clip handoff lands exactly on the cut. The one-cue flip to
Speaker 3 mid-sentence is the same turn-boundary weakness seen in the synthetic
runs — brief, and it recovers immediately.
