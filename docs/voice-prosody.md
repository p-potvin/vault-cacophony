# Voice prosody: pauses, hesitations, breaths, emotion

How a voice *speaks* (as opposed to who it is) is now stored, measured and
rendered the same way everywhere a voice is used. This page covers the
standards it follows, the data it keeps, and where it plugs in.

## The standards, and why these ones

| Concern | Standard / source | How it is used here |
|---|---|---|
| Synthesis markup | **W3C SSML 1.1**, portable core only: `<speak>`, `<prosody rate pitch range volume>`, `<break time>`, `<emphasis>`, `<mark>` | The canonical, engine-neutral form of every turn's delivery. Vendors agree on these elements; `say-as` values, emotional styles and the like are vendor extensions and are avoided. |
| Things SSML has no element for (laughs, sighs, breaths, filled pauses, delivery tone) | SSML `<mark name="…"/>` | `vocal:sigh`, `filler:um`, `tone:sarcastic`. Engines that don't know them ignore them, and they survive a round trip. |
| Emotion | **W3C EmotionML 1.0**, default dimensional set (pleasure/valence, arousal, dominance in [-1, 1]) | Each tone cue has PAD coordinates (`speech_markup.TONE_EMOTION`, overridable per `ToneAnchor`). A mild Schröder-style linear mapping turns them into SSML prosody: arousal raises pitch, rate and loudness; valence lifts pitch slightly; dominance lowers it and adds level. |
| Pause durations | Campione & Véronis, *Speech Prosody* 2002: pause length is log-normal, in brief (~100–200 ms), medium (~0.5 s) and long (~1–1.5 s) populations | `PauseModel` samples each pause from a log-normal around a per-speaker median for its class. |
| Where "uh"/"um" go | Dall, Tomalin & Wester, SSW 2016 (~83% of filled pauses are preceded by silence); Clark & Fox Tree, *Cognition* 2002 ("uh" = short delay, "um" = longer); Székely et al., SSW 2019 (spread evenly into fluent speech they read as a tic) | `humanize` puts filled pauses only at planning points (clause boundaries, before a conjunction), precedes most with silence, and follows "um" with a longer pause than "uh". Their count scales with line length. |
| Breathing | Sundaram & Narayanan, Eurospeech 2003 (breaths at phrase boundaries improve naturalness) | An inhalation before long turns when taking the floor (never on a barge-in), and again at a planning point once a breath group (~12 words) has gone by. |
| Speaking rate from audio | de Jong & Wempe, *Behavior Research Methods* 2009 (syllable nuclei = intensity peaks ≥ 2 dB over the preceding dip, voiced, above a −25 dB silence floor) | `prosody.timing` measures speech rate, articulation rate, pause count/median/p90 and log-spread on every observation. |

Engine tag vocabularies follow the engines' own docs: Fish Audio S2 reads
free-form bracket cues (`[pause]`, `[laughs]`, `[emphasis]`, …); OmniVoice has a
closed non-verbal set (`[laughter]`, `[sigh]`, `[sniff]`, `[surprise-ah]`,
`[confirmation-en]`, …) and speaks anything else aloud; Kokoro, Magpie,
Chatterbox, CosyVoice and Qwen3-TTS take plain text.

## One representation, many dialects

`scripts/speech_markup.py` holds a turn as a short token list:

```
Vocal(breath) Break(160) Text("I think the readings are wrong,") Break(240)
Filler("um") Break(520) Text("but we should check the loop") Emphasis("again")
```

and renders it per engine:

| Engine | Same turn |
|---|---|
| `ssml` | `<speak version="1.1" …><prosody rate="101%" pitch="+0.3st" range="+0.1st" volume="+0.3dB"><mark name="tone:worried"/><mark name="vocal:breath"/><break time="160ms"/> I think … <mark name="filler:um"/>um <break time="520ms"/> … <emphasis level="moderate">again</emphasis></prosody></speak>` |
| `fish_audio` | `[worried] [breath] [short pause] I think the readings are wrong, [short pause] um, [pause] but we should check the loop [emphasis] again` |
| `omnivoice` | `I think the readings are wrong, um... but we should check the loop again` |
| `kokoro`, `magpie`, `cosyvoice`, `chatterbox`, `qwen3_tts` | `I think the readings are wrong, um... but we should check the loop again` |

Pauses become punctuation where punctuation is the only pause control an engine
has (`,` for brief, `...` for longer; a sentence end already is one). Rendering
is idempotent: `sanitize_for_engine(render(u, e), e) == render(u, e)`, so the
TTS engine can sanitize every input without fighting the Smart Picker.

`parse_tagged` reads the bracket cues LLMs and engine dialects write
(`[sarcastic] Oh sure, [sighs] that is *definitely* going to work...`), so a
line written in one dialect can be rendered in any other.

## What the voice store keeps

`voices.db` schema 2 adds a `prosody` JSON column to `voice` (migrated in place
from schema 1):

```json
{
  "standard":   {"markup": "SSML 1.1", "emotion": "EmotionML 1.0 (PAD)"},
  "baseline":   {"rate": 1.04, "range_st": 0.6},
  "pauses":     {"brief_ms": 160, "medium_ms": 480, "long_ms": 1200, "sigma": 0.36},
  "disfluency": {"filled_pause_rate": 0.031, "um_ratio": 0.45},
  "measured":   {"f0_median_hz": 132.4, "articulation_rate_sps": 5.2, "pause_median_ms": 480,
                 "pause_log_sigma": 0.36, "observations": 12, "seconds": 318.5},
  "derived":    "Tue, 03 Nov 2026 14:02"
}
```

`measured` is only ever computed from observations; the other sections may be
derived or set by hand. Hand-set values are also recorded under `overrides`
and laid back over every later derivation, so re-enrolling a voice refreshes
the measurements without undoing a tuned pause length. Each turn's SSML also
carries its EmotionML coordinates as
`<mark name="emotion:arousal=…;valence=…;dominance=…"/>`.

Habits stored against a voice are never overwritten by the startup sync
(`sync_custom_voices_to_store` only fills voices that have none); pass
`overwrite_habits=True` to reset them to the built-ins. Disfluency is derived from the voice's reference transcript when it
has one with enough words.

```bash
python scripts/voices.py derive-prosody --name "Clopeux"     # or omit --name for every voice
python scripts/voices.py prosody --name "Clopeux"            # the stored document
python scripts/voices.py ssml --name "Clopeux" --text "Well, I think we should go." --engine ssml
```

`voices.py enroll` and `sync_custom_voices_to_store(embed_audio=True)` derive it
automatically.

## Where it is applied

* **Smart Picker** (`cacophony/vocal_habits.py`) builds every turn as an
  `Utterance`: persona discourse markers by grammar, then `humanize` with the
  speaker's `PauseModel`/`DisfluencyModel`, then throat noises, a trailing
  laugh/sigh and one emphasised word, under the tone's EmotionML-derived
  prosody. It returns the engine rendering, the SSML, and a `speed_scale`.
* **Profiles** (`cacophony/habit_profiles.py`): each persona has its own pause
  and disfluency models. `resolve_profile` reads the voice store first (habits
  saved there win; fields they predate fall back to the built-in persona), then
  overlays the voice's stored prosody. Set `CACOPHONY_VOICE_STORE` to point it
  at a different database.
* **Orchestrator**: every engine is annotated now (Kokoro/Magpie/CosyVoice used
  to receive raw `[tone]` tags in batch mode). The tone's SSML rate scales the
  engine speed for engines that cannot act the tone out themselves; Fish, which
  performs the `[tone]` cue, is left alone. Each run writes
  `prosody_markup.json` beside the transcript: per turn, the tone, the rendered
  text and the SSML.
* **TTS engine**: `synthesize` passes all input through
  `sanitize_for_engine`, so no caller can make an engine read markup aloud.
* **LLM prompt**: when inline cues are allowed the model is given exactly the
  vocabulary the parser understands, and told not to write "uh"/"um" itself.
* **Cascade** (`cascade_conversation.py`): Qwen3-TTS replies are sanitized, since
  agents sometimes echo SenseVoice's `[emotion]` tags.
* **UI**: the audition shelf shows each voice's measured pitch, rate, pause
  length and filler rate when the store has them.

## Limits

An `Utterance` has one prosody for the whole line. Incoming SSML whose
`<prosody>` covers only part of a line keeps its words but drops that local
rate/pitch change rather than widening it to the whole line. Bare SSML
fragments (`Wait <break time="300ms"/> now`) are parsed as SSML; markup that
does not parse is stripped before it can reach an engine.

A cascade only *simulates* disfluency: these are statistically plausible
hesitations, not ones caused by actually planning speech in real time. The PAD
mapping is deliberately mild; expressive engines act the emotion out from the
tone cue and the mapping only nudges the baseline. Syllable-nuclei rate
estimation undercounts in fast, reduced speech, which is why the derived rate is
clamped to 0.8–1.25× and not applied to cloning engines (the reference recording
already carries the speaker's rate).
