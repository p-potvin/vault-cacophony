# Black-hole investigation — every modification, and why production is unaffected

Scope: the language-switch "black hole" in `nemotron-3.5-asr-streaming-0.6b`, where
a stretch of real speech after a French→English switch is transcribed as nothing.

**Every change is diagnostic and default-off.** No shipped code path changes
behaviour unless an environment variable is set. The verification section below
shows the default output is byte-identical to the pre-change build.

Built and tested only in `build-s2s`. `build-cuda` has not been rebuilt.

---

## Files touched

| File | Commit | What |
|---|---|---|
| `src/asr/decoders/rnnt_greedy_decoder.cpp` | `8dcbae4` | Decode tracing for both greedy decoders |
| `src/asr/runner.cpp` | `8dcbae4` | Cache-aware geometry log line |
| `src/asr/model.cpp` | `b2354ad` | Offline encoder frame statistics + offline context overrides |
| `src/asr/model.h` | `b2354ad` | Declaration for the above |
| `src/asr/encoder/fastconformer.cpp` | `b2354ad` | Offline attention-mask geometry log |
| `src/asr/decoders/rnnt_greedy_decoder.{cpp,h}` | **uncommitted** | Blank-run cap (`NEMO_SPEECH_RNNT_MAX_BLANK`) |

## Environment variables added

| Variable | Default | Effect when set |
|---|---|---|
| `NEMO_SPEECH_DEBUG_RNNT` | unset (off) | Logs chunk geometry and every emitted token with its preceding blank run and symbol count, to stderr. Read-only. |
| `NEMO_SPEECH_DEBUG_ENC` | unset (off) | Logs the resolved offline attention window, a host-side recomputation of the mask (per-query visible key span and actual lookahead), and per-frame encoder-output statistics (l2, mean, min, max, cosine to previous frame). Read-only. |
| `NEMO_SPEECH_OFFLINE_LEFT_CTX` | unset | Overrides the offline encoder's left attention context. `-1` = unlimited (no mask built). |
| `NEMO_SPEECH_OFFLINE_RIGHT_CTX` | unset | Overrides the offline encoder's right context (lookahead). `-1` = unlimited. |
| `NEMO_SPEECH_RNNT_MAX_BLANK` | unset / `0` (off) | **Changes decoding.** After N consecutive blank frames, forces the best non-blank token to unfreeze the predictor. See below. |

The first four cannot alter output. The fifth can, by design, and is off by default.

## The one behavioural knob: `NEMO_SPEECH_RNNT_MAX_BLANK`

Mechanism it targets: in RNN-T the predictor is conditioned on the last *emitted*
token, and a blank is not an emission. `rnnt_greedy_decoder.cpp` is explicit about
this — "Blank paths never reach this block and retain the cache across frames and
step() calls." So across a blank run the joint is scored against one frozen
predictor state. When that state is a stale prior (the last token of the previous
language), every frame can score blank and greedy decoding has no way to perturb
the state back out. `max_symbols_per_step` cannot help: it counts emitted symbols
at one frame and is incremented only on the non-blank commit path, so during a
blank run it is pinned at zero.

When set, after N consecutive blanks the decoder runs one blank-excluded argmax
and force-emits the best non-blank. That commits a new predictor state and breaks
the lock. It deliberately emits a token the model ranked below blank, so it is a
diagnostic, not a fix.

Non-obvious implementation note: the cap must be applied to the blank run the
*vectorized scan is about to consume* (`first_emit`), not only to the
"every remaining frame is blank" branch. A full utterance decoded as one chunk
always has a non-blank somewhere ahead, so that branch is never reached and the
run is swallowed in a single hop. The first attempt got this wrong and fired only
in the streaming warm-up pass, which emits nothing and is discarded.

## Results on the failing clip (`a_60-72.wav`, 12.08 s, 151 encoder frames)

Ground truth (official `.en` subtitles, 63 s):
*"First of all, I want to say that I understand everyone very well. Like, there's over 8 million people…"*

| `MAX_BLANK` | forced emits (offline) | transcript |
|---|---|---|
| unset / 0 | 0 | French kept; **48-frame hole**; resumes at "There's over eight million" |
| 24 | 2 | French clean; "**Like** there's over eight million…" |
| 16 | 3 | French clean; "**everyone very well. Like** there's over…" |
| 10 | 3 | French damaged ("Vend"); "**That I understand everyone very well. Like**…" |
| 5 | 4 | French damaged ("Wenn" for "Même"); **"First of all, I wanna say that I understand everyone very well. Like there's over eight million…"** |

At `MAX_BLANK=5` a single forced `.` at t=45 unfreezes the predictor and every
token after it is correct. The audio was always legible to the model.

The trade-off is a real one: a low cap also fires inside genuine silence at the
start of the clip and injects spurious tokens ("Wenn" for "Même"). A production
rule would need to gate forcing on acoustic evidence rather than blank count
alone. **This knob is not production-ready and is not enabled by default.**

## Hypotheses tested and ruled out

- **Symbol cap discarding output** — the guard fires zero times at every setting.
- **Attention-mask / static-cache padding hiding real frames** — our offline mask
  matches `transformers/models/nemotron_asr_streaming` exactly (same
  `chunked_limited` function, chunk = R+1 = 4, left_context_chunks = 56/4 = 14,
  reach 60 keys, left_ctx = `sliding_window - 1` = 56). No query is fully masked.
- **Degenerate encoder features in the hole** — frames 39-87 have mean l2 3.45 and
  mean cos-to-previous 0.82, against 3.70 and 0.74 outside; zero near-duplicate
  frames anywhere. A mask leaving queries on a constant context would pin cosine
  near 1.0 and flatten the norm.
- **Word boosting** — now genuinely live on the reconverted GGUF (`boosting=on`,
  output changes), but it cannot fill a blank run: blank-vs-non-blank is decided
  on unboosted logits and boosting only re-ranks which non-blank wins.

## Also corrected

`8dcbae4`'s rc=1 vs rc=3 traces do not show what its message claims. The offline
encoder is built in `ensure_offline_path()` from the model's *trained* context, so
`--rnnt-right-context` never reaches it; both traces are the same 56/3 decode and
are byte-identical apart from the output filename. Its traces were also never
committed — `*.log` is gitignored. They are present now as `.txt` under
`docs/traces/`.

## Verification that production behaviour is unchanged

With no environment variables set, the transcript is identical to the
pre-instrumentation build:

```
Quand je parle avec quelqu'un, je vais penser en anglais. There's over eight
million people in the province of Quebec, so it's one of the most well known
ways of speaking Fr
```

`MAX_BLANK=0` (explicitly off) produces the same string, and the forced-emit path
logs zero activations.

## To revert

The diagnostics are additive and can be dropped wholesale:

```bash
git revert b2354ad 8dcbae4
```

The blank-run cap is currently uncommitted; `git checkout -- src/asr/decoders/`
removes it.
