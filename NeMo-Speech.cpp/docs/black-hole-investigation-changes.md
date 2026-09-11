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
| `src/asr/decoders/rnnt_greedy_decoder.{cpp,h}` | `b26235a` | Blank-run cap (`NEMO_SPEECH_RNNT_MAX_BLANK`) |
| `src/asr/decoders/rnnt_greedy_decoder.{cpp,h}` | `60737b2` | Shadow decode (`NEMO_SPEECH_DEBUG_BLANK_SHADOW`) |

## Environment variables added

| Variable | Default | Effect when set |
|---|---|---|
| `NEMO_SPEECH_DEBUG_RNNT` | unset (off) | Logs chunk geometry and every emitted token with its preceding blank run and symbol count, to stderr. Read-only. |
| `NEMO_SPEECH_DEBUG_ENC` | unset (off) | Logs the resolved offline attention window, a host-side recomputation of the mask (per-query visible key span and actual lookahead), and per-frame encoder-output statistics (l2, mean, min, max, cosine to previous frame). Read-only. |
| `NEMO_SPEECH_OFFLINE_LEFT_CTX` | unset | Overrides the offline encoder's left attention context. `-1` = unlimited (no mask built). |
| `NEMO_SPEECH_OFFLINE_RIGHT_CTX` | unset | Overrides the offline encoder's right context (lookahead). `-1` = unlimited. |
| `NEMO_SPEECH_RNNT_MAX_BLANK` | unset / `0` (off) | **Changes decoding.** After N consecutive blank frames, forces the best non-blank token to unfreeze the predictor. See below. |
| `NEMO_SPEECH_DEBUG_BLANK_SHADOW` | unset (off) | After a blank run of N frames (a number sets N, any other non-empty value means 3), logs what the model would have emitted with blank excluded, frame by frame. Commits nothing; transcript verified byte-identical. |

Only `NEMO_SPEECH_RNNT_MAX_BLANK` can alter output, by design, and it is off by
default. The rest are read-only.

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

## The no-code answer: segment before the ASR call

The lock only forms if the predictor has committed hard to the previous
language. It cannot form across a segment boundary, because each `transcribe`
call starts with `prev_token_ = blank`. Cutting the same audio at the switch
removes the hole entirely, with no code change and no GGUF change:

| clip | French before the switch | result |
|---|---|---|
| `a_60-72.wav` | ~3 s | 48-frame hole |
| `b_62-70.wav` | ~1 s | full English recovered |
| `c_63-68.wav` | none (cut at switch) | **exact match to ground truth** |
| `d_exact.wav` | none | exact match |

`c_63-68` returns "First of all, I want to say that I understand everyone very
well. Like there's over eight million." -- including "want to" where the
forced-emit diagnostic produced "wanna".

This is also the real explanation for `2cdd156` ("separation fixes it"): the win
came from segmenting, not from separating.

**Recommended production change: run VAD/segmentation before the ASR call, and
prefer boundaries at language or speaker changes. No C++ change required.**

A second, weaker no-code lever is the language prompt (`--language`), which is
one-hot fused into the encoder before decoding. It changes which language
survives but no value recovers both:

| `--language` | result |
|---|---|
| auto / default | French kept, English resumes late |
| `en-US` | French dropped entirely, 89-frame blank run |
| `fr-FR` / `fr-CA` | French kept, English mangled |

## Divergences found against upstream `a5b6953`

Checked by cloning `NVIDIA/NeMo-Speech.cpp` clean and applying its pinned
patch series to stock ggml. Three real differences, none yet shown to cause the
hole:

1. `CMakePresets.json` has a **committed JSON syntax error** in the `cuda-asr`
   preset -- a missing comma and a tab-indented hand edit. Upstream's parses.
   The preset is unusable as shipped, so local builds were configured by hand.
2. Our vendored `ggml/` carries `fused-relpos-attn.cu`; applying upstream's
   current series to stock ggml produces `fused-attention.cu` instead. Different
   generations of the same kernel.
3. Our `ggml-patches/` is an older series than the C++ it was imported with:
   `0014` is `cuda-relpos-extensions` vs upstream's
   `cuda-fused-attention-extensions`, `0017-cuda-stream-interop` and
   `0018-metal-tensor-api-dynamic-k` are **missing**, and `0007`/`0015` differ in
   content.

Note the vendored `ggml/` has no git metadata, so its provenance cannot be
checked in place -- the comparison above had to be made against a fresh clone.

Evidence argues these are not the cause: a miscompiled attention kernel would
garble the shadow ranking, but the ranking is exactly correct, and the encoder
features through the hole are statistically normal. Worth fixing on their own
merits.

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

Full revert of everything from this investigation:

```bash
git revert e831234 60737b2 b26235a b2354ad 8dcbae4
```
