# Changelog

## 0.8.25

* New ASR backend: GigaAM-v3 (Russian, CTC + RNN-T, punctuation-native `e2e` heads).
* New diarization method `foxnose` — WeSpeaker embeddings + spectral clustering,
  no external binary required.
* Diarization correctness: the pyannote powerset decode table had two entries
  swapped (48.21% -> 33.37% DER), and `--diarize-max-speakers` was picking the
  speaker count instead of bounding it (15.74% -> 7.81% DER).
* Diarization speed: pyannote segmentation now infers in parallel chunks
  (2888 s file, 8 cores: ~50 s -> 18.1 s); speaker embedding parallelised.
* Fixed sherpa diarization hanging indefinitely (#328) and the MSVC build
  failure in glint (#327).
* ggml synced to v0.17.0.

## 0.8.24

- **`Session.setTtsPhonemes(String)`** — synthesize a phoneme string verbatim,
  skipping the grapheme-to-phoneme stage. Honoured by `kokoro` and `piper`;
  other backends throw rather than silently synthesizing the text instead.
- **`SessionSegment.speaker`** — the native per-segment speaker label, in the
  `"(Speaker N) "` form, or `""` when the backend does not diarize natively.
  The ordinals are chunk-local: `Speaker 1` from one `transcribe` call is not
  necessarily the same voice as `Speaker 1` from the next.
- Native improvements that ship with this release (no Dart API change):
  - **English pronunciation for Kokoro** substantially fixed — numbers were
    silently dropped from synthesized text, and the phonemes were in the wrong
    alphabet for the model. Agreement with Kokoro's own G2P goes from 58% to
    99.1%.
  - **MOSS-TTS 4B** no longer generates past its stop token.
  - **CosyVoice3 cross-lingual synthesis** — a clone reading a target language
    no longer carries the reference voice's accent.
  - **VibeVoice** speaker turns and per-utterance timings are now parsed out of
    the model's answer instead of being returned as one raw JSON blob.
  - Punctuation restoration no longer double-capitalizes or doubles a full stop.

## 0.8.23

- No Dart API change. Ships the native v0.8.23 improvements: the restored
  Windows CPU binary, F5-TTS Chinese synthesis, VibeVoice voice-pack
  correctness, broader NVIDIA GPU coverage, faster VAD, the opt-in
  `--strict-pipeline` mode, and several TTS-on-Vulkan fixes.

## 0.8.22

- **TTS provenance opt-out now requires a marking attestation.** Synthesis is
  provenance-marked by default; raw (unmarked) synthesis and an explicit
  marking-responsibility affirmation are exposed for callers that take
  responsibility for labeling AI-generated audio. Marked-output flows are
  unchanged.
- Native improvements that ship with this release (no Dart API change):
  - **Tiron** — a new multi-speaker meeting-ASR model (Whisper large-v3 + inline
    speaker markers, `--diarize` cross-window speaker linking) is reachable via
    `CrispasrSession.open()` on the whisper path.
  - **Source separation ~26× faster** on Linux/Windows — `CrispasrSession.separate()`
    with mel-band-roformer no longer runs the naive scalar path (BLAS + FFT iSTFT
    + threading), output bit-identical.
  - **NaN-robust ASR decode** — canary-qwen and 7 other backends no longer emit
    garbage when a quantized weight goes non-finite.
- No breaking Dart API changes beyond the provenance gate above.

## 0.8.21

- **`--max-new-tokens` is honored across all ASR backends.** Ten backends
  (moss-diarize, canary, canary-qwen, glm-asr, funasr, mimo-asr, moss-transcribe,
  mini-omni2, higgs-stt, higgs) previously hardcoded their decode cap, so a long
  single-pass transcription truncated. Each now forwards the value while keeping
  its own default, so nothing regresses.
- **MOSS TTS synthesis params wired** — moss-tts / moss-tts-local now honor
  max-new-tokens, duration (max-speech-tokens), top-p, top-k, and
  repetition-penalty instead of staying at hardcoded defaults.
- **Chunk-local speaker scope** — diarized segments now carry a `chunk_id`;
  `(Speaker N)` labels are chunk-local, so a consumer uses `chunk_id` to tell
  continuity from an ID swap.
- No Dart API changes.

## 0.8.20

- **iOS is now supported.** The release ships an Apple `crispasr.xcframework`
  (device + simulator slices for iOS, plus macOS/tvOS/visionOS) for the first
  time, so a Flutter iOS app can link the native library. Earlier releases
  shipped no iOS artifact at all.
- **Prebuilt native library bundles load as delivered.** The macOS and Linux
  `libcrispasr` bundles had a broken embedded run-path and could not be
  `dlopen`ed after extraction; they now resolve their bundled `ggml` libraries
  relative to their own location. Android and the desktop bundles are laid out
  as a flat `lib/` directory.
- **canary-qwen long audio fixed.** Clips longer than ~30 s no longer skip
  chunking, return near-empty transcripts, or grow memory without bound — the
  backend had advertised an internal chunker it did not implement (#290).
- No Dart API changes; existing code upgrades untouched.

## 0.8.19

- Beat and downbeat tracking: `CrispasrSession.beats()` returns a beat grid
  from mono 22050 Hz float32 PCM, backed by the Beat This! backend. Adds the
  `Beat` record typedef (`timeS`, `isDownbeat`), a `beatsSampleRate`
  capability probe and a `beatsTempoBpm` median-interval tempo estimate.
  Previously beat tracking was reachable only from the CLI and the C ABI.
- Every downbeat is also reported as a beat: the postprocessor snaps each
  downbeat onto its nearest detected beat, so downbeats are a strict subset —
  filter on `isDownbeat` for the bar grid, and never merge two lists.
- Beat This! is **MIT for code and weights** and uses **no DBN** (postprocessing
  is peak-picking only), so unlike most beat trackers this arm carries none of
  madmom's patent-encumbered, non-commercially-licensed machinery.
- The native GGUF architecture auto-detect recognises `beat-this`, so plain
  `CrispasrSession.open()` works for beat models; `backend: 'beat-this'`
  remains valid and is required against an older native library.

## 0.8.17

- Source separation: `CrispasrSession.separate()` splits interleaved stereo
  PCM into named stems (`drums`/`bass`/`other`/`vocals` for htdemucs), with
  the `Stem` record typedef and a `separateSampleRate` capability probe.
  Previously separation was reachable only from the CLI, the C ABI and
  Python — never from Dart.
- Note the input contract: `separate()` takes **interleaved** stereo
  (`L,R,L,R,…`) at 44100 Hz, and the native side counts samples *per
  channel*. To feed a stem to `pitch()`, downmix to mono and resample to
  16 kHz first — the dartdoc has the recipe.
- Piano transcription: `CrispasrSession.pianoNotes()` returns structured
  note events — `PianoNote = ({int midi, double onMs, double offMs, int
  velocity})` — plus a `pianoSampleRate` probe. Previously the only route
  out of the piano backend was `transcribe()`, whose segment text reads
  like `"C4 v=80"`; parsing that back was lossy.
- Native fixes that ship with this release: htdemucs now segments long
  audio (7.8 s windows, 25% overlap, weighted overlap-add), so separating a
  full song no longer needs tens of GB — peak memory is flat instead of
  growing with length. And the htdemucs loader no longer silently zero-fills
  quantized tensors it cannot read, which had made the q4_k model produce
  garbage; unreadable weights are now fatal.

## 0.8.16

- Pitch (F0) estimation: `CrispasrSession.pitch()` returns a monophonic
  pitch track from mono 16 kHz float32 PCM, backed by the CREPE backend.
  Adds the `PitchFrame` record typedef (`timeMs`, `f0Hz`, `voicedProb`)
  and the `pitchSampleRate` capability probe.
- The native GGUF architecture auto-detect now recognises `crepe` (and
  `htdemucs`), so plain `CrispasrSession.open()` works for those models;
  previously only the CLI auto-detected them and every binding had to name
  the backend. Passing `backend: 'crepe'` explicitly remains valid and is
  required against an older native library.

## 0.8.11

- Session introspection accessors: CTC vocab, Whisper `no_speech_prob`, and
  `detected_language`, mirrored across the language bindings.
- Docs/quality: add an example, enable `lints/core`, and apply the resulting
  fixes (flow-control braces + FFI typedef casing).

## 0.8.10

- Initial pub.dev release for the current CrispASR 0.8 line.
- Exposes Dart FFI bindings for the unified CrispASR session API, Whisper-shaped legacy API, audio decode helpers, language detection, diarization, chat/text helpers, grammar support and alt-token capture.
- Keeps the package pure Dart FFI: native `libcrispasr` must be installed or bundled separately.

## 0.5.13

- **Whisper alt-token capture (`--alt N` parity)** — per-token
  top-N alternative-candidate capture for whisper greedy
  decode. Closes the last open whisper-cli equivalent gap. New
  `Word.alts` list (with a new `AltToken` value class, default
  `const []` so existing call-sites stay source-compatible)
  on both the low-level `Segment` shape (from `transcribePcm`)
  and the unified session `SessionSegment` shape (from
  `CrispasrSession.transcribe`). New
  `TranscribeOptions.altN` (default 0 = off) plumbs through
  the low-level path; sticky
  `CrispasrSession.setAltN(int)` (matches the
  `setFallbackThresholds` / `setWhisperDecodeExtras` pattern)
  threads through the session path. Pre-0.5.13 dylibs raise
  `UnsupportedError` for the session setter and report
  `Word.alts == []` from both readers, so apps stay loadable
  and just hide the affordance.
- Capture happens inside `whisper_sample_token` so the alts
  are softmax probabilities at the same decode step as the
  chosen token. Beam search is excluded — siblings are
  beam-conditional rather than greedy alternatives. Chosen
  token is not included in the alt list; entries are sorted
  descending by probability.
- Whisper session-transcribe path now also populates
  `SessionSegment.words` (it previously returned only
  segment-level text) — closes a long-standing gap with the
  parakeet / canary backends as a side-benefit of needing
  per-token data for the word-alt mapping. Whisper sub-word
  BPE means a multi-token word's alts cover the first content
  token only (e.g. "kubectl" surfaces alts for "kub"); full
  word-level enumeration is deferred.
- C-ABI symbols: `crispasr_params_set_alt_n`,
  `crispasr_session_set_alt_n`, `crispasr_token_n_alts` /
  `_alt_id` / `_alt_p` / `_alt_text`,
  `crispasr_session_result_word_n_alts` / `_alt_text` /
  `_alt_p`. New whisper getters
  `whisper_full_get_token_n_alts` / `_alt_id` / `_alt_p`
  (plus `_from_state` variants). All nine symbols pinned in
  `bindings_smoke_test.dart`.
- Live test `test/alt_tokens_live_test.dart` (tagged `live`)
  exercises the full stack against `ggml-tiny.en.bin` +
  `samples/jfk.wav`. Asserts ≥1 returned word has alts,
  every alt's p ∈ [0, 1] and the list is descending,
  chosen token excluded from its own alts, and `setAltN(0)`
  on a re-decode actually clears them. Skips silently when
  `CRISPASR_LIB` / `CRISPASR_MODEL` aren't set. Representative
  dev-box result: 22/22 words on JFK get runner-ups (e.g.
  "Americans → America(4.85%), americ(3.84%), American(3.35%)").

## 0.5.12

- **Audio enhancement (RNNoise pre-step)** — new top-level
  `enhanceAudioRnnoise(Float32List pcm)` runs xiph/rnnoise v0.1
  on a 16 kHz mono float32 buffer (upsample to 48 kHz →
  RNNoise frame loop → downsample back) and returns a fresh
  same-length `Float32List`. Backed by the new C-ABI
  `crispasr_enhance_audio_rnnoise`; pre-0.5.12 dylibs raise
  `UnsupportedError` so callers graceful-degrade. State is
  per-call so worker isolates can run enhancement concurrently
  without coordination.

## 0.5.11

- **Whisper text-suppression + prompt-carry extras** — three
  more whisper-only `wparams` knobs the CLI surfaces
  (`--suppress-nst`, `--suppress-regex`,
  `--carry-initial-prompt`) now have Dart bindings. New
  sticky setter
  `CrispasrSession.setWhisperDecodeExtras(suppressNonSpeechTokens:,
  suppressRegex:, carryInitialPrompt:)`. Empty regex clears
  any prior pattern (passes `nullptr` to wparams =
  whisper's "no suppression" sentinel). Pre-0.5.11 dylibs
  raise `UnsupportedError` so callers can graceful-degrade.
- Underlying C-ABI: `crispasr_session_set_whisper_decode_extras`;
  three new sticky session fields default to whisper's
  upstream values so an unmodified session matches stock
  whisper.cpp.

## 0.5.10

- **Whisper decoder-fallback thresholds** — `wparams`
  knobs that decide when the decoder falls back to a higher
  temperature pass (hard audio, low logprob) or treats a
  segment as silence are now exposed via the session API:
  `CrispasrSession.setFallbackThresholds(entropyThold:,
  logprobThold:, noSpeechThold:, temperatureInc:)`. Mirrors
  the CLI's `--entropy-thold` / `--logprob-thold` /
  `--no-speech-thold` / `--temperature-inc` / `--no-fallback`
  flags. `temperatureInc` is clamped to `[0, 1]`; setting
  `0.0` disables the temperature-fallback loop entirely
  (= the CLI's `--no-fallback`). Defaults match
  `whisper_full_default_params` so an unmodified session
  behaves bit-identical to stock whisper.cpp.
- Underlying C-ABI: `crispasr_session_set_fallback_thresholds`.
  Pre-0.5.10 dylibs raise `UnsupportedError`. Non-whisper
  backends silently ignore — none have an analog for these
  fields today.

## 0.4.9

- Initial pub.dev release.
- Dart FFI bindings for the CrispASR C ABI (`src/crispasr_c_api.cpp`).
- Supports all 17 backends: Whisper, Qwen3-ASR, FastConformer, Canary, Parakeet, Cohere, Granite-Speech, Voxtral (Mistral 1.0/4B), wav2vec2, GLM-ASR, Kyutai-STT, Moonshine, FireRed, OmniASR, VibeVoice-ASR, plus FireRedPunc post-processor.
- Unified `Session` API across all backends; legacy `CrispASR` Whisper-shaped API preserved.
- Word-level alignment, speaker diarization, and language ID helpers.
- Auto-download of registered models via the model registry.
