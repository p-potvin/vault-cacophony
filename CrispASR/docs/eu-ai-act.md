# CrispASR and the EU AI Act

CrispASR's position under **Regulation (EU) 2024/1689** (the EU AI Act): which
obligations attach to this software, which ones this project discharges in code,
which ones it cannot discharge for you, and where the enforcement actually lives
so it doesn't rot.

**This is not legal advice.** The classifications below are this project's
reasoned position on its own software, not a regulator's ruling. If you deploy
CrispASR commercially in the EU, get your own review — especially for the
sections marked **deployer duty**.

---

## 1. Who is who

The Act assigns duties by role, not by who wrote the code.

| Role | Who that is here |
|---|---|
| **Provider** of the AI system | Whoever *places it on the market or puts it into service under their own name*. For the upstream models, that's OpenAI / NVIDIA / Mistral / Alibaba / … For a product you ship that embeds CrispASR, **that's you**. |
| **Deployer** | Whoever uses an AI system under their own authority — you, running `crispasr` on real people's audio. Most Art. 50 duties land here. |
| This project | Publishes a runtime under MIT. It is neither the provider of the underlying models nor the deployer of your pipeline. Its job is to make sure the runtime doesn't *stop* you complying, and to hand you the machinery you need. |

The practical consequence: **CrispASR can mark its output, but it cannot inform
your users.** Sections 5 and 6 split accordingly.

---

## 2. Timeline

| Obligation | Applies from |
|---|---|
| Art. 5 prohibited practices; Art. 4 AI literacy | **2 Feb 2025** — in force |
| Ch. V general-purpose AI models | 2 Aug 2025 — in force |
| Art. 50 transparency (most of it) | **2 Aug 2026** |
| Art. 50(2) marking, for generative systems placed on the market *before* 2 Aug 2026 | **2 Dec 2026** (per the Parliament/Council provisional agreement of 7 May 2026) |
| Annex III high-risk regime (Ch. III) | **2 Dec 2027** — deferred from 2 Aug 2026 by the Digital Omnibus |

Two cautions. The Annex III deferral was agreed in June 2026 but takes effect on
publication in the Official Journal — **treat it as likely, not as law**. And the
deferral moved only the *high-risk* regime: the Art. 5 prohibitions were never
deferred, and neither were the Art. 50 duties for emotion recognition and
deepfakes.

CrispASR was placed on the market before 2 Aug 2026, so its Art. 50(2) marking
deadline is 2 Dec 2026. It has marked its output since well before that.

---

## 3. The open-source exemption, and what it does not cover

Art. 2(12) exempts AI systems released under free and open-source licences.
CrispASR is MIT. That exemption is narrower than it sounds — it **does not
apply** to:

- **prohibited practices** (Art. 5),
- **high-risk systems** (Art. 6 / Annex III),
- **Art. 50 transparency obligations.**

So the FOSS licence buys nothing for exactly the three areas that a speech
toolkit touches. Everything below is assessed as if the exemption did not exist,
because for these purposes it doesn't.

This is also why "just gate it behind a flag" is not a sufficient answer to an
Annex III capability. A flag governs the *deployer's* context; the provider
obligation attaches to placing a high-risk system on the market at all.

---

## 4. Prohibited practices (Art. 5) — in force since Feb 2025

| Practice | CrispASR |
|---|---|
| **Art. 5(1)(f)** — inferring emotions of natural persons in **workplace or education** settings | **Capability removed entirely.** See §4.1. |
| **Art. 5(1)(g)** — biometric categorisation to deduce race, political opinions, trade-union membership, religion, sex life or sexual orientation | Not implemented. CrispASR infers no protected attribute. See §4.2 on language ID. |
| **Art. 5(1)(h)** — real-time remote biometric identification in publicly accessible spaces for law enforcement | Not implemented, and structurally prevented: the named-voiceprint path refuses to run in streaming/live mode at all. See [`diarization-speakers.md`](diarization-speakers.md). |
| Art. 5(1)(a)–(e) — subliminal manipulation, exploitation of vulnerability, social scoring, crime prediction, untargeted facial scraping | Out of scope for a speech runtime. |

### 4.1 Emotion recognition — removed, not gated

SenseVoice-Small is upstream a multi-task model: its CTC head emits an
`<|HAPPY|>` / `<|ANGRY|>` / `<|SAD|>` / `<|FEARFUL|>` / … marker alongside the
transcript. Surfacing that value would make CrispASR an **emotion recognition
system** under Art. 3(39) — "an AI system for the purpose of identifying or
inferring emotions or intentions of natural persons on the basis of their
biometric data."

That is prohibited outright in workplace and education settings, and high-risk
(Annex III(1)(c)) everywhere else. CrispASR's headline use cases include meeting
transcription with speaker diarization — which is *precisely* the workplace
context Art. 5(1)(f) names.

A consent flag was considered and rejected. It would have addressed Art. 5 (the
system would no longer be placed on the market *for* workplace emotion
inference) but not Annex III(1)(c), which attaches to shipping the capability at
all and is not covered by the FOSS exemption. Retaining it would have committed
this project to the full Chapter III provider regime by Dec 2027 — risk
management system, data governance, technical documentation, human oversight,
accuracy and robustness testing, quality management system, conformity
assessment, CE marking, EU-database registration, post-market monitoring and
serious-incident reporting — or to a documented Art. 6(3) derogation assessment
that still requires registration. For one low-accuracy field on one backend out
of 54, that trade is not close.

**What was removed** (audit of 2026-08-01):

- `emotion` field dropped from `sensevoice_result` (`src/sensevoice.h`) and from
  `crispasr_segment` (`examples/cli/crispasr_backend.h`).
- `"emotion"` key dropped from both JSON writers (`cli.cpp`,
  `crispasr_output.cpp`).
- `sensevoice_transcribe()` now **strips the annotation prefix**. It previously
  returned it verbatim, so every session-ABI consumer — Python, Rust, Go, Dart,
  Java, C#, JS, WASM — received transcripts that literally began
  `<|en|><|HAPPY|><|Speech|><|withitn|>`. That was simultaneously a transcript
  corruption bug and an emotion inference leaking onto every binding.

The marker is still *parsed*, because that is what keeps it out of the
transcript text — the classified value is then discarded. `sensevoice_result.raw`
still holds the unfiltered model output for byte-exact diff-harness parity
against the FunASR reference; nothing in CrispASR surfaces it and nothing should.

Guarded by `tests/test-no-emotion-recognition.cpp` (unit tier — no model
needed). SenseVoice keeps its ASR, its 50+ language coverage and its native
language ID, which is what it was worth having for.

### 4.2 Why language identification is not biometric categorisation

CrispASR infers the *language being spoken*, not a property of the speaker.
Art. 5(1)(g) bites on categorisation that *deduces* race, ethnic origin or
another protected attribute. A language label is an attribute of the audio
signal, and the mapping from language to any protected attribute is neither
performed nor implied.

If you build something downstream that uses a detected language as a proxy for
national or ethnic origin, that inference is yours and Art. 5(1)(g) is your
problem. Don't.

---

## 5. High-risk (Annex III) — CrispASR is designed to stay outside it

| Annex III point | Status |
|---|---|
| **1(a)** remote biometric identification | **Deliberately not implemented.** See below. |
| **1(b)** biometric categorisation by sensitive attributes | Not implemented (§4.2). |
| **1(c)** emotion recognition | **Removed** (§4.1). |

Point 1(a) is the one a diarization feature could drift into. The constraints
that keep the named-voiceprint path outside it are load-bearing, not stylistic:

- **Off by default** behind `--speaker-db-consent`; enrollment hard-fails
  (exit 25) without it.
- **Closed claimed roster only** — `--expect-speakers` is mandatory. The tool
  confirms which *asserted, enrolled, consenting* participants speak. It cannot
  answer "who is this unknown voice?"; the open-ended 1:N search that defines an
  identification system is not implemented, at the CLI or at the C API.
- **Active involvement** — enrollment is a deliberate act by the enrolled person,
  with the consent attestation recorded in the `.spkr` profile. Art. 3(41)'s RBI
  definition turns on identification *without* active involvement.
- **Post-processing only** — no real-time or streaming identification path
  exists (cf. Art. 5(1)(h)).

The default and recommended path — `--diarize-speakers` — computes embeddings
per recording, clusters them into `(speaker 0)` / `(speaker 1)` labels, and
discards them. It identifies nobody and stores nothing.

Full write-up, including the GDPR Art. 9 obligations that apply to the named
path regardless of AI Act classification: [`diarization-speakers.md`](diarization-speakers.md).

**If you re-enable 1:N identification, remove the roster requirement, or add
real-time matching, you are building an Annex III(1)(a) high-risk system and
this document stops describing your software.**

---

## 6. Transparency (Art. 50)

### 6.1 Art. 50(2) — machine-readable marking of synthetic audio

**Provider duty, discharged in code.** Every path that produces synthesized or
substantially altered audio marks it, by default, on every surface:

| Surface | Marking |
|---|---|
| CLI (`--tts-output`, `--tts-stream`) | audio watermark + C2PA manifest |
| HTTP server (`/v1/audio/speech`) | audio watermark + C2PA manifest |
| C ABI (`crispasr_session_synthesize`, `_streaming`, `_speech_to_speech`) | audio watermark |
| WASM / JS (`ttsSynthesize`, `ttsSpeechToSpeech`) | audio watermark; `c2paSign()` available |
| All language bindings | inherit the C ABI — they cannot reach an unmarked path by accident |

Two marking technologies, deliberately:

- **C2PA Content Credentials** — the interoperable, standards-based manifest, and
  the one a third-party tool will actually read. Native implementation
  (`third_party/c2pa-audio`), verified interoperable with the c2pa-rs reference
  reader in both directions. WAV (RIFF `C2PA` chunk), MP3 (ID3v2.4 GEOB), M4A/MP4
  (ISO BMFF `uuid` box). Raw ADTS `.aac` and Ogg `.opus` cannot carry a manifest,
  so they are remuxed to MP4 when C2PA is active.
- **Audio watermark** — survives re-encoding, transcoding and container loss,
  which a manifest does not. Spread-spectrum by default (band-limited to
  ~1.5–4.8 kHz so it stays inaudible); **AudioSeal** neural watermarking via
  `--watermark-model auto` for the stronger option.

**The watertight floor.** Opting out is possible but cannot produce a fully
unmarked file. `crispasr_enforce_cli_watermark_floor()` forces the watermark on
whenever the chosen container can't carry a C2PA manifest — so `--no-watermark`
on a `.opus` output is overridden, with a printed notice. Any opt-out
(`--no-watermark` / `--no-c2pa` / `--no-spoken-disclaimer`) additionally requires
`--accept-marking-responsibility`, which writes a `[MARKING]` audit line
recording that the operator took the disclosure duty on themselves. The server
refuses to *start* with an opt-out flag and no attestation.

On the ABI, `crispasr_session_synthesize_raw()` returns unmarked PCM — for
callers that must resample, mix or concatenate *before* marking — and is
hard-refused (returns `nullptr`) until
`crispasr_session_accept_marking_responsibility()` has been called. Such callers
must then mark the result themselves via `crispasr_watermark_embed()`.

**Not marked, and why.** Art. 50(2) exempts systems performing "an assistive
function for standard editing" or not substantially altering the input data or
its semantics:

- **Transcription (ASR)** — produces a record of what a human actually said. No
  synthetic content is generated.
- **Translation** — generated text, but semantics-preserving by construction;
  this project reads it as standard assistive processing. The conservative
  reading is not obviously wrong, so if you publish machine-translated text at
  scale, consider marking it yourself.
- **Denoising and source separation** (rnnoise, HTDemucs, Mel-Band RoFormer) —
  removes or isolates existing signal rather than generating new content.
- **Speech restoration and upscaling** (Sidon, VoxCPM2 AudioVAE) — these *do*
  generate signal, and they **are** marked; they route through the watermarked
  S2S path.

### 6.2 Art. 50(4) — deepfake disclosure

**Deployer duty.** If you generate a voice clone, you must disclose that the
audio is artificially generated, **clearly and distinguishably, at first
exposure at the latest**. The Commission's guidance is explicit that this needs a
*visible or audible* label — a machine-readable watermark alone does not satisfy
Art. 50(4).

CrispASR gives you the audible label: voice-cloned output gets a **spoken
AI-disclosure prefix**, synthesized in a neutral voice, prepended to the clip.
It is on by default and skipping it requires an attestation
(`--no-spoken-disclaimer` + `--accept-marking-responsibility` at the CLI;
`"marking_attestation"` in the request body, or a server launched with
`--accept-marking-responsibility`, on the API).

Note the deliberate `#312` design: an unattested opt-out is **denied, not
refused**. You still get your audio — with the default disclaimer — plus
`X-Crispasr-*` response headers and an audit-log line saying so. Serving the
stronger default can never emit weaker-than-default output, while a hard 400
would only break clients one field out of date.

**On the C ABI and WASM, Art. 50(4) is yours.** Those paths watermark
(Art. 50(2) ✅) but do **not** prepend the spoken disclaimer. That is a
deliberate limit, not an oversight: the CLI produces a *neutral-voice*
disclaimer by clearing `tts_voice` per call, and several backends need
adapter-specific handling to honour it. On the ABI the voice has already been
applied to the backend context, and there is no portable way to un-apply it —
synthesizing anyway would risk speaking the disclosure **in the cloned voice**,
which makes the fake more convincing rather than less.

The ABI gives you the pieces instead:

| Call | Use |
|---|---|
| `crispasr_session_disclaimer_text()` | The canonical string, identical to the one the CLI speaks. Render it as a **visible** label — Art. 50(5) requires disclosures to meet accessibility requirements, and audio-only is not accessible to a deaf user. |
| `crispasr_session_get_disclaimer_pcm()` | The disclosure synthesized in the neutral voice, for you to prepend. **Must be called before `set_voice()`** installs a clone; it returns `NULL` afterwards rather than risk the cloned-voice failure above. |

Supported order: open session → `get_disclaimer_pcm()` → `set_voice()` →
`synthesize()` → prepend.

Synthesizing with a clone voice and no attestation logs a one-time `[MARKING]`
line naming the duty. It does not refuse —
`crispasr_session_accept_marking_responsibility()` silences it.

**Cloning consent is not gated on the ABI.** The CLI (`--i-have-rights`) and the
server (`consent_attestation`) both hard-refuse without it; the ABI logs a
`[CONSENT]` line and proceeds. Three reasons: consent to clone a voice is a
personality-rights and GDPR matter rather than an AI Act duty; the caller is an
integrator who has read the header; and #312 is the standing lesson on what a
hard refusal does to a surface with many downstream clients. The audit trail is
the deliverable there, not the block.

### 6.3 Art. 50(1) — disclosure of AI interaction

**Deployer duty.** If your system interacts directly with natural persons, tell
them they're talking to an AI, unless it's obvious to a reasonably well-informed
person. A CLI transcription tool is obvious. A voice agent built on CrispASR's
S2S backends is not — that one is on you.

### 6.4 Art. 50(3) — emotion recognition / biometric categorisation

Not applicable: CrispASR implements neither (§4.1, §4.2). Had the emotion field
been kept, every deployer would have inherited a duty to inform exposed persons
of its operation, on every recording.

### 6.5 Art. 50(5) — form of disclosure

All disclosures must be given at or before first interaction, be clear and
distinguishable, and conform to applicable accessibility requirements. The spoken
disclaimer is audible; if your product has a visual surface, mirror it there —
audio-only disclosure is not accessible to a deaf user.

---

## 7. General-purpose AI models (Ch. V)

This project publishes quantized GGUF conversions of third-party models to
Hugging Face (`cstr/*-GGUF`). That does **not** make it a GPAI provider:

- Most of the models are narrow speech models — ASR, TTS, diarization — and do
  not display the "significant generality" Art. 3(63) requires.
- Quantization is a format conversion, not training. Under the Commission's GPAI
  guidelines, a downstream modifier becomes the provider of a modified model only
  when the modification uses more than a third of the original training compute.
  Quantization uses approximately none.

Some backends derive from omni-modal bases (Voxtral, Qwen3-Omni, LFM2-Audio)
which may themselves be GPAI models — but the obligations there sit with their
original providers, not with a downstream requantizer. Model cards in
`hf_readmes/` carry upstream attribution and licence terms.

---

## 8. Feature classification, at a glance

| Feature | Classification | Enforcement |
|---|---|---|
| ASR transcription (54 backends) | Minimal risk; Art. 50(2) assistive exemption | — |
| Text translation | Minimal risk; semantics-preserving | — |
| Language identification | Not biometric categorisation | — |
| Denoise / source separation | Art. 50(2) assistive exemption | — |
| TTS synthesis (52 engines) | **Art. 50(2)** — marked | watermark + C2PA, default-on, watertight floor |
| Voice cloning | **Art. 50(2) + 50(4)** | + spoken disclaimer + `--i-have-rights` |
| Speech restoration / upscaling / S2S | **Art. 50(2)** — marked | watermark via S2S path |
| Session-scoped diarization | Not biometric identification | embeddings discarded, no names |
| Named voiceprint profiles | Kept outside Annex III(1)(a) | `--speaker-db-consent`, closed roster, offline-only |
| Voice-based emotion inference | Art. 5(1)(f) / Annex III(1)(c) | **removed**; `test-no-emotion-recognition` |

---

## 9. Deployer checklist

Things CrispASR cannot do for you:

- [ ] **Art. 50(4)** — show or speak an AI-generated label for any synthetic voice you publish. Default-on at the CLI and server; **your job** on the C ABI, WASM and bindings, using `crispasr_session_disclaimer_text()` / `crispasr_session_get_disclaimer_pcm()` (§6.2).
- [ ] **Art. 50(1)** — disclose AI interaction in conversational products.
- [ ] **Art. 4** — ensure the people operating the system have adequate AI literacy.
- [ ] **GDPR** — voice is personal data, and biometric data when used to identify. The named-voiceprint path is Art. 9 special-category: explicit consent, retention and deletion policy, transparency. Applies independently of the AI Act.
- [ ] Don't repurpose diarization or speaker matching for surveillance, law-enforcement identification, or scraped audio. Out of scope and unsupported.
- [ ] Don't reconstruct emotion inference from the raw model output. §4.1 removed it for a reason; `sensevoice_result.raw` is a parity-testing field, not a workaround.

Penalties for scale: Art. 5 breaches reach €35 M or 7 % of worldwide annual
turnover; most other breaches €15 M or 3 %.

---

## 10. Where the enforcement lives

For whoever maintains this next — the compliance behaviour is code, and code
rots:

| Concern | File |
|---|---|
| Spoken-disclaimer opt-out policy | `examples/cli/crispasr_marking_policy.h` (+ `tests/test-marking-policy.cpp`) |
| Watermark embed / detect | `examples/cli/crispasr_watermark.h`, `crispasr_watermark_dispatch.h` |
| Watertight CLI marking floor | `crispasr_enforce_cli_watermark_floor()` in `examples/cli/crispasr_run.cpp` |
| C2PA signing | `src/core/crispasr_c2pa.h`, `third_party/c2pa-audio` |
| ABI marking attestation | `crispasr_session_accept_marking_responsibility()` in `src/crispasr_c_api.cpp` |
| Voice-clone consent gate | `--i-have-rights` (`crispasr_run.cpp`); `consent_attestation` (`crispasr_server.cpp`); `[CONSENT]` audit line only on the ABI |
| ABI clone disclosure | `crispasr_session_{disclaimer_text,get_disclaimer_pcm}()` (+ `tests/test-abi-clone-disclosure.cpp`) |
| Speaker-DB consent gate | `src/speaker_db.cpp`, `crispasr_speaker_db_open/enroll2` |
| Emotion-recognition exclusion | `tests/test-no-emotion-recognition.cpp` |

Two rules learned the hard way, both worth keeping:

1. **A gate that CI can't run is a gate that ships wrong.** The `#312` marking
   policy went four days broken because its only coverage was a live server test.
   Compliance logic belongs in weight-free, model-free headers with unit tests.
2. **Prove the gate can go red.** Every guard here was verified by re-introducing
   the thing it forbids and watching it fail. A green test that cannot fail is
   indistinguishable from no test.

---

## References

- [Regulation (EU) 2024/1689 — full text](https://artificialintelligenceact.eu/the-act/)
- [Commission Guidelines on transparency obligations (Art. 50)](https://digital-strategy.ec.europa.eu/en/library/guidelines-transparency-obligations-providers-and-deployers-ai-systems)
- [Code of Practice on Transparency of AI-generated Content](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)
- [Commission FAQ on Art. 50](https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act)
- [`diarization-speakers.md`](diarization-speakers.md) — speaker labels, GDPR Art. 9, the RBI boundary
- [`tts.md`](tts.md) — synthesis, cloning, `--i-have-rights`
- [`server.md`](server.md) — `consent_attestation`, `marking_attestation`, `X-Crispasr-*` headers
