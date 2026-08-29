# Handoff — Mon, 25 Aug 2026

Written at the end of a long session, for whoever picks this up next. The goal
that session started from: **two agents holding a spoken conversation**, as the
first step toward the full-duplex loop in [ROADMAP.md](ROADMAP.md).

That works now, by a route nobody planned. Read "What failed and why" before
changing the architecture — three of the four approaches tried are dead ends
that look plausible on paper, and each one cost hours to disprove.

---

## Where things stand

**Working:** `scripts/cascade_conversation.py` — SenseVoice ASR → Ollama
`llama3.2:3b` → Qwen3-TTS voice cloning, two personas alternating. Six to eight
coherent in-character turns. This is the demo.

**Working:** `https://huggingface.co/spaces/clopeux/personaplex-baseline` —
private ZeroGPU Space, PersonaPlex. Single-turn tab, plus a "Two agents in
conversation" accordion that takes an uploaded seed and returns a stitched wav,
a per-turn table, and a zip of per-turn wavs + `manifest.json` +
`transcript.txt`. **Untested end to end** — it was built but never run, because
running it spends the user's ZeroGPU quota.

**Nothing is committed.** Branch `agent/nemotron-engine`, everything is working
tree. Eight new scripts, `ROADMAP.md`, `README.md` rewritten, and local
modifications inside two submodules (see "Submodule changes" — those are easy to
lose).

---

## What failed and why

Do not re-attempt these without new information.

**Two PersonaPlex locally.** The Q4_K GGUF is 7.86 GB on disk and **10.9 GB
resident** on a 12 GB card. Two instances cannot coexist. This is the constraint
that shapes everything else.

**One PersonaPlex playing both sides.** Feeding the model its own generated
speech as the user stream collapses it into repetition within a few turns
(*"Hi, my name is Amy. Hi, my name is Amy. Sami."*). It is trained to answer a
human; its own output is out of distribution. Session continuation makes it
worse, not better — the cache fills with its own voice on both streams.

**PersonaPlex session continuation for memory.** Implemented, works, measured
(cache 79 → 278 → 477 frames, preamble 3479 ms → 0.01 ms). Then discovered the
voice prompt and system prompt are consumed *inside* `start_conversation`, which
continuing skips — so a continued session keeps turn 1's voice and persona
forever. **Shared context and alternating personas are mutually exclusive on one
instance.** The code is still in the submodule and is worth keeping.

**LFM2-Audio 1.5B** (native ASR+TTS+S2S, would have been ideal — two fit in
3.5 GB). CUDA path segfaults during generation; CPU path completes but the model
echoes and loops on the first turn against a human clip. Not usable in the
current CrispASR build. Two real fixes came out of trying (below).

**The one that worked:** stop asking a speech model to hold the conversation.
Split it — ASR, then a text LLM, then TTS. Every end-to-end S2S model tried
degenerates when its counterpart is not a human; a text LLM is good at exactly
the part they all fail.

---

## Gotchas that cost real time

Each of these presents as something else. They are the reason this file exists.

**CrispASR appears completely broken** — every invocation exits `0xC0000135`.
Its `ggml-cuda.dll` imports `cublas64_13.dll`/`cudart64_13.dll`, which on CUDA 13
live in `bin\x64`, and only `bin` is on PATH. Add
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64`. No rebuild
needed. Will bite anything else linked against CUDA 13 on this box.

**PersonaPlex's model spec is embedded in the GGUF**, and `@gguf` takes
precedence over `model_specs/` on disk. Editing the JSON does nothing in a real
run; you need `--model-spec-override model_specs/personaplex.json`. Related:
`--list-loaders` needs `cwd=audio.cpp` *because no model is open*, so there is no
embedded spec to find — a different cause with the same symptom.

**PersonaPlex output length equals input length, and it talks over its input.**
It is lockstep full duplex. Two consequences: pad the input with silence or the
reply is truncated, and discard the overlapped region or turns grow without
bound (measured: 10.5 → 22.3 → 33.0 → 43.5 s).

**Its KV cache is preallocated at 3000 frames** (`context = 3000`,
`kFrameSamples = 1920` → 12.5 Hz → exactly 4 minutes) and **throws** on
overflow. No eviction, only full reset. VRAM is flat from step one.

**`moshi-personaplex` pins are unsatisfiable against modern Gradio** —
`huggingface-hub >=0.24,<0.25` vs gradio's `>=0.33.5`. Also `torch<2.5`,
`einops==0.7`, `numpy<2.2`. Install it with `--no-deps` at app startup; the pins
are conservative metadata, not real incompatibilities.

**Ollama silently takes the GPU.** Default context held **8 GB of 12**, starving
the TTS of a *35 MB* buffer. Pass `num_ctx: 4096` and `keep_alive: 0`. Note the
unload is **asynchronous** — sampling VRAM right after a request shows it
mid-release, which nearly caused an OOM when PersonaPlex loaded next.

**Qwen3-TTS rejects `--task clon`** ("only supports the Tts task") *and* rejects
plain `--task tts` ("requires voice clone reference audio"). The working form is
`--task tts` **with** `--voice-ref` and `--reference-text`.

**A voice reference is a pair and it fails silently.** Give text for only part
of the clip and cloning produces confident gibberish, not an error. A 14.55 s
reference labelled with its first clause (0.9 words/s) made three of four turns
unintelligible while the other agent, whose text matched, was fine. **Generate
reference text with the ASR, never type it** — `mint_voice_refs.py` does this,
and `cascade_conversation.py` warns outside 1.2–4.5 words/s.

**Autoregressive TTS can fail to stop.** A 26-word sentence produced **655
seconds** of audio, which the silence trimmer happily kept because it was not
silent. `speak()` budgets `words/2.5 × 3` seconds, retries once, then truncates.

**`nsys stats` caches a `.sqlite`** and, finding a stale one, *refuses the whole
command* rather than re-exporting — the second run of a benchmark silently lost
every GPU number while still printing a confident wall clock. Force the export.

**HF Space `/logs/run` is a live SSE stream that never terminates.** Poll
`space_info().runtime.stage` instead. `RUNNING` is itself evidence that imports
succeeded, since `app.py` imports moshi at module scope.

**A 10-minute tool timeout kills the wrapper shell, not the child.** An orphaned
`audiocpp_cli` held 4.39 GB until killed manually. Check for strays after a
timeout.

---

## Measured numbers worth not re-deriving

| | |
|---|---|
| PersonaPlex | 10.9 GB resident, **0.9× realtime** end to end |
| — its AR core alone | **1.64× realtime** (9 684 ms for 199 frames) |
| — the rest | 3 479 ms preamble + 2 282 ms encode + 1 468 ms decode, all serial |
| — model load | ~4.8 s once per process (*not* the dominant cost) |
| PersonaPlex bottleneck | **253 672 `cudaStreamSynchronize` = 79.8% of wall**; only 1.2 s of actual `cudaGraphLaunch` compute |
| Qwen3-TTS 0.6B | 1.99 GB, **2.5× realtime**, clones from ~3 s |
| SenseVoice-Small | 254 MB, emits `<\|emotion\|>`/`<\|event\|>` tags via `keep_tags=true` |
| llama3.2:3b | 2.0 GB, holds persona and thread over 8 turns |
| Nemotron ASR | ~1.5 GB, 16–26× realtime |
| htdemucs | 886 MB, 10.4× realtime |
| Speaker pass | CPU, 40× realtime |

**The headline finding:** PersonaPlex is not compute-bound. Its AR core already
beats realtime; the 0.9× comes from serialized encode → preamble → generate →
decode plus 657 synchronizations *per 80 ms frame*. Quantising smaller attacks
4% of the runtime.

---

## File map

```
scripts/
  cascade_conversation.py   THE WORKING DEMO. ASR -> Ollama -> TTS, two personas.
  mint_voice_refs.py        PersonaPlex voice -> clone reference pair (ASR-generated text)
  Start-Conversation.ps1    PersonaPlex turn loop, per-process. Transcript-in-prompt context.
  duplex_conversation.py    PersonaPlex via resident server. Works; conversation degenerates.
  trim_speech.py            silence trim + --skip for the overlapped region
  prosody.py                F0/RMS/crest via numpy autocorrelation
  conversation_tags.py      turns -> tag store (speaker track = diarization ground truth)
  Measure-Passes.ps1        per-stage timing + targeted nsys traces (sep/asr/speakers/s2s)
  voices.py                 permanent SQLite voice store: centroids + observations
  speakers.py               diarization pass, writes speaker track + observations
  voiceprint.py             CAM++ embeddings
  words_to_srt.py           cue builder (punctuation-aware splitting, orphan merge)
  tagstore.py / translate_srt.py / Start-SubtitlesAudioCpp.ps1   subtitle pipeline
```

Space files live in the scratchpad, not the repo:
`.../scratchpad/space/{app.py,conversation.py,requirements.txt}`. **Worth moving
into the repo** — they are only on HF right now.

---

## Submodule changes (easy to lose)

**`audio.cpp`** — added `continue_conversation` request option to PersonaPlex:
field in `request.h`, parsing in `request.cpp`, `resident_state_` member in
`session.h`, reuse logic + trace scalars in `session.cpp`, and the option
declared in `model_specs/personaplex.json`. Built via
`cmake --build build -j 16 --target bin/crispasr.exe`-equivalent:
`scripts/build_windows.ps1 -Preset windows-cuda-release -Target audiocpp_cli -ModelSet custom -Models "personaplex,nemotron_asr,parakeet_tdt,demucs,sortformer_diar" -CudaArchitectures 86`.
Note `silero_vad` is **not** a selectable model target; `htdemucs` is an alias of
`demucs`. Binary lands in `audio.cpp/build/windows-cuda-release/bin/`.

**`CrispASR`** — fixed a hardcoded 512 MB ggml pool in `src/lfm2_audio.cpp`
(~line 2307) that aborted the LFM2 detokenizer *after generation had already
succeeded* (179 frames needed 537 641 568 bytes vs 536 870 912 available). Now
scales as `Tu × h × 1400` with the old value as floor. **Worth sending
upstream.** Build with `cmake --build build -j 16 --target bin/crispasr.exe`
after sourcing VS18 `vcvars64.bat`; the full build fails on a pre-existing test
needing `unistd.h`.

---

## Open threads

**Prosody** is the current priority — the user's words: *"the voice is really
the most important thing, the conversation is already good."* What they want is
PersonaPlex's pauses, fillers and breathing. Their plan, which is sound:
1. Inject conversational tokens at the LLM layer (`...`, `well,`, `uhm,`)
2. Stream TTS on clause boundaries to get time-to-first-audio under 400 ms
3. Use conversational checkpoints — CrispASR has `csm` (Sesame CSM-1B, built for
   this) and `dia` (dialogue-style, `[S1]`/`[S2]` tags)

Caveat worth repeating to them: a cascade can only *simulate* disfluency.
PersonaPlex's pauses come from reacting in real time to the other speaker.

**Turn length drifts long** — 19 s by turn 6 despite "at most two short
sentences". Llama stops honouring it as history grows. Needs a hard token cap,
not a politer prompt.

**Explicit serialization.** The cascade works because stages happen to take
turns, not because anything enforces it. On a 12 GB card with a 10.9 GB model in
play, this should be a real lock or a VRAM-threshold wait.

**Per-word ASR confidence** is still discarded (model emits it, cue builder drops
it, pipeline deletes the words JSON). It is the natural input to post-ASR
correction and a better code-switch signal than a per-segment language tag.

**Diarization over-segments** — 5 speakers found in a 3-voice generated
conversation. Generated conversations are now free labelled fixtures for fixing
this, via `conversation_tags.py`.

---

## Conventions

Timestamps `DDD, dd MMM YYYY HH:mm`, never epochs. Record every session in the
agent ledger (`agent-ledger/scripts/record-agent-change.ps1`) before replying —
this session's entries are the detailed audit trail behind this summary. Do not
log secrets: the NVIDIA NIM token at
`Desktop/Github Repos/.access/nvidia-inference-token.txt` was deliberately left
unread, and the HF token went into the Space secret without ever being printed.
