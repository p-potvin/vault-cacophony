# Roadmap

Where this is going, and in what order. The ASR work so far was about learning
what the pretrained models can do and building the scaffolding around them —
cues, tag store, voice store, profiling. That part is largely in hand.

The shift from here is **towards optimisation and away from reliance on
pre-existing models.** Not all at once: we keep plugging in pretrained models
and refining the algorithms around them, and a training pipeline arrives further
down the road rather than now.

---

## Now: the full-duplex pipeline

**The goal: two audiocpp servers talking to each other, with the ability to
listen in real time.**

This is the focus, and it does not need most of what follows. The base is
already there — CrispASR, NeMo-Speech.cpp, mini-omni, and `audiocpp_server`.

### What already exists

`audiocpp_server.exe` exposes both halves of the loop:

| endpoint | what it gives us |
|---|---|
| `POST /v1/audio/transcriptions/live?model=<id>` | raw PCM in a chunked body, transcript deltas as SSE **on the same connection** |
| `POST /v1/audio/speech` | speech out, `stream_format=sse\|audio` |
| `GET /v1/audio/voices?model=<id>` | reference voice library, `--voice-dir` |
| `GET /health`, `/v1/models`, `POST /v1/models/load` | lifecycle |

Two servers, each one's speech stream feeding the other's live transcription
endpoint, is architecturally expressible today without writing a new protocol.

### The trap to avoid, learned the expensive way

The previous attempt at this (`D:\mini-omni\HANDOFF.md`, Sat 01 Aug 2026) went
through **virtual audio devices** — VB-CABLE, Voicemeeter Banana, VBAN to
phones. It never reached a model problem. It died on Windows audio drivers:
three colliding VB-Audio VAIO packages (Banana's, Banana-AUX's, Potato's),
6 errors from the VB-Audio Devices Checker, device names disambiguated with a
`(2- ` prefix, and half of session 1's device mapping invalidated because the
bridge had been writing to a Potato device while Banana was the engine running.
The handoff ends before any audio flowed.

**Route audio between the two agents digitally — PCM buffers over HTTP — and
none of that exists.** No virtual cable, no Voicemeeter, no driver collisions,
and, worth stating plainly, **no acoustic echo**: an agent cannot hear itself
through a wire it is not connected to. Acoustic echo cancellation is one of the
hardest parts of real full duplex and this design deletes it rather than solving
it.

One real audio device is still needed — the user listening in — but that is
playback only, one direction, one device. It is not the bidirectional routing
matrix that broke last time.

### The model

**PersonaPlex** — `--family personaplex`, `--task s2s`, offline and streaming.
Its own description: *a Moshi-style full-duplex speech-to-speech conversational
model with Mimi audio tokenization, packaged speaker/persona prompts, streaming
audio input, text generation, and generated assistant speech.* Eighteen packaged
voice prompts (`NATF0-3`, `NATM0-3`, `VARF0-4`, `VARM0-4`) and a system prompt
per agent, which is what makes two distinguishable personalities cheap.

This is the model the repo was named around — the tagline has said *"NVIDIA
Personaplex talking to himself via local moshi servers, eager STS models
interrupting each other"* since the first commit.

Two facts about it that shape everything below:

- **The Q4_K GGUF is 7.86 GB.** One instance fits the 3060 comfortably. **Two
  resident instances do not.** With the second card gone, simultaneous overlap
  needs either a shared resident model, a smaller quantisation, or more card.
  Alternating turns need only one instance, which is a second reason to do them
  first.
- **The vendored `audiocpp_cli.exe` cannot load it.** The binary is from
  Aug 18 and registers 50 loaders; the PersonaPlex source landed Aug 20 with the
  submodule bump. The model spec ships, the implementation ships
  (`src/models/personaplex/`), the compiled binary predates both. **audio.cpp
  has to be built from source** — it has only ever been used here as a vendored
  prebuilt.

### Order of work

Alternation first, overlap last. Something you can hear is something you can
adjust, and a strictly alternating conversation is listenable, seekable and
diffable in a way two people talking over each other is not.

1. **Build audio.cpp from source** with PersonaPlex. Blocks everything else.
2. **The turn loop.** A hears the seed and replies; B hears that and replies;
   repeat. Each turn a `.wav` on disk, the whole thing stitched into one file,
   and the turns written into a tag store with a speaker track — the same shape
   the subtitle pipeline produces for a film. That tag store is *ground truth*
   for who spoke when, which makes every generated conversation a free test
   fixture for the diarization pass.
3. **The monitor path.** Hearing each turn as it is produced, not only
   afterwards. Early, because it is what makes the rest adjustable.
4. **Endpointing.** Deciding an utterance is finished, quickly, without cutting
   people off. The `-Gap` heuristic from the subtitle path is an offline
   approximation and will not transfer directly. `silero_vad` and
   `marblenet_vad` are both registered in the runtime.
5. **Turn-taking.** When an agent starts talking, and when it stops because the
   other one has. This is the actual research problem; the rest is plumbing.
6. **Overlap and barge-in.** Detecting that the other party has started while we
   are generating, and deciding to yield or continue. Needs generation to be
   cancellable mid-stream, and two models resident at once — see the memory note
   above. **Deliberately last.**

### What the first working conversations showed

The loop runs. Six turns, two voices, self-endpointing, no truncation once the
reply window is large enough. Four things learned from listening to it:

- **Personas work, and they work well.** Given "you are Desmond, a dreamy,
  over-enthusiastic startup founder", the model introduced *itself as Desmond*
  and stayed in character with no further prompting. Voice id and system prompt
  together are enough to make two agents genuinely distinct.
- **Brevity has to be asked for.** Without "speak in at most two short
  sentences, then stop talking", agents ran to the end of every reply window and
  were cut off mid-word. With it, every turn ended on its own.
- **The seed sets the register, and a bad seed poisons the conversation.** A
  LibriSpeech literary passage made the first agent *parrot* it rather than
  answer, and the conversation took several turns to recover. Openers should
  sound like someone talking to someone.
- **There is no memory between turns, and this is the real limitation.** Each
  turn is a fresh `audiocpp_cli` invocation, so an agent hears only the single
  utterance in front of it and knows nothing of the conversation so far. Persona
  survives because it is re-sent in the system prompt every turn; *context does
  not*. This is the strongest argument for moving from CLI-per-turn to a
  resident server session, which is the stated goal anyway -- a persistent
  streaming session keeps the model's state across the whole conversation
  instead of resetting it every reply.

### Where a turn's time actually goes

One 15.9 s turn, measured with `--log` on the RTX 3060:

| stage | ms | note |
|---|---:|---|
| `graph_prepare` | 305 | |
| `prompt` | **3 479** | the preamble: 51 voice-prompt replay steps, 0.5 s of silence, 16 system-prompt tokens |
| `user_encode` | 2 282 | Mimi encoding the whole input up front |
| `ar.generate` | 9 684 | 199 output frames -- **1.64x realtime** |
| `mimi.decode` | 1 468 | decoding the whole output at the end |
| total | 17 218 | 0.92x realtime |

**The autoregressive core is already faster than realtime.** The 0.92x comes
from the offline shape around it: encode everything, replay the preamble,
generate, then decode everything, strictly in series. Three of those five stages
would overlap or disappear in a persistent streaming session. Full duplex is not
blocked on the model being too slow.

Model load, separately measured with `--request-sequence`, is only ~4.8 s once
per process -- not the dominant cost. Three requests in one session took 55.7 s
total for 17.1 + 17.4 + 16.4 s of work.

### Why `start_conversation` cannot simply stop resetting

Every request calls `main_step_graph_->reset()`, so the KV cache never survives
a turn, and the 3 479 ms preamble is re-paid every time -- 20% of each turn
bought back for nothing. Deleting that one line does not work, for three
reasons that are all about shape rather than difficulty:

- `ConversationState` is constructed fresh per call and carries `delay_state`,
  the delayed-stream alignment. Keeping the cache while restarting the delay
  offset at zero desynchronises the two.
- The preamble runs *after* the reset. Keeping the cache without also skipping
  the preamble appends a second voice prompt and system prompt mid-conversation.
- `valid_steps_` would then climb toward the fixed 3000-frame cache and throw,
  with no eviction path.

The tractable change is a continuation mode: keep `ConversationState` on the
session, and when continuing, skip both the reset and the preamble. The state is
already encapsulated in `ConversationState` and `main_step_graph_`, which is what
makes it a contained change rather than a rewrite.

Routing between agents is **file- and pipe-based, never through audio devices.**
`--audio -` streams raw PCM from stdin in streaming mode, and the server has
`/v1/tasks/run` and `/v1/tasks/stream` alongside the live transcription
endpoint, so there is a path to continuous streaming later without a virtual
cable ever existing.

### One measured thing that bears on this

TTFT is our strongest point from the outside, and sustained throughput drops
fast. The profiling in [README.md](README.md#profiling) says why, and it is
structural rather than a tuning problem: the ASR spends **1.6% of wall time in
GPU kernels** — 107 ms of compute across 28,629 launches averaging 3.7 µs —
while the host sits in `cudaStreamSynchronize` **18,507 times**, at a median of
4.5 µs each. The GPU is idle nearly always, not because it is slow but because
it is never allowed to run ahead: enqueue a little, fence, enqueue a little more.

That is exactly the pattern that hurts most when two models have to stay
resident and interleave on **one** GPU, which is now the configuration. Fixing
overlap is likely worth more to full duplex than any smaller or more heavily
quantised model — quantisation attacks the 1.6%.

---

## The three ASR pipelines

### #1 — Everyday multilingual conversation

The one geared toward actual use: parsing everyday video chats where speakers
barely share a common language, or use several at once. An average conversation
in Montreal.

The bar is **useful rather than distracting** — not 3% WER, not real time, not
production-ready. The failure mode that matters is people focusing on the funny
mistakes instead of the content.

Conditions it has to survive:

- 20 minutes to several hours long
- multilingual **code-switching**, mid-sentence
- loose enunciation
- arbitrary audio/video files

Standing in the way today: language detection outside Chinese/English (see #2),
and the language track being per-segment rather than per-word, which is the
wrong granularity for code-switching by construction.

### #2 — Real time

What we have with Nemotron and company, but better. Current standing: TTFT is
excellent and impresses people unfamiliar with the technology; memory
consumption is good; RTX is good but drops fast.

Most urgent, in order:

1. **The `.srt` files** — a big step back from Parakeet, and the most concrete
   regression on the board.
2. **Language detection** beyond Chinese/English.
3. **Diarization** — barely tested, needs a general pass. Sortformer-Diar-4spk-v1
   is already downloaded in `audio.cpp/models`, and the voice store exists
   specifically to link its window-local labels into identities across a file.

### #3 — Non-real-time, experimental

Our own architecture, roughly. Likely to land near an SSM — Mamba-shaped — with
multi-head attention, **linear scaling as the main constraint**.

Least defined, furthest out, and the one that eventually needs the training
pipeline. Worth noting that PyTorch is back on this box, so this is no longer
blocked on tooling.

---

## Features

### Have

| feature | state |
|---|---|
| Non-speech noise elimination | htdemucs gets ~90% there; wants fine-tuning |
| Diarization with learned voices | speaker pass + permanent voice store; clustering-based, pre-Sortformer |

### To develop

| feature | notes |
|---|---|
| **Far/near-field distinction** | Primarily to drop far-field noise — but it also creates the option to *switch* to a far speaker deliberately. The signal is reverberation: direct-to-reverberant ratio and C50 are the standard measures, both estimable from the audio without a model. Specific model choices need research. |
| **Post-ASR correction** | The words for this: **ASR error correction**, or **generative error correction (GER)** in its recent LLM framing. Adjacent and often bundled: **n-best / LM rescoring**, **punctuation and capitalisation restoration (PnC)**, and **inverse text normalisation (ITN)** for numbers, dates and formatting. Worth separating — PnC and ITN are well-solved and cheap, general error correction is neither. |
| **Per-word confidence** | Called out as something to really improve. See below — it is currently thrown away rather than missing. |

---

## Per-word confidence

The ASR emits it. The cue builder drops it, and the pipeline deletes the words
JSON on the way out, so nothing downstream ever sees it. It is the one thing the
models already give us that we discard.

It is cheap to recover and it is load-bearing for a lot of the above:

- it is the natural input to **post-ASR correction** — you correct what the model
  was unsure of, not everything
- low confidence concentrated in one stretch is a strong **code-switch** signal
  for #1, and a better one than a per-segment language tag
- it belongs in the tag store as its own track, and in the voice store as
  evidence about how well a voice was heard

---

## Foundations already in place

Not roadmap items — the things the roadmap can assume.

- **Tag store** — one `.tags.json` per media file, independent tracks per pass.
  New passes add tracks without disturbing existing ones.
- **Voice store** — permanent SQLite, centroids plus every observation and its
  evidence. Built specifically so Sortformer's window-local labels can become
  identities that persist across files.
- **Profiling** — `Measure-Passes.ps1`, targeted nsys traces, so optimisation
  work has numbers rather than intuitions.

## Hardware

One RTX 3060 (12 GB). The second card was pulled after it interfered with nsys,
nvcc, CUDA, PyTorch and WDDM; reporting is accurate again and PyTorch works.
Two resident models and full duplex now share one GPU, which raises the value of
the overlap problem above.
