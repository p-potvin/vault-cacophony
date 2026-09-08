# NeMo-Speech.cpp vs audio.cpp + Audio Flamingo

Wed, 02 Sep 2026

45 minutes of English podcast, one RTX 3060, every ASR model the new build
exposes, plus the two composed pipelines. The question behind it: keep the
audio.cpp + Audio Flamingo path, or move to NeMo-Speech.cpp?

Short answer: **move the transcription, subtitle, and translation work to
NeMo-Speech.cpp. Keep Audio Flamingo, but stop using it as an ASR engine — it is
not one. Keep audio.cpp for the two things NeMo-Speech.cpp cannot do at all.**

Source: `Shane Gillis This Past Weekend w Theo Von 478.mp3`, offset 600 s,
exactly 2700.0 s, 16 kHz mono. Two speakers, conversational, unscripted,
overlapping — a hard and realistic case, not a clean read.

---

## ASR

RTF is wall/audio, so lower is better and below 1.0 is faster than realtime.
VRAM is the delta over an idle desktop baseline, sampled once a second.

| model | wall | RTF | ×realtime | VRAM MiB | words |
|---|---|---|---|---|---|
| **parakeet-tdt-0.6b-v3** | **50.9 s** | **0.0189** | **53.0×** | 4436 | 8272 |
| nemotron-speech-streaming-en-0.6b `--stream` | 74.6 s | 0.0276 | 36.2× | **1078** | 8187 |
| nemotron-speech-streaming-en-0.6b | 76.8 s | 0.0285 | 35.1× | 2430 | 8190 |
| nemotron-3.5-asr-streaming-0.6b `--stream` | 81.3 s | 0.0301 | 33.2× | 1080 | 7946 |
| nemotron-3.5-asr-streaming-0.6b | 83.5 s | 0.0309 | 32.4× | 2442 | 7950 |
| parakeet-ctc-1.1b `--stream` | 293.9 s | 0.1088 | 9.2× | 1459 | 8133 |
| parakeet-ctc-1.1b | 297.9 s | 0.1104 | 9.1× | 2792 | 8119 |
| **audio-flamingo-3 Q4_K_M** | **691.7 s** | **0.2562** | **3.9×** | 5747 | 8127 |

All models were given the same 45 minute file. Nemotron 3.5 was prompted with
`--language en-US`; the others take no language prompt.

### What the table says

**Parakeet TDT is the fastest thing here by a wide margin** — 53× realtime, and
13.6× faster than Audio Flamingo. The cost is VRAM (4.4 GB, the largest of the
NeMo models) and that it is **offline-only**: it rejects `--stream` and `--live`
outright, so it cannot serve the full-duplex work in ROADMAP.md.

**Streaming is strictly better than offline on the Nemotron models.** Slightly
faster *and* less than half the VRAM — 1078 MiB versus 2430 MiB. The cache-aware
path never materializes the whole utterance. There is no tradeoff to weigh here:
if the work is streaming anyway, streaming mode is simply cheaper.

**Parakeet CTC 1.1b is the one model to skip.** It is 3.9× slower than the 0.6B
models and produces no visible benefit on this audio. Bigger is worse here.

**Audio Flamingo is last by an order of magnitude**, and that is before noting
that its 691.7 s includes ~176 s of pure process-reload overhead: it must be
chunked at 29 s (see [audio-flamingo.md](audio-flamingo.md)) and
`llama-mtmd-cli` reloads all 4.68 GB per chunk, 94 times. Even crediting that
back it lands near 6× realtime, still 9× slower than Parakeet TDT.

### Quality

There is no reference transcript for this recording, so no WER. Two things are
measurable and both matter:

- nemotron-3.5 and Audio Flamingo agree on **78.5%** of words (word-level
  sequence match, case- and punctuation-insensitive). They broadly hear the same
  thing; neither is verified correct.
- **The NeMo models emit punctuation and capitalization. Audio Flamingo does
  not** — it returns lowercase, unpunctuated text. For subtitles that is not a
  stylistic difference, it is an extra PnC pass the other path does not need.

Word counts cluster tightly (7946-8272, a 4% spread), so none of them is
dropping or hallucinating whole passages.

---

## Pipeline 1: ASR + diarization, full stack resident

`nemo-speech transcribe --diarize --format json`, Sortformer v2.

| stage | wall | RTF | ×realtime | VRAM MiB |
|---|---|---|---|---|
| ASR + diarization | 157.5 s | 0.0583 | 17.1× | 2712 |
| TTS stack load (Magpie + NanoCodec + tokenizer) | 2.6 s | - | - | 1076 |
| **total** | **160.1 s** | **0.0593** | **16.9×** | peak 8250 |

Diarization roughly doubles ASR cost — 83.5 s becomes 157.5 s — and still
finishes at **17× realtime**. It tagged all 7950 words, found **3 distinct
speakers across 172 turns**. The podcast has two hosts, so the third label is
either a guest, inserted clips, or a mis-segment; Sortformer v2 caps at four
speakers and was not given a count.

Peak GPU was 8250 MiB of 12288 with the desktop's ~1900 MiB included, so the
whole set is genuinely co-resident with ~4 GB spare.

## Pipeline 2: ASR → Riva translation → Magpie TTS

English audio in, Spanish audio out. Riva-Translate-4B Q4_K_M.

| stage | wall | RTF | ×realtime | VRAM MiB | note |
|---|---|---|---|---|---|
| ASR | 83.5 s | 0.0309 | 32.4× | 2583 | 7950 words |
| Translate | 133.8 s | 0.0496 | 20.2× | 3208 | 208 segments |
| TTS *(1200-char sample)* | 39.7 s | - | - | 1236 | 96.1 s of audio out |
| **total as measured** | **257.0 s** | **0.0952** | **10.5×** | peak 8797 | |

**The measured total understates the real cost, and the honest number matters
more than the flattering one.** TTS ran on a 1200-character sample, but the full
translated transcript is 42,379 characters. Magpie synthesizes at **RTF 0.4134
(2.4× realtime)** measured against its own output, so the whole transcript costs
about **1403 s** and produces ~56 minutes of Spanish (Spanish runs longer than
the English it came from).

    full pipeline = 83.5 + 133.8 + 1402.7 = 1620 s
    RTF 0.600, i.e. 1.67x realtime

So the answer to *can we still achieve RTF with all of this* is **yes, but the
margin collapses from 10.5× to 1.67× once TTS runs whole, and TTS is then 87% of
the wall clock.** Everything upstream of TTS is nearly free by comparison.

Translation output is fluent, correctly-accented Spanish.

### The bug that would have cost an afternoon

The first translate run split the transcript on sentence punctuation and
produced **one** segment from 179 words. Conversational ASR output runs for
hundreds of words between full stops, and Riva rejects any prompt past
`nmt.model.n_ctx` (default 1024 tokens) — so on the 45 minute file this fails
outright rather than slowly.

Capping each request at 40 words fixed it **and made translation three times
faster** (14.4 s → 4.6 s on the 60 s sample): several short prompts beat one long
one. `scripts/nemo_pipeline.py` does punctuation-first splitting with a hard word
cap as the fallback.

---

## Recommendation

**Move to NeMo-Speech.cpp for transcription, subtitles, diarization, and
translation.** It is 8-14× faster than the Audio Flamingo path, punctuates and
capitalizes without a second model, and collapses ASR + VAD + diarization + PnC +
ITN + NMT into a single command with word timestamps and native SRT/VTT output.
The whole model set is co-resident on a 12 GB card with room to spare.

Model choice depends on the surface:

| use | model | why |
|---|---|---|
| batch subtitles, offline files | `parakeet-tdt` | 53× realtime, fastest available |
| streaming / full-duplex | `nemotron-en --stream` | 36× realtime at 1078 MiB, and TDT cannot stream |
| non-English or mixed | `nemotron-3.5 --language <code>` | prompt-conditioned across 40+ locales |
| — | ~~`parakeet-ctc-1.1b`~~ | slower than the 0.6B models for no gain |

**Keep Audio Flamingo — but not as an ASR engine.** It loses on every
transcription axis: 3.9× realtime, most VRAM of the ASR set, no punctuation, and
a hard 29 s chunking limit. What it does that *nothing* in NeMo-Speech.cpp does
is understand audio — captioning, genre and instrument identification, and
open-ended questions about a recording. That is what the tag store and cue work
actually need it for, and multi-window understanding works correctly (unlike its
transcription). Point it at that and stop benchmarking it against ASR models.

**Keep audio.cpp for two things NeMo-Speech.cpp cannot do at all:**

- **Source separation** — BS-RoFormer / HTDemucs have no equivalent here.
- **Voice cloning** — Magpie selects from baked speaker names or indices. There
  is no reference-audio cloning, so the Qwen3-TTS clone path and the voice store
  built around it are not replaced.

If TTS output volume matters, Magpie's 2.4× realtime is the ceiling on any
speech-to-speech work and is where optimization effort belongs.

## Gotchas

**`nemo-speech.exe` exits 53 with no output whatsoever** unless CUDA 13's
`bin\x64` is on PATH — not just `bin`. Identical to the CrispASR failure in
`HANDOFF.md`, different binary. There is no error message and the shell swallows
the code, so it presents as a program that runs and prints nothing.

**`nemo-speech transcribe` writes results to stdout and diagnostics to stderr**,
so redirection is safe, but merging the streams corrupts parsed output.

## Reproducing

```bash
ffmpeg -ss 600 -t 2700 -i podcast.mp3 -ac 1 -ar 16000 -c:a pcm_s16le gillis45.wav

python scripts/bench_asr_nemo.py --audio gillis45.wav \
  --models nemotron-3.5,nemotron-en,parakeet-tdt,parakeet-ctc \
  --stream-models nemotron-en,parakeet-ctc --with-af3

python scripts/nemo_pipeline.py --audio gillis45.wav --mode stack
python scripts/nemo_pipeline.py --audio gillis45.wav --mode translate --target-lang es
```

Raw results: `D:/HuggingFace/bench/results/*.json`,
`D:/HuggingFace/bench/pipeline/*/pipeline.json`.

---

# Addendum: real WER on LibriSpeech test-clean

Wed, 02 Sep 2026, later the same day

The podcast benchmark above measured speed and had no reference transcript, so
it could not rank accuracy. This addendum does, using the Open ASR leaderboard's
own English path -- Whisper `EnglishTextNormalizer` then `evaluate.load('wer')`
-- over all 2620 utterances of LibriSpeech test.clean (5.4 h).

**Two claims in the section above are wrong and are corrected here.**

## Results

| model | WER | RTFx | mode |
|---|---|---|---|
| audio-flamingo-3 Q4_K_M | **1.489%** *(250 utt; 1.542% over 1723)* | ~3.9× | chunked 29 s |
| **parakeet-ctc-1.1b** | **1.852%** | 163.5 | offline |
| parakeet-tdt-0.6b-v3 | 1.931% | 195.8 | offline |
| nemotron-speech-streaming-en-0.6b | 2.649% | 33.6 | `--stream` |
| nemotron-3.5-asr-streaming-0.6b | 3.511% | 30.6 | `--stream`, `--language en-US` |

**The Q8_0 GGUF conversion is lossless.** NVIDIA publishes **1.93%** for
parakeet-tdt-0.6b-v3 on this split; the Q8 GGUF scored **1.9310%**. That the
number reproduces to three decimals validates both the quantization and this
harness, and means a GGUF number here can be compared directly against the
published leaderboard.

**Not apples to apples on the last two rows.** The Nemotron models could only be
run in `--stream` (see the memory section below), and cache-aware streaming
costs some accuracy against offline decoding because the right context is
bounded. Their offline WER is unmeasured and would likely be better. Do not
read 2.649% and 3.511% as their ceiling.

## Correction 1: parakeet-ctc is the most accurate NeMo model

The section above recommended skipping `parakeet-ctc-1.1b` because it ran 3.9×
slower than the 0.6B models on the podcast. That was a conclusion about one long
file, not about the model, and it was wrong.

On short utterances CTC reaches **163.5 RTFx** -- within 20% of Parakeet TDT --
and it is the **most accurate NeMo model tested**, before any tuning at all.

The podcast slowness has a specific and fixable cause. Buffered streaming
defaults to `chunk_size` 0.16 s with `ctc_left_padding` and
`ctc_right_padding` both 1.92 s, so every decode step processes

    0.16 + 1.92 + 1.92 = 4.0 s of audio to emit 0.16 s of output

which is **25× redundant compute**. That is a tuning parameter, not a property
of the model, and raising `chunk_size` should recover most of it: at
`chunk_size` 1.0 s the same padding gives 4.84 s per 1.0 s, roughly 5× overhead
instead of 25×. Untested so far.

## Correction 2: the crash is host RAM, not a file-count limit

An earlier note claimed the cache-aware RNNT models abort above ~128 input
files. **That was wrong** -- an artifact of free memory drifting between tests.

The real cause: `ggml_init`'s `mem_buffer` is a **host** allocation. The offline
RNNT path requests far more host memory than the CTC and TDT paths, and when
malloc returns NULL the process dies on

    GGML_ASSERT(ctx->mem_buffer != NULL) failed   ggml/src/ggml.c:1610
    exit 3221226505 (0xC0000409)

which reads like a GPU or model fault and is neither. Measured on this box
(31 GiB total, ~73-76% consumed by browsers and IDEs): the 2620-file run
**succeeds above ~8.4 GiB available and fails below ~8.0 GiB**. Ruled out by
experiment, each tested directly:

| suspected cause | verdict |
|---|---|
| GPU memory | no -- 3 `audiocpp_server.exe` held 7.4 GB of 12 GB throughout, and TDT/CTC ran fine |
| file count | no -- all 2620 succeeded in a single call when RAM was free |
| `--concurrency` | no -- 1, 2 and 4 behave identically |
| hardlink vs copy staging | no -- both succeed |
| CUDA `bin` / `bin\x64` PATH order | no -- all three orderings succeed |
| pyarrow / soundfile imports and memory pool | no -- pool is empty, releasing it changes nothing |

**Workaround: `--stream`.** It allocates far less host memory and completed at
7.75 GiB available where offline failed at 8.02 GiB. It is also the mode the
full-duplex work needs, so this is not much of a concession.

## What this changes

On read speech every model lands within **1.2 WER points** while the speed
spread is **50×**. Audio Flamingo buys 0.36 points over parakeet-ctc for roughly
40× the compute, which is a poor trade for bulk subtitling and a reasonable one
for a small high-value corpus.

Revised model guidance, superseding the table above:

| use | model | why |
|---|---|---|
| bulk subtitles, offline files | `parakeet-tdt` | 195.8 RTFx, 1.931%, matches published |
| accuracy-first offline | `parakeet-ctc` | best NeMo WER at 1.852%, untuned; raise `chunk_size` for long files |
| streaming / full-duplex | `nemotron-en --stream` | 33.6 RTFx, low host and GPU memory |
| non-English | `nemotron-3.5 --language <code>` | 40+ locales; the only one taking word boosting for code-switching |
| small high-value corpus | audio-flamingo-3 | 1.489%, best measured, ~40× the cost |

Note that **Parakeet TDT ignores word boosting** entirely, so multilingual
code-switching work has to go through nemotron-3.5 or flashlight CTC.

## Still outstanding

- **Tuned CTC.** Flashlight beam search is compiled in (`doctor` lists it,
  `kenlm.dll` ships) and the OpenSLR 3-gram LM is in `D:/HuggingFace/lm`, but
  its lexicon is phonetic ARPAbet and flashlight needs word-to-subword mappings
  for a SentencePiece model. The lexicon must be regenerated from the model's
  own tokenizer before this can run.
- **`chunk_size` sweep** for CTC on long files, per the 25× figure above.
- **test-other**, where these models actually separate.
- **Offline Nemotron WER**, blocked by host RAM in this box's current state.

---

# Addendum 2: patched vs stock ggml — it is one flag, not the series

Thu, 03 Sep 2026

`NEMO_SPEECH_GGML_PATCHED` defaults to **ON**, so `build-cuda` was already fully
patched and every number above was measured on the patched path. The comparison
build is therefore the *stock* one, not the other way round.

Three builds, 45 minutes of podcast, best of two runs each:

| build | parakeet-tdt (offline) | nemotron-en (`--stream`) |
|---|---|---|
| patched, all options ON | 48.9 s · RTFx 55.2 | **75.5 s · RTFx 35.7** |
| **patched, `FUSED_RELPOS_ATTN=OFF`** | **22.7 s · RTFx 118.9** | 108.8 s · RTFx 24.8 |
| stock, pristine ggml, all OFF | 26.0 s · RTFx 103.9 | 125.2 s · RTFx 21.6 |

Word counts agree within 0.1% across all three (8272-8275 offline, 8187-8197
streaming), so this is speed only.

## The whole effect is `NEMO_SPEECH_FUSED_RELPOS_ATTN`

Turning that one option off is worth **2.15× on the offline path** and costs
**44% on the streaming path**. Nothing else in the series accounts for a
meaningful share. Ruled out by direct measurement on the patched build:

| switch | offline result |
|---|---|
| `GGML_SKINNY_Q8=0` | 47.4 s — 3% |
| `GGML_CUDA_DISABLE_GRAPHS=1` | 52.8 s — worse |
| `GGML_CUDA_DISABLE_FUSION=1` | 48.6 s — unchanged |

This is exactly what patch 0001 documents about itself: the kernel is tuned for
the cache-aware streaming shape (`d_k=128`, `q=2`, `kv=72`) with an occupancy
query choosing between one and two queries per block, and "other CUDA shapes
retain the generic fused kernel". A 45-minute offline full-context pass is as
far from a two-frame streaming window as the workload gets, and there the
generic kernel loses to stock's batched cuBLAS. The option is doing precisely
what it was built for; it is simply on by default for a workload it was not
built for.

## Stock is never the right build

`patched -relpos` beats fully stock offline (118.9 vs 103.9) *and* patched-all-on
beats stock streaming (35.7 vs 21.6). The other fifteen patches earn their keep
on both paths. There is no workload here where a pristine ggml build wins, so
the stock tree is a bisection tool rather than a deployment artifact.

## What the commands use

Two builds that differ in exactly one CMake option:

| build | option | used by |
|---|---|---|
| `NeMo-Speech.cpp/build-cuda` | `FUSED_RELPOS_ATTN=ON` | `vw live-subs` (streaming) |
| `nemo-stock/build-norelpos` | `FUSED_RELPOS_ATTN=OFF` | `vw better-subtitles` (offline) |

`Start-BetterSubtitles.ps1` searches `build-norelpos` before `build-cuda`;
`utils/nemo_asr.py` searches `build-cuda` first. Both fall back to the other, so
a missing build degrades in speed rather than failing.

Reproducing the bisect build: copy the tree, apply the patch series from
**inside `ggml/`** (the patches are relative to that directory, not the repo
root — applying them from the top fails with `src/ggml.c: No such file`), then
configure with `-DNEMO_SPEECH_GGML_PATCHED=ON -DNEMO_SPEECH_FUSED_RELPOS_ATTN=OFF`.
`scripts/windows/build.ps1 -Backend cuda` always applies the patches and never
passes either flag, so it needs a local edit for anything but the default.

## Worth reporting upstream

A default-ON option that costs 2.15× on offline recognition is a reasonable
issue to file, and it is cleanly reproducible: three builds one flag apart, one
input, matching transcripts. The fix is probably a shape check that falls back
to the unfused path when the query count is large, rather than a default change.

## Note on batching

`docs/development/asr-batching.md` describes microbatching across *concurrent
streams through one model*, not across models: only work with the same
graph-shaping dimensions and options can share a microbatch, each caller keeps
its own stream and decoder state, and stateful caches live in indexed device
rows bounded by `state_arena_slots` (exhausting it rejects new stateful work
rather than reusing another stream's state). Backend submission stays serialized
behind a compute mutex, so throughput comes from fewer and wider submissions.

It is therefore unlikely to address the host-RAM aborts in Addendum 1: that is a
host allocation in `ggml_init`, and batching bounds device state. Untested.

---

# Addendum 3: the fused-attention option, shaped instead of disabled

Thu, 03 Sep 2026

Addendum 2 concluded that `NEMO_SPEECH_FUSED_RELPOS_ATTN` had to be traded
between the offline and streaming paths, and that two builds were needed. That
is no longer true, and the fix is smaller than the workaround.

## The two paths were never the same code

The fused op is emitted from three call sites, each behind the same `#ifdef`:

| file | path |
|---|---|
| `src/asr/encoder/rel_pos_attention.cpp` | offline / full context |
| `src/asr/encoder/fastconformer.cpp:909` | cache-aware streaming |
| `src/asr/encoder/cache_aware_encoder.cpp:268` | cache-aware streaming |

The compile-time option turns off **all three**, which is why disabling it fixed
offline and broke streaming. Gating one of them fixes offline and leaves
streaming alone.

`rel_pos_attention.cpp` already had both a fused and an unfused branch plus a
runtime gate (`use_fused = session->params.use_gpu`), so this is a condition,
not a kernel rewrite:

```cpp
const int64_t q_len = input_tensor.tensor->ne[1];
const bool use_fused = session->params.use_gpu && q_len <= fused_max_q;
```

`NEMO_SPEECH_RELPOS_MAX_Q` overrides the threshold (default 512, `0` disables
the fused path) so it can be swept without rebuilding.

## Measured

Parakeet TDT offline, 45 minutes, best of two, shaped build:

| `NEMO_SPEECH_RELPOS_MAX_Q` | wall | RTFx |
|---|---|---|
| 0 (fused off) | 21.8 s | 123.9 |
| 128 | 21.7 s | 124.1 |
| 512 (default) | 21.8 s | 123.9 |
| 999999 (always fused) | 49.0 s | 55.1 |

The 999999 row reproduces the ungated patched build to within 0.1 s, which is
what proves the gate is the only variable and that this call site really is the
offline hot path. The threshold value itself barely matters: an offline query
length is orders of magnitude past any of these, so 512 is a safe default rather
than a tuned one.

nemotron-en `--stream`, same build:

| configuration | wall | RTFx |
|---|---|---|
| `MAX_Q=512` (default) | 79.3 s | 34.0 |
| `MAX_Q=0` | 80.4 s | 33.6 |
| ungated patched build | 81.0 s | 33.3 |

All three within noise — the gate does not reach the streaming call sites, as
intended.

## One build, both paths

| build | offline RTFx | streaming RTFx |
|---|---|---|
| patched, ungated | 55.1 | 33.3 |
| patched, `FUSED_RELPOS_ATTN=OFF` | 118.9 | 24.8 |
| stock | 103.9 | 21.6 |
| **patched + query-length gate** | **~118-124** | **~34** |

Best or tied on both. The gated build also beats the `FUSED_RELPOS_ATTN=OFF`
build offline, because the compile-time switch also removed the fused op from
the shapes where it wins, while the gate only declines the large ones.

`build-cuda` in the main checkout now carries this (an incremental rebuild:
one translation unit and one relink, since the ASR code lives in
`nemo_speech_asr.dll`). Both `vw` commands point at it again; `build-norelpos`
and `build-stock` remain as fallbacks and as the bisection record.

## Caveat on the streaming numbers

Streaming measurements drifted between sessions -- the ungated patched build
read 75.5 s in Addendum 2 and 81.0 s here, about 7%. Both were best-of-two on
the same audio and binary, so treat streaming differences under ~10% as noise.
The offline effect is 2.2x and far outside that band.

## Worth upstreaming

A default-ON option costing 2.15x on offline recognition is a real issue, and
the fix is small and local: gate the offline call site on query length rather
than changing the default. The reproduction is three builds one flag apart plus
an env sweep, with matching transcripts throughout (8272-8275 words).
