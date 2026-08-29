# #324 — FoxNoseTech/diarize port

Issue #324 (external reporter) asks for FoxNoseTech/diarize as an alternative
to pyannote. It is not a model port: it is ~1,350 lines of Python glue over
pip packages. The genuinely new work was one embedding model plus ~800 lines
of numerics.

## NOW — active work

Embedder, clustering, smoothing, pipeline, DER harness, CLI wiring and
word-aligned segment splitting all landed. **Automatic speaker counting is the open problem** (below). Not yet
done: WeSpeaker GGUF upload + CC-BY attribution, THIRD_PARTY_NOTICES entry,
docs/architecture.md section, session ABI, real-audio DER.

## What was reused vs. built

| stage | source |
|---|---|
| VAD | already in-tree (Silero) — not rebuilt |
| speaker embedding | **new**: WeSpeaker ResNet34-LM, `src/wespeaker.{h,cpp}` |
| clustering | **new**: `src/core/spectral_diarize.{h,cpp}` |
| temporal smoothing | **new**: `src/core/diarize_smooth.{h,cpp}` |
| orchestration | **new**: `src/core/foxnose_pipeline.{h,cpp}` |
| metric | **new**: `src/core/der.h` |

## Licensing

The repo stays MIT. Three separate questions, only one a constraint:

* **WeSpeaker weights are CC-BY-4.0**, not Apache-2.0 as FoxNose's README
  states. Weights are not code, so this does not touch our source licence —
  but redistributing the GGUF REQUIRES attribution in the model card and
  THIRD_PARTY_NOTICES.txt.
* **FoxNose and wenet-e2e/wespeaker code are Apache-2.0.** Apache is
  compatible with MIT but does not become MIT; copied expression would stay
  Apache and make the repo mixed-licence.
* **Clean-room keeps it uniformly MIT**, and here that was nearly free:
  `clustering.py` is a sequence of scikit-learn CALLS, not implementations, so
  there was nothing to translate. What was taken is the recipe and the tuned
  constants — ideas and parameters, not copyrightable expression. The upstream
  is imported by the reference dumper as an ORACLE only, never vendored.

## Acceptance

Bit-exact sklearn parity is unachievable and always would be: k-means++
seeding, GMM init and the ARPACK eigensolver all ride sklearn's RNG stream.
So the gates are known-answer unit tests plus DER, not label equality.

WeSpeaker embedder, per-stage vs the upstream oracle on samples/jfk.wav:

| stage | cos_mean |
|---|---|
| fbank | 0.999999 |
| stem / layer1-4 | 0.99997 - 0.999995 |
| stats | 0.999999 |
| embedding | 0.999997, cosine(emb, ref) = **0.99999747** |

Clustering + smoothing + pipeline: **375 assertions over 57 hermetic cases**
(no model, no audio, no network), including an end-to-end synthetic
2-speaker timeline at **DER 0.0**.

## ⚠ Real-data benchmark — and a reversed conclusion

**8 VoxConverse dev files, human labels, 0.25 s collar, optimal 1:1 mapping.**
Pooled over 1109 s of scored reference speech:

| system | miss | FA | confusion | **DER** |
|---|---|---|---|---|
| upstream Python `diarize` 0.1.2 | 0.6 | 4.7 | 29.0 | **3.1 %** |
| this port, `bic` (upstream estimator) | 0.0 | 26.1 | 32.7 | **5.3 %** |
| this port, `eigengap` | 0.0 | 26.1 | 101.3 | **11.4 %** |

### The eigengap default was wrong and has been reverted

An earlier revision made eigengap the default on the strength of five
synthetic blob configurations (5/5 exact vs BIC's 4/5) and one 31.5 s clip.
Both were unrepresentative — `samples/multispeaker.wav` is the same speech
re-read by different speakers, an unusually easy case — and the real benchmark
reversed the verdict.

Eigengap systematically UNDER-counts on real speech:

    reference speakers   4  7  2  5  5  4  4  5
    eigengap             3  5  2  3  3  2  2  3
    bic                  4  6  3  5  4  3  2  5

and the confusion term triples. It is retained behind
`CRISPASR_DIARIZE_COUNT=eigengap` — it is genuinely better on well-separated
data and costs less — but it is NOT the default, and synthetic evidence must
not be used to make it one again. The unit test now pins the default.

### With VAD, this port is at parity with upstream

The first benchmark handed our pipeline WHOLE FILES as a single speech region
while upstream ran Silero VAD first, charging us 26.1 s of false alarm the real
CLI path never incurs (it takes the caller's ASR/VAD segments). Re-run with the
same Silero VAD and upstream's parameters (threshold 0.45, min speech 200 ms,
min silence 50 ms, pad 20 ms):

| system | miss | FA | confusion | **DER** |
|---|---|---|---|---|
| upstream Python `diarize` 0.1.2 | 0.6 | 4.7 | 29.0 | **3.07 %** |
| **this port, `bic` + Silero VAD** | 0.0 | 9.0 | **26.5** | **3.18 %** |
| this port, `bic`, no VAD | 0.0 | 26.1 | 32.7 | 5.27 % |

0.11 points apart, and our speaker CONFUSION is actually lower (26.5 s vs
29.0 s) — the residual gap is false alarm, not diarization. Estimated speaker
counts now differ on one file of eight:

    reference   4 7 2 5 5 4 4 5
    this port   4 6 2 5 4 4 2 5
    upstream    4 7 2 5 4 4 2 5

### Reproducing the benchmark

```bash
# 1. audio + human labels (one dev shard, 44 files; 8 used here)
python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download('diarizers-community/voxconverse','data/dev-00000-of-00005.parquet',
                repo_type='dataset', local_dir='voxconverse')
EOF
# 2. the upstream reference implementation, in its own venv
python -m venv foxvenv && ./foxvenv/bin/pip install diarize==0.1.2
# 3. score both with tools/der_score.py (0.25 s collar, optimal 1:1 mapping)
```

## Blueprint parity — measured end to end

The upstream Python pipeline (`pip install diarize==0.1.2`) was run on the same
`samples/multispeaker.wav` and compared against this port.

**With the speaker count pinned to 2 on both sides**, the speaker assignment is
identical and the boundaries agree within ~1 s:

| upstream Python | this port |
|---|---|
| 0.30-10.60 SPEAKER_00 (3 pieces, VAD-split) | 0.00-10.50 SPEAKER_00 |
| 11.50-15.60 SPEAKER_01 | 10.50-15.90 SPEAKER_01 |
| 16.30-26.60 SPEAKER_00 (3 pieces) | 15.90-26.70 SPEAKER_00 |
| 27.50-31.50 SPEAKER_01 | 26.70-31.50 SPEAKER_01 |

Scored with the DER harness (0.25 s collar, optimal 1:1 mapping), treating the
upstream output as reference:

    missed        0.00 s
    false alarm   1.05 s
    confusion     0.00 s
    DER           3.93 %

**Zero speaker confusion**: wherever both assign a speaker, they agree. The
entire residual is false alarm, and it is explained — upstream runs its own
Silero VAD and drops silence gaps, while this port tiles the caller's speech
regions contiguously. That is a difference in where the speech segmentation
comes from, not a diarization disagreement.

**On automatic counting this port is better than upstream.** On the same clip
upstream emits 11 speakers across 25 segments (its default `max_speakers=20`);
this port emits 2. The gated `CRISPASR_DIARIZE_COUNT=bic` path reproduces
upstream's failure mode (7-8 speakers), which is what confirms the port is
faithful — the improvement comes from the eigengap switch, not from a
divergence in the shared parts.

### What would still settle it properly

A DER number on labelled audio. There is none in the repo and none in
`cstr/crispasr-regression-fixtures` — this was checked. VoxConverse dev (what
upstream benchmarks on) needs a ~1-2 GB audio download plus RTTM wiring, which
is its own task. Until then the honest recommendation is a **conservative
`--diarize-max-speakers` default (4-6, not 20)**, and pinning
`--diarize-num-speakers` when the count is known.

## Wiring

`--diarize-method foxnose --diarize-embedder <wespeaker.gguf>`, with
`--diarize-max-speakers` / `--diarize-num-speakers`.

The method plugs into the existing `crispasr_diarize_segments` contract: the
caller's segments ARE the speech regions (they come from ASR/VAD upstream), so
FoxNose deliberately runs no VAD of its own — re-segmenting would duplicate
work and desynchronise labels from the segments they attach to.

Segment splitting IS implemented (`split_segments_on_foxnose_turns`): the
method now returns its derived turns through `crispasr_diarize_segments`'s
`out_turns`, and the CLI splits any caller segment spanning several speakers
at word-aligned boundaries. It reuses the same `group_words_into_speaker_runs`
grouping and sub-segment emission as the pyannote splitter — only the per-word
labelling differs (turn-interval lookup instead of posterior scoring).
Measured effect on `samples/multispeaker.wav`: before, both ASR segments
collapsed to `(speaker 0)`; after, the first slice correctly reads
0 -> 1 -> 0 across the ~11 s boundary.

### Cross-slice speaker identity (fixed)

Per-slice diarization cannot give consistent identities: each slice clusters
independently and restarts numbering at 0, so `speaker 0` in one slice is a
different person from `speaker 0` in the next. On `samples/multispeaker.wav`
(2 slices) the final turn came out `speaker 0` where the whole-file pipeline
says SPEAKER_01.

Fixed by running FoxNose in ONE global pass after transcription
(`crispasr_apply_foxnose_global`), using the final segment list as its speech
regions. The pyannote path solves the same problem with a pre-computed
posterior cache (#107); FoxNose does it afterwards instead, because it needs
the segments as speech regions and they do not exist beforehand. The per-slice
path stands down via `params.diarize_foxnose_global`, so the embedder is
loaded once rather than per slice.

Result on `samples/multispeaker.wav` — CLI output now matches the whole-file
pipeline's turns exactly:

| CLI | pipeline truth |
|---|---|
| 0.28-10.84 speaker 0 | 0.00-10.50 SPEAKER_00 |
| 11.64-15.88 speaker 1 | 10.50-15.90 SPEAKER_01 |
| 17.04-26.80 speaker 0 | 15.90-26.70 SPEAKER_00 |
| 26.52-31.52 speaker 1 | 26.70-31.50 SPEAKER_01 |

Note this runs on the `crispasr_run` unified path. The legacy `cli.cpp`
whisper path still diarizes per slice; it is the same fallback situation the
pyannote cache has there.

## Env gates

| var | effect |
|---|---|
| `CRISPASR_DIARIZE_BIC_WINDOW=1` | score silhouette only in `[k-2, k+3]` around the BIC anchor instead of the full `[min,max]` range (the default) |
| `CRISPASR_WESPEAKER_BENCH=1` | per-stage embedder timings |
| `CRISPASR_WESPEAKER_DEBUG=1` | embedder diagnostics |

## Performance — what was measured, and what not to bother trying

All numbers: M1, Release, `esrit.wav` (215.1 s, 3 VAD regions, 352 embedding
windows), through the real `crispasr --diarize-method foxnose` path. The
machine was noisy early on, so every A/B below was re-run interleaved on a
quiet machine; the paired numbers repeat to ~1%.

Where the time goes on a 215 s file (~45 s wall, ~4.8x realtime):

| stage | share |
|---|---|
| WeSpeaker ResNet34 forward | ~70% (94% of the embedder) |
| speaker-count estimation (GMM/BIC + silhouette) | ~12% (58 s -> 51 s when `--diarize-num-speakers` pins it) |
| Kaldi fbank | ~4% |
| VAD, clustering, smoothing, I/O | remainder |

### Levers that were tried and LOST — do not re-litigate without new evidence

| lever | result |
|---|---|
| **Metal / GPU** (`use_gpu = true`) | **2.0x SLOWER** — 89.6 / 89.6 s vs 45.1 / 44.6 s CPU, interleaved. The graph is submitted 352 times, once per 1.2 s window, and each one is tiny (80 mels x ~120 frames). Per-dispatch overhead and the host<->device round trip swamp the conv work. This is why `crispasr_diarize.cpp` deliberately does NOT set `cp.use_gpu`. It would likely flip if the windows were batched — see below. |
| **F16 conv kernels** | **2.2x SLOWER** — 297 vs 133 ms/window. Correctness is fine now (our fork's `00285218` removed the `src1->type == F32` assert; cosine 0.99999724 vs the oracle) and the GGUF drops 23.9 -> 13.3 MB, but ggml's CPU conv path is far slower on F16 input. The converter keeps 4-D kernels at F32 and says so. |
| **Persistent graph + `gallocr`** (the `bananamind_tts` / `beat_this` trick) | **~0.1% available.** Instrumented build/alloc/compute per call: **0.050 ms build, 0.049 ms alloc, 103.430 ms compute.** There is no per-call overhead to remove. The graph is already rebuilt-and-thrown-away for free. |
| **More threads** | wash. `-t 4` 44.72 / 44.58 s vs `-t 8` 45.05 / 44.60 s. The M1's 4 E-cores contribute nothing; the default `n_threads = 4` is already right. |

BLAS is not a factor on this path: with `use_gpu = false` the embedder runs on
a plain `ggml_backend_cpu_init()` backend and never schedules to the BLAS
backend, even though the binary reports `backends: cpu,metal,blas`.

### The lever that is actually left: stop embedding the same audio twice

`kEmbeddingWindowSeconds = 1.2` with `kEmbeddingStepSeconds = 0.6` means every
sample except the first and last half-window goes through ResNet34 **twice** —
352 windows x 1.2 s = 422 s of audio pushed through the net for 211 s of
speech, exactly 2x by construction.

ResNet34 here is fully convolutional in time, so in principle the net could run
**once per VAD region** and TSTP could then pool over each window's slice of
the layer4 time axis. That is close to a 2x cut of the dominant 70%.

It is NOT a free refactor and must not be assumed equivalent:

* conv padding at a *region* edge is not the padding at a *window* edge, so
  embeddings near boundaries will change;
* layer4 is downsampled 8x in time, so a 1.2 s window is only ~15 frames there
  and window boundaries no longer land on frame boundaries;
* it changes memory from O(window) to O(region), which matters on long files.

So it needs a DER run on the VoxConverse shard (see the benchmark section
above), not a cosine check. A cheaper variant with the same shape: batch N
windows into one graph along `ne[3]` — the `parakeet` / `nemotron` batched
decode trick — which changes no arithmetic at all, improves CPU GEMM shapes,
and is the change most likely to make the GPU path win instead of lose.
