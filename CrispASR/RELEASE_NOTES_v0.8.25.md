# CrispASR v0.8.25

140 commits since v0.8.24. Two new backends — **GigaAM-v3** (Russian ASR) and
**foxnose** (speaker diarization) — but the through-line of this release is
measurement: several long-standing numbers turned out to be wrong, and finding
that out was worth more than any single fix.

The sharpest example: pyannote diarization has been mis-reading its own model
output since it was written. The segmentation head emits a **powerset** over
locally-active speakers, and our decode table had two entries transposed — so
every frame in which a *third* speaker was talking alone was credited to the
first two as if they were talking over each other. On VoxConverse dev that cost
**15 DER points**. It survived because every test used at most two speakers,
where the wrong table and the right one agree.

Drop-in from v0.8.24 — existing flags are unchanged.

## New — GigaAM-v3 Russian ASR

`ai-sage/GigaAM-v3`, all four revisions: CTC and RNN-T, each in a
character-level and an SPM (`e2e`) variant. Rotary Conformer encoder, and the
`e2e` heads emit punctuation and capitalisation natively.

```bash
crispasr -m gigaam-v3-e2e-rnnt-q4_k.gguf -f russian.wav
```

Reached cosine **1.000000 at every stage** against the PyTorch reference on the
first diff run, with a byte-identical transcript. Weights:
[`cstr/gigaam-v3-GGUF`](https://huggingface.co/cstr/gigaam-v3-GGUF).

The backend declares `CAP_PUNCTUATION_NATIVE` and reports Russian as its sole
language, so the auto-punctuation restorer stays out of the way and language
detection is skipped.

## New — `--diarize-method foxnose` (#324)

A full diarization pipeline with **no external binary and no Python**: WeSpeaker
ResNet34-LM embeddings over sliding windows → PCA + full-covariance GMM/BIC
speaker counting → Ng-Jordan-Weiss spectral clustering → Viterbi temporal
smoothing.

```bash
crispasr -m auto -f meeting.wav \
    --diarize --diarize-method foxnose --diarize-embedder auto
```

`--diarize-embedder auto` fetches a 24 MB GGUF
([`cstr/wespeaker-resnet34-lm-GGUF`](https://huggingface.co/cstr/wespeaker-resnet34-lm-GGUF),
⚠ CC-BY-4.0 weights — see `THIRD_PARTY_NOTICES.txt`). The embedding matches the
PyTorch oracle to cosine **0.99999747**. Scored the way the upstream reference
scores itself, the port is at parity: **3.18% DER against its 3.07%** on
VoxConverse dev, with lower speaker confusion (26.5 s vs 29.0 s).

Clean-room: the clustering half is standard published numerical linear algebra
and statistics; what came from upstream is the recipe and the tuned constants.

## Fixed — the pyannote powerset table had two entries swapped

`SPK_MASK` read class 3 as "spk0 + spk1" and class 4 as "spk2 alone". The head
enumerates subsets by increasing size, so 3 is **spk2 alone** and 4 is the first
**pair**. Every frame of the third local speaker was attributed to the wrong two
speakers.

Ground truth gives the layout away — the old table only holds if a speaker can
be active most of a file yet never once alone:

| file | GT overlap | as `{4,5,6}` (correct) | as `{3,5,6}` (old) |
|---|---|---|---|
| fsaal | 0.42% | 0.17% | 62.65% |
| jyirt | 0.09% | 0.00% | 28.32% |
| mesob | 28.84% | 34.05% | 0.00% |
| nnqfq | 14.40% | 10.14% | 48.40% |

Mean DER on that shard: **48.21% → 33.37%**. The layout now lives once in
`src/core/powerset.h`, derived rather than transcribed at both call sites, with
`tests/test-powerset.cpp` guarding the singleton/pair split and permutation
closure.

## Fixed — `--diarize-max-speakers` was picking the speaker count, not bounding it

The embedder path clustered with single-linkage agglomerative at a fixed 0.5
cosine threshold. Single linkage chains, and a fixed threshold does not adapt,
so the merge loop never reached the threshold and the **cap became the answer**
— it returned exactly 8 speakers on 4 of 8 files with 4–7 real ones. Now routed
through the same spectral estimator `foxnose` uses, which *estimates* the count.

`--diarize-cluster-threshold` still selects the old path, but only when you pass
it explicitly: the threshold means nothing to the estimator, so honouring its
default would have reinstated the bug.

**pyannote + TitaNet: 15.74% → 7.81% DER.**

That exposed a second defect in the estimator itself — the component ceiling is
`n/(pca_dim+1)`, so a short recording with few embedding windows could not
represent more than one component and silently returned "1 speaker". `pca_dim`
now drops until at least four components are representable.

## Faster — pyannote segmentation runs in parallel (#326)

Reported as diarization taking a 24-core box from 70× to 26.3× realtime.
`pyannote_seg_run` pushed the whole file through as **one sequence**, so the
four stacked BiLSTMs became a single 171k-timestep scan — and because the
recurrence is sequential it used exactly two threads regardless of `-t`.

Audio is now inferred in fixed 60 s chunks concurrently, each with 5 s of real
audio spliced either side and trimmed after. On a 2888 s file (M1, 8 cores):

| `-t` | before | after |
|---|---|---|
| 1 | ~50 s | 47.0 s |
| 2 | ~50 s | 25.2 s |
| 4 | ~50 s | 18.5 s |
| 8 | 49.8–56.1 s | **18.1 s** |

Chunking is decided by audio **length**, never by thread count, so `-t 1/2/4/8`
produce byte-identical posteriors. Accuracy does not regress — every chunked
setting beat the single scan on the dev shard. Tunable via
`CRISPASR_PYANNOTE_CHUNK_S` (0 restores the old single scan).

## Faster — speaker embedding

`wespeaker_context::n_threads` was stored and **never applied** — no
`ggml_backend_cpu_set_n_threads` call existed in the file, so every context ran
at ggml's default no matter what `-t` said. Fixed, and windows are now embedded
concurrently (one ggml thread each, sharing one copy of the weights) rather than
one at a time. Measured interleaved, threads × workers on an 85 s file: 4×1
10.27 s, 8×1 10.80 s, 2×4 8.67 s, **1×8 8.09 s**. On a 215 s file the
diarization stage goes 23.8 s → 19.2 s.

Also opt-in: `CRISPASR_DIARIZE_SPAN_EMBED=1` runs one network pass per span of
windows instead of one per window — **1.78× less diarization CPU for +0.30 mean
DER**. Off by default because accuracy is the better default for a diarizer;
`docs/cli.md#diarization` carries both numbers so the trade is informed.

## Fixed — CI was executing 1 unit test out of 162

The unit tier was configured but never actually ran on push. Turning it on
immediately surfaced two genuine pre-existing failures:

- **VAD fed a transposed view to a matmul.** `ggml_transpose` swaps `nb[0]/nb[1]`,
  so the row stride collapsed to `sizeof(float)` and `llamafile_sgemm`'s
  `ldb >= k` precondition was violated. Release defines `NDEBUG`, so the assert
  was compiled out and the matmul **ran anyway** with a violated precondition,
  producing plausible-looking segments. A/B verified byte-identical output after
  the fix.
- **ggml SVE used data from inactive lanes.** `svmad_f32_m` merges on the *first*
  operand, so inactive lanes took the zeroed predicated load and wiped what the
  leftover loop had accumulated. Only bites when `n >= epr && n % epr != 0` —
  ordinary LLM dims never hit it; `core_adaln`'s `dim=6` does. Fixed upstream in
  our ggml fork.

Several workflows also had checks that **could not go red** — `continue-on-error`
made `needs.result == 'success'` true for failed jobs. Audited and fixed across
every workflow, plus a new CI check that version files agree with `VERSION`.

## Fixed — Windows/MSVC build failure in glint (#327)

`glint/src/simd.hpp` called `__cpuidex` without including `<intrin.h>`; GCC and
Clang take the `__builtin_cpu_supports` branch, so the path was never exercised
off Windows. Fixed upstream in `CrispStrobe/glint` first and re-synced, because
`tools/sync-glint.sh` overwrites `glint/` from upstream and an in-tree-only fix
would have been reverted by the next sync.

Two further bugs in the same six lines: it tested the **AVX2** bit to gate a
code path that only uses `_mm256_*_pd` (AVX1 — so every Sandy/Ivy Bridge part
fell back to SSE2 under MSVC while GCC ran it with AVX), and it never checked
`OSXSAVE`/`XGETBV` before enabling AVX.

## Fixed — sherpa diarization hung indefinitely on Windows (#328)

The subprocess was spawned with **single-quoted** arguments, which `cmd.exe`
does not interpret — sherpa received literal quote characters in the model
paths — and with `2>/dev/null`, which is not valid Windows redirection. Now
spawned via `CreateProcessA` with correct MSVC quoting, stderr to `NUL`, and a
timeout.

The POSIX branch accepted the same `timeout_sec` parameter and **ignored it**,
still using `popen`, so the identical hang was reachable on Linux and macOS
behind a signature that said otherwise. Now `fork`/`exec` + `poll(2)` against a
deadline. `CRISPASR_SHERPA_TIMEOUT_SEC` overrides; the default is
`max(120, audio_seconds × 5)`.

## Fixed — whisper vocabularies with unserialized special tokens (#322)

A model can ship a vocab where the special-token ids are half-written. The
legacy id path silently produced wrong ids rather than failing. Restructured so
the half-serialized state is **unrepresentable**, with
`tests/test-whisper-special-tokens.cpp` pinning it. On our fixtures the bug was
latent — it did not change decode — which is exactly why it needed a test rather
than a spot check.

## Also in this release

- **Static builds link again (#316).** `crispasr_cache` moved to `crispasr-core`
  so a `BUILD_SHARED_LIBS=OFF` build of `crispasr-diff` resolves it — the static
  linker discards an archive member before the reference that needs it.
- **Split weight buffers sized by backend allocation, not `ggml_nbytes`.** The
  two differ when a backend pads, and the shortfall corrupted the tail.
- **`stb_vorbis` freed uninitialised pointers** on a truncated comment header.
- **ggml synced to v0.17.0**, with a TTS→ASR roundtrip gate for the pin and GPU
  validation.
- **Binding parity (#321)** — Rust, C#, Ruby, Java and WASM gained
  `speechToSpeech`, `synthesizeRaw`, `acceptMarkingResponsibility` and
  `inputSampleRate`; the parity rule is now documented so surfaces stop drifting.
- **`--diarize-method moss-diarize` no longer truncates its tagged transcript log
  to 200 characters (#318).**
- **dots-tts fast CFG and low-latency ODE profiles (#319).**
- **piper honours `--seed`**, so synthesis is reproducible.
- **CosyVoice3 honours the `default`/`auto` voice sentinel.**
- **LID classifies VAD speech for long server uploads.**
- Diarization knobs documented in `docs/environment-variables.md`; foxnose added
  to `README.md` and `docs/cli.md#diarization`.

## ⚠ Calibration — earlier diarization DER figures came from an easy subset

Numbers near 3–7% DER quoted in this project's history came from an 8-file
VoxConverse subset that turned out to be unrepresentative. Identical code scores
**7.3% there and 33.1%** on a 40-file sample of the full dev set, where speaker
counts reach 20.

On that fuller corpus the speaker count is exactly right on **18/40 (45%)** of
files. Estimating *how many* speakers are present is the weak link — not the
embeddings, which match their oracle to cosine 0.99999747. Pass
`--diarize-num-speakers N` when you know the count.

A reusable harness now exists for this: `tools/voxconverse_extract.py` builds
the corpus from HuggingFace parquet, and `tools/diarize_eval.py` reports
**speaker-count accuracy and DER** over a stable tune/holdout split. It was
built after DER alone proved unable to see the failure — one file predicts 2
speakers against a true 5 and still scores 6.88% DER.

## Known gaps

- **NME-SC speaker counting is implemented but loses.** Auto-tuning the affinity
  binarisation (`CRISPASR_DIARIZE_COUNT=nme-sc`) does cut under-counting 15 → 11,
  the failure it was chosen for, but converts those into over-counts 7 → 12:
  count exact 17/40 vs BIC's 18/40, DER 39.26% vs 33.06%. It relocates the error
  rather than removing it. Default unchanged, and the holdout split remains
  unspent.
- Speaker counting at 45% exact is the open problem in diarization. It is a
  criterion problem, not a threshold one — three of four failures are
  under-counts by wide margins.
- The legacy `cli.cpp` whisper path still diarizes per slice; the unified
  `crispasr_run` path does not.
