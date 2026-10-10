# Segmentation audit + corrected step 2, Tue, 08 Sep 2026

Follow-up to [plurilingual-step1-results.md](plurilingual-step1-results.md),
prompted by the observation that segment boundaries were dropping and repeating
words. Several of those observations are confirmed; one premise is not.

## The WER table never used segmentation

**Step 2 transcribed the 155 original clips whole**, one file each, 5.8-14.7 s.
Segmentation was only ever used for step 1 (detection over the spliced file) and
for the French timeline. Boundary damage therefore cannot explain the 23.99%
WER — there were no boundaries in that run.

## ffmpeg is not losing audio

The 70 French segments reassemble to PCM **bit-identical** to the source
(22 275 552 bytes, md5 `1fa1964e…`, 0 frames lost). `ffmpeg -f segment` on WAV
is sample-exact and non-overlapping. Whatever is being dropped is dropped
downstream of it.

## What the boundaries actually do

There is no merging, overlap or stitching logic — not in the segmentation, and
none in the transcribe path. That is the problem rather than the fix: a word
straddling a hard cut has its first half in segment *i* and its second half in
*i+1*, and each half is decoded independently into a whole-word guess. The same
mechanism produces **both** symptoms at once — a repeat when both halves decode
to the same word, an omission when neither does. Measured: 2 of 69 boundaries
repeat a token outright, which is the visible tail of the effect.

The systematic cost is at segment heads:

| | head gap (first word) | tail gap (last word) |
|---|---|---|
| whole file (696 s) | 0.64 s, once | 0.03 s |
| 10 s segments (x70) | mean 0.83 s, median 0.72 s, max 3.52 s | -0.07 s |

Segmenting turns one 0.64 s warmup into **70 of them, ~58 s of dead head** on a
696 s file. Word count drops 2332 to 2309.

Two side findings: word timestamps overshoot the audio by ~1% (last word at
10.08 s on a 9.98 s file), and the negative tail gaps fall out of that, so the
timestamp scale itself is slightly wrong.

## The multi-second drops are not segmentation

The ~5 s drop after `residency.` at 6:52 is present in the **whole-file**
transcript too, as a 4.96 s inter-word gap, and the audio there is not silent
(RMS ~3400, same as surrounding speech). Same for the other long gaps. These are
the model or the decoder dropping speech, and they survive whichever way the
file is cut.

## A confound I introduced

`NEMO_SPEECH_RELPOS_MAX_Q` (default 512) routes short inputs to the fused
attention kernel and long ones to cuBLAS. A 10 s segment is ~125 encoder frames
and takes the **fused** path; the 696 s whole file takes **cuBLAS**. So the
whole-file and segmented runs were never comparing only segmentation.

Re-running the French segments with `NEMO_SPEECH_RELPOS_MAX_Q=0`: **10 of 70
segments change text**, and cuBLAS is the better transcript in each case seen —
`I also re relate` becomes `I also relate`, `download the aerolog using` becomes
`download the aerolog app using`, `Y va a trouver la recette ma hec` becomes
`Il va trouver la recette Mahick`. The handoff records this gate as "2.15x with
no effect on streaming"; that is true of streaming but **it is not
accuracy-neutral offline**.

Aggregate WER impact is small, under 0.6 points, so it does not explain the
headline numbers — but it does mean short-input runs are silently on a lossier
kernel.

## Corrected step 2

All 155 clips, whole, `--normalizer basic`, fused path disabled.

| condition | fused | cuBLAS | enspa (151) | spa (4) | empty |
|---|---|---|---|---|---|
| nemotron-3.5 `auto` | 23.99% | **23.87%** | 23.89% | 22.03% | 0 |
| nemotron-3.5 `es-US` | 27.40% | 27.19% | 27.25% | 22.03% | 1 |
| parakeet-tdt | 26.15% | 26.19% | 26.22% | 23.73% | 2 |
| nemotron-3.5 `en-US` | 45.61% | 46.23% | 45.61% | 100.00% | 10 |
| parakeet-ctc | — | 55.58% | 55.06% | 100.00% | 0 |

`parakeet-ctc` is not a candidate for this material: 55.58% overall and 100% on
the monolingual Spanish clips.

### auto vs es-US

The aggregate says `auto`, and so does the head-to-head: over 155 clips,
**auto better on 71, es-US better on 26, 58 ties**. That holds on both kernels
and on both subsets (`spa` is an exact tie at 22.03%). The es-US advantage is
real where it appears — `two thousand twenty seven` against auto's `twenty
twenty seven`, which is what the audio says — but it does not generalise to the
corpus.

Number formatting is confirmed **ASR-side**, not the scorer: the basic
normalizer passes both `twenty twenty seven` and `two thousand twenty seven`
through unchanged. It is also nearly irrelevant to WER here — the references
contain 5 digit tokens in 4999 words.

## Step 1 survives

The enspa and French runs used the **same** segmentation code, the same window,
the same model and the same flags, and returned 58.3% vs 12.9% empty
`languages`. A shared technique cannot produce a 45-point split, so the enspa
detection result is a property of the audio, not of how it was cut. The
"can't tag two languages inside one sentence, probably waits for an
end-of-turn" reading fits every number here.

## Next

1. Segment on silence (VAD) rather than fixed windows, so cuts land between
   words. That removes the per-segment head gap and the boundary word-splitting
   at once.
2. Decide whether `NEMO_SPEECH_RELPOS_MAX_Q=512` should stay the default given
   it changes offline output.
3. The multi-second drops on non-silent speech need their own investigation;
   they are the largest unexplained loss and they are not a segmentation bug.
