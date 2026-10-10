# Plurilingual steps 1 and 2 — measured results, Tue, 08 Sep 2026

> **Superseded in part.** The step 2 table here was produced with the fused
> attention kernel; see
> [plurilingual-segmentation-audit.md](plurilingual-segmentation-audit.md) for
> corrected numbers, a parakeet-ctc result, and a segmentation audit. The step 1
> conclusion is unchanged.

Answers steps 1 and 2 of [HANDOFF-plurilingual.md](HANDOFF-plurilingual.md).

**Conclusion: do not build per-segment routing.** Step 1 shows the detection
signal is not usable at any segment length on this material, and step 2 shows
routing would not pay for itself even if it were — `--language auto` already
beats both pinned languages.

---

## Step 1 — detection on real code-switched audio

155 `enspa_dev` clips spliced in TSV row order (turn order, confirmed by ear)
into one 35.9 min file, segmented, each segment through nemotron-3.5
`--language auto`.

| segment | n | empty `languages` | exactly 1 | ≥2 | both en+es | outside en/es | longest empty run |
|---|---|---|---|---|---|---|---|
| 5 s | 431 | 214 (49.7%) | 204 (47.3%) | 13 (3.0%) | 11 (2.6%) | 15 (3.5%) | 6 (30 s) |
| 10 s | 216 | 126 (58.3%) | 79 (36.6%) | 11 (5.1%) | 9 (4.2%) | 5 (2.3%) | 9 (90 s) |
| 20 s | 108 | 62 (57.4%) | 33 (30.6%) | 13 (12.0%) | 12 (11.1%) | 3 (2.8%) | 9 (180 s) |
| 30 s | 72 | 41 (56.9%) | 18 (25.0%) | 13 (18.1%) | 11 (15.3%) | 1 (1.4%) | 5 (150 s) |

**There is no sweet spot.** The empty rate is flat at 50–58% across a 6×
range of window sizes. Longer windows do not recover the signal, they only
aggregate it — the rise in "≥2 languages" from 3.0% to 18.1% buys nothing for
routing, which needs to know *which span* is which language.

Answering the handoff's four questions:

- **How often is `languages` empty?** 58.3% at 10 s, and roughly that at every
  length tested.
- **Consecutive empties?** Up to 9 segments — 90 s at 10 s windows, 180 s at
  20 s. A sticky carry-forward rule would hold stale state for minutes.
- **Does it report the wrong language rather than none?** Yes. `pt-PT`, `vi-VN`,
  `sl-SI` and a literal `unk` all appear. Separately, the Spanish tag is not
  stable: 36 segments came back `es-ES` and 13 `es-US` on the same speakers, so
  a router keyed on `es-US` would silently miss the majority of Spanish spans.
- **Does carry-forward survive mid-sentence switching?** The question is moot at
  a 58% empty rate; only 4.2% of 10 s segments reported both languages actually
  present.

The handoff's warning was right: the earlier 6/8 detection result came from
synthesized clips with hard cuts and does not transfer.

### Where detection *does* work

The same model on `French Girl Reacts to Quebecois Canadian French.mp3`
(11.6 min, English narration interleaved with French clips) is a different
story: **12.9% empty**, longest empty run 20 s, whole-file `["en-US","fr-FR"]`,
and the 10 s timeline tracks the actual structure — narration tagged `en-US`,
played clips tagged `fr-FR`, switch segments tagged both. Transcript and full
timeline in [french-reacts-transcript.md](french-reacts-transcript.md). Two
spurious `vi-VN` tags.

So the failure is specific to **intra-sentential** switching, not to language
detection in general. Routing may well be viable for material where a speaker
holds one language for a stretch. It is not viable for enspa-style audio where
the switch happens mid-clause with no pause.

---

## Step 2 — WER with and without routing

155 clips individually, scored with `evaluate_openasr.py --normalizer basic`.
Both apostrophe variants reported, per the request to see whether apostrophe
handling matters here.

| condition | WER (as-is) | WER (apostrophes removed) | enspa (151) | spa (4) | empty predictions |
|---|---|---|---|---|---|
| nemotron-3.5 `--language auto` | **23.99%** | **24.10%** | 24.03% | 20.34% | 0 |
| nemotron-3.5 pinned `en-US` | 45.61% | 46.36% | 44.98% | 100.00% | 9 |
| nemotron-3.5 pinned `es-US` | 27.40% | 27.38% | 27.48% | 20.34% | 1 |
| parakeet-tdt (multilingual) | 26.15% | 26.14% | 26.18% | 23.73% | 2 |

**`auto` does not cost accuracy — it is the best condition.** This inverts the
premise the routing design rested on. `auto` beats the Spanish pin by 3.4 points
and the English pin by 21.6, and is the only condition that never returns an
empty transcript.

Pinning English is actively destructive: 100% WER on the four monolingual
Spanish clips and 9 empty transcripts overall. Whatever routing would have done,
sending a Spanish span to an English-pinned decode is worse than not routing.

**Apostrophe stripping does not matter on this material** — every condition
moves by ≤0.75 points, and not consistently in one direction. The `what's` →
`what s` concern was real in principle and negligible in practice. Recorded so
it does not get re-litigated.

These numbers are comparisons between conditions scored identically, not
absolute WERs; the basic normalizer is not the leaderboard's English path and
these are not leaderboard-comparable.

---

## What this means for step 3

Step 3 (`-EnglishOnly` and per-segment routing) should not be built as designed.
`nemotron-3.5 --language auto` is the recommendation for plurilingual material:
best WER, no empty transcripts, no routing machinery, no detection dependency.

Still open, and now the useful question: whether routing helps on
**inter-sentential** material like the French file, where detection is reliable.
That is a different experiment from the one the handoff specified, and it needs
reference transcripts that this repo does not currently have for such material.
