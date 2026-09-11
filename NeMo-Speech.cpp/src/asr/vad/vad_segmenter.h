// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Offline VAD segmentation configuration.
//
// This is a different job from the other two VAD consumers in this directory.
// VadMasker blanks non-speech *features* inside one decode; VadEndpointer
// decides when a *live stream* has stopped talking. Neither one splits a
// complete recording into independently decoded pieces, which is what the
// offline path needs.
//
// Why the offline path needs it. An RNN-T predictor is conditioned on the last
// emitted token, and a blank is not an emission, so the predictor state does
// not change across a run of blanks. A decoder that has committed to one
// language can therefore score blank on every subsequent frame and never leave
// that state on its own -- the run ends only when the acoustics alone outvote
// the stale prior, which on real code-switched audio can take seconds. Cutting
// at silence and re-entering with a blank predictor removes the failure mode,
// because the state that caused it cannot cross the boundary.
//
// Two properties of the cut matter and are easy to get wrong:
//
//   * Segments must PARTITION the audio -- contiguous, no overlap and no gaps.
//     The decode loop advances a cumulative frame offset per segment, so
//     overlapping segments would duplicate words and gaps would desynchronize
//     every timing after the first cut.
//   * The cut goes in the MIDDLE of a silent run, not at its edge. That leaves
//     each segment with silence at its head and tail, which is what the encoder
//     expects, and removes any need for explicit padding (which would overlap
//     and duplicate).
#ifndef NEMO_SPEECH_ASR_VAD_VAD_SEGMENTER_H
#define NEMO_SPEECH_ASR_VAD_VAD_SEGMENTER_H

#include "parameter_parser.h"

namespace nemo_speech::asr {

struct VadSegmenterCfg {
    // Off by default: enabling it changes transcripts on the offline path.
    // Requires a VAD model (--vad-model); without one this is inert.
    bool enable = false;

    // Speech/silence hysteresis on the Silero window probabilities. Onset is
    // deliberately the higher bar: a false "speech" only costs a missed cut,
    // while a false "silence" could cut mid-word.
    float onset = 0.5f;
    float offset = 0.35f;

    // A silent run must be at least this long to be worth cutting at. Too
    // small and cuts land in the pauses inside a sentence, throwing away
    // context the model uses; too large and long monolingual stretches never
    // get a boundary. 300 ms is a comfortable inter-sentence pause.
    float min_silence_ms = 300.0f;

    // A segment is closed at the last eligible silence before this length. This
    // is the real control: it bounds how long a stale predictor state can
    // survive, and it is also the upper bound on damage from one bad decode.
    float max_segment_sec = 30.0f;

    // Cuts closer together than this are skipped, so a burst of short pauses
    // does not shred the audio into fragments too small to decode well.
    float min_segment_sec = 2.0f;

    // What to do when max_segment_sec is reached and there was no pause to cut
    // at. Continuous speech really does occur -- on the clip that motivated
    // this work, 371 of 375 windows are speech and the longest silent run is
    // 64 ms, so a language switch happens mid-breath with nothing to cut at.
    //
    // false (default): cut at the quietest point but CARRY the decoder context
    //   across, which is the historical behaviour and never makes a transcript
    //   worse by itself.
    // true: also reset the context there. The cut lands at the lowest-energy
    //   window available, so it is the least-bad boundary rather than a good
    //   one -- this trades the risk of clipping a word against a hard bound on
    //   how long a stale predictor state can suppress audio. Turn it on when
    //   the input is known to be code-switched and dense.
    bool reset_on_forced_cut = false;

    void Register(common::ParameterParser& p) {
        p.Register(
            "enable", &enable, "Split offline audio at VAD silence and reset decoder context",
            {"--vad-segmentation"});
        p.Register("onset", &onset, "Segmenter speech onset threshold", {"--vad-seg-onset"});
        p.Register("offset", &offset, "Segmenter speech offset threshold", {"--vad-seg-offset"});
        p.Register(
            "min_silence_ms", &min_silence_ms, "Minimum silence to cut at (ms)",
            {"--vad-seg-min-silence-ms"});
        p.Register(
            "max_segment_sec", &max_segment_sec, "Maximum segment length (s)",
            {"--vad-seg-max-sec"});
        p.Register(
            "min_segment_sec", &min_segment_sec, "Minimum segment length (s)",
            {"--vad-seg-min-sec"});
        p.Register(
            "reset_on_forced_cut", &reset_on_forced_cut,
            "Reset decoder context even when a cut is forced mid-speech",
            {"--vad-seg-reset-forced"});
    }
};

}  // namespace nemo_speech::asr

#endif  // NEMO_SPEECH_ASR_VAD_VAD_SEGMENTER_H
