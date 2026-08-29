#!/usr/bin/env python3
"""What a voice sounds like, in numbers a person can read.

The CAM++ embedding is the thing that actually identifies a speaker, but it is
512 opaque floats: it can tell you two clips are the same person and it can
never tell you why, or let you ask "which voices in the corpus are low and
slow". These measurements can. They are not identification and they are not
meant to be -- pitch overlaps heavily between people and moves within one person
across a scene -- they are description, kept beside the embedding so the store
holds something interpretable next to something merely effective.

All of it is numpy over audio that is already in memory by the time the speaker
pass runs, so it adds no model, no dependency and no meaningful time.
"""

from __future__ import annotations

import numpy as np

SR = 16000
# 60-400 Hz spans a deep male voice to a raised female one. Wider invites
# octave errors at both ends, and the tails are not where the information is.
F0_MIN, F0_MAX = 60.0, 400.0
FRAME_S = 0.032
HOP_S = 0.010
# Autocorrelation peak below this is noise finding a pattern in itself.
VOICED_R = 0.35


def _frames(audio, frame, hop):
    if len(audio) < frame:
        return np.zeros((0, frame), dtype=np.float32)
    n = 1 + (len(audio) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return audio[idx]


def f0_track(audio, sr=SR):
    """Fundamental frequency per frame, NaN where the frame is not voiced.

    Plain normalised autocorrelation. It is the oldest pitch detector there is
    and it is enough here: we want the distribution over a whole turn, not a
    per-frame contour good enough to resynthesise, and the median of a few
    hundred frames is unbothered by the ones it gets wrong.
    """
    frame, hop = int(FRAME_S * sr), int(HOP_S * sr)
    windows = _frames(np.asarray(audio, dtype=np.float32), frame, hop)
    if not len(windows):
        return np.zeros(0, dtype=np.float32)
    windows = windows - windows.mean(axis=1, keepdims=True)
    energy = np.sum(windows * windows, axis=1)

    lag_min, lag_max = int(sr / F0_MAX), min(int(sr / F0_MIN), frame - 1)
    out = np.full(len(windows), np.nan, dtype=np.float32)
    if lag_max <= lag_min:
        return out

    # One FFT-based autocorrelation for every frame at once; the loop version of
    # this was the only slow thing in the pass.
    size = 1 << int(np.ceil(np.log2(2 * frame)))
    spectrum = np.fft.rfft(windows, size, axis=1)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), size, axis=1)[:, :lag_max + 1]

    band = corr[:, lag_min:lag_max + 1]
    lags = np.argmax(band, axis=1) + lag_min
    peaks = band[np.arange(len(band)), lags - lag_min]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(energy > 0, peaks / energy, 0.0)
    voiced = (r >= VOICED_R) & (energy > 1e-8)
    out[voiced] = sr / lags[voiced]
    return out


def measure(audio, sr=SR):
    """A dict of descriptors for one stretch of speech.

    Pitch is reported in semitones as well as hertz because hertz is not the
    scale hearing works on: 100->120 Hz and 200->240 Hz are the same interval
    and differ by 20 against 40 in hertz. A spread quoted in semitones compares
    across voices; one quoted in hertz flatters low ones.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < int(0.1 * sr):
        return {}

    rms = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))
    out = {
        "seconds": round(len(audio) / sr, 3),
        # dBFS, so 0 is a full-scale square wave and speech sits around -25.
        "rms_db": round(20 * np.log10(max(rms, 1e-9)), 2),
        # Peak over RMS. Speech runs 10-18 dB; a much lower number means the
        # audio was compressed or clipped, which is a property of the recording
        # rather than the person and worth knowing before comparing loudness.
        "crest_db": round(20 * np.log10(max(peak, 1e-9) / max(rms, 1e-9)), 2),
    }

    f0 = f0_track(audio, sr)
    voiced = f0[~np.isnan(f0)]
    out["voiced_ratio"] = round(float(len(voiced)) / max(len(f0), 1), 3)
    if len(voiced) >= 10:
        semitones = 12 * np.log2(voiced / 55.0)  # 55 Hz = A1, an arbitrary floor
        out["f0_median_hz"] = round(float(np.median(voiced)), 1)
        out["f0_p10_hz"] = round(float(np.percentile(voiced, 10)), 1)
        out["f0_p90_hz"] = round(float(np.percentile(voiced, 90)), 1)
        # Interquartile, not full range: one octave-halving error at either end
        # would otherwise set the number by itself.
        out["f0_iqr_semitones"] = round(
            float(np.percentile(semitones, 75) - np.percentile(semitones, 25)), 2)
    return out


if __name__ == "__main__":
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from voiceprint import read_wav

    ap = argparse.ArgumentParser(description="Measure a stretch of a 16 kHz mono wav")
    ap.add_argument("wav")
    ap.add_argument("--span", metavar="START:END", help="seconds")
    args = ap.parse_args()
    start, end = (float(x) for x in args.span.split(":")) if args.span else (None, None)
    for key, value in measure(read_wav(args.wav, start, end)).items():
        print(f"  {key:20} {value}")
