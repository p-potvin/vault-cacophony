#!/usr/bin/env python3
"""What a voice sounds like, in numbers a person can read.

The CAM++ embedding is the thing that actually identifies a speaker, but it is
512 opaque floats: it can tell you two clips are the same person and it can
never tell you why, or let you ask "which voices in the corpus are low and
slow". These measurements can. They are not identification and they are not
meant to be -- pitch overlaps heavily between people and moves within one person
across a scene -- they are description, kept beside the embedding so the store
holds something interpretable next to something merely effective.

Alongside pitch and level it measures timing -- pauses (count, median length,
log-normal spread) and speaking rate from syllable nuclei (de Jong & Wempe,
Behavior Research Methods 2009) -- because rate and pausing are as much of what
makes a voice recognisably *that person* as pitch is, and they are exactly what
the synthesis side needs to sound like them.

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


# Pauses and rate. Silence is judged relative to the loud end of the clip, not
# an absolute level, so a quiet recording is not one long pause: de Jong &
# Wempe (2009) use 25 dB below the 99th-percentile intensity. A gap shorter than
# 150 ms is articulation (stop closures, word joins), not a pause -- the bottom
# of Campione & Veronis's "brief" population.
SILENCE_DB_BELOW = 25.0
MIN_PAUSE_S = 0.15
# A syllable nucleus is an intensity peak standing at least this far above the
# dip before it (de Jong & Wempe's 2 dB), and voiced.
NUCLEUS_DIP_DB = 2.0


def intensity_db(audio, sr=SR):
    """Frame intensity in dB, 10 ms hop, 32 ms frames -- aligned with `f0_track`."""
    frame, hop = int(FRAME_S * sr), int(HOP_S * sr)
    windows = _frames(np.asarray(audio, dtype=np.float32), frame, hop)
    if not len(windows):
        return np.zeros(0, dtype=np.float32)
    power = np.mean(windows * windows, axis=1)
    return (10 * np.log10(np.maximum(power, 1e-12))).astype(np.float32)


def _runs(mask):
    """(start, end) index pairs of the True runs in a boolean array."""
    if not len(mask):
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def timing(audio, sr=SR, f0=None):
    """Pause and speaking-rate descriptors: where the silences are, and how fast the speech between them is."""
    db = intensity_db(audio, sr)
    if len(db) < 10:
        return {}
    threshold = float(np.percentile(db, 99)) - SILENCE_DB_BELOW
    sounding = db > threshold
    idx = np.flatnonzero(sounding)
    if not len(idx):
        return {}
    first, last = int(idx[0]), int(idx[-1])
    inner = ~sounding[first:last + 1]
    min_frames = int(round(MIN_PAUSE_S / HOP_S))
    pauses = [(e - b) * HOP_S for b, e in _runs(inner) if e - b >= min_frames]
    span_s = (last - first + 1) * HOP_S
    phonation_s = max(span_s - sum(pauses), HOP_S)

    # Syllable nuclei: smoothed intensity peaks, high enough, prominent over the
    # preceding dip, and voiced.
    kernel = np.ones(5, dtype=np.float32) / 5
    smooth = np.convolve(db, kernel, mode="same")
    peaks = np.flatnonzero((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])) + 1
    voiced = None
    if f0 is not None and len(f0):
        voiced = ~np.isnan(np.asarray(f0))
    nuclei = 0
    last_peak = None
    for p in peaks:
        if smooth[p] <= threshold:
            continue
        if voiced is not None and (p >= len(voiced) or not voiced[p]):
            continue
        lo = last_peak if last_peak is not None else max(0, p - 30)
        dip = float(np.min(smooth[lo:p + 1]))
        if smooth[p] - dip >= NUCLEUS_DIP_DB or last_peak is None:
            nuclei += 1
            last_peak = p

    out = {
        "phonation_ratio": round(float(phonation_s / span_s), 3) if span_s else None,
        "pause_count": len(pauses),
        "pause_rate_per_min": round(float(len(pauses) / span_s * 60.0), 2) if span_s else None,
        "syllables_est": int(nuclei),
        # Speech rate counts the pauses, articulation rate does not; the gap
        # between them is how much of a speaker's slowness is hesitation.
        "speech_rate_sps": round(float(nuclei / span_s), 2) if span_s else None,
        "articulation_rate_sps": round(float(nuclei / phonation_s), 2),
    }
    if pauses:
        ms = np.asarray(pauses) * 1000.0
        out["pause_median_ms"] = round(float(np.median(ms)), 1)
        out["pause_p90_ms"] = round(float(np.percentile(ms, 90)), 1)
        # Pause length is log-normal, so its spread is reported in log space --
        # the parameter a pause model actually samples with.
        out["pause_log_sigma"] = round(float(np.std(np.log(ms))), 3) if len(ms) > 1 else None
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
    out.update({k: v for k, v in timing(audio, sr, f0).items() if v is not None})
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
