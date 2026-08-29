#!/usr/bin/env python3
"""Cut a PersonaPlex turn down to the part where somebody is actually talking.

PersonaPlex is a full-duplex model: it emits one output frame per input frame,
so its reply arrives inside a stream as long as whatever it was given, and the
reply occupies only part of it. Measured on a 15.86 s input holding a 6 s
question followed by silence, the output was:

    0.0 - 5.0 s   -73 dB   silent, because it is listening
    5.0 - 11.5 s  -23 dB   its answer, beginning as the question ends
    11.5 - 15.9 s -73 dB   silent again, because it has finished

Both silences carry real information and neither belongs in the conversation.
The leading one is the model's turn-taking latency and the trailing one is the
evidence it decided to stop -- both are reported rather than merely discarded,
because they are the two numbers that say whether the loop is behaving.

The gap between speech and silence here is fifty decibels, so the threshold is
not delicate and does not want tuning.
"""

from __future__ import annotations

import json
import sys
import wave

import numpy as np

FRAME_S = 0.02
# Speech sits near -23 dB and silence near -73 dB, so anywhere between is safe
# for finding *where* speech is. The end of an utterance is the delicate part:
# a word's final consonant and the breath after it trail off well under -50 dB,
# and cutting at the first quiet frame audibly clips the last word. The
# threshold sits low and the margin is generous, because keeping a fraction of a
# second of silence costs nothing and losing the end of a word is obvious.
SILENCE_DB = -58.0
MARGIN_S = 0.40


def read(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def write(path, audio, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def speech_bounds(audio, sr, threshold_db=SILENCE_DB):
    """(start, end) seconds of the speech, or None if the turn is all silence."""
    frame = max(1, int(FRAME_S * sr))
    n = len(audio) // frame
    if n == 0:
        return None
    frames = audio[:n * frame].reshape(n, frame)
    rms = np.sqrt((frames * frames).mean(axis=1))
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(np.maximum(rms, 1e-9))
    loud = np.flatnonzero(db > threshold_db)
    if not len(loud):
        return None
    return float(loud[0] * FRAME_S), float((loud[-1] + 1) * FRAME_S)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Trim a turn to its speech")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--threshold-db", type=float, default=SILENCE_DB)
    ap.add_argument("--margin", type=float, default=MARGIN_S)
    ap.add_argument("--skip", type=float, default=0.0,
                    help="ignore this many seconds at the start: the stretch during "
                         "which the other party was still talking")
    ap.add_argument("--json", action="store_true", help="report as JSON on stdout")
    args = ap.parse_args()

    audio, sr = read(args.src)
    # Being full duplex, the model talks *over* its input rather than waiting for
    # it -- so the reply proper is what it says after the other party stopped.
    # Dropping the overlapped stretch is what keeps alternating turns
    # alternating: without it a turn contains the answer *and* the running
    # commentary made during the question, each turn is nearly as long as its
    # input, and the conversation grows without bound (measured: 10.5s, 22.3s,
    # 33.0s, 43.5s, each one a reply window longer than the last).
    offset = max(0.0, args.skip)
    audio = audio[int(offset * sr):]
    total = len(audio) / sr
    bounds = speech_bounds(audio, sr, args.threshold_db)
    if bounds is None:
        # A turn with nothing in it is a real outcome -- the model declining to
        # speak -- so it is reported rather than treated as a failure.
        report = {"ok": False, "total": round(total, 3), "speech": 0.0,
                  "latency": None, "tail": None}
        print(json.dumps(report) if args.json else "  silent turn")
        return 2

    start, end = bounds
    a = max(0.0, start - args.margin)
    b = min(total, end + args.margin)
    write(args.dst, audio[int(a * sr):int(b * sr)], sr)

    # Speech running to within a hair of the end means the reply did not finish
    # -- it was cut off by the end of the audio it was given, not by the model
    # deciding to stop. The cure is a longer reply window, not a looser
    # threshold, so the two cases are reported apart rather than averaged into
    # one "tail" number a caller has to interpret.
    tail = total - end
    truncated = tail <= 2 * FRAME_S

    report = {
        "ok": True,
        "total": round(total, 3),
        "speech": round(b - a, 3),
        # How long the model waited before answering: its turn-taking latency.
        "latency": round(start, 3),
        # Silence left after finishing -- proof it endpointed itself.
        "tail": round(tail, 3),
        "truncated": truncated,
        "sample_rate": sr,
    }
    if args.json:
        print(json.dumps(report))
    else:
        note = "  [!] cut off by the end of the input" if truncated else ""
        print(f"  {report['speech']:.2f}s speech "
              f"(waited {report['latency']:.2f}s, left {report['tail']:.2f}s){note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
