#!/usr/bin/env python3
"""Write a conversation's turns into a tag store.

A generated conversation is media like any other, so it gets the same shape as
a film does rather than a bespoke log format: cues for what was said, a
`speaker` track for who said it. That means every consumer already written --
the tag store CLI, anything that intersects tracks -- works on it unchanged, and
the diarization pass can later be run over the same audio and compared against
this, which is ground truth rather than an estimate.

Reads a turns JSON written by Start-Conversation.ps1:

    [{"agent": "A", "voice": "NATF2", "wav": "turn001.wav",
      "text": "...", "start": 0.0, "end": 4.2}, ...]
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tagstore


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Write conversation turns into a tag store")
    ap.add_argument("--turns", required=True, help="turns JSON")
    ap.add_argument("--out", required=True, help="tag store to write")
    ap.add_argument("--media", help="the stitched conversation wav")
    ap.add_argument("--model", default="personaplex-7b-v1")
    args = ap.parse_args()

    with open(args.turns, encoding="utf-8") as f:
        turns = json.load(f)
    if not turns:
        print("  no turns")
        return 1

    store = tagstore.load(args.out)
    tagstore.set_media(store, args.media, turns[-1]["end"])
    # One cue per turn. A turn is not a subtitle cue -- it is far too long to
    # read on screen -- but it is the unit that was actually generated, and
    # splitting it here would invent boundaries the model never expressed.
    tagstore.put_cues(store, [(t["start"], t["end"], " ".join(t.get("text", "").split()))
                              for t in turns])
    tagstore.put_track(store, "speaker",
                       tagstore.collapse([{"start": t["start"], "end": t["end"],
                                           "value": t["agent"]} for t in turns]),
                       "conversation", model=args.model)
    # The voice each agent spoke in is a property of the generation, not
    # something inferred from the audio, so it is its own track rather than a
    # detail buried in the speaker one.
    tagstore.put_track(store, "voice",
                       tagstore.collapse([{"start": t["start"], "end": t["end"],
                                           "value": t.get("voice", "?")} for t in turns]),
                       "conversation", model=args.model)
    tagstore.save(store, args.out)
    print(f"    {len(turns)} turns -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
