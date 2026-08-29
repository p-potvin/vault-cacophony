#!/usr/bin/env python3
"""The tag store: what every pass over a file writes into, keyed by time.

One `<media>.tags.json` per media file. Each analysis pass owns one *track* --
`language` from the ASR, `speaker` from diarization, `emotion` or `accent` from
whatever comes next -- and writes only that track, leaving the others alone. A
pass can therefore run late, run again, or not run at all, and the file stays
valid either way.

Tracks are independent spans rather than fields on a cue, because the passes do
not agree on boundaries and never will: a speaker turn runs across cue breaks, a
language tag covers a sentence, a cue is sized to be read in the time it is on
screen. Forcing them into one row means the first pass to write decides the
segmentation for all the others. Consumers intersect instead.

    {
      "schema": 1,
      "media": {"path": "episode.mkv", "duration": 1356.85},
      "tracks": {
        "language": {
          "source": {"pass": "asr", "model": "nemotron-3.5-asr-streaming-0.6b",
                     "written": "Sun, 23 Aug 2026 11:04"},
          "spans": [{"start": 0.88, "end": 4.40, "value": "en-US"}]
        }
      },
      "cues": [{"start": 0.88, "end": 4.40, "text": "Mister Quilter is..."}]
    }

`cues` is kept beside the tracks because it is what the tracks are usually
looked up against, and because a consumer that wants "the French lines spoken by
speaker 2" should not have to open three files to get it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

SCHEMA = 1


def _stamp():
    # The house format: no epochs anywhere a human might read them.
    return datetime.now().strftime("%a, %d %b %Y %H:%M")


def load(path):
    """Read a store, or hand back an empty one shaped the same way."""
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("schema") == SCHEMA:
                data.setdefault("tracks", {})
                data.setdefault("cues", [])
                return data
        except (json.JSONDecodeError, OSError):
            pass  # A corrupt store is replaced, not repaired.
    return {"schema": SCHEMA, "media": {}, "tracks": {}, "cues": []}


def set_media(store, path=None, duration=None):
    media = store.setdefault("media", {})
    if path:
        media["path"] = os.path.basename(path)
    if duration:
        media["duration"] = round(float(duration), 3)
    return store


def put_track(store, name, spans, source_pass, model=None, **extra):
    """Replace one track. Whatever else is in the store is left untouched.

    `spans` is [{start, end, value, ...}] in seconds. Empty spans remove the
    track, so a pass that found nothing leaves no trace rather than an empty
    list a reader has to interpret.
    """
    tracks = store.setdefault("tracks", {})
    if not spans:
        tracks.pop(name, None)
        return store
    source = {"pass": source_pass, "written": _stamp()}
    if model:
        source["model"] = model
    source.update(extra)
    tracks[name] = {"source": source, "spans": spans}
    return store


def put_cues(store, cues):
    """cues: [(start, end, text)] or [{start, end, text}]."""
    out = []
    for c in cues:
        if isinstance(c, dict):
            out.append({"start": round(c["start"], 3), "end": round(c["end"], 3), "text": c["text"]})
        else:
            start, end, text = c[0], c[1], c[2]
            out.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    store["cues"] = out
    return store


def value_at(store, track, seconds, default=None):
    """The value of `track` covering `seconds`, or `default`."""
    spans = store.get("tracks", {}).get(track, {}).get("spans", [])
    for s in spans:
        if s["start"] <= seconds < s["end"]:
            return s["value"]
    return default


def collapse(spans):
    """Merge neighbouring spans that carry the same value.

    A per-cue language track is mostly one long run of the same code; collapsing
    turns three hundred spans into two and makes the file readable by a human,
    which is most of what it is for early on.
    """
    out = []
    for s in spans:
        if out and out[-1]["value"] == s["value"] and s["start"] - out[-1]["end"] < 5.0:
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(s))
    return out


def save(store, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=1)
    return path


def summary(store):
    """One line for the console: what this file knows about itself."""
    bits = []
    for name, track in sorted(store.get("tracks", {}).items()):
        values = {s["value"] for s in track["spans"]}
        bits.append(f"{name}={'/'.join(sorted(values)) if len(values) <= 3 else f'{len(values)} values'}")
    return ", ".join(bits)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect a tag store")
    ap.add_argument("path")
    ap.add_argument("--at", type=float, help="print every track's value at this time")
    args = ap.parse_args()
    store = load(args.path)
    media = store.get("media", {})
    print(f"{media.get('path', args.path)}  {media.get('duration', '?')}s  "
          f"{len(store.get('cues', []))} cues")
    for name, track in sorted(store.get("tracks", {}).items()):
        src = track["source"]
        print(f"  {name:10} {len(track['spans']):4d} spans  "
              f"[{src.get('pass')}{'/' + src['model'] if src.get('model') else ''}, {src.get('written')}]")
        if args.at is not None:
            print(f"             at {args.at}s: {value_at(store, name, args.at)}")
