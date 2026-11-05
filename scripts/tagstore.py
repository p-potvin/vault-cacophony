#!/usr/bin/env python3
"""The tag store: what every pass over a media file writes into, keyed by time.

The tags live in the file's main sidecar, `<media>.<ext>.json` (the shared v2 record that vault-raggae,
vault-explorer and vault-commander also keep), under one `tags` key:

    "tags": {
      "duration": 1356.85,
      "tracks": {
        "language": {
          "source": {"pass": "asr", "model": "nemotron-3.5-asr-streaming-0.6b",
                     "written": "Sun, 23 Aug 2026 11:04"},
          "spans": [{"start": 0.88, "end": 4.40, "value": "en-US"}]
        }
      }
    }

Each analysis pass owns one *track* -- `language` from the ASR, `speaker` from diarization, `voice`, `face` -- and
writes only that track. A pass can therefore run late, run again, or not run at all, and the record stays valid.
`save()` re-reads the sidecar and replaces only the tracks this run changed, so two passes and the other tools
writing the same sidecar do not undo each other.

Tracks are independent spans rather than fields on a cue, because the passes do not agree on boundaries: a speaker
turn runs across cue breaks, a language tag covers a sentence. Consumers intersect.

What does *not* go here (Thu 05 Nov 2026, with the operator): text that grows with the length of the media. The
transcript is the `.srt` beside the file (read it with `read_srt`); VSS chunk captions stay in `.captions.jsonl` and
its events in `.summary.json`. Track values are short labels, so a sidecar stays a few KB however long the video.
The old separate `<media>.tags.json` files are gone (migrated once); nothing reads them.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif", ".tiff",
    ".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv",
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".opus", ".ogg",
}
VIDEO = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv", ".mpg", ".mpeg", ".ts", ".m2ts"}
IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif", ".tiff", ".tif", ".heic"}
AUDIO = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".opus", ".ogg", ".wma", ".aiff"}


def _stamp():
    # The house format: no epochs anywhere a human might read them.
    return datetime.now().strftime("%a, %d %b %Y %H:%M")


def _now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---- where the sidecar is (vault-raggae app/files/records/sidecar_names.py, vault-explorer src/sidecar-names.js) ----

def sidecar_path(media):
    """The media file's existing sidecar under any of its four names, else the canonical `<media>.<ext>.json`.
    The stem forms (`<stem>.json`, `<stem>.meta.json`) count only when no other media file shares the stem."""
    stem, ext = os.path.splitext(media)
    full = [media + ".json", media + ".meta.json"]
    for p in full:
        if os.path.isfile(p):
            return p
    stems = [stem + ".json", stem + ".meta.json"]
    if any(os.path.isfile(p) for p in stems):
        shared = any(os.path.exists(stem + e) for e in MEDIA_EXTENSIONS if e != ext.lower())
        if not shared:
            return next(p for p in stems if os.path.isfile(p))
    return media + ".json"


def _read_record(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _kind(media):
    ext = os.path.splitext(media)[1].lower()
    return "video" if ext in VIDEO else "image" if ext in IMAGE else "audio" if ext in AUDIO else "other"


# ---- the store ----

def load(media):
    """The tags of `media` (a media file path), as `{"duration"?, "tracks"}`; empty when it has none yet."""
    tags = _read_record(sidecar_path(media)).get("tags") or {}
    store = {"duration": tags.get("duration"), "tracks": dict(tags.get("tracks") or {})}
    store["_media"] = media
    store["_dirty"] = set()
    return store


def set_media(store, path=None, duration=None):
    if path:
        store["_media"] = path
    if duration:
        store["duration"] = round(float(duration), 3)
        store["_dirty"].add("duration")
    return store


def put_track(store, name, spans, source_pass, model=None, **extra):
    """Replace one track. Whatever else is in the store is left untouched.

    `spans` is [{start, end, value, ...}] in seconds, `value` a short label. Empty spans remove the track, so a pass
    that found nothing leaves no trace rather than an empty list a reader has to interpret.
    """
    tracks = store.setdefault("tracks", {})
    store["_dirty"].add(name)
    if not spans:
        tracks.pop(name, None)
        return store
    source = {"pass": source_pass, "written": _stamp()}
    if model:
        source["model"] = model
    source.update(extra)
    tracks[name] = {"source": source, "spans": spans}
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

    A per-cue language track is mostly one long run of the same code; collapsing turns three hundred spans into two
    and keeps the sidecar small and readable.
    """
    out = []
    for s in spans:
        if out and out[-1]["value"] == s["value"] and s["start"] - out[-1]["end"] < 5.0:
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(s))
    return out


def save(store, media=None):
    """Write the tracks this run changed into the media's sidecar, keeping everything else in it (other tracks, and
    every other tool's keys), as it is on disk now. A new sidecar gets the v2 envelope. Returns the sidecar path."""
    media = media or store["_media"]
    path = sidecar_path(media)
    record = _read_record(path) if os.path.isfile(path) else {}
    tags = record.get("tags") if isinstance(record.get("tags"), dict) else {}
    tracks = tags.get("tracks") if isinstance(tags.get("tracks"), dict) else {}
    for name in store["_dirty"]:
        if name == "duration":
            continue
        if name in store["tracks"]:
            tracks[name] = store["tracks"][name]
        else:
            tracks.pop(name, None)
    if "duration" in store["_dirty"] and store.get("duration"):
        tags["duration"] = store["duration"]
    tags["tracks"] = tracks
    if tracks:
        record["tags"] = tags
    else:
        record.pop("tags", None)
    # v2 envelope (vault-raggae record.py): only what this module can know; vw.id is the media index's
    record.setdefault("schema", 2)
    vw = record.setdefault("vw", {})
    vw.setdefault("kind", _kind(media))
    vw.setdefault("created", _now_iso())
    vw["updated"] = _now_iso()
    master = record.setdefault("master", {})
    master.setdefault("basename", os.path.basename(media))
    master.setdefault("ext", os.path.splitext(media)[1].lower())
    tmp = f"{path}.tmp_{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    store["_dirty"] = set()
    return path


def summary(store):
    """One line for the console: what this file knows about itself."""
    bits = []
    for name, track in sorted(store.get("tracks", {}).items()):
        values = {s["value"] for s in track["spans"]}
        bits.append(f"{name}={'/'.join(sorted(values)) if len(values) <= 3 else f'{len(values)} values'}")
    return ", ".join(bits)


# ---- the transcript: the .srt beside the media ----

_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def srt_path(media):
    return os.path.splitext(media)[0] + ".srt"


def read_srt(path):
    """[{start, end, text}] from a SubRip file (seconds); [] when it is missing."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            blocks = re.split(r"\r?\n\s*\r?\n", f.read().strip())
    except OSError:
        return []
    cues = []
    for b in blocks:
        lines = b.splitlines()
        for i, ln in enumerate(lines):
            m = _TIME.search(ln)
            if m:
                g = [int(x) for x in m.groups()]
                start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                text = " ".join(x.strip() for x in lines[i + 1:] if x.strip())
                cues.append({"start": round(start, 3), "end": round(end, 3), "text": text})
                break
    return cues


def _ts(seconds):
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(path, cues):
    """Write cues ([(start, end, text)] or [{start, end, text}]) as SubRip; for media whose transcript is generated
    rather than recognised (a stitched conversation), so it has the same .srt beside it as any other file."""
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(cues, 1):
            start, end, text = (c["start"], c["end"], c["text"]) if isinstance(c, dict) else (c[0], c[1], c[2])
            f.write(f"{i}\n{_ts(start)} --> {_ts(end)}\n{' '.join(str(text).split())}\n\n")
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect the tags of a media file (its sidecar's `tags`)")
    ap.add_argument("media")
    ap.add_argument("--at", type=float, help="print every track's value at this time")
    args = ap.parse_args()
    store = load(args.media)
    print(f"{args.media}  {store.get('duration') or '?'}s  ({sidecar_path(args.media)})")
    for name, track in sorted(store.get("tracks", {}).items()):
        src = track["source"]
        print(f"  {name:10} {len(track['spans']):4d} spans  "
              f"[{src.get('pass')}{'/' + src['model'] if src.get('model') else ''}, {src.get('written')}]")
        if args.at is not None:
            print(f"             at {args.at}s: {value_at(store, name, args.at)}")
