#!/usr/bin/env python3
"""The voice store: names for embeddings, remembered between files.

A diarizer can only ever say "speaker 0" and "speaker 1", and it says it afresh
for every file -- speaker 0 in tonight's episode has nothing to do with speaker 0
in last night's. This is what carries identity across that boundary: enrol a
voice once, and every later pass that sees it close enough in cosine distance
gets the name back.

The store is one JSON file of unit-length centroids. Enrolling the same person
again averages the new vector into their centroid rather than replacing it, so a
voice heard in a dozen scenes ends up represented by all of them.

    %LOCALAPPDATA%\\VaultWares\\voices.json

Thresholds come from measurement, not taste. Over 24 clips from 6 LibriSpeech
speakers this embedder scores 0.808 mean cosine within a speaker and 0.124
between speakers, and 0.52 separates them with one error in 276 pairs. The
default match threshold is deliberately above that crossing point: a wrong name
is worse than no name, because no name is visibly missing and a wrong one is
believed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

MATCH_THRESHOLD = 0.60
MARGIN = 0.05  # how far the best match must lead the runner-up


def default_path():
    return os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                        "VaultWares", "voices.json")


def load(path=None):
    path = path or default_path()
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for v in data.get("voices", {}).values():
                v["centroid"] = np.asarray(v["centroid"], dtype=np.float32)
            return data
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return {"schema": 1, "voices": {}}


def save(store, path=None):
    path = path or default_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {"schema": store.get("schema", 1), "voices": {}}
    for name, v in store["voices"].items():
        out["voices"][name] = {
            "centroid": [round(float(x), 6) for x in v["centroid"]],
            "samples": int(v.get("samples", 1)),
            "updated": v.get("updated") or datetime.now().strftime("%a, %d %b %Y %H:%M"),
            "notes": v.get("notes", ""),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return path


def enroll(store, name, embedding, notes=""):
    """Add a vector to a voice, averaging it into whatever is already there."""
    embedding = np.asarray(embedding, dtype=np.float32)
    voice = store["voices"].get(name)
    if voice is None:
        centroid, samples = embedding, 1
    else:
        samples = int(voice.get("samples", 1)) + 1
        centroid = voice["centroid"] * (samples - 1) / samples + embedding / samples
    norm = np.linalg.norm(centroid)
    store["voices"][name] = {
        "centroid": (centroid / norm).astype(np.float32) if norm else centroid,
        "samples": samples,
        "updated": datetime.now().strftime("%a, %d %b %Y %H:%M"),
        "notes": notes or (store["voices"].get(name, {}) or {}).get("notes", ""),
    }
    return store


def identify(store, embedding, threshold=MATCH_THRESHOLD, margin=MARGIN):
    """(name, score) for the best match, or (None, best score) if nothing fits.

    Two guards, because a store grows and near-misses get more likely as it
    does: the score has to clear `threshold`, and it has to lead the runner-up
    by `margin`. Two voices that both score 0.62 are a question, not an answer.
    """
    if embedding is None or not store["voices"]:
        return None, 0.0
    scored = sorted(
        ((float(np.dot(v["centroid"], embedding)), name) for name, v in store["voices"].items()),
        reverse=True)
    best_score, best_name = scored[0]
    if best_score < threshold:
        return None, best_score
    if len(scored) > 1 and best_score - scored[1][0] < margin:
        return None, best_score
    return best_name, best_score


def forget(store, name):
    store["voices"].pop(name, None)
    return store


def main():
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from voiceprint import Embedder

    ap = argparse.ArgumentParser(description="Enrol and look up voices")
    ap.add_argument("action", choices=["list", "enroll", "identify", "forget"])
    ap.add_argument("--name")
    ap.add_argument("--audio", help="16 kHz mono wav")
    ap.add_argument("--span", metavar="START:END", help="seconds within --audio")
    ap.add_argument("--store")
    ap.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    store = load(args.store)

    if args.action == "list":
        if not store["voices"]:
            print(f"  no voices in {args.store or default_path()}")
        for name, v in sorted(store["voices"].items()):
            print(f"  {name:24} {v['samples']:3d} sample(s)  {v.get('updated','')}"
                  f"{'  ' + v['notes'] if v.get('notes') else ''}")
        return 0

    if args.action == "forget":
        forget(store, args.name)
        save(store, args.store)
        print(f"  forgot {args.name}")
        return 0

    if not args.audio:
        print("  --audio is required", flush=True)
        return 2
    start, end = (float(x) for x in args.span.split(":")) if args.span else (None, None)
    vector = Embedder().embed_file(args.audio, start, end)
    if vector is None:
        print("  not enough audio to embed")
        return 1

    if args.action == "enroll":
        enroll(store, args.name, vector, args.notes)
        path = save(store, args.store)
        print(f"  {args.name}: {store['voices'][args.name]['samples']} sample(s) -> {path}")
    else:
        name, score = identify(store, vector, args.threshold)
        print(f"  {name or 'unknown'}  (best score {score:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
