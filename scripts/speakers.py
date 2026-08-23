#!/usr/bin/env python3
"""Who spoke when: the speaker pass, written into the tag store.

Embeds each cue with CAM++, clusters the embeddings by cosine distance, and
writes one `speaker` track. Names come from the voice store when a cluster
matches something enrolled; otherwise a cluster is SPEAKER_00 and stays
anonymous, which is the honest answer.

This is not yet diarization. It attributes speech that the ASR already
segmented, so it inherits the ASR's boundaries and cannot represent two people
talking at once. Sortformer replaces the segmentation later -- it predicts
per-frame, per-speaker activity and so handles overlap -- but its labels are
local to a 20-second window, and this is what will link them across windows and
across files. Building it first is what makes that possible.

    python speakers.py --audio episode.16k.wav --tags episode.tags.json
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tagstore
import voices as voicestore
from voiceprint import Embedder, read_wav

# Under two seconds the embedding carries no speaker information at all. On a
# two-speaker file, cues of 0.8-2.0 s scored 0.297 within a speaker and 0.294
# between -- a gap of 0.004. From 2-4 s it is 0.618 against 0.015, and past 4 s
# 0.744 against 0.013. The model's window is 2 s, so a shorter cue yields one
# truncated window and nothing to average. Hence turns: consecutive cues are
# pooled until there is enough audio to say something.
MIN_SPAN_S = 1.0
WINDOW_S = 2.0
HOP_S = 1.0
# With turns of 2 s and up, different speakers score near zero and the same
# speaker 0.6-0.75, so the boundary sits low and is not delicate.
CLUSTER_THRESHOLD = 0.40
# A cluster this small is a window straddling a speaker change, not a speaker.
MIN_CLUSTER_WINDOWS = 3


def windows(cues, length=WINDOW_S, hop=HOP_S):
    """Slide a fixed window across the speech, ignoring cue boundaries.

    Pooling whole cues into turns was the first attempt and it cannot work: on
    back-to-back dialogue the pause between two speakers is shorter than the
    pause inside one speaker's sentence, so a gap rule merges across the change
    and the file comes back with one speaker. A window that ignores the
    transcript finds the change instead, at the cost of resolution -- a boundary
    is located to within one hop.

    The window is the model's own 2 s crop, which is also the shortest span that
    carries any speaker information at all.
    """
    spans = []
    for cue in cues:
        if spans and cue["start"] - spans[-1][1] <= 0.4:
            spans[-1][1] = max(spans[-1][1], cue["end"])
        else:
            spans.append([cue["start"], cue["end"]])
    out = []
    for start, end in spans:
        if end - start < length:
            if end - start >= length / 2:
                out.append({"start": start, "end": end})
            continue
        t = start
        while t + length <= end + 1e-6:
            out.append({"start": t, "end": t + length})
            t += hop
        if end - (t - hop + length) > hop / 2:
            out.append({"start": max(start, end - length), "end": end})
    return out


def label_cues(cues, spans, labels):
    """Give each cue the label of whichever windows cover it most."""
    out = []
    for cue in cues:
        votes = {}
        for span, label in zip(spans, labels):
            overlap = min(cue["end"], span["end"]) - max(cue["start"], span["start"])
            if overlap > 0:
                votes[label] = votes.get(label, 0.0) + overlap
        if votes:
            out.append((cue, max(votes.items(), key=lambda kv: kv[1])[0]))
    return out


def cluster(vectors, threshold=CLUSTER_THRESHOLD):
    """Average-linkage agglomerative clustering on cosine similarity.

    Merges the closest pair until nothing is closer than `threshold`, so the
    number of speakers comes out of the audio rather than being declared up
    front -- which matters because nobody knows it in advance, and guessing
    wrong is worse than either error it prevents.

    The n^2 similarity matrix is fine at this scale: an hour of dialogue is a
    few thousand cues, and the whole thing is milliseconds next to the ASR pass
    that produced them.
    """
    n = len(vectors)
    if n == 0:
        return []
    groups = [[i] for i in range(n)]
    sim = np.dot(np.asarray(vectors), np.asarray(vectors).T)
    np.fill_diagonal(sim, -1.0)

    while len(groups) > 1:
        best, pair = -1.0, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                score = float(np.mean(sim[np.ix_(groups[a], groups[b])]))
                if score > best:
                    best, pair = score, (a, b)
        if best < threshold:
            break
        a, b = pair
        groups[a] = groups[a] + groups[b]
        groups.pop(b)

    labels = [0] * n
    # Order clusters by first appearance so SPEAKER_00 is whoever speaks first.
    for label, group in enumerate(sorted(groups, key=min)):
        for i in group:
            labels[i] = label
    return labels


def drop_strays(spans, vectors, labels, min_windows=MIN_CLUSTER_WINDOWS):
    """Discard windows that belong to no real cluster.

    A window straddling a speaker change holds both voices, so its embedding
    sits near neither and starts a cluster of its own: two speakers came back as
    five. Those strays are not a third speaker and they are not cleanly either
    of the first two -- measured, one such cluster scored 0.20 against one
    centroid and 0.10 against the other, far from both. So they are dropped
    rather than assigned, and the cues they covered fall back on the votes of
    the windows around them, which do know who was speaking.

    The cost is a real speaker with a single short line, whose windows look
    exactly like this. At two seconds the two cases are not distinguishable, and
    inventing a speaker per boundary is the louder error.
    """
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    keep = {label for label, n in counts.items() if n >= min_windows}
    if not keep:
        return spans, vectors, labels
    kept = [(sp, v, l) for sp, v, l in zip(spans, vectors, labels) if l in keep]
    order = {}
    for _, _, label in kept:
        order.setdefault(label, len(order))
    return ([sp for sp, _, _ in kept],
            [v for _, v, _ in kept],
            [order[l] for _, _, l in kept])


def run(audio_path, tags_path, store_path=None, threshold=CLUSTER_THRESHOLD,
        match_threshold=voicestore.MATCH_THRESHOLD, min_span=MIN_SPAN_S, quiet=False):
    store = tagstore.load(tags_path)
    cues = store.get("cues", [])
    if not cues:
        raise SystemExit(f"no cues in {tags_path}; run the subtitle pass first")

    embedder = Embedder()
    audio = read_wav(audio_path)
    usable, vectors = [], []
    for span in windows(cues):
        if span["end"] - span["start"] < min_span:
            continue
        vector = embedder.embed(audio[int(span["start"] * 16000):int(span["end"] * 16000)])
        if vector is not None:
            usable.append(span)
            vectors.append(vector)
    if not vectors:
        raise SystemExit("no window was long enough to embed")

    labels = cluster(vectors, threshold)
    usable, vectors, labels = drop_strays(usable, vectors, labels)
    known = voicestore.load(store_path)

    # Name a cluster once, from its centroid, rather than per cue: a centroid is
    # built from every span the speaker said and is steadier than any one of them.
    names, scores = {}, {}
    for label in sorted(set(labels)):
        members = [v for v, l in zip(vectors, labels) if l == label]
        centroid = np.mean(members, axis=0)
        norm = np.linalg.norm(centroid)
        if norm:
            centroid = centroid / norm
        name, score = voicestore.identify(known, centroid, match_threshold)
        names[label] = name or f"SPEAKER_{label:02d}"
        scores[label] = score

    # One span per cue: a consumer asking "who said this line" should not have
    # to know that the attribution was done on windows.
    voted = label_cues(cues, usable, labels)
    spans = [{"start": cue["start"], "end": cue["end"], "value": names[label]}
             for cue, label in voted]
    tagstore.put_track(store, "speaker", tagstore.collapse(spans), "speaker",
                       model="campplus-voxceleb-lm",
                       clustered_at=round(threshold, 3))
    tagstore.save(store, tags_path)

    if not quiet:
        print(f"    {len(set(labels))} speaker(s) over {len(voted)} of {len(cues)} cues "
              f"({len(usable)} windows) -> {tags_path}")
        for label in sorted(set(labels)):
            spoken = sum(c["end"] - c["start"] for c, l in voted if l == label)
            named = not names[label].startswith("SPEAKER_")
            detail = f"matched {scores[label]:.2f}" if named else f"best {scores[label]:.2f}, unknown"
            print(f"      {names[label]:16} {spoken:6.1f}s  ({detail})")
    return store


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Attribute cues to speakers and write the speaker track")
    ap.add_argument("--audio", required=True, help="16 kHz mono wav of the same media")
    ap.add_argument("--tags", required=True, help="tag store written by the subtitle pass")
    ap.add_argument("--voices", help="voice store (default: %%LOCALAPPDATA%%\\VaultWares\\voices.json)")
    ap.add_argument("--threshold", type=float, default=CLUSTER_THRESHOLD,
                    help="cosine similarity below which two cues are different speakers")
    ap.add_argument("--match-threshold", type=float, default=voicestore.MATCH_THRESHOLD,
                    help="cosine similarity required to attach a name from the voice store")
    ap.add_argument("--min-span", type=float, default=MIN_SPAN_S,
                    help="windows shorter than this are left unattributed")
    args = ap.parse_args()
    run(args.audio, args.tags, args.voices, args.threshold, args.match_threshold, args.min_span)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
