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
import prosody
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


def unit(vector):
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=np.float32) / norm if norm else np.asarray(vector, np.float32)


# Prosody over a whole film's worth of one speaker buys nothing over a few
# minutes of them, and the concatenation is the only thing here that grows with
# file length. Past this, windows are sampled evenly across the file rather than
# truncated, so the measurement still covers every scene the speaker is in.
PROSODY_CAP_S = 180.0


def gather(audio, spans, cap=PROSODY_CAP_S):
    """The speaker's audio, back to back, bounded.

    Overlapping spans are merged first. The windows overlap by half by
    construction, so concatenating them raw plays every second of speech twice
    -- which does not move a median but does make the reported duration a lie.
    """
    merged = []
    for s in sorted(spans, key=lambda s: s["start"]):
        if merged and s["start"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s["end"])
        else:
            merged.append([s["start"], s["end"]])
    if not merged:
        return np.zeros(0, dtype=np.float32)
    total = sum(e - s for s, e in merged)
    step = max(1, int(np.ceil(total / cap))) if total > cap else 1
    return np.concatenate([audio[int(s * 16000):int(e * 16000)] for s, e in merged[::step]])


def record(known, store, audio, audio_path, tags_path, spans, vectors, labels, ordered,
           members, centroids, names, scores, voted, threshold, match_threshold):
    """Write one observation per cluster: the embedding, and why to believe it.

    Two numbers say how much the cluster is worth, and they are the reason this
    is stored rather than recomputed later. `cohesion` is the mean cosine of the
    cluster's own windows to its centroid -- how much one person it is. It is
    high for a real speaker and drops when clustering has merged two people who
    sound alike. `separation` is the best cosine to any *other* cluster in the
    same file -- how distinct it was from the competition it was actually judged
    against. A centroid at 0.85 cohesion and 0.10 separation is a clean answer;
    the same 0.85 against a 0.55 neighbour is a coin toss that happened to land,
    and only the file it came from ever knew that.
    """
    media = store.get("media", {})
    # The wav handed to this pass is a temp file the pipeline deletes on its way
    # out, so recording it would leave the store pointing at nothing. The tag
    # store sits beside the media it describes, and knows its name.
    if media.get("path"):
        media_path = os.path.abspath(os.path.join(os.path.dirname(tags_path), media["path"]))
    else:
        media_path = os.path.abspath(audio_path)
    recorded = {}
    for label in ordered:
        languages = {}
        mine = [sp for sp, l in zip(spans, labels) if l == label]
        cues = [c for c, l in voted if l == label]
        centroid = centroids[label]
        cohesion = float(np.mean(np.dot(np.stack(members[label]), centroid)))
        others = [float(np.dot(centroids[o], centroid)) for o in ordered if o != label]
        text = " ".join(c["text"] for c in cues).strip()
        spoken = sum(c["end"] - c["start"] for c in cues)
        for c in cues:
            code = tagstore.value_at(store, "language", (c["start"] + c["end"]) / 2)
            if code:
                languages[code] = languages.get(code, 0.0) + c["end"] - c["start"]

        metrics = prosody.measure(gather(audio, mine))
        metrics.update({
            "words": len(text.split()),
            "characters": len(text),
            # Words per minute over time actually spoken, not wall clock: the
            # pauses between this speaker's cues belong to whoever filled them.
            "wpm": round(len(text.split()) / (spoken / 60.0), 1) if spoken > 0 else None,
            "cues": len(cues),
            "languages": sorted(languages, key=languages.get, reverse=True)[:3] or None,
            "first_line": text[:120] or None,
        })
        recorded[label] = known.observe(
            centroid, windows=np.stack(members[label]),
            voice=names[label] if not names[label].startswith("SPEAKER_") else None,
            label=names[label],
            media=media.get("path") or os.path.basename(audio_path),
            media_path=media_path,
            media_duration=media.get("duration"),
            start_s=min(c["start"] for c in cues) if cues else None,
            end_s=max(c["end"] for c in cues) if cues else None,
            speech_seconds=round(spoken, 3),
            cohesion=round(cohesion, 4),
            separation=round(max(others), 4) if others else None,
            match_score=round(scores[label], 4),
            matched=names[label] if not names[label].startswith("SPEAKER_") else None,
            metrics={k: v for k, v in metrics.items() if v is not None},
            source={"pass": "speaker", "model": "campplus-voxceleb-lm",
                    "tags": os.path.basename(tags_path),
                    "window_s": WINDOW_S, "hop_s": HOP_S,
                    "cluster_threshold": threshold, "match_threshold": match_threshold,
                    "speakers_in_file": len(ordered)})
    return recorded


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
    ordered = sorted(set(labels))
    members = {l: [v for v, k in zip(vectors, labels) if k == l] for l in ordered}
    centroids = {l: unit(np.mean(members[l], axis=0)) for l in ordered}
    names, scores = {}, {}
    for label in ordered:
        name, score = known.identify(centroids[label], match_threshold)
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

    # Deposit the evidence. This runs after the tag store is written, and its
    # failure is caught below, because the subtitle pass calling this must not
    # lose a finished speaker track to a locked database.
    recorded = {}
    try:
        recorded = record(known, store, audio, audio_path, tags_path, usable, vectors,
                          labels, ordered, members, centroids, names, scores, voted,
                          threshold, match_threshold)
    except Exception as exc:  # sqlite errors, a read-only store, a full disk
        if not quiet:
            print(f"    [!] voice store not updated: {exc}")

    if not quiet:
        print(f"    {len(ordered)} speaker(s) over {len(voted)} of {len(cues)} cues "
              f"({len(usable)} windows) -> {tags_path}")
        for label in ordered:
            spoken = sum(c["end"] - c["start"] for c, l in voted if l == label)
            named = not names[label].startswith("SPEAKER_")
            detail = f"matched {scores[label]:.2f}" if named else f"best {scores[label]:.2f}, unknown"
            obs = f"  #{recorded[label]}" if label in recorded else ""
            print(f"      {names[label]:16} {spoken:6.1f}s  ({detail}){obs}")
        if recorded:
            print(f"      {len(recorded)} observation(s) -> {known.path}")
    known.close()
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
