#!/usr/bin/env python3
"""The voice store: everything the passes have ever learned about a voice.

A diarizer can only ever say "speaker 0" and "speaker 1", and it says it afresh
for every file -- speaker 0 in tonight's episode has nothing to do with speaker 0
in last night's. This is what carries identity across that boundary: enrol a
voice once, and every later pass that sees it close enough in cosine distance
gets the name back.

    %LOCALAPPDATA%\\VaultWares\\voices.db

Two tables, and the distinction between them is the whole design.

`voice` is the answer: one row per person, holding the centroid that naming is
done against. It is small, it is what `identify` reads, and it is derived.

`observation` is the evidence: one row per stretch of speech any pass has ever
attributed, with the embedding it produced, every 2 s window that fed it, how
tightly those windows agreed, how far the nearest other speaker in that file
was, what was said, and what it sounded like. Nothing is discarded, because the
things worth asking of a corpus are not knowable in advance -- "which voices
appear in more than one film", "did this person's pitch move between seasons",
"what did the pass get wrong, and what did the windows look like when it did" --
and every one of them needs the evidence rather than the summary. A centroid
cannot be un-averaged.

That also makes a mistake recoverable. Attributing an observation to the wrong
person, under a store that only kept centroids, permanently poisoned that
centroid. Here the link is a column: point the observation at the right voice
and rebuild, and the store is as though it never happened.

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
import sqlite3
from datetime import datetime

import numpy as np

SCHEMA = 1
MATCH_THRESHOLD = 0.60
MARGIN = 0.05  # how far the best match must lead the runner-up
EMB_DIM = 512


def default_path():
    return os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                        "VaultWares", "voices.db")


def _stamp():
    # The house format: no epochs anywhere a human might read them.
    return datetime.now().strftime("%a, %d %b %Y %H:%M")


def _blob(vector, dtype=np.float32):
    return np.asarray(vector, dtype=dtype).tobytes() if vector is not None else None


def _vector(blob, dtype=np.float32):
    return np.frombuffer(blob, dtype=dtype).astype(np.float32) if blob else None


def _matrix(blob, dim=EMB_DIM):
    """Window embeddings come back out at float32 whatever they went in as."""
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(-1, dim)


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS voice (
    id       INTEGER PRIMARY KEY,
    name     TEXT UNIQUE NOT NULL,
    centroid BLOB NOT NULL,
    samples  INTEGER NOT NULL DEFAULT 1,
    created  TEXT,
    updated  TEXT,
    notes    TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS observation (
    id             INTEGER PRIMARY KEY,
    voice_id       INTEGER REFERENCES voice(id) ON DELETE SET NULL,
    label          TEXT,
    media          TEXT,
    media_path     TEXT,
    media_duration REAL,
    start_s        REAL,
    end_s          REAL,
    speech_seconds REAL,
    embedding      BLOB,
    windows        BLOB,
    n_windows      INTEGER,
    cohesion       REAL,
    separation     REAL,
    match_score    REAL,
    matched        TEXT,
    metrics        TEXT,
    source         TEXT,
    written        TEXT
);

CREATE INDEX IF NOT EXISTS observation_voice ON observation(voice_id);
CREATE INDEX IF NOT EXISTS observation_media ON observation(media);
"""


class VoiceStore:
    """One SQLite file. Open it, ask it things, close it."""

    def __init__(self, path=None):
        self.path = path or default_path()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(DDL)
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('schema', ?)", (str(SCHEMA),))
        self.db.commit()
        self._migrate_json()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- reading -----------------------------------------------------------

    def voices(self):
        """[(name, centroid, row)] for every enrolled voice."""
        rows = self.db.execute("SELECT * FROM voice ORDER BY name").fetchall()
        return [(r["name"], _vector(r["centroid"]), r) for r in rows]

    def identify(self, embedding, threshold=MATCH_THRESHOLD, margin=MARGIN):
        """(name, score) for the best match, or (None, best score) if nothing fits.

        Two guards, because a store grows and near-misses get more likely as it
        does: the score has to clear `threshold`, and it has to lead the
        runner-up by `margin`. Two voices that both score 0.62 are a question,
        not an answer.
        """
        if embedding is None:
            return None, 0.0
        known = self.voices()
        if not known:
            return None, 0.0
        embedding = np.asarray(embedding, dtype=np.float32)
        scored = sorted(((float(np.dot(c, embedding)), n) for n, c, _ in known), reverse=True)
        best_score, best_name = scored[0]
        if best_score < threshold:
            return None, best_score
        if len(scored) > 1 and best_score - scored[1][0] < margin:
            return None, best_score
        return best_name, best_score

    def observations(self, name=None, media=None, limit=None):
        sql = ("SELECT o.*, v.name AS voice FROM observation o "
               "LEFT JOIN voice v ON v.id = o.voice_id WHERE 1=1")
        args = []
        if name:
            sql += " AND v.name = ?"
            args.append(name)
        if media:
            sql += " AND o.media LIKE ?"
            args.append(f"%{media}%")
        sql += " ORDER BY o.id DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(int(limit))
        return self.db.execute(sql, args).fetchall()

    # -- writing -----------------------------------------------------------

    def enroll(self, name, embedding, notes="", observation_id=None):
        """Add a vector to a voice, averaging it into whatever is already there.

        Enrolling the same person again averages rather than replaces, so a
        voice heard in a dozen scenes ends up represented by all of them.
        """
        embedding = _unit(embedding)
        row = self.db.execute("SELECT * FROM voice WHERE name = ?", (name,)).fetchone()
        if row is None:
            centroid, samples, created = embedding, 1, _stamp()
        else:
            samples = int(row["samples"]) + 1
            centroid = _unit(_vector(row["centroid"]) * (samples - 1) / samples
                             + embedding / samples)
            created = row["created"]
            notes = notes or row["notes"]
        self.db.execute(
            "INSERT INTO voice (name, centroid, samples, created, updated, notes) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "centroid=excluded.centroid, samples=excluded.samples, "
            "updated=excluded.updated, notes=excluded.notes",
            (name, _blob(centroid), samples, created, _stamp(), notes or ""))
        if observation_id is not None:
            self.attribute(observation_id, name)
        self.db.commit()
        return samples

    def observe(self, embedding, windows=None, **fields):
        """Record one stretch of attributed speech. Returns its row id.

        Everything the caller knows goes in. A field it does not know is left
        null rather than defaulted, so a later query can tell "the pass did not
        measure this" apart from "the pass measured this and got zero".
        """
        name = fields.pop("voice", None)
        voice_id = None
        if name:
            row = self.db.execute("SELECT id FROM voice WHERE name = ?", (name,)).fetchone()
            voice_id = row["id"] if row else None
        metrics = fields.pop("metrics", None)
        source = fields.pop("source", None)
        columns = ("label", "media", "media_path", "media_duration", "start_s", "end_s",
                   "speech_seconds", "cohesion", "separation", "match_score", "matched")
        values = [fields.get(c) for c in columns]
        cur = self.db.execute(
            f"INSERT INTO observation (voice_id, {', '.join(columns)}, embedding, windows, "
            "n_windows, metrics, source, written) VALUES "
            f"(?, {', '.join('?' * len(columns))}, ?, ?, ?, ?, ?, ?)",
            [voice_id] + values + [
                _blob(embedding),
                # float16 halves the store for no measurable cost: these are
                # unit vectors, so every component is within [-1, 1] where
                # float16 has ~3 decimal digits, and cosine over 512 of them
                # moves in the fourth.
                _blob(windows, np.float16) if windows is not None else None,
                len(windows) if windows is not None else None,
                json.dumps(metrics, ensure_ascii=False) if metrics else None,
                json.dumps(source, ensure_ascii=False) if source else None,
                _stamp()])
        self.db.commit()
        return cur.lastrowid

    def attribute(self, observation_id, name):
        """Point an observation at a voice, creating the voice if it is new."""
        row = self.db.execute("SELECT id FROM voice WHERE name = ?", (name,)).fetchone()
        if row is None:
            obs = self.db.execute("SELECT embedding FROM observation WHERE id = ?",
                                  (observation_id,)).fetchone()
            if obs is None:
                raise KeyError(f"no observation {observation_id}")
            self.db.execute(
                "INSERT INTO voice (name, centroid, samples, created, updated, notes) "
                "VALUES (?,?,?,?,?,'')",
                (name, obs["embedding"], 1, _stamp(), _stamp()))
            row = self.db.execute("SELECT id FROM voice WHERE name = ?", (name,)).fetchone()
        self.db.execute("UPDATE observation SET voice_id = ? WHERE id = ?",
                        (row["id"], observation_id))
        self.db.commit()
        return row["id"]

    def rebuild(self, name):
        """Recompute a voice's centroid from the observations linked to it.

        Weighted by speech length: an observation built from forty windows says
        more about a person than one built from two, and averaging unit vectors
        without weights lets a two-second aside count as much as a monologue.

        This is the repair path. It is also what makes `attribute` worth having
        -- correcting a link is only a correction if something re-derives the
        answer from the links.
        """
        rows = self.db.execute(
            "SELECT o.embedding, o.n_windows FROM observation o "
            "JOIN voice v ON v.id = o.voice_id WHERE v.name = ? AND o.embedding IS NOT NULL",
            (name,)).fetchall()
        if not rows:
            return 0
        vectors = np.stack([_vector(r["embedding"]) for r in rows])
        weights = np.asarray([max(int(r["n_windows"] or 1), 1) for r in rows], dtype=np.float32)
        centroid = _unit(np.average(vectors, axis=0, weights=weights))
        self.db.execute("UPDATE voice SET centroid = ?, samples = ?, updated = ? WHERE name = ?",
                        (_blob(centroid), len(rows), _stamp(), name))
        self.db.commit()
        return len(rows)

    def forget(self, name):
        """Drop the voice. Its observations survive, unlinked.

        Deleting the evidence with the label would mean an accidental `forget`
        costs the recordings too, and the evidence is the expensive half.
        """
        self.db.execute("DELETE FROM voice WHERE name = ?", (name,))
        self.db.commit()

    # -- one-time carry-over ------------------------------------------------

    def _migrate_json(self):
        """Fold an old voices.json in, once, and leave it on disk.

        The JSON store kept centroids and nothing else, so the observations
        behind them are gone and cannot be reconstructed -- these voices arrive
        with a centroid and no evidence, which `rebuild` would wipe. They are
        marked so that is visible rather than surprising.
        """
        if self.db.execute("SELECT value FROM meta WHERE key='migrated_json'").fetchone():
            return
        legacy = os.path.join(os.path.dirname(self.path), "voices.json")
        if not os.path.exists(legacy):
            return
        try:
            with open(legacy, encoding="utf-8") as f:
                data = json.load(f)
            for name, v in data.get("voices", {}).items():
                if self.db.execute("SELECT 1 FROM voice WHERE name=?", (name,)).fetchone():
                    continue
                self.db.execute(
                    "INSERT INTO voice (name, centroid, samples, created, updated, notes) "
                    "VALUES (?,?,?,?,?,?)",
                    (name, _blob(_unit(v["centroid"])), int(v.get("samples", 1)),
                     v.get("updated"), v.get("updated"),
                     (v.get("notes", "") + " [imported from voices.json]").strip()))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('migrated_json', ?)", (_stamp(),))
            self.db.commit()
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            pass  # A store that will not parse is not worth failing an ASR run over.


# Module-level shims, so a caller that just wants an answer does not have to
# think about connection lifetime.

def load(path=None):
    return VoiceStore(path)


def identify(store, embedding, threshold=MATCH_THRESHOLD, margin=MARGIN):
    return store.identify(embedding, threshold, margin)


def enroll(store, name, embedding, notes=""):
    store.enroll(name, embedding, notes)
    return store


def forget(store, name):
    store.forget(name)
    return store


def main():
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    ap = argparse.ArgumentParser(description="The permanent voice store")
    ap.add_argument("action", choices=["list", "show", "enroll", "identify", "forget",
                                       "observations", "attribute", "rebuild", "export"])
    ap.add_argument("--name")
    ap.add_argument("--audio", help="16 kHz mono wav")
    ap.add_argument("--span", metavar="START:END", help="seconds within --audio")
    ap.add_argument("--store")
    ap.add_argument("--media", help="filter observations by media file name")
    ap.add_argument("--id", type=int, help="observation id, for attribute")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    store = VoiceStore(args.store)

    if args.action == "list":
        known = store.voices()
        if not known:
            print(f"  no voices in {store.path}")
        for name, _, row in known:
            heard = store.db.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(speech_seconds),0) s FROM observation "
                "WHERE voice_id = ?", (row["id"],)).fetchone()
            print(f"  {name:24} {row['samples']:3d} sample(s)  "
                  f"{heard['n']:3d} obs / {heard['s']:7.1f}s  {row['updated'] or ''}"
                  f"{'  ' + row['notes'] if row['notes'] else ''}")
        return 0

    if args.action == "show":
        rows = store.observations(name=args.name, limit=args.limit)
        if not rows:
            print(f"  nothing recorded for {args.name}")
            return 0
        media = {r["media"] for r in rows if r["media"]}
        print(f"  {args.name}: {len(rows)} observation(s) over {len(media)} file(s)")
        # A column the pass did not measure prints as a dash, not as zero: the
        # store keeps that difference and the display would otherwise lose it.
        def num(value, fmt="{:.2f}", width=4):
            return fmt.format(value) if value is not None else "-".rjust(width)

        for r in rows:
            m = json.loads(r["metrics"]) if r["metrics"] else {}
            pitch = f"{m['f0_median_hz']:5.0f} Hz" if "f0_median_hz" in m else "     -   "
            print(f"    #{r['id']:<5} {(r['media'] or '?')[:34]:34} "
                  f"{num(r['speech_seconds'], '{:6.1f}', 6)}s {pitch}  "
                  f"coh {num(r['cohesion'])}  match {num(r['match_score'])}  "
                  f"win {num(r['n_windows'], '{:d}', 3)}")
        return 0

    if args.action == "observations":
        rows = store.observations(media=args.media, limit=args.limit)
        for r in rows:
            print(f"    #{r['id']:<5} {r['voice'] or r['label'] or '?':16} "
                  f"{(r['media'] or '?')[:30]:30} {r['speech_seconds'] or 0:6.1f}s  "
                  f"{r['written'] or ''}")
        return 0

    if args.action == "attribute":
        store.attribute(args.id, args.name)
        n = store.rebuild(args.name)
        print(f"  #{args.id} -> {args.name}; centroid rebuilt from {n} observation(s)")
        return 0

    if args.action == "rebuild":
        print(f"  {args.name}: centroid rebuilt from {store.rebuild(args.name)} observation(s)")
        return 0

    if args.action == "forget":
        store.forget(args.name)
        print(f"  forgot {args.name} (its observations are kept, unlinked)")
        return 0

    if args.action == "export":
        out = {"voices": [], "observations": []}
        for name, centroid, row in store.voices():
            out["voices"].append({"name": name, "samples": row["samples"],
                                  "updated": row["updated"], "notes": row["notes"],
                                  "centroid": [round(float(x), 6) for x in centroid]})
        for r in store.observations(limit=args.limit):
            out["observations"].append({
                k: r[k] for k in r.keys() if k not in ("embedding", "windows")})
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0

    # Both remaining actions need audio.
    from voiceprint import Embedder
    import prosody
    from voiceprint import read_wav

    if not args.audio:
        print("  --audio is required")
        return 2
    start, end = (float(x) for x in args.span.split(":")) if args.span else (None, None)
    audio = read_wav(args.audio, start, end)
    embedder = Embedder()
    windows = embedder.embed_windows(audio)
    vector = embedder.embed(audio)
    if vector is None:
        print("  not enough audio to embed")
        return 1

    if args.action == "enroll":
        obs = store.observe(
            vector, windows=windows,
            cohesion=round(float(np.mean(np.dot(windows, vector))), 4) if len(windows) else None,
            voice=args.name, label=args.name,
            media=os.path.basename(args.audio), media_path=os.path.abspath(args.audio),
            start_s=start or 0.0, end_s=end, speech_seconds=len(audio) / 16000,
            metrics=prosody.measure(audio),
            source={"pass": "enroll", "model": "campplus-voxceleb-lm"})
        samples = store.enroll(args.name, vector, args.notes, observation_id=obs)
        print(f"  {args.name}: {samples} sample(s), observation #{obs} -> {store.path}")
    else:
        name, score = store.identify(vector, args.threshold)
        print(f"  {name or 'unknown'}  (best score {score:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
