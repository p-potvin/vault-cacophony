#!/usr/bin/env python3
"""Tests for the local VSS pipeline. No model, no ffmpeg: the pure parts only."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import sqlite3
import numpy as np

import faces  # noqa: E402
import tagstore  # noqa: E402
import vss_pipeline as vp  # noqa: E402


class TestChunkPlan(unittest.TestCase):
    def test_exact_multiple(self):
        chunks = vp.plan_chunks(30.0, 10)
        self.assertEqual([(c["start"], c["end"]) for c in chunks], [(0, 10), (10, 20), (20, 30)])
        self.assertTrue(chunks[0]["is_first"] and chunks[-1]["is_last"])
        self.assertEqual([c["chunkIdx"] for c in chunks], [0, 1, 2])

    def test_short_tail_is_merged(self):
        # 31 s: the 1 s sliver joins the last chunk instead of becoming one.
        chunks = vp.plan_chunks(31.0, 10)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[-1]["end"], 31.0)

    def test_long_tail_keeps_its_chunk(self):
        chunks = vp.plan_chunks(34.0, 10)
        self.assertEqual(len(chunks), 4)
        self.assertEqual((chunks[-1]["start"], chunks[-1]["end"]), (30.0, 34.0))

    def test_overlap_reaches_back(self):
        chunks = vp.plan_chunks(20.0, 10, overlap=2)
        self.assertEqual((chunks[0]["start"], chunks[0]["end"]), (0.0, 10.0))
        self.assertEqual((chunks[1]["start"], chunks[1]["end"]), (8.0, 20.0))

    def test_shorter_than_a_chunk(self):
        chunks = vp.plan_chunks(4.0, 10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["end"], 4.0)

    def test_empty(self):
        self.assertEqual(vp.plan_chunks(0, 10), [])


class TestFrameTimes(unittest.TestCase):
    def test_bin_centres(self):
        self.assertEqual(vp.frame_times(10, 20, 4), [11.25, 13.75, 16.25, 18.75])

    def test_never_on_the_end(self):
        times = vp.frame_times(0, 166.0, 8)
        self.assertLess(times[-1], 166.0)
        self.assertGreater(times[0], 0.0)

    def test_single(self):
        self.assertEqual(vp.frame_times(0, 10, 1), [5.0])


class TestJsonExtraction(unittest.TestCase):
    def test_fenced_with_think(self):
        text = ('<think>hmm</think>Here you go:\n```json\n{"events": [{"start_time": 1, '
                '"end_time": 2, "type": "typing", "description": "keys pressed"}]}\n```')
        parsed = vp.extract_json(text)
        self.assertEqual(parsed["events"][0]["type"], "typing")

    def test_bare_list_with_template_comments_and_trailing_comma(self):
        text = ('[\n{\n "start_time": 3.0, #(MANDATORY)\n "end_time": 4.5,\n'
                ' "type": "typing",\n "description": "text is typed",\n},\n]')
        parsed = vp.extract_json(text)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["end_time"], 4.5)

    def test_no_json(self):
        self.assertIsNone(vp.extract_json("A person walks across the room."))

    def test_truncated_by_num_predict(self):
        text = ('{"events": [{"start_time": 1, "end_time": 2, "type": "typing", '
                '"description": "done"}, {"start_time": 3, "end_time": 4, "type": "typing", '
                '"description": "cut off mid sen')
        parsed = vp.extract_json(text)
        self.assertEqual([e["description"] for e in parsed["events"]], ["done"])


class TestNormalizeEvents(unittest.TestCase):
    chunk = {"start": 20.0, "end": 30.0}

    def test_absolute_times_kept(self):
        evs = vp.normalize_events([{"start_time": 22, "end_time": 25, "type": "typing",
                                    "description": "x"}], self.chunk, ["typing"])
        self.assertEqual((evs[0]["start_time"], evs[0]["end_time"]), (22.0, 25.0))

    def test_relative_times_shifted(self):
        evs = vp.normalize_events({"events": [{"start_time": 1.0, "end_time": 4.0,
                                               "type": "typing", "description": "x"}]},
                                  self.chunk, ["typing"])
        self.assertEqual((evs[0]["start_time"], evs[0]["end_time"]), (21.0, 24.0))

    def test_out_of_window_clamped(self):
        evs = vp.normalize_events([{"start_time": 5, "end_time": 99, "type": "typing",
                                    "description": "x"}], self.chunk, ["typing"])
        # 5 fits as relative (25); 99 is clamped to the chunk end.
        self.assertEqual((evs[0]["start_time"], evs[0]["end_time"]), (25.0, 30.0))

    def test_timestamp_strings(self):
        evs = vp.normalize_events([{"start_time": "00:00:21,500", "end_time": "00:00:23,000",
                                    "type": "typing", "description": "x"}], self.chunk, ["typing"])
        self.assertEqual(evs[0]["start_time"], 21.5)

    def test_type_snapped_to_allowed(self):
        evs = vp.normalize_events([{"start_time": 21, "end_time": 22, "type": "Window Change event",
                                    "description": "x"}], self.chunk, ["typing", "window change"])
        self.assertEqual(evs[0]["type"], "window change")
        evs = vp.normalize_events([{"start_time": 21, "end_time": 22, "type": "nonsense",
                                    "description": "x"}], self.chunk, ["typing", "window change"])
        self.assertEqual(evs[0]["type"], "typing")

    def test_empty_description_dropped(self):
        self.assertEqual(vp.normalize_events([{"start_time": 1, "end_time": 2, "type": "t",
                                               "description": ""}], self.chunk), [])
        self.assertEqual(vp.normalize_events(None, self.chunk), [])


class TestPrompt(unittest.TestCase):
    def test_structured_prompt_substitutes(self):
        cfg = vp.load_config(os.devnull)
        p = vp.build_caption_prompt(cfg, "screen recording", ["typing", "scrolling"], ["cursor"],
                                    structured=True, enable_audio=True)
        self.assertIn("analyzing screen recording", p)
        self.assertIn("typing, scrolling", p)
        self.assertIn("cursor", p)
        self.assertIn("audio transcript", p.lower())
        self.assertNotIn("{scenario}", p)

    def test_free_form_prompt(self):
        cfg = vp.load_config(os.devnull)
        p = vp.build_caption_prompt(cfg, "x", None, None, structured=False, enable_audio=False)
        self.assertEqual(p, cfg["prompts"]["caption"])

    def test_repo_config_loads(self):
        cfg = vp.load_config()
        self.assertEqual(cfg["chunking"]["chunk_duration"], 10)
        self.assertIn("start_timestamp-end_timestamp", cfg["prompts"]["caption_summarization"])

    def test_chunk_context_lists_frames_and_cues(self):
        text = vp.chunk_context({"start": 0.0, "end": 10.0}, [1.25, 3.75],
                                [{"start": 2, "end": 4, "text": "hello"}], True)
        self.assertIn("1.2 s, 3.8 s", text)
        self.assertIn("[2.0-4.0] hello", text)


class TestSrt(unittest.TestCase):
    def test_parse_and_window(self):
        srt = ("1\n00:00:01,000 --> 00:00:03,500\nHello <i>there</i>\n\n"
               "2\n00:00:12,000 --> 00:00:14,000\nSecond\nline\n\n")
        with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False, encoding="utf-8") as f:
            f.write(srt)
        try:
            cues = vp.parse_srt(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["text"], "Hello there")
        self.assertEqual(cues[1]["text"], "Second line")
        self.assertEqual([c["text"] for c in vp.cues_in_window(cues, 0, 10)], ["Hello there"])
        self.assertEqual([c["text"] for c in vp.cues_in_window(cues, 3, 13)],
                         ["Hello there", "Second line"])


class TestSearch(unittest.TestCase):
    def _records(self):
        mk = lambda i, cap, ev=None: {"caption": cap, "events": ev or [], "audio": "",
                                      "chunk": {"chunkIdx": i, "start": i * 10.0, "end": i * 10.0 + 10}}
        return [mk(0, "A person opens a red door and walks in."),
                mk(1, "The screen shows a terminal with scrolling text."),
                mk(2, "Someone types into an editor.", [{"start_time": 21, "end_time": 24,
                                                          "type": "typing", "description": "keys pressed"}])]

    def test_bm25_ranks_the_matching_chunk(self):
        recs = self._records()
        scores = vp.BM25([vp.doc_text(r) for r in recs]).score("red door")
        self.assertEqual(scores.index(max(scores)), 0)
        scores = vp.BM25([vp.doc_text(r) for r in recs]).score("typing keys")
        self.assertEqual(scores.index(max(scores)), 2)

    def test_search_over_files(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a", "b"):
                with open(os.path.join(d, f"{name}.captions.jsonl"), "w", encoding="utf-8") as f:
                    for r in self._records():
                        r["chunk"]["file"] = name + ".mp4"
                        f.write(json.dumps(r) + "\n")
            hits = vp.search([d], "terminal scrolling", top_k=3)
            self.assertEqual(len(hits), 2)
            self.assertTrue(all(h["chunkIdx"] == 1 for h in hits))
            self.assertEqual(vp.search([d], "zebra"), [])

    def test_stopwords(self):
        self.assertEqual(vp.tokenize("What is the door"), ["door"])


class TestOutputs(unittest.TestCase):
    def test_caption_lines_and_batches(self):
        recs = [{"caption": "one", "events": [], "chunk": {"start": 0.0, "end": 10.0}},
                {"caption": "", "events": [{"start_time": 12, "end_time": 14, "type": "t",
                                            "description": "two"}], "chunk": {"start": 10.0, "end": 20.0}}]
        lines = vp.caption_lines(recs)
        self.assertEqual(lines, ["**0.0-10.0** one", "**12.0-14.0** two"])
        self.assertEqual(len(vp.batched(list(range(120)), 50)), 3)

    def test_write_tags_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            video = os.path.join(d, "clip.mp4")
            recs = [{"caption": "cap", "events": [], "chunk": {"chunkIdx": 0, "start": 0.0, "end": 10.0}}]
            summary = {"video_summary": "It happened.",
                       "events": [{"start_time": 1.0, "end_time": 2.0, "type": "t", "description": "d"}]}
            tags = vp.write_tags(video, recs, summary, "vlm", "llm", 10.0)
            store = tagstore.load(tags)
            self.assertEqual(store["tracks"]["caption"]["spans"][0]["value"], "cap")
            self.assertEqual(store["tracks"]["event"]["spans"][0]["description"], "d")
            self.assertEqual(store["media"]["duration"], 10.0)
            out, md = vp.write_summary(video, summary, {"scenario": "s", "events": ["t"]},
                                       {"total_chunks_processed": 1}, "llm")
            with open(out, encoding="utf-8") as f:
                env = json.load(f)
            content = json.loads(env["choices"][0]["message"]["content"])
            self.assertEqual(content["video_summary"], "It happened.")
            self.assertEqual(env["usage"]["total_chunks_processed"], 1)
            self.assertTrue(os.path.exists(md))


class TestFaceTrackBridge(unittest.TestCase):
    def _unit_vec(self, seed: int, noise: float = 0.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.normal(size=512).astype(np.float32)
        v /= np.linalg.norm(v)
        if noise > 0:
            n = rng.normal(scale=noise, size=512).astype(np.float32)
            v = v + n
            v /= np.linalg.norm(v)
        return v

    def test_pose_estimation_2d106_and_5pts(self):
        # Frontal 5-point face: eyes at x=38, 74, nose centered at x=56
        frontal_5 = np.array([
            [38.0, 52.0], [74.0, 52.0], [56.0, 72.0], [42.0, 92.0], [70.0, 92.0]
        ], dtype=np.float32)
        pose_f = faces.estimate_pose_degrees(landmarks_5pts=frontal_5)
        self.assertAlmostEqual(pose_f["yaw_deg"], 0.0, places=1)
        self.assertAlmostEqual(pose_f["roll_deg"], 0.0, places=1)

        # Profile 2D-106 face turned sharply to the right
        pts106 = np.zeros((106, 2), dtype=np.float32)
        pts106[1] = [20.0, 60.0]       # left cheek
        pts106[31] = [120.0, 60.0]     # right cheek
        pts106[33:43] = [45.0, 50.0]   # left eye
        pts106[87:97] = [95.0, 50.0]   # right eye
        pts106[86] = [110.0, 70.0]     # nose tip shifted near right cheek
        pose_p = faces.estimate_pose_degrees(landmarks_106=pts106)
        self.assertGreater(abs(pose_p["yaw_deg"]), 45.0)

    def test_gallery_sqlite_matching_and_clustering_exemplars(self):
        alice_base = self._unit_vec(101)
        bob_unknown_base = self._unit_vec(202)

        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "gallery.db")
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE identities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL DEFAULT 'locked',
                    threshold REAL DEFAULT 0.50
                );
                CREATE TABLE face_crops (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_id INTEGER,
                    model_name TEXT NOT NULL,
                    embedding BLOB NOT NULL
                );
            """)
            conn.execute("INSERT INTO identities (id, name, status, threshold) VALUES (1, 'Alice', 'locked', 0.55)")
            conn.execute("INSERT INTO identities (id, name, status, threshold) VALUES (2, 'QuarantinedPerson', 'quarantined', 0.45)")
            conn.execute("INSERT INTO face_crops (identity_id, model_name, embedding) VALUES (1, 'Alice', ?)",
                         (alice_base.tobytes(),))
            conn.execute("INSERT INTO face_crops (identity_id, model_name, embedding) VALUES (2, 'QuarantinedPerson', ?)",
                         (bob_unknown_base.tobytes(),))
            conn.commit()
            conn.close()

            gallery = faces.FaceGallery.load(db_path)
            self.assertEqual(gallery.unique_names, ["Alice"])

            # Create observations for Alice (3 frames: 1 profile, 1 small box, 1 large frontal)
            # and an unknown person (2 frames, since QuarantinedPerson was excluded)
            frontal_5 = np.array([[38, 52], [74, 52], [56, 72], [42, 92], [70, 92]], dtype=np.float32)
            profile_5 = np.array([[38, 52], [74, 52], [78, 72], [42, 92], [70, 92]], dtype=np.float32)

            obs = [
                # Alice frame 1: turned profile (high yaw)
                faces.FaceObservation(
                    timestamp=1.5, frame_path="f001.jpg",
                    embedding=self._unit_vec(101, noise=0.01),
                    bbox=[10, 10, 150, 150], prob=0.95, landmarks_5pts=profile_5,
                ),
                # Alice frame 2: frontal and large box -> should win #1 exemplar
                faces.FaceObservation(
                    timestamp=3.5, frame_path="f002.jpg",
                    embedding=self._unit_vec(101, noise=0.005),
                    bbox=[10, 10, 210, 210], prob=0.98, landmarks_5pts=frontal_5,
                ),
                # Alice frame 3: frontal but tiny box
                faces.FaceObservation(
                    timestamp=6.0, frame_path="f003.jpg",
                    embedding=self._unit_vec(101, noise=0.01),
                    bbox=[10, 10, 60, 60], prob=0.92, landmarks_5pts=frontal_5,
                ),
                # Unknown person frames
                faces.FaceObservation(
                    timestamp=12.0, frame_path="f004.jpg",
                    embedding=self._unit_vec(202, noise=0.01),
                    bbox=[20, 20, 140, 140], prob=0.96, landmarks_5pts=frontal_5,
                ),
                faces.FaceObservation(
                    timestamp=14.0, frame_path="f005.jpg",
                    embedding=self._unit_vec(202, noise=0.01),
                    bbox=[20, 20, 145, 145], prob=0.95, landmarks_5pts=frontal_5,
                ),
            ]

            annotated, identities = faces.cluster_observations(obs, gallery=gallery)
            self.assertIn("Alice", identities)
            self.assertIn("FACE_0", identities)
            self.assertEqual(identities["Alice"]["source"], "gallery")
            # Best exemplar for Alice must be the large frontal frame f002.jpg at t=3.5s
            best_ex = identities["Alice"]["exemplars"][0]
            self.assertEqual(best_ex["frame_path"], "f002.jpg")
            self.assertEqual(best_ex["timestamp"], 3.5)
            self.assertLessEqual(abs(best_ex["yaw_deg"]), 5.0)

            # Write face track into clip.tags.json beside existing caption track, then search for "Alice"
            video = os.path.join(d, "clip.mp4")
            recs = [
                {"caption": "A person walks into the hallway.", "events": [],
                 "chunk": {"chunkIdx": 0, "file": "clip.mp4", "start": 0.0, "end": 10.0}},
                {"caption": "Another person sits at a desk.", "events": [],
                 "chunk": {"chunkIdx": 1, "file": "clip.mp4", "start": 10.0, "end": 20.0}},
            ]
            with open(os.path.join(d, "clip.captions.jsonl"), "w", encoding="utf-8") as cf:
                for r in recs:
                    cf.write(json.dumps(r) + "\n")
            vp.write_tags(video, recs, {"video_summary": "test", "events": []}, "vlm", "llm", 20.0)
            faces.write_face_track(video, annotated, identities, duration=20.0, gallery_path=db_path)

            store = tagstore.load(os.path.join(d, "clip.tags.json"))
            self.assertIn("caption", store["tracks"])
            self.assertIn("face", store["tracks"])
            self.assertEqual(tagstore.value_at(store, "face", 3.5), "Alice")
            self.assertEqual(tagstore.value_at(store, "face", 13.0), "FACE_0")

            # Searching for "Alice" across the directory finds chunk 0 (where Alice appears)
            hits = vp.search([d], "Alice", top_k=5)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["chunkIdx"], 0)
            self.assertIn("Alice", hits[0]["faces"])


if __name__ == "__main__":
    unittest.main()
