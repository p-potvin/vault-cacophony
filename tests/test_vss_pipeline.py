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


if __name__ == "__main__":
    unittest.main()
