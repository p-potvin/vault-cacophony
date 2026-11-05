"""The tag store lives in the media's main sidecar (`tags`), never in a separate .tags.json."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import tagstore  # noqa: E402


def touch(path, data=b"x"):
    with open(path, "wb") as f:
        f.write(data)
    return path


class TagStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def side(self, media):
        with open(tagstore.sidecar_path(media), encoding="utf-8") as f:
            return json.load(f)

    def test_a_new_sidecar_gets_the_tags_and_the_v2_envelope(self):
        media = touch(os.path.join(self.d, "ep.mkv"))
        store = tagstore.load(media)
        self.assertEqual(store["tracks"], {})
        tagstore.set_media(store, media, 12.5)
        tagstore.put_track(store, "language", [{"start": 0, "end": 12.5, "value": "en-US"}], "asr", model="m")
        path = tagstore.save(store)
        self.assertEqual(path, media + ".json")
        side = self.side(media)
        self.assertEqual(side["schema"], 2)
        self.assertEqual(side["vw"]["kind"], "video")
        self.assertEqual(side["master"]["basename"], "ep.mkv")
        self.assertEqual(side["tags"]["duration"], 12.5)
        self.assertEqual(side["tags"]["tracks"]["language"]["spans"][0]["value"], "en-US")
        self.assertEqual(os.listdir(self.d), ["ep.mkv", "ep.mkv.json"])

    def test_other_keys_and_other_passes_survive(self):
        media = touch(os.path.join(self.d, "ep.mp4"))
        with open(media + ".json", "w", encoding="utf-8") as f:
            json.dump({"schema": 2, "vw": {"id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"}, "duration": 99.0}, f)
        a, b = tagstore.load(media), tagstore.load(media)          # two passes read the same state
        tagstore.put_track(a, "language", [{"start": 0, "end": 1, "value": "fr"}], "asr")
        tagstore.put_track(b, "speaker", [{"start": 0, "end": 1, "value": "SPEAKER_00"}], "speaker")
        tagstore.save(a)
        tagstore.save(b)                                           # must not drop a's track
        side = self.side(media)
        self.assertEqual(sorted(side["tags"]["tracks"]), ["language", "speaker"])
        self.assertEqual(side["vw"]["id"], "01ARZ3NDEKTSV4RRFFQ69G5FAV")
        self.assertEqual(side["duration"], 99.0)                   # vault-explorer's ffprobe key
        c = tagstore.load(media)
        tagstore.put_track(c, "speaker", [], "speaker")            # an empty pass removes its track
        tagstore.save(c)
        self.assertEqual(sorted(self.side(media)["tags"]["tracks"]), ["language"])

    def test_an_existing_sidecar_name_is_kept(self):
        media = touch(os.path.join(self.d, "clip.mp4"))
        touch(media + ".meta.json", b'{"isValid": true}')
        store = tagstore.load(media)
        tagstore.put_track(store, "face", [{"start": 0, "end": 2, "value": "Alice"}], "face")
        self.assertEqual(tagstore.save(store), media + ".meta.json")
        self.assertFalse(os.path.exists(media + ".json"))
        self.assertTrue(self.side(media)["isValid"])

    def test_a_stem_sidecar_counts_only_when_no_other_media_shares_the_stem(self):
        media = touch(os.path.join(self.d, "a.mp4"))
        touch(os.path.join(self.d, "a.json"), b"{}")
        self.assertEqual(tagstore.sidecar_path(media), os.path.join(self.d, "a.json"))
        touch(os.path.join(self.d, "a.jpg"))                        # a poster now shares the stem
        self.assertEqual(tagstore.sidecar_path(media), media + ".json")

    def test_srt_round_trip(self):
        srt = os.path.join(self.d, "c.srt")
        tagstore.write_srt(srt, [(0.0, 1.5, "hello\nthere"), {"start": 61.25, "end": 3600.0, "text": "bye"}])
        self.assertEqual(tagstore.read_srt(srt), [{"start": 0.0, "end": 1.5, "text": "hello there"},
                                                  {"start": 61.25, "end": 3600.0, "text": "bye"}])
        self.assertEqual(tagstore.read_srt(os.path.join(self.d, "missing.srt")), [])


if __name__ == "__main__":
    unittest.main()
