"""Prosody / natural-delivery layer: markup, measurement, voice store, and the places voices are used."""

import json
import random
import re
import sqlite3
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import speech_markup as sm  # noqa: E402


class _FixedToneBridge:
    """Stands in for vault-raggae: deterministic tone choice, no network."""

    def score_tones_batch(self, texts, profiles):
        return [(p.tone_palette[i % len(p.tone_palette)], 0.5, "lexical") for i, p in enumerate(profiles)]


LINES = [
    "I think the telemetry is wrong, but we should check the coolant loop again because the sensor drifted overnight.",
    "[sarcastic] Oh sure, [sighs] that is *definitely* going to work... uh, totally. [laughing]",
    "Wait, hold on, the valve was open when I left, so who closed it while we were all downstairs?",
    "No way.",
]


# -- markup -------------------------------------------------------------------

def test_ssml_is_valid_and_round_trips():
    utt = sm.humanize(LINES[0], random.Random(1), tone="excited",
                      prosody=sm.TONE_EMOTION["excited"].to_prosody())
    utt.add(sm.Emphasis("again"))
    utt.add(sm.Vocal("chuckle"))
    ssml = utt.to_ssml()
    root = __import__("xml.etree.ElementTree", fromlist=["x"]).fromstring(ssml)
    assert root.tag == f"{{{sm.SSML_NS}}}speak"
    assert root.get("version") == "1.1"
    assert 'rate="' in ssml and "<break time=" in ssml and "<emphasis" in ssml
    back = sm.parse_ssml(ssml)
    assert back.to_ssml() == ssml
    assert back.tone == "excited"
    assert back.words() == utt.words()


def test_parse_tagged_understands_llm_and_engine_dialects():
    utt = sm.parse_tagged(LINES[1])
    kinds = [type(t).__name__ for t in utt.tokens]
    assert utt.tone == "sarcastic"
    assert sm.Vocal("sigh") in utt.tokens and sm.Vocal("laugh") in utt.tokens
    assert sm.Emphasis("definitely") in utt.tokens
    assert sm.Filler("uh") in utt.tokens
    assert "Break" in kinds
    assert utt.words() == "Oh sure, that is definitely going to work, uh, totally."


@pytest.mark.parametrize("engine", ["kokoro", "magpie", "cosyvoice", "chatterbox", "qwen3_tts"])
def test_plain_engines_never_receive_markup(engine):
    for line in LINES + ["<speak><prosody rate='110%'>Hi <break time='300ms'/> there</prosody></speak>"]:
        out = sm.sanitize_for_engine(line, engine)
        assert not re.search(r"[\[\]<>*]", out), out
        assert out


def test_omnivoice_and_fish_dialects():
    utt = sm.parse_tagged(LINES[1])
    omni = sm.render(utt, "omnivoice")
    assert set(re.findall(r"\[[^\]]+\]", omni)) <= sm.OMNIVOICE_ALLOWED
    assert "*" not in omni
    fish = sm.render(utt, "fish_audio")
    assert fish.startswith("[sarcastic]")
    assert "[emphasis] definitely" in fish and "[sighs]" in fish


def test_sentence_end_absorbs_a_following_pause():
    assert sm.sanitize_for_engine("[icy tone] Fine. [break] Whatever.", "kokoro") == "Fine. Whatever."


def test_emotion_moves_prosody_in_the_expected_direction():
    excited = sm.TONE_EMOTION["excited"].to_prosody()
    calm = sm.TONE_EMOTION["calm"].to_prosody()
    assert excited.rate > 1.0 > calm.rate
    assert excited.pitch_st > 0 > calm.pitch_st
    assert sm.Prosody.from_ssml_attrs(excited.ssml_attrs()).rate == pytest.approx(excited.rate, abs=0.01)


def test_humanize_follows_the_disfluency_literature():
    dis = sm.DisfluencyModel(filled_pause_rate=0.2, um_ratio=0.5)
    preceded = total = 0
    after = {"uh": [], "um": []}
    for seed in range(300):
        utt = sm.humanize(LINES[2], random.Random(seed), disfluency=dis)
        toks = utt.tokens
        for i, t in enumerate(toks):
            if isinstance(t, sm.Filler):
                total += 1
                preceded += isinstance(toks[i - 1], sm.Break) if i else 0
                if i + 1 < len(toks) and isinstance(toks[i + 1], sm.Break):
                    after[t.word].append(toks[i + 1].ms)
    assert total > 50
    # Silence before most filled pauses (Dall et al. 2016: ~83%).
    assert 0.65 < preceded / total < 0.95
    # "um" announces a longer delay than "uh" (Clark & Fox Tree 2002).
    assert np.median(after["um"]) > np.median(after["uh"])


def test_humanize_breathes_before_long_turns_but_not_on_barge_in():
    long_line = " ".join(["word"] * 30)
    taking = sm.humanize(long_line, random.Random(0))
    assert taking.tokens[0] == sm.Vocal("breath")
    barging = sm.humanize(long_line, random.Random(0), interrupting=True)
    assert sm.Vocal("breath") not in barging.tokens


def test_humanize_keeps_inline_cues_in_place():
    utt = sm.humanize(sm.parse_tagged("Fine, [sighs] I will do it."), random.Random(3))
    i = utt.tokens.index(sm.Vocal("sigh"))
    assert any(isinstance(t, sm.Text) and "Fine" in t.text for t in utt.tokens[:i])


# -- the Smart Picker on top of it ---------------------------------------------------

def test_smart_picker_renderings_are_stable_under_engine_sanitizing():
    from cacophony.vocal_habits import SmartVocalPicker

    for seed in range(12):
        picker = SmartVocalPicker(bridge=_FixedToneBridge(), seed=seed)
        for engine in ("fish_audio", "omnivoice", "kokoro", "cosyvoice"):
            for speaker in ("MARGOT", "DESMOND", "KAI", "CHLOE", "CLOPEUX"):
                for n, line in enumerate(LINES):
                    ann = picker.annotate_turn(line, speaker, engine_type=engine, turn_num=seed * 10 + n,
                                               transition_type="interruption" if n % 2 else "turn_switch")
                    assert sm.sanitize_for_engine(ann.rendered_text, engine) == ann.rendered_text
                    assert sm.parse_ssml(ann.ssml).to_ssml() == ann.ssml
                    assert ann.speed_scale > 0
                    if engine in ("kokoro", "cosyvoice"):
                        assert not re.search(r"[\[\]*]", ann.rendered_text)


def test_inline_llm_tone_outranks_the_scorer():
    from cacophony.vocal_habits import SmartVocalPicker

    ann = SmartVocalPicker(bridge=_FixedToneBridge(), seed=0).annotate_turn(
        "[worried] The pressure is still climbing.", "KAI", engine_type="fish_audio")
    assert ann.selected_tone == "worried" and ann.tone_method == "inline"
    assert ann.rendered_text.startswith("[worried]")


# -- measurement ------------------------------------------------------------------

def _syllables(n, sr=16000):
    t = np.arange(int(sr * n / 4)) / sr
    env = 0.2 + 0.8 * np.abs(np.sin(np.pi * 4 * t))
    return (0.3 * env * np.sin(2 * np.pi * 150 * t)).astype(np.float32)


def test_prosody_measures_pauses_and_rate():
    import prosody

    sr = 16000
    audio = np.concatenate([_syllables(8), np.zeros(int(0.4 * sr)), _syllables(6),
                            np.zeros(int(0.7 * sr)), _syllables(8)]).astype(np.float32)
    m = prosody.measure(audio)
    assert m["pause_count"] == 2
    assert 400 <= m["pause_median_ms"] <= 700
    assert 20 <= m["syllables_est"] <= 24
    assert 3.5 <= m["articulation_rate_sps"] <= 4.5
    assert m["speech_rate_sps"] < m["articulation_rate_sps"]
    json.dumps(m)  # everything stored must be plain JSON


# -- the voice store --------------------------------------------------------------

def test_store_migrates_a_v1_database(tmp_path):
    import voices

    path = tmp_path / "voices.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO meta VALUES ('schema', '1');
        CREATE TABLE voice (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, centroid BLOB NOT NULL,
                            samples INTEGER NOT NULL DEFAULT 1, created TEXT, updated TEXT, notes TEXT DEFAULT '');
    """)
    db.execute("INSERT INTO voice (name, centroid) VALUES ('old', ?)", (np.ones(512, np.float32).tobytes(),))
    db.commit()
    db.close()
    with voices.VoiceStore(str(path)) as store:
        cols = {r["name"] for r in store.db.execute("PRAGMA table_info(voice)")}
        assert {"habits", "prosody", "ref_wav", "ref_text"} <= cols
        assert store.db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0] == str(voices.SCHEMA)
        assert store.get_prosody("old") is None


def test_store_derives_prosody_from_observations(tmp_path):
    import prosody
    import voices

    sr = 16000
    audio = np.concatenate([_syllables(8), np.zeros(int(0.5 * sr)), _syllables(8)]).astype(np.float32)
    vec = np.random.default_rng(0).normal(size=512).astype(np.float32)
    with voices.VoiceStore(str(tmp_path / "v.db")) as store:
        obs = store.observe(vec, voice=None, label="X", speech_seconds=len(audio) / sr,
                            metrics=prosody.measure(audio))
        store.enroll("X", vec, observation_id=obs,
                     ref_text="So um I was thinking uh that we could maybe try the other valve first and "
                              "then uh check the loop again before anyone goes home tonight okay")
        doc = store.derive_prosody("X")
        assert doc["standard"]["markup"] == "SSML 1.1"
        assert doc["measured"]["observations"] == 1
        assert 0.8 <= doc["baseline"]["rate"] <= 1.25
        assert doc["pauses"]["medium_ms"] >= 250
        assert doc["disfluency"]["filled_pause_rate"] > 0
        assert 0 < doc["disfluency"]["um_ratio"] < 1

        store.set_prosody("X", {"pauses": {"long_ms": 2000}})
        merged = store.get_prosody("X")
        assert merged["pauses"]["long_ms"] == 2000 and "medium_ms" in merged["pauses"]
        summary = store.prosody_summary("X")
        assert summary["f0_hz"] and summary["pause_ms"]
        with pytest.raises(KeyError):
            store.set_prosody("nobody", {})


def test_resolve_profile_reads_the_store(tmp_path, monkeypatch):
    import voices
    from cacophony import habit_profiles

    path = tmp_path / "v.db"
    monkeypatch.setenv(habit_profiles.VOICE_STORE_ENV, str(path))
    builtin_kai = habit_profiles.builtin_profiles()["KAI"]
    with voices.VoiceStore(str(path)) as store:
        # Habits saved before pauses/disfluency existed: those must fall back to the built-in.
        legacy = builtin_kai.to_dict()
        legacy.pop("pauses")
        legacy.pop("disfluency")
        legacy["filler_rate"] = 0.11
        store.set_habits("normal-male-slightly-faster-cadence", legacy)
        store.set_prosody("normal-male-slightly-faster-cadence",
                          {"pauses": {"medium_ms": 777.0}, "baseline": {"rate": 1.2, "pitch_st": -1.0}})

    prof = habit_profiles.resolve_profile("KAI", "normal-male-slightly-faster-cadence")
    assert prof.filler_rate == 0.11                       # store habits win
    assert prof.disfluency == builtin_kai.disfluency      # missing fields fall back to the persona
    assert prof.pauses.medium_ms == 777.0                 # stored prosody overlays
    assert prof.baseline.pitch_st == -1.0
    assert prof.baseline.rate == 1.0                      # reference rate is the clone's business

    monkeypatch.setenv(habit_profiles.VOICE_STORE_ENV, str(tmp_path / "missing.db"))
    assert habit_profiles.resolve_profile("KAI").filler_rate == builtin_kai.filler_rate


# -- HF brain streaming (Devin review, PR #29) ---------------------------------------

def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def test_batch_generation_streams_when_asked():
    from cacophony.hf_brain import HfDialogueBrain

    seen = {}

    class FakeCompletions:
        def create(self, **kw):
            seen.update(kw)
            body = "1. MARGOT: The loop is failing.\n2. DESMOND: Obviously, it always does."
            return iter([_chunk(body[i:i + 7]) for i in range(0, len(body), 7)])

    brain = HfDialogueBrain.__new__(HfDialogueBrain)
    brain.model = "test-model"
    brain.temperature = 0.5
    brain.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    lines = []
    out = brain.generate_turns_batch(
        personas={"MARGOT": "", "DESMOND": ""},
        turn_plan=[{"speaker": "MARGOT"}, {"speaker": "DESMOND"}],
        history=[], stream=True, on_line=lambda i, line: lines.append(line))
    assert seen["stream"] is True
    assert out == ["The loop is failing.", "Obviously, it always does."]
    assert lines == ["1. MARGOT: The loop is failing.", "2. DESMOND: Obviously, it always does."]


def test_orchestrator_batch_path_passes_stream(tmp_path):
    from cacophony.acoustic_mixer import AcousticRoomMixer
    from cacophony.orchestrator import STORY_SCRIPTS, CacophonyOrchestrator, Persona
    from cacophony.turn_taker import TurnTakingScheduler
    from cacophony.vocal_habits import SmartVocalPicker

    class Brain:
        stream_flags = []

        def generate_turns_batch(self, personas, turn_plan, history, narrative_hint=None,
                                 max_tokens=1200, stream=False, on_line=None):
            self.stream_flags.append(stream)
            return ["[excited] The pressure is rising, but we can still, uh, isolate the loop."] * len(turn_plan)

    class Tts:
        gpu_devices = [0]
        jobs = []

        def synthesize_batch_parallel(self, jobs):
            self.jobs.extend(jobs)
            return [(np.full(12000, 0.1, np.float32), 24000) for _ in jobs]

    orch = CacophonyOrchestrator.__new__(CacophonyOrchestrator)
    orch.args = SimpleNamespace(turns=4, batch_size=4, duration_sec=120.0, rt60=0.2, enable_ramblers=False,
                                rambler_gain_db=-18.0, engine_type="kokoro", history_turns=8, stream=True)
    orch.script = STORY_SCRIPTS["testbench"]
    orch.seed_topic = orch.script["seed_topic"]
    orch.out_dir = tmp_path
    orch.work_dir = tmp_path / "work"
    orch.work_dir.mkdir()
    orch.personas = [Persona(i, n, f"voice-{i}", "p", f"SPEAKER_{i}") for i, n in enumerate(["MARGOT", "KAI"])]
    orch.brain = Brain()
    orch.tts = Tts()
    orch.scheduler = TurnTakingScheduler()
    orch.mixer = AcousticRoomMixer(sample_rate=24000, rt60=0.2)
    orch.rambler_pool = None
    orch.run_error = None
    orch.smart_picker = SmartVocalPicker(bridge=_FixedToneBridge(), seed=0)
    orch.run()

    assert orch.brain.stream_flags == [True]
    # Kokoro takes plain text: no tone tag, no bracket, no asterisk reaches it.
    for job in orch.tts.jobs:
        assert not re.search(r"[\[\]*<>]", job["text"]), job["text"]
    markup = json.loads((tmp_path / "prosody_markup.json").read_text(encoding="utf-8"))
    assert markup["standard"] == "SSML 1.1" and len(markup["turns"]) == 4
    assert all(t["ssml"].startswith("<speak") for t in markup["turns"])


# -- voice preview endpoint (Devin review, PR #29) -------------------------------------

def test_voice_preview_is_streamed_cached_and_bounded(tmp_path, monkeypatch):
    from http.server import ThreadingHTTPServer
    from cacophony import ui_server

    import soundfile as sf
    (tmp_path / "voices-store").mkdir()
    sf.write(tmp_path / "voices-store" / "alice.wav", np.zeros(48000, np.float32), 24000)
    monkeypatch.setattr(ui_server, "REPO_DIR", tmp_path)
    assert ui_server.find_ref_wav("alice") == tmp_path / "voices-store" / "alice.wav"

    server = ThreadingHTTPServer(("127.0.0.1", 0), ui_server.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/api/voice-preview?name="
    try:
        with urllib.request.urlopen(base + "alice") as resp:
            body = resp.read()
            etag = resp.headers["ETag"]
            assert resp.headers["Content-Type"] == "audio/wav"
        assert len(body) == (tmp_path / "voices-store" / "alice.wav").stat().st_size

        req = urllib.request.Request(base + "alice", headers={"If-None-Match": etag})
        with pytest.raises(urllib.error.HTTPError) as not_modified:
            urllib.request.urlopen(req)
        assert not_modified.value.code == 304

        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(base + "nobody")
        assert missing.value.code == 404

        held = [ui_server._preview_slots.acquire(blocking=False) for _ in range(ui_server.PREVIEW_MAX_CONCURRENT)]
        assert all(held)
        try:
            with pytest.raises(urllib.error.HTTPError) as busy:
                urllib.request.urlopen(base + "alice")
            assert busy.value.code == 503
        finally:
            for _ in held:
                ui_server._preview_slots.release()
    finally:
        server.shutdown()
        server.server_close()


# -- PersonaPlex download source (Devin review, PR #29) ---------------------------------

def test_personaplex_download_follows_the_model_spec():
    import mint_voice_refs

    spec = json.loads((REPO_DIR / "audio.cpp" / "model_specs" / "personaplex.json").read_text(encoding="utf-8"))
    package = mint_voice_refs.personaplex_package()
    assert package["repo"] == spec["package_defaults"]["download"]["repo"]
    assert package["id"] == next(p["id"] for p in spec["packages"] if p.get("default"))
    assert all(Path(f).parent.name == "PersonaPlex-GGUF" for f in package["local_files"])
    assert any(f.endswith("personaplex-7b-v1-q4_k.gguf") for f in package["local_files"])


# -- Devin review, PR #30 -------------------------------------------------------------

def test_ssml_fragments_and_broken_markup_never_reach_plain_engines():
    assert sm.sanitize_for_engine('Wait <break time="300ms"/> now', "kokoro") == "Wait... now"
    assert sm.sanitize_for_engine('Wait <break time="300ms"> now', "kokoro") == "Wait now"
    assert sm.sanitize_for_engine("x < y and y > z", "kokoro") == "x < y and y > z"


def test_scoped_prosody_is_not_widened_to_the_whole_line():
    whole = sm.parse_any('<speak><prosody rate="120%">A B C</prosody></speak>')
    assert whole.prosody.rate == pytest.approx(1.2)
    nested = sm.parse_any('<speak><prosody rate="120%">A <prosody rate="50%">B</prosody> C</prosody></speak>')
    assert nested.prosody.rate == pytest.approx(1.2) and nested.words() == "A B C"
    partial = sm.parse_any('<speak>A <prosody rate="50%">B</prosody> C</speak>')
    assert partial.prosody.rate == 1.0 and partial.words() == "A B C"


def test_emotion_coordinates_survive_ssml():
    from cacophony.vocal_habits import SmartVocalPicker

    ann = SmartVocalPicker(bridge=_FixedToneBridge(), seed=0).annotate_turn(
        "[worried] The pressure is still climbing.", "KAI", engine_type="ssml")
    back = sm.parse_ssml(ann.ssml)
    assert back.emotion == sm.TONE_EMOTION["worried"]
    assert ann.markup["emotion"]["arousal"] == sm.TONE_EMOTION["worried"].arousal


def test_hand_set_prosody_survives_rederivation(tmp_path):
    import prosody
    import voices

    sr = 16000
    audio = np.concatenate([_syllables(8), np.zeros(int(0.5 * sr)), _syllables(8)]).astype(np.float32)
    vec = np.random.default_rng(1).normal(size=512).astype(np.float32)
    with voices.VoiceStore(str(tmp_path / "v.db")) as store:
        obs = store.observe(vec, label="X", speech_seconds=len(audio) / sr, metrics=prosody.measure(audio))
        store.enroll("X", vec, observation_id=obs)
        derived_long = store.derive_prosody("X")["pauses"]["long_ms"]
        assert derived_long != 2000
        store.set_prosody("X", {"pauses": {"long_ms": 2000}})
        again = store.derive_prosody("X")
        assert again["pauses"]["long_ms"] == 2000
        assert again["overrides"] == {"pauses": {"long_ms": 2000}}
        assert again["pauses"]["medium_ms"] and again["measured"]["observations"] == 1


def test_startup_sync_keeps_edited_habits(tmp_path, monkeypatch):
    import voices
    from cacophony import habit_profiles
    from cacophony.raggae_bridge import sync_custom_voices_to_store

    path = str(tmp_path / "v.db")
    monkeypatch.setenv(habit_profiles.VOICE_STORE_ENV, path)
    voice = "normal-male-slightly-faster-cadence"
    sync_custom_voices_to_store(store_path=path)
    with voices.VoiceStore(path) as store:
        edited = store.get_habits(voice)
        edited["filler_rate"] = 0.11
        store.set_habits(voice, edited)

    sync_custom_voices_to_store(store_path=path)  # what every orchestrator start does
    assert habit_profiles.resolve_profile("KAI", voice).filler_rate == 0.11

    sync_custom_voices_to_store(store_path=path, overwrite_habits=True)
    builtin = habit_profiles.builtin_profiles()["KAI"].filler_rate
    assert habit_profiles.resolve_profile("KAI", voice).filler_rate == builtin
