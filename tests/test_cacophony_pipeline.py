"""Comprehensive test suite for the Cacophony multi-speaker pipeline."""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

REPO_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cacophony.hf_brain import HfDialogueBrain
from cacophony.rambler_pool import RamblerPool, AudioCut
from cacophony.turn_taker import TurnTakingScheduler, TransitionType
from cacophony.acoustic_mixer import AcousticRoomMixer
from cacophony.orchestrator import (
    CacophonyOrchestrator,
    Persona,
    STORY_SCRIPTS,
    _generate_turn_once,
    _hf_429_diagnostic,
    _recent_history,
    _write_run_status,
)


def test_hf_brain_live():
    """Verify live remote HF Inference Client dialogue generation under real conditions."""
    brain = HfDialogueBrain()
    history = [{"role": "user", "content": "What is 2 + 2? Answer in one short sentence."}]
    reply = brain.generate_turn(
        speaker_name="TEST_BOT",
        system_persona="A concise testing assistant.",
        history=history,
        max_tokens=40,
        stream=False,
    )
    assert reply, "Dialogue brain must return non-empty reply"
    assert "*" not in reply, "Reply must not contain markdown asterisks"
    assert "4" in reply.lower() or "four" in reply.lower(), "Reply must correctly answer the prompt"


def test_rambler_pool_real_audio():
    """Verify RamblerPool discovers real LibriSpeech audio cuts and loads 24kHz audio."""
    pool = RamblerPool()
    assert len(pool.speakers) > 0, "Pool must discover speakers in LibriSpeech dataset"
    
    cut = pool.sample_cut()
    assert cut is not None, "Pool must return a sampled AudioCut"
    assert cut.file_path.exists(), f"Sampled file must exist: {cut.file_path}"
    
    audio = cut.load_audio(target_sr=24000, max_duration=2.0)
    assert isinstance(audio, np.ndarray), "Loaded audio must be numpy array"
    assert audio.ndim == 1, "Audio cut must be single-channel mono"
    assert len(audio) > 0, "Audio array must not be empty"
    assert not np.isnan(audio).any(), "Audio array must not contain NaNs"
    assert not np.isinf(audio).any(), "Audio array must not contain Infs"


def test_turn_taking_scheduler():
    """Verify turn-taking timing calculations, interruptions, and backchannel offsets."""
    scheduler = TurnTakingScheduler(interruption_prob=0.0)
    # With interruption_prob=0.0, turn switch should always happen after previous turn
    offset, trans = scheduler.sample_turn_switch_offset(prev_offset=2.0, prev_duration=3.0)
    assert trans == TransitionType.TURN_SWITCH
    assert offset >= 5.0, "Turn switch must start after prev_offset + prev_duration"

    # Test backchannel landing position
    bc_offset = scheduler.sample_backchannel_offset(utt_offset=10.0, utt_duration=4.0)
    assert 10.0 < bc_offset < 14.0, "Backchannel must fall inside utterance bounds"


def test_acoustic_room_mixer_render_and_persistence(tmp_path):
    """Verify 3D room simulation, RIR convolution, stereo rendering, and disk persistence."""
    mixer = AcousticRoomMixer(sample_rate=24000, rt60=0.25)
    
    # Generate synthetic speech burst
    t = np.linspace(0, 1.0, 24000, endpoint=False, dtype=np.float32)
    sig_a = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    sig_b = 0.5 * np.sin(2 * np.pi * 880 * t)  # 880 Hz tone

    mixer.add_event(
        speaker_id="SPEAKER_A",
        audio=sig_a,
        offset_seconds=0.2,
        gain_db=0.0,
        text="Speaker A tone",
    )
    mixer.add_event(
        speaker_id="SPEAKER_B",
        audio=sig_b,
        offset_seconds=1.0,
        gain_db=-3.0,
        text="Speaker B tone",
    )

    stereo_mix, manifest = mixer.render(apply_reverb=True)
    
    assert stereo_mix.shape[0] == 2, "Rendered mix must be 2-channel stereo"
    assert stereo_mix.shape[1] > 24000 * 1.5, "Mix length must span all scheduled events"
    assert not np.isnan(stereo_mix).any(), "Stereo mix must have no NaNs"
    assert not np.isinf(stereo_mix).any(), "Stereo mix must have no Infs"
    
    rms = np.sqrt(np.mean(stereo_mix ** 2))
    assert rms > 0.01, "Mix must contain audible energy"

    # Test persistence to disk and reload
    out_wav = tmp_path / "test_render.wav"
    mixer.save_wav(out_wav, stereo_mix)
    assert out_wav.exists(), "Saved wav file must exist on disk"

    # Reload from disk and verify format
    reloaded_data, sr = sf.read(str(out_wav))
    assert sr == 24000, "Reloaded sample rate must be 24kHz"
    assert reloaded_data.shape[1] == 2, "Reloaded audio must have 2 channels"
    assert len(reloaded_data) == stereo_mix.shape[1], "Reloaded length must match rendered length"

    # Verify manifest structure
    assert len(manifest) == 2, "Manifest must record both events"
    assert manifest[0]["speaker"] == "SPEAKER_A"
    assert manifest[1]["speaker"] == "SPEAKER_B"
    assert manifest[0]["start"] == 0.2
    assert manifest[1]["start"] == 1.0


def test_interleaved_multi_speaker_scheduling():
    """Verify barge-in interruptions (0.10s-0.20s) and max-2 concurrent speaker staggering."""
    scheduler = TurnTakingScheduler()

    # Interruption barge-in: must overlap by 0.10s - 0.20s
    prev_end = 1.0 + 2.5  # 3.5s
    barge_in = scheduler.sample_interruption_offset(prev_offset=1.0, prev_duration=2.5, min_overlap=0.10, max_overlap=0.20)
    overlap = prev_end - barge_in
    assert 0.09 <= overlap <= 0.21, f"Barge-in overlap ({overlap:.3f}s) must be within 0.10s - 0.20s"

    # Fast reply offset
    fast_reply = scheduler.sample_fast_reply_offset(target_offset=1.0, target_duration=2.5, min_gap=0.08, max_gap=0.20)
    assert 3.58 <= fast_reply <= 3.75, f"Fast reply offset ({fast_reply}) must trigger immediately after target finishes"

    # Staggering: if Speaker 0 ([1.0, 3.0]) and Speaker 1 ([2.0, 4.0]) are both active at t=2.5,
    # Speaker 2 wanting to speak at t=2.5 must be shifted to >= 3.05s (when Speaker 0 ends)
    scheduled = [(1.0, 3.0, 0), (2.0, 4.0, 1)]
    staggered_start = scheduler.stagger_turn(
        proposed_start=2.5,
        duration=2.0,
        speaker_idx=2,
        scheduled_turns=scheduled,
        max_concurrent=2,
    )
    assert staggered_start >= 3.05, f"Staggered start ({staggered_start}) must wait until Speaker 0 finishes at 3.0"


def test_four_speaker_acoustic_room_mixing(tmp_path):
    """Verify 4-speaker spatial positioning and mixing with tuned RT60=0.20s."""
    mixer = AcousticRoomMixer(sample_rate=24000, rt60=0.20)
    for spk_id in ["SPEAKER_0", "SPEAKER_1", "SPEAKER_2", "SPEAKER_3"]:
        assert spk_id in mixer.predefined_positions

    t = np.linspace(0, 0.5, 12000, endpoint=False, dtype=np.float32)
    sig = 0.3 * np.sin(2 * np.pi * 440 * t)

    for i in range(4):
        mixer.add_event(
            speaker_id=f"SPEAKER_{i}",
            audio=sig,
            offset_seconds=i * 0.4,
            gain_db=0.0,
            text=f"Speaker {i} line",
        )

    stereo_mix, manifest = mixer.render(apply_reverb=True)
    assert stereo_mix.shape[0] == 2
    assert len(manifest) == 4
    assert not np.isnan(stereo_mix).any()


def test_recent_history_uses_bounded_recent_turns():
    history = [{"content": f"speaker-{i}"} for i in range(20)]
    assert [item["content"] for item in _recent_history(history, 8)] == [f"speaker-{i}" for i in range(12, 20)]
    with pytest.raises(ValueError, match="at least 1"):
        _recent_history(history, 0)


def test_hf_429_diagnostic_includes_provider_details_without_retry():
    from types import SimpleNamespace

    response = SimpleNamespace(
        status_code=429,
        headers={"Retry-After": "30", "X-Request-Id": "request-123"},
        text='{"error":"provider rate limit"}',
    )
    diagnostic = _hf_429_diagnostic(SimpleNamespace(response=response))
    assert "HTTP 429" in diagnostic
    assert "Retry-After=30" in diagnostic
    assert "request_id=request-123" in diagnostic
    assert "provider rate limit" in diagnostic
    assert _hf_429_diagnostic(RuntimeError("not an HTTP response")) is None


def test_hf_429_stops_after_three_attempts(monkeypatch):
    from types import SimpleNamespace
    import cacophony.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.time, "sleep", lambda _: None)

    class RateLimitedBrain:
        calls = 0

        def generate_turn(self, **kwargs):
            self.calls += 1
            error = RuntimeError("429 Too Many Requests")
            error.response = SimpleNamespace(status_code=429, headers={}, text="rate limited")
            raise error

    brain = RateLimitedBrain()
    text, diagnostic = _generate_turn_once(brain, speaker_name="TEST")
    assert text is None
    assert "HTTP 429" in diagnostic
    assert brain.calls == 3


def test_partial_run_status_is_written(tmp_path):
    error = "HF router returned HTTP 429; Retry-After=30"
    status = _write_run_status(tmp_path, requested_turns=20, completed_turns=19, error=error)
    saved = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert status == saved
    assert saved == {
        "status": "partial",
        "requested_turns": 20,
        "completed_turns": 19,
        "error": error,
    }


def test_orchestrator_finalizes_completed_turns_after_hf_429(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import cacophony.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.time, "sleep", lambda _: None)

    class RateLimitedBrain:
        calls = 0

        def generate_turn(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "[worried] Pressure is rising on deck two."
            error = RuntimeError("429 Too Many Requests")
            error.response = SimpleNamespace(status_code=429, headers={"Retry-After": "30"}, text="rate limited")
            raise error

    class FakeTTS:
        def synthesize(self, **kwargs):
            return np.ones(24000, dtype=np.float32) * 0.1, 24000

    class FakeMixer:
        def __init__(self):
            self.events = []

        def add_event(self, speaker_id, audio, offset_seconds, gain_db, text, is_background):
            self.events.append({
                "speaker": speaker_id,
                "start": offset_seconds,
                "end": offset_seconds + len(audio) / 24000,
                "duration": len(audio) / 24000,
                "gain_db": gain_db,
                "text": text,
                "is_background": is_background,
            })

        def render(self, apply_reverb=True):
            return np.zeros((2, 24000), dtype=np.float32), self.events

        def save_wav(self, path, audio):
            sf.write(str(path), audio.T, 24000)

    args = SimpleNamespace(
        turns=10, duration_sec=120.0, rt60=0.2, enable_ramblers=False,
        rambler_gain_db=-18.0, engine_type="fish_audio", history_turns=8,
    )
    orch = CacophonyOrchestrator.__new__(CacophonyOrchestrator)
    orch.args = args
    orch.script = STORY_SCRIPTS["testbench"]
    orch.seed_topic = orch.script["seed_topic"]
    orch.out_dir = tmp_path
    orch.work_dir = tmp_path / "work"
    orch.work_dir.mkdir()
    orch.personas = [
        Persona(i, name, f"voice-{i}", "test persona", f"SPEAKER_{i}")
        for i, name in enumerate(["MARGOT", "DESMOND", "KAI", "CHLOE", "CLOPEUX"])
    ]
    orch.brain = RateLimitedBrain()
    orch.tts = FakeTTS()
    orch.scheduler = TurnTakingScheduler()
    orch.mixer = FakeMixer()
    orch.rambler_pool = None
    orch.run_error = None

    master = orch.run()

    assert master.exists()
    assert (tmp_path / "timeline_manifest.json").exists()
    assert (tmp_path / "transcript.txt").exists()
    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "partial"
    assert status["completed_turns"] == 1
    assert status["requested_turns"] == 10
    assert orch.brain.calls == 4


def test_vocal_habits_smart_picker_and_engine_rendering():
    """Verify VocalHabitProfile, grammar-aware post-verb filler placement, throat cooldowns, and engine tag rendering."""
    from cacophony.habit_profiles import resolve_profile
    from cacophony.vocal_habits import SmartVocalPicker, OMNIVOICE_ALLOWED_TAGS

    kai_prof = resolve_profile("KAI")
    assert kai_prof.voice_name == "normal-male-slightly-faster-cadence"
    assert any(f.word == "like" and "post_verb" in f.slots for f in kai_prof.filler_lexicon)

    picker = SmartVocalPicker(seed=42)
    # Test Fish Audio rendering: should begin with a single [<tone>] tag and support *emphasis* / [break]
    fish_ann = picker.annotate_turn(
        clean_text="Wait a second, sensor three is jumping fast because the valve is stuck!",
        speaker_name="KAI",
        engine_type="fish_audio",
        transition_type="turn_switch",
        turn_num=1,
    )
    assert fish_ann.rendered_text.startswith(f"[{fish_ann.selected_tone}]")
    assert fish_ann.selected_tone in {t.tag for t in kai_prof.tone_palette}

    # Test OmniVoice rendering: must strip non-OmniVoice bracket tags while keeping valid OmniVoice tags & fillers
    omni_ann = picker.annotate_turn(
        clean_text="That pressure spike is honestly crazy, we need to check the main scrubber loop.",
        speaker_name="KAI",
        engine_type="omnivoice",
        transition_type="turn_switch",
        turn_num=3,
    )
    import re
    brackets = re.findall(r"\[[^\]]+\]", omni_ann.rendered_text)
    for b in brackets:
        assert b in OMNIVOICE_ALLOWED_TAGS, f"Unsupported bracket tag {b} leaked into omnivoice text"
    assert "*" not in omni_ann.rendered_text, "Asterisks must be stripped for omnivoice"


def test_voices_store_habits_and_custom_sync(tmp_path):
    """Verify VoiceStore habits column migration, set_habits/get_habits, and custom voice roster sync."""
    import voices as voices_mod
    from cacophony.raggae_bridge import sync_custom_voices_to_store

    db_path = str(tmp_path / "test_voices.db")
    synced = sync_custom_voices_to_store(store_path=db_path, embed_audio=False)
    assert len(synced) == 5
    assert "softwaredev-30yomale-drama3" in synced
    assert "high-woman-voice-excited" in synced

    with voices_mod.VoiceStore(db_path) as store:
        habits = store.get_habits("softwaredev-30yomale-drama3")
        assert habits is not None
        assert habits["persona_name"] == "CHLOE"
        assert len(habits["tone_palette"]) >= 2


def test_batched_orchestrator_with_parallel_tts_mock(tmp_path):
    """Verify batched 5-10 turn generation, batched Smart Picker annotation, and parallel TTS synthesis."""
    from types import SimpleNamespace

    class FakeBatchBrain:
        batch_calls = 0

        def generate_turns_batch(self, personas, turn_plan, history, narrative_hint=None, max_tokens=1200):
            self.batch_calls += 1
            return [
                f"Sensor reading {i + 1} is rising fast, we need to isolate the loop."
                for i in range(len(turn_plan))
            ]

    class FakeParallelTTS:
        gpu_devices = [0, 1]
        parallel_batches = []

        def synthesize_batch_parallel(self, jobs):
            self.parallel_batches.append(len(jobs))
            t = np.linspace(0, 0.6, 14400, endpoint=False, dtype=np.float32)
            tone = 0.2 * np.sin(2 * np.pi * 220 * t)
            return [(tone, 24000) for _ in jobs]

    args = SimpleNamespace(
        turns=6,
        batch_size=6,
        duration_sec=120.0,
        rt60=0.2,
        enable_ramblers=False,
        rambler_gain_db=-18.0,
        engine_type="omnivoice",
        history_turns=8,
    )
    orch = CacophonyOrchestrator.__new__(CacophonyOrchestrator)
    orch.args = args
    orch.script = STORY_SCRIPTS["testbench"]
    orch.seed_topic = orch.script["seed_topic"]
    orch.out_dir = tmp_path
    orch.work_dir = tmp_path / "work"
    orch.work_dir.mkdir()
    orch.personas = [
        Persona(i, name, f"voice-{i}", "test persona", f"SPEAKER_{i}")
        for i, name in enumerate(["MARGOT", "DESMOND", "KAI", "CHLOE", "CLOPEUX"])
    ]
    orch.brain = FakeBatchBrain()
    orch.tts = FakeParallelTTS()
    orch.scheduler = TurnTakingScheduler()
    orch.mixer = AcousticRoomMixer(sample_rate=24000, rt60=0.2)
    orch.rambler_pool = None
    orch.run_error = None
    orch.smart_picker = None

    master = orch.run()
    assert master.exists()
    assert orch.brain.batch_calls == 1
    assert orch.tts.parallel_batches == [6]
    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["completed_turns"] == 6
