"""Master orchestrator for multi-speaker bot conversations with background cacophony."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Ensure repository scripts path is accessible
REPO_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_DIR / "scripts"))

from cacophony.hf_brain import HfDialogueBrain
from cacophony.tts_engine import TtsEngine
from cacophony.rambler_pool import RamblerPool
from cacophony.acoustic_mixer import AcousticRoomMixer
from cacophony.turn_taker import TurnTakingScheduler, TransitionType
from cacophony.vocal_habits import SmartVocalPicker
from cacophony.raggae_bridge import sync_custom_voices_to_store


def _recent_history(history: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    if limit < 1:
        raise ValueError("history-turns must be at least 1")
    return history[-limit:]


def _hf_429_diagnostic(exc: Exception) -> Optional[str]:
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) != 429:
        return None
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    request_id = headers.get("x-request-id") or headers.get("X-Request-Id")
    body = getattr(response, "text", "") or ""
    body = " ".join(body.split())[:500]
    details = ["HF router returned HTTP 429"]
    if retry_after:
        details.append(f"Retry-After={retry_after}")
    if request_id:
        details.append(f"request_id={request_id}")
    if body:
        details.append(f"detail={body}")
    return "; ".join(details)


def _generate_turn_once(brain, max_429_attempts: int = 3, **kwargs) -> Tuple[Optional[str], Optional[str]]:
    last_diag: Optional[str] = None
    for attempt in range(1, max_429_attempts + 1):
        try:
            return brain.generate_turn(**kwargs), None
        except Exception as exc:
            diagnostic = _hf_429_diagnostic(exc)
            if diagnostic is None:
                raise
            last_diag = diagnostic
            if attempt < max_429_attempts:
                wait_s = 2.0 * attempt
                print(f"  [!] HF HTTP 429 (attempt {attempt}/{max_429_attempts}); retrying in {wait_s:.1f}s...")
                time.sleep(wait_s)
    return None, last_diag


def _generate_batch_once(brain, max_429_attempts: int = 3, **kwargs) -> Tuple[Optional[List[str]], Optional[str]]:
    last_diag: Optional[str] = None
    for attempt in range(1, max_429_attempts + 1):
        try:
            if hasattr(brain, "generate_turns_batch"):
                return brain.generate_turns_batch(**kwargs), None
            # Fallback if a mock/test brain only implements generate_turn
            turn_plan = kwargs.get("turn_plan") or []
            personas = kwargs.get("personas") or {}
            history = list(kwargs.get("history") or [])
            lines = []
            for step in turn_plan:
                spk = step.get("speaker") or "SPEAKER"
                line = brain.generate_turn(
                    speaker_name=spk,
                    system_persona=personas.get(spk, ""),
                    history=history[-8:],
                    target_speaker=step.get("target"),
                    bounce_mode=step.get("bounce_mode"),
                    narrative_hint=kwargs.get("narrative_hint"),
                    allow_prosody_tags=False,
                )
                lines.append(line)
                history.append({"role": "assistant", "content": f"{spk}: {line}"})
            return lines, None
        except Exception as exc:
            diagnostic = _hf_429_diagnostic(exc)
            if diagnostic is None:
                raise
            last_diag = diagnostic
            if attempt < max_429_attempts:
                wait_s = 2.0 * attempt
                print(f"  [!] HF HTTP 429 (attempt {attempt}/{max_429_attempts}); retrying in {wait_s:.1f}s...")
                time.sleep(wait_s)
    return None, last_diag


def _write_run_status(out_dir: Path, requested_turns: int, completed_turns: int, error: Optional[str]) -> dict:
    status = {
        "status": "partial" if error else "complete",
        "requested_turns": requested_turns,
        "completed_turns": completed_turns,
        "error": error,
    }
    (out_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


@dataclass
class Persona:
    index: int
    name: str
    voice_name: str
    description: str
    speaker_tag: str
    speed_ratio: float = 1.0
    voice_ref_wav: Optional[Path] = None
    voice_ref_text: Optional[str] = None
    engine: Optional[str] = None
    voice_pack_gguf: Optional[Path] = None
    preset_voice: Optional[str] = None


STORY_SCRIPTS = {
    "testbench": {
        "label": "Test Bench Opinions",
        "seed_topic": "Desmond, what is your opinion on yellow hats?",
        "persona_descriptions": {
            "MARGOT": "Blunt, skeptical lead physicist analyzing yellow hats and sus opinions.",
            "DESMOND": "Speculative tech investigator who believes the yellow hats are part of a larger pattern.",
            "KAI": "Fast, observant audio tech with opinions suspiciously similar to Joe Rogan's.",
            "CHLOE": "Pragmatic systems engineer who thinks they're individualistic but actually is more bland than sliced bread.",
            "CLOPEUX": "Deadpan, sharp-witted director who cuts into the group's tangents with dry observations and unexpected takes.",
        },
    },
}


@dataclass
class TurnRecord:
    turn_num: int
    speaker: Persona
    target: Optional[Persona]
    text: str
    start_time: float
    duration: float
    end_time: float
    transition_type: str


def persona_presets() -> List[dict]:
    """Default persona roster (5 entries; UI + CLI share this single source of truth).

    Uses the custom cloned reference voices in voices-store/ and samples/voices/
    (designed with Audio Fish Drama 3 / Clopeux samples). Descriptions are per-script
    templates (`{NAME}`) resolved against STORY_SCRIPTS[...]["persona_descriptions"].
    """
    kokoro_dir = Path.home() / ".cache" / "crispasr"
    vs_dir = REPO_DIR / "voices-store"
    return [
        {
            "index": 0, "name": "MARGOT", "voice_name": "high-woman-voice-excited",
            "description": "{MARGOT}", "speaker_tag": "SPEAKER_0", "speed_ratio": 1.15,
            "voice_pack_gguf": str(kokoro_dir / "kokoro-voice-af_heart.gguf"),
            "voice_ref_wav": str(vs_dir / "high-woman-voice-excited.wav"),
        },
        {
            "index": 1, "name": "DESMOND", "voice_name": "sarcastic-male-voice",
            "description": "{DESMOND}", "speaker_tag": "SPEAKER_1", "speed_ratio": 1.08,
            "voice_pack_gguf": str(kokoro_dir / "kokoro-voice-am_michael.gguf"),
            "voice_ref_wav": str(vs_dir / "sarcastic-male-voice.wav"),
        },
        {
            "index": 2, "name": "KAI", "voice_name": "normal-male-slightly-faster-cadence",
            "description": "{KAI}", "speaker_tag": "SPEAKER_2", "speed_ratio": 1.20,
            "voice_pack_gguf": str(kokoro_dir / "kokoro-voice-am_puck.gguf"),
            "voice_ref_wav": str(vs_dir / "normal-male-slightly-faster-cadence.wav"),
        },
        {
            "index": 3, "name": "CHLOE", "voice_name": "softwaredev-30yomale-drama3",
            "description": "{CHLOE}", "speaker_tag": "SPEAKER_3", "speed_ratio": 1.10,
            "voice_pack_gguf": str(kokoro_dir / "kokoro-voice-bf_emma.gguf"),
            "voice_ref_wav": str(vs_dir / "softwaredev-30yomale-drama3.wav"),
        },
        {
            "index": 4, "name": "CLOPEUX", "voice_name": "clopeux-voice-samples1",
            "description": "{CLOPEUX}",
            "speaker_tag": "SPEAKER_4", "speed_ratio": 1.08,
            "voice_ref_wav": str(REPO_DIR / "samples" / "voices" / "clopeux-voice-samples1.wav"),
        },
    ]


def parse_args():
    ap = argparse.ArgumentParser(description="Multi-Speaker Cacophony & Bot Conversation Orchestrator")
    ap.add_argument("--duration-sec", type=float, default=300.0, help="Target conversation duration in seconds (default 300.0s = 5 minutes)")
    ap.add_argument("--turns", type=int, default=100, help="Maximum number of dialogue turns (default 100)")
    ap.add_argument("--history-turns", type=int, default=8, help="Maximum recent dialogue turns sent to HF (default 8)")
    ap.add_argument("--batch-size", type=int, default=8, help="Number of dialogue turns generated, embedded, and synthesized in parallel per batch (default 8; 1 = sequential)")
    ap.add_argument("--dual-gpu", action="store_true", default=True, help="Run 2 warm TTS engines in parallel across GPU 0 and GPU 1 when available (default True)")
    ap.add_argument("--no-dual-gpu", dest="dual_gpu", action="store_false", help="Disable dual-GPU parallel TTS synthesis")
    ap.add_argument("--out-dir", type=str, default="./cacophony_output", help="Output directory")
    ap.add_argument("--engine-type", type=str, default="omnivoice", choices=["omnivoice", "fish_audio", "chatterbox", "cosyvoice", "magpie", "kokoro"], help="TTS backend (default omnivoice)")
    ap.add_argument("--gpu-device", type=int, default=-1, help="GPU index for TTS (-1 = auto dual-GPU [0, 1] for omnivoice)")
    ap.add_argument("--model", type=str, default="openai/gpt-oss-120b:cheapest", help="HF LLM model")
    ap.add_argument("--num-speakers", type=int, default=5, choices=[3, 4, 5], help="Number of active conversational speakers")
    ap.add_argument("--rt60", type=float, default=0.20, help="Room reverberation RT60 in seconds (toned down echo)")
    ap.add_argument("--enable-ramblers", action="store_true", default=False, help="Inject background incoherent ramblers (disabled by default)")
    ap.add_argument("--rambler-gain-db", type=float, default=-18.0, help="Relative gain of background ramblers in dB")
    ap.add_argument("--seed-topic", type=str, default=None, help="Opening conversation starter (defaults to the selected script's seed)")
    ap.add_argument("--script", type=str, default="testbench", choices=list(STORY_SCRIPTS.keys()), help="Story script blueprint (default testbench)")
    ap.add_argument("--personas-json", type=str, default=None, help="JSON file of per-persona overrides [{index, engine, voice_name, voice_pack_gguf, voice_ref_wav, voice_ref_text, speed_ratio, description}]")
    ap.add_argument("--answering-mode", action="store_true", default=False, help="Have the LLM generate responses directly in character as the answering persona instead of a director script batch")
    ap.add_argument("--stream", action="store_true", default=True, help="Stream LLM tokens in real time (default True)")
    ap.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming LLM tokens")
    ap.add_argument("--allow-prosody-tags", action="store_true", default=True, help="Allow inline vocal prosody tags from the LLM (default True)")
    ap.add_argument("--no-prosody-tags", dest="allow_prosody_tags", action="store_false", help="Strip prosody tags from LLM responses")
    return ap.parse_args()


class CacophonyOrchestrator:
    """Coordinates remote LLM dialogue, fast local neural TTS, and 3D acoustic room mixing."""

    def __init__(self, args):
        self.args = args
        self.run_error: Optional[str] = None
        self.script = STORY_SCRIPTS[args.script]
        self.seed_topic = args.seed_topic or self.script["seed_topic"]
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = self.out_dir / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        print(f"[*] Initializing HF Dialogue Brain ({args.model})...")
        self.brain = HfDialogueBrain(model=args.model)

        gpu_dev = None if args.gpu_device < 0 else args.gpu_device
        dual_gpu = getattr(args, "dual_gpu", True) and gpu_dev is None and args.engine_type.lower() in ("omnivoice", "omni")
        gpu_devices = [0, 1] if dual_gpu else ([gpu_dev] if gpu_dev is not None else [0])
        gpu_str = f"Dual-GPU {gpu_devices}" if len(gpu_devices) > 1 else ("auto" if gpu_dev is None else f"GPU {gpu_dev}")
        print(f"[*] Initializing local TTS Engine ({args.engine_type}) on {gpu_str}...")
        self.tts = TtsEngine(engine_type=args.engine_type, gpu_device=gpu_dev, gpu_devices=gpu_devices)

        print("[*] Initializing Smart Vocal Habit Picker & syncing custom voices to voices.db...")
        try:
            sync_custom_voices_to_store()
        except Exception as sync_err:
            print(f"  [!] VoiceStore sync warning: {sync_err}")
        self.smart_picker = SmartVocalPicker()

        print("[*] Initializing FastMSS Turn-Taking Scheduler...")
        self.scheduler = TurnTakingScheduler()

        rambler_str = f"Ramblers={args.enable_ramblers}" if not args.enable_ramblers else f"Rambler Gain={args.rambler_gain_db}dB"
        print(f"[*] Initializing Acoustic Room Mixer (RT60={args.rt60}s, {rambler_str})...")
        self.mixer = AcousticRoomMixer(sample_rate=24000, rt60=args.rt60)

        if self.args.enable_ramblers:
            print("[*] Initializing Background Rambler Pool...")
            self.rambler_pool = RamblerPool()
            print(f"    Loaded {len(self.rambler_pool.speakers)} background rambler speakers.")
        else:
            print("[*] Background ramblers disabled (clean studio room dialogue).")
            self.rambler_pool = None

        # Configure personas from presets (Kokoro voice packs, baked Magpie voices, or cloned references).
        # Kokoro GGUF voice packs live in the shared model store (~/.cache/crispasr/).
        descriptions = {
            "MARGOT": "Sharp, opinionated lead voice in the room.",
            "DESMOND": "Speculative, contrarian conversationalist.",
            "KAI": "Fast-talking, excitable observer who jumps in with wild angles.",
            "CHLOE": "Pragmatic, dry-humored realist.",
            "CLOPEUX": "Calm, decisive voice who cuts through tangents.",
            **self.script.get("persona_descriptions", {}),
        }
        global_eng = args.engine_type.lower()
        clone_engines = ("omnivoice", "omni", "fish_audio", "fish", "fish-s2", "cosyvoice")
        self.personas = [
            Persona(
                index=p["index"],
                name=p["name"],
                voice_name=Path(p["voice_ref_wav"]).stem if global_eng in clone_engines and p.get("voice_ref_wav") else p["voice_name"],
                description=p["description"].format(**descriptions),
                speaker_tag=p["speaker_tag"],
                speed_ratio=1.0 if global_eng in ("fish_audio", "fish", "fish-s2") else p["speed_ratio"],
                engine=None if global_eng in clone_engines else p.get("engine"),
                voice_pack_gguf=Path(p["voice_pack_gguf"]) if p.get("voice_pack_gguf") else None,
                voice_ref_wav=Path(p["voice_ref_wav"]) if p.get("voice_ref_wav") else None,
            )
            for p in persona_presets()[: args.num_speakers]
        ]
        self._apply_persona_overrides()

    def _apply_persona_overrides(self):
        """Apply per-persona overrides from --personas-json (used by the web UI voice pickers)."""
        if not getattr(self.args, "personas_json", None):
            return
        with open(self.args.personas_json, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        by_index = {int(o.get("index", i)): o for i, o in enumerate(overrides)}
        for p in self.personas:
            o = by_index.get(p.index)
            if not o:
                continue
            for field in ("engine", "voice_name", "voice_ref_text", "description", "preset_voice"):
                if o.get(field) is not None:
                    setattr(p, field, o[field])
            if o.get("speed_ratio") is not None:
                p.speed_ratio = float(o["speed_ratio"])
            for field in ("voice_pack_gguf", "voice_ref_wav"):
                if o.get(field):
                    setattr(p, field, Path(o[field]))

    def _plan_turn_step(
        self,
        turn_idx: int,
        spoken_indices: set[int],
        prev_speaker: Optional[Persona],
        prev_target: Optional[Persona],
        earlier_speaker: Optional[Persona],
    ) -> Tuple[Persona, Optional[Persona], str, Optional[str]]:
        """Select (speaker, target, action_type, bounce_mode) for turn_idx."""
        if turn_idx == 0:
            speaker = self.personas[0]
            target = next(
                (p for p in self.personas[1:] if p.name.lower() in self.seed_topic.lower()),
                self.personas[1] if len(self.personas) > 1 else None,
            )
            action_type = "seed"
        else:
            prev_spk_idx = prev_speaker.index if prev_speaker else -1
            unheard = [p for p in self.personas if p.index not in spoken_indices and p.index != prev_spk_idx]
            candidates = [p for p in self.personas if p.index != prev_spk_idx] or list(self.personas)

            if prev_target and prev_target.index != prev_spk_idx and np.random.rand() < 0.50:
                speaker = prev_target
            elif unheard and np.random.rand() < 0.75:
                speaker = unheard[0]
            else:
                speaker = np.random.choice(candidates)

            if earlier_speaker and np.random.rand() < 0.30:
                target = earlier_speaker if earlier_speaker.index != speaker.index else prev_speaker
            else:
                target = prev_speaker

            action_type = "interject" if np.random.rand() < 0.35 else "reply"

        spoken_indices.add(speaker.index)
        bounce_mode = None
        if prev_speaker and target and prev_speaker.index != speaker.index and prev_speaker.index != target.index:
            bounce_mode = str(np.random.choice(["bridge", "uno_reverse", "continuation", "contrast"]))
        return speaker, target, action_type, bounce_mode

    def _schedule_and_mix_turn(
        self,
        turn_num: int,
        speaker: Persona,
        target: Optional[Persona],
        action_type: str,
        utterance_text: str,
        audio_data: np.ndarray,
        sr: int,
        turn_records: List[TurnRecord],
    ) -> None:
        duration = len(audio_data) / sr

        if turn_num == 1:
            proposed_start = 0.5
            trans_type = TransitionType.TURN_HOLD.value
        elif action_type == "interject":
            prev_rec = turn_records[-1]
            proposed_start = self.scheduler.sample_interruption_offset(
                prev_offset=prev_rec.start_time,
                prev_duration=prev_rec.duration,
                min_overlap=0.125,
                max_overlap=0.25,
            )
            trans_type = TransitionType.INTERRUPTION.value
        else:
            latest_active_rec = max(turn_records, key=lambda r: r.end_time)
            proposed_start = self.scheduler.sample_interruption_offset(
                prev_offset=latest_active_rec.start_time,
                prev_duration=latest_active_rec.duration,
                min_overlap=0.125,
                max_overlap=0.25,
            )
            trans_type = TransitionType.INTERLEAVED_REPLY.value

        scheduled_intervals = [(rec.start_time, rec.end_time, rec.speaker.index) for rec in turn_records]
        start_offset = self.scheduler.stagger_turn(
            proposed_start=proposed_start,
            duration=duration,
            speaker_idx=speaker.index,
            scheduled_turns=scheduled_intervals,
            max_concurrent=2,
            max_overlap_sec=0.25,
            min_speaker_gap=0.08,
        )
        if start_offset > proposed_start + 0.01:
            print(f"  > [STAGGER] Shifted start from t={proposed_start:.2f}s to t={start_offset:.2f}s (max 2 active speakers limit)")

        end_offset = start_offset + duration
        record = TurnRecord(
            turn_num=turn_num,
            speaker=speaker,
            target=target,
            text=utterance_text,
            start_time=start_offset,
            duration=duration,
            end_time=end_offset,
            transition_type=trans_type,
        )
        turn_records.append(record)
        print(f"  > Scheduled {speaker.name} at t={start_offset:.2f}s - {end_offset:.2f}s ({duration:.2f}s, {trans_type})")

        self.mixer.add_event(
            speaker_id=speaker.speaker_tag,
            audio=audio_data,
            offset_seconds=start_offset,
            gain_db=0.0,
            text=f"{speaker.name}: {utterance_text}",
            is_background=False,
        )
        if self.args.enable_ramblers and self.rambler_pool and duration > 1.8:
            self._inject_background_rambler(start_offset, duration)

    def run(self):
        batch_size = max(1, int(getattr(self.args, "batch_size", 1) or 1))
        if not hasattr(self, "smart_picker") or self.smart_picker is None:
            self.smart_picker = SmartVocalPicker()

        print("\n=======================================================")
        print(f"  Starting Staggered Multi-Speaker Orchestration")
        print(f"  Script: {self.script['label']}")
        print(f"  Speakers ({len(self.personas)}): {', '.join(f'{p.name} ({p.engine or self.args.engine_type}:{p.voice_name})' for p in self.personas)}")
        print(f"  Engine: {self.args.engine_type} | Turns: {self.args.turns} | Batch Size: {batch_size}")
        print(f"  Acoustics: RT60={self.args.rt60}s | Ramblers: {self.args.enable_ramblers}")
        print(f"  Constraints: Max concurrent speakers = 2 | Barge-in = 0.12s-0.25s")
        print("=======================================================\n")

        # Global conversation history for HF LLM
        history: List[Dict[str, str]] = []
        turn_records: List[TurnRecord] = []
        spoken_indices: set[int] = set()

        turn_idx = 0
        while turn_idx < self.args.turns:
            current_timeline_end = max([r.end_time for r in turn_records], default=0.0)
            if self.args.duration_sec > 0 and current_timeline_end >= self.args.duration_sec and turn_idx >= len(self.personas):
                print(f"\n[*] Target duration reached: {current_timeline_end:.2f}s >= {self.args.duration_sec:.2f}s after {turn_idx} turns.")
                break

            # Batched Producer + Dual-GPU Parallel TTS Pipeline when batch_size > 1 and not in answering mode
            use_batch = (
                batch_size > 1
                and not getattr(self.args, "answering_mode", False)
                and hasattr(self.brain, "generate_turns_batch")
                and hasattr(self.tts, "synthesize_batch_parallel")
            )
            if use_batch:
                remaining = self.args.turns - turn_idx
                cur_batch_n = min(batch_size, remaining)
                planned_steps = []
                sim_prev_spk = turn_records[-1].speaker if turn_records else None
                sim_prev_tgt = turn_records[-1].target if turn_records else None
                sim_earlier_spk = turn_records[-2].speaker if len(turn_records) >= 2 else None

                for b_i in range(cur_batch_n):
                    t_idx = turn_idx + b_i
                    spk, tgt, act_type, b_mode = self._plan_turn_step(
                        t_idx, spoken_indices, sim_prev_spk, sim_prev_tgt, sim_earlier_spk
                    )
                    trans_hint = (
                        TransitionType.TURN_HOLD.value if t_idx == 0
                        else (TransitionType.INTERRUPTION.value if act_type == "interject" else TransitionType.INTERLEAVED_REPLY.value)
                    )
                    planned_steps.append({
                        "turn_num": t_idx + 1,
                        "speaker_obj": spk,
                        "target_obj": tgt,
                        "action_type": act_type,
                        "bounce_mode": b_mode,
                        "transition_type": trans_hint,
                        "speaker": spk.name,
                        "target": tgt.name if tgt else None,
                    })
                    sim_earlier_spk = sim_prev_spk
                    sim_prev_spk = spk
                    sim_prev_tgt = tgt

                persona_map = {p.name: p.description for p in self.personas}
                print(f"\n[Batch Turns {turn_idx + 1}..{turn_idx + cur_batch_n} | t={current_timeline_end:.1f}s] Generating {cur_batch_n} dialogue lines via HF Brain...")
                raw_lines, diagnostic = _generate_batch_once(
                    self.brain,
                    personas=persona_map,
                    turn_plan=[{"speaker": s["speaker"], "target": s["target"], "bounce_mode": s["bounce_mode"]} for s in planned_steps],
                    history=_recent_history(history, self.args.history_turns) if history else [],
                    narrative_hint=self.seed_topic,
                )
                if diagnostic or raw_lines is None:
                    self.run_error = diagnostic
                    print(f"  [!] {diagnostic}; stopping HF requests and finalizing the {len(turn_records)} completed turns.")
                    break

                # Batch-score tones via vault-raggae (:8750) EmbeddingGemma-2 and inject Smart Picker vocal habits
                picker_items = []
                for step, clean_line in zip(planned_steps, raw_lines):
                    spk = step["speaker_obj"]
                    eff_eng = (spk.engine or self.args.engine_type).lower()
                    picker_items.append({
                        "text": clean_line,
                        "speaker_name": spk.name,
                        "voice_name": spk.voice_name,
                        "engine_type": eff_eng,
                        "transition_type": step["transition_type"],
                        "turn_num": step["turn_num"],
                    })
                annotated_batch = self.smart_picker.annotate_batch(picker_items, engine_type=self.args.engine_type)

                tts_jobs = []
                for step, ann in zip(planned_steps, annotated_batch):
                    spk = step["speaker_obj"]
                    tgt = step["target_obj"]
                    t_num = step["turn_num"]
                    print(
                        f"  > [Turn {t_num} | {spk.name} -> {tgt.name if tgt else 'ALL'} | tone={ann.selected_tone} ({ann.tone_method})]: "
                        f"\"{ann.rendered_text}\""
                    )
                    history.append({
                        "role": "user" if spk.index == 0 else "assistant",
                        "content": f"{spk.name}: {ann.clean_text}",
                    })
                    ref_text = spk.voice_ref_text
                    if not ref_text and spk.voice_ref_wav and spk.voice_ref_wav.with_suffix(".txt").exists():
                        ref_text = spk.voice_ref_wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
                    tts_jobs.append({
                        "text": ann.rendered_text,
                        "speaker_idx": spk.index,
                        "voice_name": spk.voice_name,
                        "speed_ratio": spk.speed_ratio,
                        "voice_ref_wav": spk.voice_ref_wav,
                        "voice_ref_text": ref_text,
                        "engine_type": spk.engine,
                        "voice_pack_gguf": spk.voice_pack_gguf,
                        "preset_voice": spk.preset_voice,
                        "output_wav": self.work_dir / f"turn_{t_num:03d}_{spk.name}.wav",
                    })

                print(f"  > Synthesizing {len(tts_jobs)} turns in parallel across GPU pool {getattr(self.tts, 'gpu_devices', [0])}...")
                t_synth0 = time.perf_counter()
                synth_outputs = self.tts.synthesize_batch_parallel(tts_jobs)
                synth_elapsed = time.perf_counter() - t_synth0
                print(f"  > Parallel TTS batch ({len(tts_jobs)} turns) finished in {synth_elapsed:.2f}s ({synth_elapsed / max(1, len(tts_jobs)):.2f}s/turn avg)")

                for step, ann, (audio_data, sr) in zip(planned_steps, annotated_batch, synth_outputs):
                    self._schedule_and_mix_turn(
                        turn_num=step["turn_num"],
                        speaker=step["speaker_obj"],
                        target=step["target_obj"],
                        action_type=step["action_type"],
                        utterance_text=ann.rendered_text,
                        audio_data=audio_data,
                        sr=sr,
                        turn_records=turn_records,
                    )
                    turn_idx += 1
                    cur_end = max([r.end_time for r in turn_records], default=0.0)
                    if self.args.duration_sec > 0 and cur_end >= self.args.duration_sec and turn_idx >= len(self.personas):
                        break
                continue

            # Sequential single-turn path (when batch_size == 1 or custom test mock brain is used)
            turn_num = turn_idx + 1
            prev_rec = turn_records[-1] if turn_records else None
            earlier_rec = turn_records[-2] if len(turn_records) >= 2 else None
            speaker, target, action_type, bounce_mode = self._plan_turn_step(
                turn_idx,
                spoken_indices,
                prev_rec.speaker if prev_rec else None,
                prev_rec.target if prev_rec else None,
                earlier_rec.speaker if earlier_rec else None,
            )

            effective_engine = (speaker.engine or self.args.engine_type).lower()
            allow_prosody = effective_engine in ("fish_audio", "fish", "fish-s2", "cosyvoice")

            target_utt = None
            if target:
                for rec in reversed(turn_records):
                    if rec.speaker.index == target.index:
                        target_utt = rec.text
                        break

            prev_speaker = prev_rec.speaker.name if prev_rec else None
            prev_utt = prev_rec.text if prev_rec else None

            if bounce_mode and prev_speaker and target:
                print(f"  > [BOUNCE] {speaker.name} bouncing off {prev_speaker} ({bounce_mode}) regarding {target.name}'s point")

            mode_tag = "Answering" if getattr(self.args, "answering_mode", False) else "Improvised"
            print(f"\n[Turn {turn_num} | {mode_tag}: {speaker.name} -> {target.name if target else 'ALL'} | t={current_timeline_end:.1f}s] ({speaker.voice_name}, speed={speaker.speed_ratio:.2f}x)...")
            stream_enabled = getattr(self.args, "stream", True)
            allow_prosody = getattr(self.args, "allow_prosody_tags", True)
            utterance_text, diagnostic = _generate_turn_once(
                self.brain,
                speaker_name=speaker.name,
                system_persona=speaker.description,
                history=_recent_history(history, self.args.history_turns),
                target_speaker=target.name if target else None,
                target_utterance=target_utt,
                prev_speaker=prev_speaker,
                prev_utterance=prev_utt,
                bounce_mode=bounce_mode,
                narrative_hint=self.seed_topic,
                max_tokens=250,
                stream=stream_enabled,
                allow_prosody_tags=allow_prosody,
            )
            if diagnostic:
                self.run_error = diagnostic
                print(f"  [!] {diagnostic}; stopping HF requests and finalizing the {len(turn_records)} completed turns.")
                break

            trans_hint = (
                TransitionType.TURN_HOLD.value if turn_num == 1
                else (TransitionType.INTERRUPTION.value if action_type == "interject" else TransitionType.INTERLEAVED_REPLY.value)
            )
            if effective_engine in ("omnivoice", "omni", "fish_audio", "fish", "fish-s2"):
                ann = self.smart_picker.annotate_turn(
                    clean_text=utterance_text,
                    speaker_name=speaker.name,
                    voice_name=speaker.voice_name,
                    engine_type=effective_engine,
                    transition_type=trans_hint,
                    turn_num=turn_num,
                )
                utterance_text = ann.rendered_text

            print(f"  > [{speaker.name} -> {target.name if target else 'ALL'}]: \"{utterance_text}\"")

            history.append({
                "role": "user" if speaker.index == 0 else "assistant",
                "content": f"{speaker.name}: {utterance_text}",
            })

            turn_wav_path = self.work_dir / f"turn_{turn_num:03d}_{speaker.name}.wav"
            print(f"  > Synthesizing speech via {speaker.engine or self.args.engine_type} (voice={speaker.voice_name}, speed={speaker.speed_ratio:.2f}x)...")

            ref_text = speaker.voice_ref_text
            if not ref_text and speaker.voice_ref_wav and speaker.voice_ref_wav.with_suffix(".txt").exists():
                ref_text = speaker.voice_ref_wav.with_suffix(".txt").read_text(encoding="utf-8").strip()

            try:
                audio_data, sr = self.tts.synthesize(
                    text=utterance_text,
                    speaker_idx=speaker.index,
                    voice_name=speaker.voice_name,
                    speed_ratio=speaker.speed_ratio,
                    voice_ref_wav=speaker.voice_ref_wav,
                    voice_ref_text=ref_text,
                    engine_type=speaker.engine,
                    voice_pack_gguf=speaker.voice_pack_gguf,
                    preset_voice=speaker.preset_voice,
                    output_wav=turn_wav_path,
                )
            except Exception as e:
                print(f"  [!] Turn synthesis transient warning ({e}). Retrying after pause...")
                time.sleep(1.0)
                audio_data, sr = self.tts.synthesize(
                    text=utterance_text,
                    speaker_idx=speaker.index,
                    voice_name=speaker.voice_name,
                    speed_ratio=speaker.speed_ratio,
                    voice_ref_wav=speaker.voice_ref_wav,
                    voice_ref_text=ref_text,
                    engine_type=speaker.engine,
                    voice_pack_gguf=speaker.voice_pack_gguf,
                    preset_voice=speaker.preset_voice,
                    output_wav=turn_wav_path,
                )

            self._schedule_and_mix_turn(
                turn_num=turn_num,
                speaker=speaker,
                target=target,
                action_type=action_type,
                utterance_text=utterance_text,
                audio_data=audio_data,
                sr=sr,
                turn_records=turn_records,
            )
            turn_idx += 1

        # Final Room Acoustic Simulation & Mixing
        print("\n[*] Simulating 3D Room Acoustics & Mixing master stem (FastMSS engine)...")
        stereo_mix, manifest = self.mixer.render(apply_reverb=True)

        master_wav_path = self.out_dir / "cacophony_master.wav"
        manifest_path = self.out_dir / "timeline_manifest.json"
        transcript_path = self.out_dir / "transcript.txt"
        status_path = self.out_dir / "run_status.json"

        self.mixer.save_wav(master_wav_path, stereo_mix)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        _write_run_status(self.out_dir, self.args.turns, len(turn_records), self.run_error)

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("=== MULTI-SPEAKER CONVERSATION TRANSCRIPT ===\n")
            if self.run_error:
                f.write(f"=== PARTIAL RUN: {len(turn_records)} turns completed; {self.run_error} ===\n")
            f.write("\n")
            for item in manifest:
                tag = "[BG]" if item["is_background"] else "[MAIN]"
                f.write(f"[{item['start']:.2f}s - {item['end']:.2f}s] {tag} {item['text']}\n")

        total_sec = stereo_mix.shape[1] / 24000
        if self.run_error:
            print(f"\n[!] PARTIAL RUN: {len(turn_records)} turns rendered; stopped after HTTP 429.")
        else:
            print("\n[+] SUCCESS!")
        print(f"    Master Stereo WAV: {master_wav_path} ({total_sec:.2f}s)")
        print(f"    Manifest Timeline: {manifest_path}")
        print(f"    Transcript Log:    {transcript_path}")
        print(f"    Run Status:        {status_path}")
        return master_wav_path

    def _inject_background_rambler(self, base_offset: float, base_duration: float):
        """Pulls a random speech cut from the RamblerPool and positions it in the room background."""
        if not self.rambler_pool:
            return
        cut = self.rambler_pool.sample_cut()
        if not cut:
            return

        rambler_audio = cut.load_audio(target_sr=24000, max_duration=min(3.5, base_duration * 0.85))
        if len(rambler_audio) < 12000:
            return

        rambler_offset = self.scheduler.sample_backchannel_offset(base_offset, base_duration)
        spk_tag = np.random.choice(["RAMBLER_1", "RAMBLER_2", "RAMBLER_3"])

        self.mixer.add_event(
            speaker_id=spk_tag,
            audio=rambler_audio,
            offset_seconds=rambler_offset,
            gain_db=self.args.rambler_gain_db,
            text=f"({spk_tag} murmurs)",
            is_background=True,
        )


if __name__ == "__main__":
    args = parse_args()
    orch = CacophonyOrchestrator(args)
    orch.run()
    if orch.run_error:
        raise SystemExit(2)
