"""Acoustic room simulator & software mixer powered by pyroomacoustics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pyroomacoustics as pra
from scipy.signal import convolve
import soundfile as sf


@dataclass
class AudioEvent:
    speaker_id: str
    audio: np.ndarray  # 1D float32 audio
    offset_seconds: float
    duration: float
    gain_db: float = 0.0
    text: Optional[str] = None
    is_background: bool = False


class AcousticRoomMixer:
    """Simulates a 3D acoustic room, computes RIRs for multiple speaker positions,

    and mixes conversation turns + ambient ramblers into a spatialized stereo stream.
    Zero virtual audio cables required.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        room_dim: Tuple[float, float, float] = (6.0, 5.0, 3.0),
        rt60: float = 0.20,  # Crisper conversational room acoustics (toned down echo)
        mic_pos: Tuple[float, float, float] = (3.0, 2.5, 1.2),
        mic_spacing: float = 0.20,  # 20cm binaural / stereo mic separation
    ):
        self.sample_rate = sample_rate
        self.room_dim = list(room_dim)
        self.rt60 = rt60
        self.events: List[AudioEvent] = []

        # Setup stereo microphones
        mx, my, mz = mic_pos
        half_spacing = mic_spacing / 2.0
        self.mic_locs = np.array([
            [mx - half_spacing, mx + half_spacing],  # Left and Right
            [my, my],
            [mz, mz],
        ])

        # Pre-assign realistic 3D locations for speakers
        # Speaker 0 (front-left), Speaker 1 (front-right), Speaker 2 (center-back), Speaker 3 (left-mid)
        self.predefined_positions = {
            "SPEAKER_A": np.array([1.8, 2.5, 1.4]),
            "SPEAKER_B": np.array([4.2, 2.5, 1.4]),
            "SPEAKER_0": np.array([1.8, 2.5, 1.4]),
            "SPEAKER_1": np.array([4.2, 2.5, 1.4]),
            "SPEAKER_2": np.array([3.0, 3.8, 1.3]),
            "SPEAKER_3": np.array([2.2, 1.5, 1.3]),
            "SPEAKER_4": np.array([3.8, 1.5, 1.3]),
            "RAMBLER_1": np.array([1.2, 4.2, 1.3]),
            "RAMBLER_2": np.array([4.8, 4.0, 1.3]),
            "RAMBLER_3": np.array([1.0, 1.2, 1.2]),
        }
        self.speaker_positions: Dict[str, np.ndarray] = {}
        self.speaker_rirs: Dict[str, List[np.ndarray]] = {}

        self._build_room()

    def _build_room(self):
        """Build pyroomacoustics ShoeBox and precompute absorption."""
        try:
            self.e_absorption, max_order = pra.inverse_sabine(self.rt60, self.room_dim)
        except ValueError:
            self.e_absorption, max_order = 0.2, 3

        self.max_order = min(max_order, 4)
        self.room = pra.ShoeBox(
            self.room_dim,
            fs=self.sample_rate,
            max_order=self.max_order,
            materials=pra.Material(self.e_absorption),
        )
        self.room.add_microphone_array(self.mic_locs)

    def register_speaker(self, speaker_id: str, position: Optional[np.ndarray] = None):
        """Assign or generate a 3D position and compute stereo RIRs for a speaker."""
        if speaker_id in self.speaker_rirs:
            return

        if position is not None:
            pos = position
        elif speaker_id in self.predefined_positions:
            pos = self.predefined_positions[speaker_id]
        else:
            # Sample random position in the room
            l, w, h = self.room_dim
            pos = np.array([
                np.random.uniform(0.5, l - 0.5),
                np.random.uniform(0.5, w - 0.5),
                np.random.uniform(1.0, 1.8),
            ])

        self.speaker_positions[speaker_id] = pos

        # Add temporary source to room and compute RIR
        room_copy = pra.ShoeBox(
            self.room_dim,
            fs=self.sample_rate,
            max_order=self.max_order,
            materials=pra.Material(self.e_absorption),
        )
        room_copy.add_microphone_array(self.mic_locs)
        room_copy.add_source(pos)
        room_copy.compute_rir()

        # RIR for channel 0 (Left) and channel 1 (Right)
        rir_l = room_copy.rir[0][0]
        rir_r = room_copy.rir[1][0]
        self.speaker_rirs[speaker_id] = [rir_l, rir_r]

    def add_event(
        self,
        speaker_id: str,
        audio: np.ndarray,
        offset_seconds: float,
        gain_db: float = 0.0,
        text: Optional[str] = None,
        is_background: bool = False,
    ):
        """Queue an audio utterance into the simulation timeline."""
        duration = len(audio) / self.sample_rate
        self.register_speaker(speaker_id)
        event = AudioEvent(
            speaker_id=speaker_id,
            audio=audio.astype(np.float32),
            offset_seconds=offset_seconds,
            duration=duration,
            gain_db=gain_db,
            text=text,
            is_background=is_background,
        )
        self.events.append(event)

    def render(
        self,
        apply_reverb: bool = True,
        headroom_db: float = -1.0,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Mix all events into a final stereo audio array (2, N_samples).

        Returns:
            Tuple of (stereo_audio_array, manifest_records).
        """
        if not self.events:
            return np.zeros((2, int(self.sample_rate * 1.0)), dtype=np.float32), []

        # Find total session duration
        total_dur = max(ev.offset_seconds + ev.duration + (0.5 if apply_reverb else 0.0) for ev in self.events)
        total_samples = int(np.ceil(total_dur * self.sample_rate)) + 1024

        master_l = np.zeros(total_samples, dtype=np.float32)
        master_r = np.zeros(total_samples, dtype=np.float32)

        manifest = []

        for ev in sorted(self.events, key=lambda x: x.offset_seconds):
            # Scale gain
            gain_linear = 10.0 ** (ev.gain_db / 20.0)
            sig = ev.audio * gain_linear

            if apply_reverb and ev.speaker_id in self.speaker_rirs:
                rir_l, rir_r = self.speaker_rirs[ev.speaker_id]
                ch_l = convolve(sig, rir_l, mode="full")
                ch_r = convolve(sig, rir_r, mode="full")
            else:
                ch_l = sig
                ch_r = sig

            start_idx = int(round(ev.offset_seconds * self.sample_rate))
            end_idx_l = start_idx + len(ch_l)
            end_idx_r = start_idx + len(ch_r)

            if end_idx_l > total_samples:
                ch_l = ch_l[: total_samples - start_idx]
            if end_idx_r > total_samples:
                ch_r = ch_r[: total_samples - start_idx]

            master_l[start_idx : start_idx + len(ch_l)] += ch_l
            master_r[start_idx : start_idx + len(ch_r)] += ch_r

            manifest.append({
                "speaker": ev.speaker_id,
                "start": round(ev.offset_seconds, 3),
                "end": round(ev.offset_seconds + ev.duration, 3),
                "duration": round(ev.duration, 3),
                "gain_db": ev.gain_db,
                "is_background": ev.is_background,
                "text": ev.text or "",
            })

        # Stack into (2, N)
        stereo_mix = np.stack([master_l, master_r], axis=0)

        # Normalize with headroom
        peak = np.max(np.abs(stereo_mix))
        if peak > 1e-6:
            target_peak = 10.0 ** (headroom_db / 20.0)
            stereo_mix = (stereo_mix / peak) * target_peak

        return stereo_mix, manifest

    def save_wav(self, path: str | Path, stereo_audio: np.ndarray):
        """Save (2, N) audio array as 16-bit or 24-bit wav."""
        sf.write(str(path), stereo_audio.T, self.sample_rate)
