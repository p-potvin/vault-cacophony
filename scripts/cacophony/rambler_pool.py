"""Rambler & background filler audio pool for multi-speaker cacophony."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass
class AudioCut:
    speaker_id: str
    file_path: Path
    duration: float
    sample_rate: int
    text: Optional[str] = None

    def load_audio(self, target_sr: Optional[int] = None, max_duration: Optional[float] = None) -> np.ndarray:
        """Load audio array, optionally resampled and trimmed."""
        audio, sr = sf.read(str(self.file_path))
        if audio.ndim > 1:
            audio = audio[:, 0]  # Take first channel
        if target_sr is not None and sr != target_sr:
            # Resample using rational fraction with resample_poly
            from math import gcd
            g = gcd(int(target_sr), int(sr))
            up = int(target_sr) // g
            down = int(sr) // g
            audio = resample_poly(audio.astype(np.float32), up, down)
            sr = target_sr
        if max_duration is not None:
            max_len = int(max_duration * sr)
            if len(audio) > max_len:
                audio = audio[:max_len]
        return audio.astype(np.float32)


class RamblerPool:
    """Manages an indexed collection of ambient speech utterances for incoherent background ramblers."""

    def __init__(
        self,
        source_dir: Optional[Path] = None,
        min_duration: float = 1.0,
        max_duration: float = 6.0,
    ):
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.source_dir = source_dir or (
            repo_root / "samples" / "dev-clean" / "LibriSpeech" / "dev-clean"
        )
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.spk2cuts: Dict[str, List[AudioCut]] = {}
        self._index_dataset()

    def _index_dataset(self):
        """Find and group flac/wav speech clips by speaker ID."""
        if not self.source_dir.exists():
            return

        patterns = [str(self.source_dir / "**" / "*.flac"), str(self.source_dir / "**" / "*.wav")]
        all_files = []
        for pat in patterns:
            all_files.extend(glob.glob(pat, recursive=True))

        for fpath_str in all_files:
            fpath = Path(fpath_str)
            # Infer speaker ID from directory structure or filename prefix
            parts = fpath.stem.split("-")
            speaker_id = parts[0] if parts else fpath.parent.name

            try:
                info = sf.info(str(fpath))
                if self.min_duration <= info.duration <= self.max_duration:
                    cut = AudioCut(
                        speaker_id=speaker_id,
                        file_path=fpath,
                        duration=info.duration,
                        sample_rate=info.samplerate,
                    )
                    self.spk2cuts.setdefault(speaker_id, []).append(cut)
            except Exception:
                continue

    @property
    def speakers(self) -> List[str]:
        return list(self.spk2cuts.keys())

    def sample_cut(
        self,
        speaker_id: Optional[str] = None,
        exclude_speakers: Optional[List[str]] = None,
    ) -> Optional[AudioCut]:
        """Sample a random cut from the pool."""
        candidates = self.speakers
        if exclude_speakers:
            candidates = [s for s in candidates if s not in exclude_speakers]
        if not candidates:
            return None

        spk = speaker_id if (speaker_id and speaker_id in self.spk2cuts) else np.random.choice(candidates)
        cuts = self.spk2cuts.get(spk, [])
        if not cuts:
            return None
        return np.random.choice(cuts)
