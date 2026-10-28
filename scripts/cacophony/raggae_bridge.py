"""Bridge between vault-cacophony, vault-raggae (:8750), and voices.db.

Provides:
1. `batch_score_tones`: Batches 5–10 dialogue turns + unique `ToneAnchor` semantic descriptions into a
   single `POST http://127.0.0.1:8750/v1/embed` request (`dim=256`, `project="vault-cacophony"`) and
   computes cosine similarity so each turn gets the best-matching opening tone from its persona's
   `tone_palette`. Includes a fast lexical/punctuation fallback when `:8750` is unreachable.
2. `sync_custom_voices_to_store`: Ensures the custom reference voices in `voices-store/` and
   `samples/voices/` are enrolled in `voices.db` along with their `VocalHabitProfile` JSON.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cacophony.habit_profiles import ToneAnchor, VocalHabitProfile, builtin_profiles
import voices as voices_mod


DEFAULT_RAGGAE_URL = os.environ.get("RAGGAE_URL", "http://127.0.0.1:8750")


def _unit_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return mat / norms


def _lexical_score_tone(text: str, tone: ToneAnchor) -> float:
    """Fast lexical + punctuation scorer used either as a tie-breaker or when vault-raggae is offline."""
    lower = text.lower()
    score = 0.0
    for kw in tone.lexical_keywords:
        if kw.lower() in lower:
            score += 0.28
    if "!" in text and tone.tag in ("excited", "in a hurry tone", "surprised"):
        score += 0.22
    if "?" in text and tone.tag in ("skeptical", "surprised", "sarcastic"):
        score += 0.18
    if "..." in text and tone.tag in ("soft tone", "worried", "skeptical", "sarcastic"):
        score += 0.15
    words = text.split()
    if len(words) <= 10 and tone.tag in ("in a hurry tone", "icy tone", "calm"):
        score += 0.10
    return score


class RaggaeBridge:
    """Client for vault-raggae (:8750) batched EmbeddingGemma-2 tone scoring and voices.db synchronization."""

    def __init__(self, base_url: str = DEFAULT_RAGGAE_URL, timeout_s: float = 3.5):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._anchor_vec_cache: Dict[str, np.ndarray] = {}

    def embed_texts(self, texts: List[str], dim: int = 256, caller: str = "smart_picker") -> Optional[np.ndarray]:
        """Embed a batch of texts in a single HTTP request to vault-raggae `/v1/embed`."""
        if not texts:
            return np.zeros((0, dim), dtype=np.float32)
        payload = {
            "inputs": [{"text": t} for t in texts],
            "dim": dim,
            "project": "vault-cacophony",
            "caller": caller,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-vw-project": "vault-cacophony",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                if resp.status != 200:
                    return None
                body = json.loads(resp.read().decode("utf-8"))
                vecs = body.get("embeddings")
                if not vecs or len(vecs) != len(texts):
                    return None
                return _unit_rows(np.asarray(vecs, dtype=np.float32))
        except Exception:
            return None

    def score_tones_batch(
        self,
        texts: List[str],
        profiles: List[VocalHabitProfile],
    ) -> List[Tuple[ToneAnchor, float, str]]:
        """Score a batch of (turn_text, VocalHabitProfile) pairs in one batched EG2 call.

        Returns a list of `(selected_ToneAnchor, confidence_score, method)` where `method`
        is `"eg2+lexical"` when vault-raggae `/v1/embed` responded, or `"lexical"` fallback.
        """
        if not texts:
            return []

        # Collect any uncached anchor descriptions so we embed turns + new anchors in ONE request
        missing_anchors: List[str] = []
        for prof in profiles:
            for ta in prof.tone_palette:
                if ta.anchor not in self._anchor_vec_cache and ta.anchor not in missing_anchors:
                    missing_anchors.append(ta.anchor)

        combined_inputs = list(texts) + missing_anchors
        # EmbeddingGemma-2 accepts up to 64 inputs per call
        vecs = self.embed_texts(combined_inputs[:64], dim=256, caller="score_tones_batch") if len(combined_inputs) <= 64 else None

        turn_vecs: Optional[np.ndarray] = None
        if vecs is not None and len(vecs) == len(combined_inputs):
            turn_vecs = vecs[: len(texts)]
            for idx, anchor_str in enumerate(missing_anchors):
                self._anchor_vec_cache[anchor_str] = vecs[len(texts) + idx]

        results: List[Tuple[ToneAnchor, float, str]] = []
        for i, (text, prof) in enumerate(zip(texts, profiles)):
            palette = prof.tone_palette
            if not palette:
                fallback = ToneAnchor(tag=prof.default_tone, anchor=prof.default_tone)
                results.append((fallback, 1.0, "default"))
                continue

            best_tone = palette[0]
            best_score = -1e9
            method = "lexical"

            for ta in palette:
                lex = _lexical_score_tone(text, ta)
                if turn_vecs is not None and ta.anchor in self._anchor_vec_cache:
                    eg2_sim = float(np.dot(turn_vecs[i], self._anchor_vec_cache[ta.anchor]))
                    total = 0.72 * eg2_sim + 0.28 * lex
                    method = "eg2+lexical"
                else:
                    total = lex
                if ta.tag == prof.default_tone:
                    total += 0.04  # slight prior toward persona's signature baseline tone
                if total > best_score:
                    best_score = total
                    best_tone = ta

            results.append((best_tone, round(float(best_score), 4), method))
        return results


def sync_custom_voices_to_store(store_path: Optional[str] = None, embed_audio: bool = False) -> List[str]:
    """Ensure the 5 core custom reference voices and their VocalHabitProfiles exist in voices.db."""
    vs_dir = REPO_DIR / "voices-store"
    samples_dir = REPO_DIR / "samples" / "voices"
    custom_roster = [
        ("high-woman-voice-excited", vs_dir / "high-woman-voice-excited.wav", "MARGOT"),
        ("sarcastic-male-voice", vs_dir / "sarcastic-male-voice.wav", "DESMOND"),
        ("normal-male-slightly-faster-cadence", vs_dir / "normal-male-slightly-faster-cadence.wav", "KAI"),
        ("softwaredev-30yomale-drama3", vs_dir / "softwaredev-30yomale-drama3.wav", "CHLOE"),
        ("clopeux-voice-samples1", samples_dir / "clopeux-voice-samples1.wav", "CLOPEUX"),
    ]
    profiles = builtin_profiles()
    synced: List[str] = []

    embedder = None
    if embed_audio:
        try:
            from voiceprint import Embedder, read_wav
            import prosody
            embedder = (Embedder(), read_wav, prosody)
        except Exception:
            embedder = None

    with voices_mod.VoiceStore(store_path) as store:
        for voice_name, wav_path, persona_name in custom_roster:
            txt_path = wav_path.with_suffix(".txt")
            ref_text = txt_path.read_text(encoding="utf-8", errors="replace").strip() if txt_path.exists() else ""
            prof = profiles.get(persona_name) or profiles.get(voice_name)
            habits_dict = prof.to_dict() if prof else {}

            if embedder is not None and wav_path.exists():
                emb_obj, _, prosody_mod = embedder
                import soundfile as sf
                from scipy.signal import resample_poly
                from math import gcd

                audio_raw, sr = sf.read(str(wav_path), dtype="float32")
                if audio_raw.ndim > 1:
                    audio_raw = np.mean(audio_raw, axis=1)
                if sr != 16000:
                    g = gcd(16000, int(sr))
                    audio = resample_poly(audio_raw, 16000 // g, int(sr) // g).astype(np.float32)
                else:
                    audio = audio_raw.astype(np.float32)
                vec = emb_obj.embed(audio)
                windows = emb_obj.embed_windows(audio)
                if vec is not None:
                    obs_id = store.observe(
                        vec,
                        windows=windows,
                        cohesion=round(float(np.mean(np.dot(windows, vec))), 4) if len(windows) else None,
                        voice=voice_name,
                        label=persona_name,
                        media=wav_path.name,
                        media_path=str(wav_path.resolve()),
                        start_s=0.0,
                        end_s=len(audio) / 16000.0,
                        speech_seconds=len(audio) / 16000.0,
                        metrics=prosody_mod.measure(audio),
                        source={"pass": "custom_voice_sync", "persona": persona_name},
                    )
                    store.enroll(
                        voice_name,
                        vec,
                        notes=f"Custom reference voice for {persona_name}",
                        observation_id=obs_id,
                        ref_wav=str(wav_path.resolve()),
                        ref_text=ref_text,
                        habits=habits_dict,
                    )
                    synced.append(voice_name)
                    continue

            store.set_habits(
                voice_name,
                habits=habits_dict,
                ref_wav=str(wav_path.resolve()) if wav_path.exists() else "",
                ref_text=ref_text,
            )
            synced.append(voice_name)

    return synced
