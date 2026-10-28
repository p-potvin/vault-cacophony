"""Modular TTS engine wrapper supporting Magpie TTS (NeMo-Speech.cpp), Chatterbox, and CosyVoice3."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import resample_poly

# Path resolution relative to repo root
REPO_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import trim_speech


def time_stretch_wsola(audio: np.ndarray, rate: float, sr: int = 24000) -> np.ndarray:
    """Pitch-preserving time scale modification via Waveform Similarity Overlap-Add (WSOLA)."""
    if abs(rate - 1.0) < 0.02:
        return audio
    win_len = int(0.025 * sr)  # 25ms window
    if win_len % 2 != 0:
        win_len += 1
    hop_syn = win_len // 2
    hop_ana = int(hop_syn * rate)
    delta = win_len // 4

    window = np.hanning(win_len).astype(np.float32)
    num_syn_frames = int(np.ceil((len(audio) - win_len - delta) / hop_ana))
    if num_syn_frames <= 0:
        return audio

    out_len = num_syn_frames * hop_syn + win_len
    output = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    syn_pos = 0
    ana_pos = 0
    last_frame = np.zeros(win_len, dtype=np.float32)

    for i in range(num_syn_frames):
        search_start = max(0, ana_pos - delta)
        search_end = min(len(audio) - win_len, ana_pos + delta)

        if i == 0 or search_end <= search_start:
            best_pos = min(max(0, ana_pos), len(audio) - win_len)
        else:
            target = last_frame[hop_syn:]
            best_corr = -1e10
            best_pos = search_start
            for pos in range(search_start, search_end, 2):
                seg = audio[pos : pos + len(target)]
                corr = np.dot(target, seg)
                if corr > best_corr:
                    best_corr = corr
                    best_pos = pos

        frame = audio[best_pos : best_pos + win_len] * window
        output[syn_pos : syn_pos + win_len] += frame
        norm[syn_pos : syn_pos + win_len] += window
        last_frame = audio[best_pos : best_pos + win_len]

        syn_pos += hop_syn
        ana_pos += hop_ana

    valid = norm > 1e-4
    output[valid] /= norm[valid]
    return output


def normalize_speech_loudness(
    audio: np.ndarray,
    target_rms_db: float = -18.0,
    peak_ceiling_db: float = -1.5,
    max_gain_db: float = 20.0,
) -> np.ndarray:
    """Normalize active speech RMS to target_rms_db with a smooth soft-knee limiter at peak_ceiling_db."""
    if len(audio) == 0:
        return audio
    y = audio.astype(np.float32, copy=True)
    y -= float(np.mean(y))

    # Compute RMS over active speech samples (above -45 dBFS gate) so pauses don't skew loudness
    active = np.abs(y) > 0.0056
    rms = float(np.sqrt(np.mean(y[active] ** 2))) if np.any(active) else float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-6:
        return y

    target_rms = 10.0 ** (target_rms_db / 20.0)
    max_gain = 10.0 ** (max_gain_db / 20.0)
    gain = min(target_rms / rms, max_gain)
    y *= gain

    # Smooth soft-knee peak ceiling so loud transients never clip or jump above peak_ceiling_db
    ceiling = 10.0 ** (peak_ceiling_db / 20.0)
    peak = float(np.max(np.abs(y)))
    if peak > ceiling:
        knee = 0.75 * ceiling
        above = np.abs(y) > knee
        sign = np.sign(y[above])
        excess = (np.abs(y[above]) - knee) / max(1e-6, ceiling - knee)
        y[above] = sign * (knee + (ceiling - knee) * np.tanh(excess))
    return y.astype(np.float32)


def ensure_voice_ref_text(wav_path: Path, crispasr_bin: Optional[Path] = None) -> str:
    """Return the reference transcript for wav_path, auto-transcribing with Parakeet if .txt is missing."""
    txt_path = wav_path.with_suffix(".txt")
    if txt_path.exists():
        cached = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if cached:
            return cached
    bin_path = crispasr_bin or (REPO_DIR / "CrispASR" / "build" / "bin" / "crispasr.exe")
    if not bin_path.exists() or not wav_path.exists():
        return ""
    proc = subprocess.run(
        [str(bin_path), "--backend", "parakeet", "-f", str(wav_path.resolve()), "-nt", "-np"],
        cwd=str(bin_path.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = [
        ln.strip()
        for ln in (proc.stdout or "").splitlines()
        if ln.strip() and not ln.startswith("crispasr") and not ln.startswith("whisper")
    ]
    transcript = " ".join(lines).strip()
    if transcript:
        try:
            txt_path.write_text(transcript + "\n", encoding="utf-8")
        except OSError:
            pass
    return transcript


class TtsEngine:
    """Invokes audio.cpp (OmniVoice Q8_0 / Fish Audio S2 Pro), NeMo-Speech.cpp (Magpie), or CrispASR (CosyVoice3/Chatterbox/Kokoro).

    Supports a Dual-GPU warm `audiocpp_server.exe` pool (`gpu_devices=[0, 1]`) so two OmniVoice
    engines synthesize batch turns simultaneously across GPU 0 (RTX 3060) and GPU 1 (RTX 2060).
    """

    def __init__(
        self,
        engine_type: str = "omnivoice",  # "omnivoice", "fish_audio", "chatterbox", "cosyvoice", "magpie", or "kokoro"
        gpu_device: Optional[int] = None,
        gpu_devices: Optional[List[int]] = None,
        sample_rate: int = 24000,
    ):
        self.engine_type = engine_type.lower()
        self.gpu_device = gpu_device
        if gpu_devices is not None:
            self.gpu_devices = list(gpu_devices)
        elif gpu_device is not None and gpu_device >= 0:
            self.gpu_devices = [int(gpu_device)]
        elif self.engine_type in ("omnivoice", "omni"):
            # OmniVoice Q8_0 uses ~3.1 GB VRAM, so run 2 warm engines across GPU 0 + GPU 1 by default
            self.gpu_devices = [0, 1]
        else:
            self.gpu_devices = [0]
        self.sample_rate = sample_rate

        self.nemo_bin = REPO_DIR / "NeMo-Speech.cpp" / "build-cuda" / "bin" / "nemo-speech.exe"
        self.crispasr_bin = REPO_DIR / "CrispASR" / "build" / "bin" / "crispasr.exe"
        self.audiocpp_bin = REPO_DIR / "audio.cpp" / "audiocpp_cli.exe"
        self.audiocpp_server_bin = REPO_DIR / "audio.cpp" / "audiocpp_server.exe"
        self.fish_model = REPO_DIR / "audio.cpp" / "models" / "Fish-Audio-S2-Pro-GGUF" / "fish-audio-s2-pro-q8_0.gguf"
        self.omnivoice_model = REPO_DIR / "audio.cpp" / "models" / "OmniVoice-GGUF" / "omnivoice-q8_0.gguf"
        self.fish_server_port = int(os.environ.get("AUDIOCPP_SERVER_PORT", "8759"))
        self.omnivoice_base_port = int(os.environ.get("AUDIOCPP_OMNIVOICE_PORT", "8761"))
        self._fish_server_proc: Optional[subprocess.Popen] = None
        self._server_procs: Dict[int, subprocess.Popen] = {}
        self._server_lock = threading.Lock()
        self._rr_counter = 0

        if self.engine_type == "magpie" and not self.nemo_bin.exists():
            raise FileNotFoundError(f"nemo-speech binary not found at {self.nemo_bin}")
        if self.engine_type in ("chatterbox", "cosyvoice", "kokoro") and not self.crispasr_bin.exists():
            raise FileNotFoundError(f"crispasr binary not found at {self.crispasr_bin}")
        if self.engine_type in ("fish_audio", "fish", "fish-s2", "omnivoice", "omni") and not self.audiocpp_bin.exists():
            raise FileNotFoundError(f"audiocpp_cli binary not found at {self.audiocpp_bin}")

    @staticmethod
    def _port_healthy(port: int, expected_model_id: Optional[str] = None) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.5) as resp:
                if resp.status != 200:
                    return False
            if expected_model_id:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1.5) as mresp:
                    if mresp.status != 200:
                        return False
                    body = mresp.read().decode("utf-8", errors="replace")
                    if expected_model_id not in body:
                        return False
            return True
        except Exception:
            return False

    def _fish_server_healthy(self) -> bool:
        return self._port_healthy(self.fish_server_port, "fish_audio")

    def _ensure_audiocpp_server(
        self,
        family: str,
        model_path: Path,
        port: int,
        device: int,
        session_options: Optional[ dict ] = None,
    ) -> str:
        """Start or reuse a persistent audiocpp_server.exe instance on a specific port and CUDA device."""
        base_url = f"http://127.0.0.1:{port}"
        if self._port_healthy(port, family):
            return base_url

        with self._server_lock:
            if self._port_healthy(port, family):
                return base_url

            if not self.audiocpp_server_bin.exists():
                raise FileNotFoundError(f"audiocpp_server.exe not found at {self.audiocpp_server_bin}")

            cfg_path = Path(tempfile.gettempdir()) / f"audiocpp_{family}_gpu{device}_{port}.json"
            cfg = {
                "host": "127.0.0.1",
                "port": port,
                "backend": "cuda",
                "device": int(device),
                "threads": 2,
                "lazy_load": True,
                "models": [
                    {
                        "id": family,
                        "family": family,
                        "path": str(model_path.resolve()).replace("\\", "/"),
                        "task": "tts",
                        "mode": "offline",
                        "session_options": session_options or {},
                    }
                ],
            }
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

            print(f"  [*] Starting warm audiocpp_server on {base_url} ({family} on GPU {device})...")
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen(
                [str(self.audiocpp_server_bin), "--config", str(cfg_path), "--no-ui"],
                cwd=str(self.audiocpp_server_bin.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self._server_procs[port] = proc
            if family == "fish_audio":
                self._fish_server_proc = proc

            deadline = time.time() + 35.0
            while time.time() < deadline:
                if self._port_healthy(port, family):
                    return base_url
                if proc.poll() is not None:
                    break
                time.sleep(0.25)
            raise RuntimeError(f"audiocpp_server ({family}, GPU {device}) failed to become healthy on {base_url}")

    def _ensure_fish_server(self, fish_model: Path, device: Optional[int] = None) -> str:
        """Start or reuse a persistent audiocpp_server.exe instance so Fish Audio S2 Pro stays warm in VRAM."""
        dev = device if (device is not None and device >= 0) else (0 if (self.gpu_device is None or self.gpu_device < 0) else int(self.gpu_device))
        port = self.fish_server_port + (dev if dev > 0 else 0)
        return self._ensure_audiocpp_server(
            family="fish_audio",
            model_path=fish_model,
            port=port,
            device=dev,
            session_options={"reference_cache_slots": "8"},
        )

    def _ensure_omnivoice_server(self, omni_model: Path, device: int = 0) -> str:
        """Start or reuse a persistent audiocpp_server.exe instance for OmniVoice Q8_0 on `device`."""
        dev = max(0, int(device))
        port = self.omnivoice_base_port + dev
        return self._ensure_audiocpp_server(
            family="omnivoice",
            model_path=omni_model,
            port=port,
            device=dev,
            session_options={},
        )

    def warm_pool(self, engine_type: Optional[str] = None) -> List[str]:
        """Pre-warm audiocpp_server instances across all configured `gpu_devices` in parallel."""
        eng = (engine_type or self.engine_type).lower()
        if eng not in ("omnivoice", "omni"):
            return []
        omni_model = Path(os.environ.get("AUDIOCPP_OMNIVOICE_MODEL", str(self.omnivoice_model)))
        if not omni_model.exists() or not self.audiocpp_server_bin.exists():
            return []
        devices = self.gpu_devices or [0]
        urls: List[str] = []
        with ThreadPoolExecutor(max_workers=len(devices)) as ex:
            futs = [ex.submit(self._ensure_omnivoice_server, omni_model, dev) for dev in devices]
            for f in futs:
                try:
                    urls.append(f.result())
                except Exception as exc:
                    print(f"  [!] Warning warming OmniVoice server: {exc}")
        return urls

    def synthesize(
        self,
        text: str,
        speaker_idx: int = 0,
        voice_name: Optional[str] = None,
        speed_ratio: float = 1.0,
        voice_ref_wav: Optional[str | Path] = None,
        voice_ref_text: Optional[str] = None,
        output_wav: str | Path = "out.wav",
        preset_voice: Optional[str] = None,
        engine_type: Optional[str] = None,
        voice_pack_gguf: Optional[str | Path] = None,
        gpu_device_override: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize text into speech audio array at target sample rate with cadence acceleration.

        Args:
            text: Text utterance to speak.
            speaker_idx: Integer speaker index (for Magpie: 0, 1, 2, 3...).
            voice_name: Baked speaker identity name (e.g. "Aria", "Jason", "John", "Sofia", "Leo"
                for Magpie; Kokoro voice pack name like "af_heart"; or any label for cloning engines).
            speed_ratio: Cadence time stretch ratio (1.0 = normal, >1.0 = faster speech without pitch shift).
            voice_ref_wav: Voice reference wav (for cloning models).
            voice_ref_text: Transcript of voice reference.
            output_wav: Destination file path.
            preset_voice: Voice identifier.
            engine_type: Per-call engine override (defaults to the engine configured at init).
            voice_pack_gguf: Kokoro voice pack .gguf path (--voice pack).

        Returns:
            Tuple of (audio_numpy_array, sample_rate).
        """
        output_path = Path(output_wav).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        engine = (engine_type or self.engine_type).lower()
        target_gpu = gpu_device_override if gpu_device_override is not None else self.gpu_device
        if target_gpu is None and self.gpu_devices:
            with self._server_lock:
                target_gpu = self.gpu_devices[self._rr_counter % len(self.gpu_devices)]
                self._rr_counter += 1

        env = os.environ.copy()
        if target_gpu is not None and target_gpu >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(target_gpu)
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)

        if engine == "magpie":
            cmd = [
                str(self.nemo_bin),
                "synthesize",
                text,
            ]
            if voice_name:
                cmd.extend(["--voice", voice_name])
            else:
                cmd.extend(["--speaker", str(speaker_idx)])
            cmd.extend([
                "-o", str(output_path),
                "--force",
            ])
            if self.gpu_device is not None:
                cmd.extend(["--device", f"cuda:{self.gpu_device}"])
            else:
                cmd.extend(["--device", "auto"])
            cwd = self.nemo_bin.parent
        elif engine == "chatterbox":
            cmd = [
                str(self.crispasr_bin),
                "--backend", "chatterbox",
                "--tts", text,
                "--tts-output", str(output_path),
            ]
            cwd = self.crispasr_bin.parent
        elif engine == "kokoro":
            kokoro_model = os.environ.get("CRISPASR_KOKORO_MODEL", "")
            if not kokoro_model:
                kokoro_model = Path.home() / ".cache" / "crispasr" / "kokoro-82m-f16.gguf"
            kokoro_model = Path(kokoro_model)
            if not kokoro_model.exists():
                raise FileNotFoundError(
                    f"Kokoro model not found at {kokoro_model}; set CRISPASR_KOKORO_MODEL or download the GGUF."
                )
            if not voice_pack_gguf or not Path(voice_pack_gguf).exists():
                raise FileNotFoundError(f"Kokoro voice pack not found: {voice_pack_gguf}")
            cmd = [
                str(self.crispasr_bin),
                "--backend", "kokoro",
                "-m", str(kokoro_model),
                "--voice", str(voice_pack_gguf),
                "--tts", text,
                "--tts-output", str(output_path),
            ]
            cwd = self.crispasr_bin.parent
        elif engine in ("omnivoice", "omni"):
            omni_model = Path(os.environ.get("AUDIOCPP_OMNIVOICE_MODEL", str(self.omnivoice_model)))
            if not omni_model.exists():
                raise FileNotFoundError(
                    f"OmniVoice Q8_0 GGUF not found at {omni_model}; install omnivoice_q8_0 via model_manager_v2.py."
                )
            ref_wav_abs = Path(voice_ref_wav).resolve() if (voice_ref_wav and Path(voice_ref_wav).exists()) else None
            ref_txt = (voice_ref_text or ensure_voice_ref_text(ref_wav_abs, self.crispasr_bin)) if ref_wav_abs else None
            dev = int(target_gpu) if (target_gpu is not None and target_gpu >= 0) else 0

            synthesized_via_server = False
            if self.audiocpp_server_bin.exists():
                try:
                    base_url = self._ensure_omnivoice_server(omni_model, device=dev)
                    payload: Dict[str, object] = {
                        "model": "omnivoice",
                        "input": text,
                    }
                    if ref_wav_abs:
                        payload["voice_ref"] = str(ref_wav_abs).replace("\\", "/")
                    if ref_txt:
                        payload["reference_text"] = str(ref_txt)
                    if speed_ratio is not None and abs(speed_ratio - 1.0) >= 0.02:
                        payload["options"] = {"speed": str(round(float(speed_ratio), 3))}
                    req = urllib.request.Request(
                        f"{base_url}/v1/audio/speech",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=180.0) as resp:
                        wav_bytes = resp.read()
                    if wav_bytes and len(wav_bytes) > 256:
                        output_path.write_bytes(wav_bytes)
                        synthesized_via_server = True
                        # OmniVoice already applied native speed multiplier
                        speed_ratio = 1.0
                except Exception as srv_err:
                    print(f"  [!] audiocpp_server (omnivoice GPU {dev}) warning ({srv_err}); falling back to audiocpp_cli...")

            cmd = None
            cwd = self.audiocpp_bin.parent
            if not synthesized_via_server:
                env_cli = os.environ.copy()
                env_cli.pop("CUDA_VISIBLE_DEVICES", None)
                env = env_cli
                cmd = [
                    str(self.audiocpp_bin),
                    "--task", "tts",
                    "--family", "omnivoice",
                    "--model", str(omni_model),
                    "--backend", "cuda",
                    "--device", str(dev),
                    "--text", text,
                    "--out", str(output_path),
                ]
                if ref_wav_abs:
                    cmd.extend(["--voice-ref", str(ref_wav_abs)])
                    if ref_txt:
                        cmd.extend(["--reference-text", str(ref_txt)])
                if speed_ratio is not None and abs(speed_ratio - 1.0) >= 0.02:
                    cmd.extend(["--request-option", f"speed={speed_ratio:.3f}"])
                    speed_ratio = 1.0
        elif engine in ("fish_audio", "fish", "fish-s2"):
            fish_model = Path(os.environ.get("AUDIOCPP_FISH_MODEL", str(self.fish_model)))
            if not fish_model.exists():
                raise FileNotFoundError(
                    f"Fish Audio S2 Pro GGUF not found at {fish_model}; download Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf."
                )
            ref_wav_abs = Path(voice_ref_wav).resolve() if (voice_ref_wav and Path(voice_ref_wav).exists()) else None
            ref_txt = (voice_ref_text or ensure_voice_ref_text(ref_wav_abs, self.crispasr_bin)) if ref_wav_abs else None

            # Synthesize via warm audiocpp_server HTTP endpoint first so the 6.3GB model is never cold-reloaded per turn
            synthesized_via_server = False
            if self.audiocpp_server_bin.exists():
                try:
                    base_url = self._ensure_fish_server(fish_model)
                    payload = {
                        "model": "fish_audio",
                        "input": text,
                    }
                    if ref_wav_abs:
                        payload["voice_ref"] = str(ref_wav_abs).replace("\\", "/")
                    if ref_txt:
                        payload["reference_text"] = str(ref_txt)
                    req = urllib.request.Request(
                        f"{base_url}/v1/audio/speech",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=180.0) as resp:
                        wav_bytes = resp.read()
                    if wav_bytes and len(wav_bytes) > 256:
                        output_path.write_bytes(wav_bytes)
                        synthesized_via_server = True
                except Exception as srv_err:
                    print(f"  [!] audiocpp_server synthesis warning ({srv_err}); falling back to audiocpp_cli...")

            cmd = None
            cwd = self.audiocpp_bin.parent
            if not synthesized_via_server:
                cmd = [
                    str(self.audiocpp_bin),
                    "--task", "tts",
                    "--family", "fish_audio",
                    "--model", str(fish_model),
                    "--backend", "cuda",
                    "--text", text,
                    "--out", str(output_path),
                ]
                if self.gpu_device is not None and self.gpu_device >= 0:
                    cmd.extend(["--device", str(self.gpu_device)])
                if ref_wav_abs:
                    cmd.extend(["--voice-ref", str(ref_wav_abs)])
                    if ref_txt:
                        cmd.extend(["--reference-text", str(ref_txt)])
        else:  # cosyvoice
            cmd = [
                str(self.crispasr_bin),
                "--backend", "cosyvoice3-tts",
                "--tts", text,
                "--tts-output", str(output_path),
            ]
            if voice_ref_wav and Path(voice_ref_wav).exists():
                ref_wav_abs = Path(voice_ref_wav).resolve()
                ref_txt = voice_ref_text or ensure_voice_ref_text(ref_wav_abs, self.crispasr_bin)
                cmd.extend(["--voice", str(ref_wav_abs), "--i-have-rights", "--no-spoken-disclaimer", "--accept-marking-responsibility"])
                if ref_txt:
                    cmd.extend(["--ref-text", str(ref_txt)])
            elif preset_voice:
                cmd.extend(["--voice", preset_voice])
            cwd = self.crispasr_bin.parent

        if cmd is not None:
            proc = None
            for attempt in range(3):
                current_cmd = list(cmd)
                if attempt == 2 and engine == "magpie":
                    if "--device" in current_cmd:
                        dev_idx = current_cmd.index("--device")
                        current_cmd[dev_idx + 1] = "cpu"
                    else:
                        current_cmd.extend(["--device", "cpu"])

                proc = subprocess.run(
                    current_cmd,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                if proc.returncode == 0 and output_path.exists():
                    break
                time.sleep(0.5)

            if proc is None or proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip() if proc else "No process executed"
                raise RuntimeError(f"TTS synthesis ({engine}) failed (code {getattr(proc, 'returncode', -1)}): {err[-300:]}")

        if not output_path.exists():
            raise RuntimeError(f"TTS finished but output wav {output_path} was not created.")

        audio, sr = trim_speech.read(str(output_path))

        # Resample to target sample_rate if different (e.g. 22050/44100 Hz -> 24000 Hz)
        if sr != self.sample_rate:
            g = gcd(int(self.sample_rate), int(sr))
            up = int(self.sample_rate) // g
            down = int(sr) // g
            audio = resample_poly(audio.astype(np.float32), up, down)
            sr = self.sample_rate

        # Trim silence
        bounds = trim_speech.speech_bounds(audio, sr)
        if bounds is not None:
            start_s = max(0.0, bounds[0] - trim_speech.MARGIN_S)
            end_s = min(len(audio) / sr, bounds[1] + trim_speech.MARGIN_S)
            audio = audio[int(start_s * sr) : int(end_s * sr)]

        # Apply cadence time-stretch if speed_ratio != 1.0
        if speed_ratio is not None and abs(speed_ratio - 1.0) >= 0.02:
            audio = time_stretch_wsola(audio, rate=speed_ratio, sr=sr)

        # Normalize active speech loudness and peak ceiling across all voices
        audio = normalize_speech_loudness(audio, target_rms_db=-18.0, peak_ceiling_db=-1.5)
        trim_speech.write(str(output_path), audio, sr)

        return audio, sr

    def synthesize_batch_parallel(
        self,
        jobs: List[Dict[str, object]],
    ) -> List[Tuple[np.ndarray, int]]:
        """Synthesize a batch of turns concurrently across all configured GPUs (`self.gpu_devices`).

        When `self.gpu_devices == [0, 1]` (e.g. RTX 3060 + RTX 2060), Worker 0 dispatches even/assigned
        jobs to GPU 0's warm `audiocpp_server` (`:8761`) while Worker 1 simultaneously dispatches to
        GPU 1's warm `audiocpp_server` (`:8762`).
        """
        if not jobs:
            return []
        devices = self.gpu_devices or [0]
        if len(devices) <= 1 or len(jobs) == 1:
            dev = devices[0]
            return [
                self.synthesize(
                    text=str(job["text"]),
                    speaker_idx=int(job.get("speaker_idx", 0)),
                    voice_name=job.get("voice_name"),  # type: ignore[arg-type]
                    speed_ratio=float(job.get("speed_ratio", 1.0)),
                    voice_ref_wav=job.get("voice_ref_wav"),  # type: ignore[arg-type]
                    voice_ref_text=job.get("voice_ref_text"),  # type: ignore[arg-type]
                    output_wav=job["output_wav"],  # type: ignore[arg-type]
                    preset_voice=job.get("preset_voice"),  # type: ignore[arg-type]
                    engine_type=job.get("engine_type"),  # type: ignore[arg-type]
                    voice_pack_gguf=job.get("voice_pack_gguf"),  # type: ignore[arg-type]
                    gpu_device_override=dev,
                )
                for job in jobs
            ]

        # Pre-warm all GPU servers concurrently before dispatching synthesis jobs
        self.warm_pool()

        results: List[Optional[Tuple[np.ndarray, int]]] = [None] * len(jobs)

        def _run_one(idx: int, job: Dict[str, object]) -> Tuple[int, np.ndarray, int]:
            assigned_gpu = devices[idx % len(devices)]
            try:
                audio, sr = self.synthesize(
                    text=str(job["text"]),
                    speaker_idx=int(job.get("speaker_idx", 0)),
                    voice_name=job.get("voice_name"),  # type: ignore[arg-type]
                    speed_ratio=float(job.get("speed_ratio", 1.0)),
                    voice_ref_wav=job.get("voice_ref_wav"),  # type: ignore[arg-type]
                    voice_ref_text=job.get("voice_ref_text"),  # type: ignore[arg-type]
                    output_wav=job["output_wav"],  # type: ignore[arg-type]
                    preset_voice=job.get("preset_voice"),  # type: ignore[arg-type]
                    engine_type=job.get("engine_type"),  # type: ignore[arg-type]
                    voice_pack_gguf=job.get("voice_pack_gguf"),  # type: ignore[arg-type]
                    gpu_device_override=assigned_gpu,
                )
                return idx, audio, sr
            except Exception as first_err:
                fallback_gpu = devices[(idx + 1) % len(devices)]
                print(f"  [!] Turn synthesis on GPU {assigned_gpu} had transient error ({first_err}); retrying on GPU {fallback_gpu}...")
                audio, sr = self.synthesize(
                    text=str(job["text"]),
                    speaker_idx=int(job.get("speaker_idx", 0)),
                    voice_name=job.get("voice_name"),  # type: ignore[arg-type]
                    speed_ratio=float(job.get("speed_ratio", 1.0)),
                    voice_ref_wav=job.get("voice_ref_wav"),  # type: ignore[arg-type]
                    voice_ref_text=job.get("voice_ref_text"),  # type: ignore[arg-type]
                    output_wav=job["output_wav"],  # type: ignore[arg-type]
                    preset_voice=job.get("preset_voice"),  # type: ignore[arg-type]
                    engine_type=job.get("engine_type"),  # type: ignore[arg-type]
                    voice_pack_gguf=job.get("voice_pack_gguf"),  # type: ignore[arg-type]
                    gpu_device_override=fallback_gpu,
                )
                return idx, audio, sr

        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futs = [pool.submit(_run_one, idx, job) for idx, job in enumerate(jobs)]
            for fut in futs:
                idx, audio, sr = fut.result()
                results[idx] = (audio, sr)

        return [r for r in results if r is not None]
