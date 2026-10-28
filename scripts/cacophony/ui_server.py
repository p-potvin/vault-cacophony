"""Web UI backend for the Cacophony orchestrator.

Serves the built React UI (cacophony-ui/dist) and exposes a small JSON API:

    GET  /                -> built UI (index.html / assets)
    GET  /api/options     -> discoverable run options (scripts, models, voices, defaults)
    POST /api/start       -> launch an orchestrator run from a config JSON
    POST /api/stop        -> terminate the current run
    GET  /api/events      -> Server-Sent Events: live console lines + done event
    GET  /api/file?path=  -> serve an output file (master wav, transcript, manifest)

Run with the repo venv:
    python scripts/cacophony/ui_server.py [--port 8753] [--host 127.0.0.1]

Dev mode: `npm run dev` in cacophony-ui proxies /api to this server.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

REPO_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from cacophony import orchestrator as orch  # noqa: E402
from cacophony.tts_engine import ensure_voice_ref_text  # noqa: E402

DIST_DIR = REPO_DIR / "cacophony-ui" / "dist"
PYTHON = sys.executable

MODEL_PRESETS = [
    "openai/gpt-oss-120b:cheapest",
    "openai/gpt-oss-20b:cheapest"
]
MAGPIE_VOICES = ["Aria", "Jason", "Leo", "Sofia", "John"]
COSYVOICE_BAKED = ["zero_shot", "fleurs-en", "fleurs-de", "fleurs-zh", "fleurs-ja", "fleurs-fr", "fleurs-es", "fleurs-ko"]
ENGINES = ["omnivoice", "fish_audio", "cosyvoice", "chatterbox"]


class RunManager:
    """Owns the single active orchestrator subprocess and its console log."""

    def __init__(self):
        self.lock = threading.Lock()
        self.lines: list[str] = []
        self.subscribers: list[queue.Queue] = []
        self.proc: subprocess.Popen | None = None
        self._starting: bool = False
        self.run_id: str | None = None
        self.result: dict | None = None
        self.out_dir: str | None = None

    # -- log plumbing -------------------------------------------------------
    def append(self, line: str):
        with self.lock:
            self.lines.append(line)
            for q in self.subscribers:
                q.put(line)
            self._trim_locked()

    def _trim_locked(self):
        if len(self.lines) > 4000:
            self.lines = self.lines[-2000:]

    def subscribe(self) -> queue.Queue:
        q = queue.Queue()
        with self.lock:
            for line in self.lines:
                q.put(line)
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def broadcast_event(self, ev: dict):
        payload = f"data: {json.dumps(ev)}\n\n"
        with self.lock:
            for q in self.subscribers:
                q.put(("__EVENT__", payload))

    # -- lifecycle ----------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._starting or (self.proc is not None and self.proc.poll() is None)

    def start(self, config: dict) -> tuple[str | None, str | None]:
        """Launch a run. Returns (run_id, error)."""
        with self.lock:
            if self.running:
                return None, "a run is already in progress"
            self._starting = True
            self.lines = []
            self.result = None
            self.run_id = uuid.uuid4().hex[:8]
            rid = self.run_id

        try:
            cmd, out_dir = build_command(config)
            self.out_dir = str(out_dir)

            # Clean stale prior run artifacts so failed runs never display previous audio
            if out_dir.exists():
                for artifact in ("cacophony_master.wav", "timeline_manifest.json", "transcript.txt", "run_status.json"):
                    stale = out_dir / artifact
                    if stale.exists():
                        try:
                            stale.unlink()
                        except OSError:
                            pass

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self.lock:
                self.proc = proc
                self._starting = False
        except Exception as e:
            with self.lock:
                self._starting = False
                self.run_id = None
            return None, f"failed to launch: {e}"

        self.broadcast_event({"kind": "started", "run_id": rid, "out_dir": str(self.out_dir)})
        self.append(f"[ui] run {rid} started: {os.path.basename(cmd[0])} " + " ".join(cmd[1:]))

        def pump():
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if line:
                    self.append(line)
            code = proc.wait()
            out_p = Path(self.out_dir) if self.out_dir else None
            status_path = out_p / "run_status.json" if out_p else None
            try:
                run_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path and status_path.exists() else None
            except (OSError, json.JSONDecodeError):
                run_status = None
            result = {
                "code": code,
                "run_status": run_status,
                "run_id": rid,
                "out_dir": str(out_p) if out_p else None,
                "master_wav": str(out_p / "cacophony_master.wav") if out_p and (out_p / "cacophony_master.wav").exists() else None,
                "transcript": str(out_p / "transcript.txt") if out_p and (out_p / "transcript.txt").exists() else None,
                "manifest": str(out_p / "timeline_manifest.json") if out_p and (out_p / "timeline_manifest.json").exists() else None,
                "transcript_text": (out_p / "transcript.txt").read_text(encoding="utf-8") if out_p and (out_p / "transcript.txt").exists() else "",
            }
            with self.lock:
                if self.run_id == rid:
                    self.result = result
                    self.proc = None
            self.append(f"[ui] run {rid} finished (code {code})")
            self.broadcast_event({"kind": "done", **result})

        threading.Thread(target=pump, daemon=True).start()
        return rid, None

    def stop(self) -> bool:
        with self.lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return False
        self.append("[ui] stop requested; terminating run process tree...")
        try:
            # Kill the process and any child processes (including warm tts servers)
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            else:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        return True


RUNS = RunManager()


def build_command(config: dict) -> tuple[list[str], Path]:
    """Translate the UI config JSON into orchestrator CLI args."""
    raw_out = (config.get("out_dir") or "cacophony_output/ui_run").strip().replace("\\", "/")
    base_out = (REPO_DIR / "cacophony_output").resolve()
    target_path = (REPO_DIR / raw_out).resolve()
    try:
        target_path.relative_to(base_out)
        out_path = target_path
    except ValueError:
        out_path = base_out / "ui_run"
    # keep the value relative for display but resolve against the repo
    out_arg = str(out_path.relative_to(REPO_DIR)) if out_path.is_relative_to(REPO_DIR) else str(out_path)

    global_engine = config.get("engine_type", "omnivoice").lower()
    cmd = [
        PYTHON,
        "-u",
        str(SCRIPTS_DIR / "cacophony" / "orchestrator.py"),
        "--script", config.get("script", "testbench"),
        "--model", config.get("model", "openai/gpt-oss-120b:cheapest"),
        "--engine-type", global_engine,
        "--num-speakers", str(int(config.get("num_speakers", 5))),
        "--turns", str(int(config.get("turns", 100))),
        "--history-turns", str(int(config.get("history_turns", 8))),
        "--batch-size", str(int(config.get("batch_size", 8))),
        "--duration-sec", str(float(config.get("duration_sec", 300.0))),
        "--rt60", str(float(config.get("rt60", 0.20))),
        "--gpu-device", str(int(config.get("gpu_device", -1))),
        "--out-dir", out_arg,
    ]
    if config.get("enable_ramblers"):
        cmd += ["--enable-ramblers", "--rambler-gain-db", str(float(config.get("rambler_gain_db", -18.0)))]
    seed = (config.get("seed_topic") or "").strip()
    if seed:
        cmd += ["--seed-topic", seed]

    personas = config.get("personas") or []
    overrides = []
    kokoro_dir = Path.home() / ".cache" / "crispasr"
    refs_by_name = {r["name"]: r for r in discover_refs()}
    for pc in personas:
        try:
            idx = int(pc.get("index"))
        except (TypeError, ValueError):
            continue
        eng = (pc.get("engine") or global_engine).lower()
        vname = (pc.get("voice_name") or "").strip()
        entry: dict = {"index": idx}
        if eng != global_engine:
            entry["engine"] = eng
        if vname and vname != "default":
            entry["voice_name"] = vname
        if pc.get("speed_ratio") is not None:
            entry["speed_ratio"] = float(pc["speed_ratio"])
        if eng == "kokoro" and vname and vname != "default":
            pack = kokoro_dir / f"kokoro-voice-{vname}.gguf"
            if pack.exists():
                entry["voice_pack_gguf"] = str(pack)
        elif eng in ("omnivoice", "omni", "fish_audio", "fish", "fish-s2", "cosyvoice"):
            if vname in refs_by_name:
                ref = refs_by_name[vname]
                entry["voice_ref_wav"] = str(REPO_DIR / ref["wav"])
                if ref.get("text"):
                    entry["voice_ref_text"] = ref["text"]
            elif eng == "cosyvoice" and vname and vname != "default":
                entry["preset_voice"] = vname
                entry["voice_ref_wav"] = None
        elif eng == "magpie" and vname and vname != "default":
            entry["voice_name"] = vname
        overrides.append(entry)
    if overrides:
        pj = Path(os.environ.get("TEMP", ".")) / f"cacophony_personas_{uuid.uuid4().hex[:8]}.json"
        pj.write_text(json.dumps(overrides, ensure_ascii=False), encoding="utf-8")
        cmd += ["--personas-json", str(pj)]
    return cmd, out_path


def discover_refs() -> list[dict]:
    """Discover all reference WAVs in voices-store/ and samples/voices/, auto-transcribing missing .txt sidecars."""
    refs = []
    vs_dir = REPO_DIR / "voices-store"
    if vs_dir.exists():
        for wav in sorted(vs_dir.glob("*.wav")):
            text = ensure_voice_ref_text(wav)
            refs.append({
                "name": wav.stem,
                "wav": str(wav.relative_to(REPO_DIR)).replace("\\", "/"),
                "text": text,
                "source": "voices-store",
            })
    voices_dir = REPO_DIR / "samples" / "voices"
    if voices_dir.exists():
        for wav in sorted(voices_dir.glob("*.wav")):
            text = ensure_voice_ref_text(wav)
            refs.append({
                "name": wav.stem,
                "wav": str(wav.relative_to(REPO_DIR)).replace("\\", "/"),
                "text": text,
                "source": "samples/voices",
            })
    return refs


def discover_options() -> dict:
    """Everything the UI needs to render its form."""
    kokoro_dir = Path.home() / ".cache" / "crispasr"
    kokoro_voices = sorted(
        p.stem.replace("kokoro-voice-", "") for p in kokoro_dir.glob("kokoro-voice-*.gguf")
    )
    cosyvoice_refs = discover_refs()

    presets = orch.persona_presets()
    personas = []
    for p in presets:
        personas.append({
            "index": p["index"],
            "name": p["name"],
            "voice_name": p["voice_name"],
            "speed_ratio": p["speed_ratio"],
            "engine": p.get("engine"),
            "voice_pack_gguf": p.get("voice_pack_gguf"),
            "voice_ref_wav": p.get("voice_ref_wav"),
        })

    return {
        "scripts": [
            {"id": sid, "label": s["label"], "seed": s["seed_topic"]}
            for sid, s in orch.STORY_SCRIPTS.items()
        ],
        "models": MODEL_PRESETS,
        "engines": ENGINES,
        "magpie_voices": MAGPIE_VOICES,
        "kokoro_voices": kokoro_voices,
        "cosyvoice_baked": COSYVOICE_BAKED,
        "cosyvoice_refs": cosyvoice_refs,
        "personas": personas,
        "num_speakers": [3, 4, 5],
        "defaults": {
            "turns": 100,
            "batch_size": 8,
            "duration_sec": 300.0,
            "rt60": 0.20,
            "rambler_gain_db": -18.0,
            "gpu_device": -1,
            "enable_ramblers": False,
            "out_dir": "cacophony_output/ui_run",
            "model": "openai/gpt-oss-120b:cheapest",
            "engine_type": "omnivoice",
            "num_speakers": 5,
        },
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "CacophonyUI/0.1"

    # -- helpers ------------------------------------------------------------
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, msg, status=400):
        self._json({"error": msg}, status)

    def _send_file(self, path: Path, content_type: str):
        try:
            data = path.read_bytes()
        except Exception:
            return self._error("file not found", 404)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/options":
            return self._json(discover_options())

        if path == "/api/events":
            return self._sse()

        if path == "/api/file":
            qs = parse_qs(parsed.query)
            rel = (qs.get("path") or [""])[0]
            target = (REPO_DIR / rel).resolve()
            allowed_root = (REPO_DIR / "cacophony_output").resolve()
            if not target.is_relative_to(allowed_root) or not target.exists():
                return self._error("file not found", 404)
            if target.suffix == ".wav":
                return self._send_file(target, "audio/wav")
            if target.suffix == ".json":
                return self._send_file(target, "application/json")
            return self._send_file(target, "text/plain; charset=utf-8")

        # static UI
        rel = path.lstrip("/") or "index.html"
        candidate = (DIST_DIR / rel).resolve()
        if not candidate.is_relative_to(DIST_DIR) or not candidate.exists():
            candidate = DIST_DIR / "index.html"
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript",
            ".css": "text/css",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".ico": "image/x-icon",
        }.get(candidate.suffix, "application/octet-stream")
        return self._send_file(candidate, ctype)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/start":
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0 or length > 2 * 1024 * 1024:
                    return self._error("request entity too large or invalid", 413)
                config = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                return self._error("invalid JSON body")
            rid, err = RUNS.start(config)
            if err:
                return self._error(err, 409)
            return self._json({"run_id": rid, "out_dir": str(RUNS.out_dir)})

        if parsed.path == "/api/stop":
            stopped = RUNS.stop()
            return self._json({"stopped": stopped})

        return self._error("unknown route", 404)

    # -- SSE -----------------------------------------------------------------
    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = RUNS.subscribe()
        # snapshot replay so a refresh mid/post-run doesn't land on an empty screen
        if RUNS.running and RUNS.run_id:
            self.wfile.write(f"data: {json.dumps({'kind': 'started', 'run_id': RUNS.run_id, 'out_dir': str(RUNS.out_dir)}, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()
        elif RUNS.result:
            self.wfile.write(f"data: {json.dumps({'kind': 'done', **RUNS.result}, ensure_ascii=False, default=str)}\n\n".encode("utf-8"))
            self.wfile.flush()
        try:
            while True:
                try:
                    item = q.get(timeout=8)
                except queue.Empty:
                    # heartbeat keeps proxies/browsers from dropping the idle stream
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if isinstance(item, tuple) and item[0] == "__EVENT__":
                    self.wfile.write(item[1].encode("utf-8"))
                else:
                    self.wfile.write(f"data: {json.dumps({'kind': 'line', 'text': item}, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            RUNS.unsubscribe(q)

    def log_message(self, fmt, *args):  # quiet by default
        if os.environ.get("CACOPHONY_UI_VERBOSE"):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description="Cacophony UI server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8753)
    args = ap.parse_args()

    if not DIST_DIR.exists():
        print(f"[!] Built UI not found at {DIST_DIR}; run `npm run build` in cacophony-ui/")
    print(f"[*] Cacophony UI server: http://{args.host}:{args.port}")
    print(f"[*] Serving built UI from {DIST_DIR}")
    print(f"[*] Options: {json.dumps(discover_options(), indent=None)[:200]}...")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()