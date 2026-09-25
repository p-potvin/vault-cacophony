"""Minimal ComfyUI HTTP + WebSocket client.

Talks to a running ComfyUI server (the Comfy Desktop instance on :8188, or the
headless fallback started by run.ps1 on :8189). Deliberately dependency-light:
`httpx` and `websockets` are already present in the repo venv via gradio.

Completion is observed over the WebSocket event stream, not by polling
/history -- ComfyUI pushes `executing` / `execution_error` events, so one
connection per job is the idiomatic pattern and avoids a request loop.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from dataclasses import dataclass, field

import httpx
from PIL import Image
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

DEFAULT_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


class ComfyError(RuntimeError):
    pass


@dataclass
class JobResult:
    prompt_id: str
    images: list[Image.Image] = field(default_factory=list)
    node_errors: dict = field(default_factory=dict)


class ComfyClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    # -- discovery -------------------------------------------------------------

    def system_stats(self) -> dict:
        return self._http.get("/system_stats").json()

    def is_alive(self) -> bool:
        try:
            return "system" in self.system_stats()
        except Exception:
            return False

    def models(self, folder: str) -> list[str]:
        r = self._http.get(f"/models/{folder}")
        r.raise_for_status()
        return r.json()

    def has_node(self, class_type: str) -> bool:
        r = self._http.get(f"/object_info/{class_type}")
        return r.status_code == 200 and bool(r.json())

    # -- inputs ----------------------------------------------------------------

    def upload_image(self, img: Image.Image, name: str,
                     subfolder: str = "text-style-transfer") -> str:
        """Upload a PIL image to the server's input dir; returns the LoadImage name."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        r = self._http.post(
            "/upload/image",
            files={"image": (name, buf, "image/png")},
            data={"overwrite": "true", "type": "input", "subfolder": subfolder},
        )
        r.raise_for_status()
        j = r.json()
        sub = j.get("subfolder") or ""
        return f"{sub}/{j['name']}" if sub else j["name"]

    # -- execution -------------------------------------------------------------

    def run(self, graph: dict, timeout_s: float = 1800.0,
            progress=None) -> JobResult:
        """Queue an API-format graph and block until it finishes.

        `progress(step, total, node_id)` is called on sampler progress events.
        """
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        # ping_interval=None: ComfyUI's event loop is blocked while the sampler runs,
        # so keepalive pings go unanswered and would kill a perfectly healthy job.
        with ws_connect(f"{ws_url}/ws?clientId={self.client_id}",
                        max_size=None, open_timeout=30,
                        ping_interval=None, ping_timeout=None,
                        close_timeout=5) as ws:
            r = self._http.post("/prompt", json={"prompt": graph,
                                                 "client_id": self.client_id})
            if r.status_code != 200:
                raise ComfyError(f"/prompt rejected: {r.status_code} {r.text[:2000]}")
            body = r.json()
            prompt_id = body["prompt_id"]
            result = JobResult(prompt_id=prompt_id, node_errors=body.get("node_errors") or {})

            # Single subscription, event-driven; no /history polling.
            while True:
                try:
                    msg = ws.recv(timeout=timeout_s)
                except ConnectionClosed:
                    # Socket lost mid-job. The job itself is unaffected -- fall
                    # through and read the outcome from /history below.
                    break
                if isinstance(msg, (bytes, bytearray)):
                    continue  # binary preview frames
                ev = json.loads(msg)
                t, data = ev.get("type"), ev.get("data", {})
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                if t == "progress" and progress:
                    progress(data.get("value"), data.get("max"), data.get("node"))
                elif t == "execution_error":
                    raise ComfyError(
                        f"{data.get('node_type')}#{data.get('node_id')}: "
                        f"{data.get('exception_message')}")
                elif t == "execution_interrupted":
                    raise ComfyError("execution interrupted")
                elif t == "executing" and data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break
                elif t == "execution_success":
                    break

        hist = self._http.get(f"/history/{prompt_id}").json().get(prompt_id, {})
        if not hist:
            raise ComfyError(
                f"lost the event stream and {prompt_id} is not in /history yet; "
                f"the job may still be running -- check the ComfyUI queue")
        status = hist.get("status", {})
        if status.get("status_str") == "error":
            raise ComfyError(f"job failed: {json.dumps(status)[:2000]}")
        for node_out in hist.get("outputs", {}).values():
            for im in node_out.get("images", []):
                if im.get("type") != "output":
                    continue
                v = self._http.get("/view", params={"filename": im["filename"],
                                                    "subfolder": im.get("subfolder", ""),
                                                    "type": im["type"]})
                v.raise_for_status()
                result.images.append(Image.open(io.BytesIO(v.content)).convert("RGB"))
        return result

    # -- batch execution -------------------------------------------------------

    def submit(self, graph: dict) -> str:
        """Queue a graph without waiting. Returns the prompt id."""
        r = self._http.post("/prompt", json={"prompt": graph, "client_id": self.client_id})
        if r.status_code != 200:
            raise ComfyError(f"/prompt rejected: {r.status_code} {r.text[:2000]}")
        return r.json()["prompt_id"]

    def collect(self, prompt_ids: list[str], timeout_s: float = 7200.0,
                on_done=None) -> dict[str, JobResult]:
        """Wait for a batch of already-queued prompts and fetch their images.

        Submitting the whole batch up front is what keeps the model resident:
        ComfyUI runs the queue back to back, so the ~60 s GGUF dequantisation is
        paid once instead of once per letter.
        """
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        pending, results = set(prompt_ids), {}
        with ws_connect(f"{ws_url}/ws?clientId={self.client_id}", max_size=None,
                        open_timeout=30, ping_interval=None, ping_timeout=None,
                        close_timeout=5) as ws:
            while pending:
                try:
                    msg = ws.recv(timeout=timeout_s)
                except ConnectionClosed:
                    break
                if isinstance(msg, (bytes, bytearray)):
                    continue
                ev = json.loads(msg)
                t, data = ev.get("type"), ev.get("data", {})
                pid = data.get("prompt_id")
                if pid not in pending:
                    continue
                if t == "execution_error":
                    results[pid] = ComfyError(
                        f"{data.get('node_type')}#{data.get('node_id')}: "
                        f"{data.get('exception_message')}")
                    pending.discard(pid)
                elif (t == "execution_success"
                      or (t == "executing" and data.get("node") is None)):
                    pending.discard(pid)
                    if on_done:
                        on_done(pid, len(prompt_ids) - len(pending), len(prompt_ids))

        for pid in prompt_ids:
            if isinstance(results.get(pid), Exception):
                continue
            results[pid] = self._fetch(pid)
        return results

    def _fetch(self, prompt_id: str) -> JobResult:
        hist = self._http.get(f"/history/{prompt_id}").json().get(prompt_id, {})
        res = JobResult(prompt_id=prompt_id)
        for node_out in hist.get("outputs", {}).values():
            for im in node_out.get("images", []):
                if im.get("type") != "output":
                    continue
                v = self._http.get("/view", params={"filename": im["filename"],
                                                    "subfolder": im.get("subfolder", ""),
                                                    "type": im["type"]})
                v.raise_for_status()
                res.images.append(Image.open(io.BytesIO(v.content)).convert("RGB"))
        return res
