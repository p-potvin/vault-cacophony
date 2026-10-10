#!/usr/bin/env python3
"""VSS-shaped video captioning, summarization, search and Q&A on one consumer GPU.

The NVIDIA VSS blueprint's summarization profile is three services and two
NIMs: rt-vlm chunks a video and captions each chunk with Cosmos Reason, the
lvs-server aggregates the captions through an LLM into a structured summary,
and Elasticsearch holds the captions for search. That stack is sized for 48 to
80 GB cards and needs NGC entitlements. This file is the same pipeline shaped
the way the rest of this repo is shaped: one script, GGUF models through a
ggml runtime, and the results written beside the media file.

    video file
        |
    ffprobe / ffmpeg      chunk plan, N frames per chunk as JPEG
        |
    Ollama  (VLM)         dense caption per chunk, or the VSS JSON event list
        |                 when --event is given; the .srt beside the file is
        |                 folded into the prompt with --enable-audio
        |
    <base>.captions.jsonl one line per chunk, the rt-vlm dense caption shape
        |
    Ollama  (LLM)         caption_summarization -> summary_aggregation ->
        |                 {video_summary, events}, the LVS response shape
        |
    <base>.summary.json   CompletionResponse envelope, content is a JSON string
    <base>.summary.md     the same, readable
    <base>.tags.json      `caption` and `event` tracks in the tag store

`search` ranks chunks across many files with BM25 over their captions, which
is what VSS's Elasticsearch leg does before the agent reads the hits. `ask`
retrieves the chunks that match the question and shows the VLM their frames
again, which is VSS's VLM Q&A tool.

The only runtime is Ollama, which is llama.cpp: every model here is a GGUF.
The default VLM is qwen3-vl:2b because it is already pulled; Cosmos-Reason2-2B
is the same architecture and has GGUF builds, see docs/vss-local-pipeline.md.

    python vss_pipeline.py summarize clip.mp4 --scenario "screen recording" \\
        --event "typing" --event "window change" --enable-audio
    python vss_pipeline.py search D:\\Media "person opens a door"
    python vss_pipeline.py ask clip.mp4 "what happens at the end?"
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tagstore

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO, "config", "vss", "lvs-local.yaml")
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts", ".mpg", ".mpeg", ".wmv"}
VSS_DEFAULT_SCENARIO = "activity monitoring"
VSS_DEFAULT_EVENTS = ["notable activity"]

# Everything the config file can set, with the value used when it does not.
DEFAULTS = {
    "ollama": {
        "url": "http://127.0.0.1:11434",
        "vlm_model": "qwen3-vl:2b",
        "llm_model": None,
        "num_ctx": 16384,
        "keep_alive": "10m",
        "temperature": 0.2,
        "top_p": 0.7,
        "seed": 1,
        "num_predict": 1024,
        "repeat_penalty": 1.1,
        "timeout_s": 900,
    },
    "chunking": {
        "chunk_duration": 10,
        "chunk_overlap_duration": 0,
        "num_frames_per_chunk": 8,
        "frame_max_side": 512,
        "jpeg_quality": 4,
    },
    "summarization": {"max_events_per_batch": 50, "structured_output": True},
    "prompts": {
        "caption": "Write a concise and clear dense caption for the provided video.",
        "vlm_role": "You are an advanced intelligent video analysis system specialized in "
                    "analyzing {scenario} and generating captions.",
        "vlm_instruction": "Focus on detecting these events: {events}. Also watch for these "
                           "objects of interest: {objects_of_interest}. Describe all events "
                           "and objects of interest throughout the video.",
        "vlm_audio_instruction": "Additionally, the audio transcript for this chunk is provided "
                                 "below. Use it to describe what is said and incorporate it "
                                 "into your event descriptions.",
        "vlm_constraints": "",
        "vlm_structured_output": "Provide the result in json format with 'seconds' for time "
                                 "depiction for each event. Use keywords 'start_time', "
                                 "'end_time', 'description', \"type\" in the json output. "
                                 "\"type\" should be chosen from [{events}] only.",
        "caption_summarization": "Aggregate the following captions in the format "
                                 "**start_timestamp-end_timestamp**event_description.",
        "summary_aggregation": "Aggregate the following captions in the format "
                               "**start_timestamp-end_timestamp**event_description",
        "final_summary": "Summarize the video for the scenario \"{scenario}\" from the "
                         "time-stamped descriptions below, then list notable events.",
        "qa": "Answer the question about the video using the frames and the descriptions.",
    },
}

EVENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["start_time", "end_time", "type", "description"],
            },
        }
    },
    "required": ["events"],
}

# The merge passes return the same event list without the type constraint.
MERGED_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "description": {"type": "string"},
                },
                "required": ["start_time", "end_time", "description"],
            },
        }
    },
    "required": ["events"],
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "evidence_times": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["answer", "evidence_times"],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "video_summary": {"type": "string"},
        "events": EVENTS_SCHEMA["properties"]["events"],
    },
    "required": ["video_summary", "events"],
}


# ----------------------------------------------------------------------------
# config


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


def load_config(path=None):
    cfg = json.loads(json.dumps(DEFAULTS))
    path = path or DEFAULT_CONFIG
    if os.path.exists(path):
        try:
            import yaml  # PyYAML is in the venv; the script still runs without it.
            with open(path, encoding="utf-8") as f:
                cfg = _merge(cfg, yaml.safe_load(f) or {})
        except ImportError:
            print(f"  [!] PyYAML missing, {os.path.basename(path)} ignored", file=sys.stderr)
    return cfg


# ----------------------------------------------------------------------------
# media


def run(args, what, check=True):
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and proc.returncode != 0:
        raise RuntimeError(f"{what} failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")
    return proc


def probe(path):
    """duration, dimensions, fps and whether there is an audio stream."""
    proc = run(["ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=codec_type,width,height,r_frame_rate",
                "-of", "json", path], "ffprobe")
    info = json.loads(proc.stdout or "{}")
    out = {"path": path, "duration": float(info.get("format", {}).get("duration") or 0.0),
           "width": 0, "height": 0, "fps": 0.0, "has_audio": False, "has_video": False}
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and not out["has_video"]:
            out["has_video"] = True
            out["width"], out["height"] = int(s.get("width") or 0), int(s.get("height") or 0)
            num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
            try:
                out["fps"] = float(num) / float(den or 1)
            except (ValueError, ZeroDivisionError):
                out["fps"] = 0.0
        elif s.get("codec_type") == "audio":
            out["has_audio"] = True
    return out


def plan_chunks(duration, chunk_duration, overlap=0.0):
    """Fixed-length windows over the file, the way rt-vlm slices a video.

    Chunk i covers [i*d - overlap, (i+1)*d], clamped to the file. The overlap
    reaches backwards so a scene cut on a boundary is seen from both sides.
    Two chunk_durations of tail get one chunk rather than a sliver.
    """
    if duration <= 0:
        return []
    d = float(chunk_duration)
    n = max(1, int(math.ceil(duration / d)))
    # A trailing chunk shorter than a quarter of the others is merged into the
    # previous one; the VLM has nothing to say about half a second of video.
    if n > 1 and duration - (n - 1) * d < d * 0.25:
        n -= 1
    chunks = []
    for i in range(n):
        start = max(0.0, i * d - float(overlap))
        end = duration if i == n - 1 else min(duration, (i + 1) * d)
        chunks.append({"chunkIdx": i, "start": round(start, 3), "end": round(end, 3),
                       "is_first": i == 0, "is_last": i == n - 1})
    return chunks


def frame_times(start, end, n):
    """n timestamps spread over the window, at bin centres so none sits on EOF."""
    n = max(1, int(n))
    span = max(0.0, end - start)
    return [round(start + (k + 0.5) * span / n, 3) for k in range(n)]


def extract_frames(video, times, out_dir, max_side=512, quality=4):
    """One JPEG per timestamp. ffmpeg seeks before demuxing, so each call is
    a keyframe seek plus a decode of a few frames rather than a scan."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    vf = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    for k, t in enumerate(times):
        out = os.path.join(out_dir, f"f{k:03d}.jpg")
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", video,
             "-frames:v", "1", "-vf", vf, "-q:v", str(quality), out], "ffmpeg frame")
        if os.path.exists(out):
            paths.append(out)
    return paths


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ----------------------------------------------------------------------------
# subtitles, for --enable-audio

_SRT_TIME = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})")


def _srt_seconds(s):
    m = _SRT_TIME.search(s)
    if not m:
        return None
    h, mi, se, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt(path):
    cues = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        block = []
        for line in list(f) + ["\n"]:
            if line.strip():
                block.append(line.rstrip("\n"))
                continue
            if len(block) >= 2:
                times = block[1] if "-->" in block[1] else (block[0] if "-->" in block[0] else None)
                if times:
                    a, _, b = times.partition("-->")
                    start, end = _srt_seconds(a), _srt_seconds(b)
                    text = " ".join(l for l in block[(block.index(times) + 1):]).strip()
                    if start is not None and end is not None and text:
                        cues.append({"start": start, "end": end, "text": re.sub(r"<[^>]+>", "", text)})
            block = []
    return cues


def cues_in_window(cues, start, end):
    return [c for c in cues if c["end"] > start and c["start"] < end]


def sidecar(video, suffix):
    base, _ = os.path.splitext(video)
    return base + suffix


# ----------------------------------------------------------------------------
# Ollama

_CAPS = {}


def ollama_capabilities(url, model):
    if model not in _CAPS:
        try:
            data = _post(url, "/api/show", {"model": model}, timeout=60)
            _CAPS[model] = set(data.get("capabilities") or [])
        except Exception:
            _CAPS[model] = set()
    return _CAPS[model]


def _post(url, route, payload, timeout):
    req = urllib.request.Request(url.rstrip("/") + route, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_chat(cfg, model, prompt, images=None, system=None, schema=None):
    """One non-streaming chat turn. Returns (text, {eval_count, eval_duration_s, ...})."""
    o = cfg["ollama"]
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    user = {"role": "user", "content": prompt}
    if images:
        user["images"] = [b64(p) for p in images]
    msgs.append(user)
    payload = {
        "model": model, "messages": msgs, "stream": False, "keep_alive": o["keep_alive"],
        "options": {"num_ctx": int(o["num_ctx"]), "temperature": float(o["temperature"]),
                    "top_p": float(o["top_p"]), "seed": int(o["seed"]),
                    "num_predict": int(o["num_predict"]),
                    "repeat_penalty": float(o["repeat_penalty"])},
    }
    # Qwen3-VL reasons before answering. Under a JSON schema think=false is
    # the right call: 300 tokens instead of 1800 and the JSON arrives (in the
    # thinking field, see below). For free text it backfires: the model then
    # writes its deliberation into content as a literal <think> block and
    # runs into num_predict before the answer. So free text keeps thinking on
    # and the thoughts stay in message.thinking, which is dropped.
    if schema and "thinking" in ollama_capabilities(o["url"], model):
        payload["think"] = False
    if schema:
        payload["format"] = schema
    try:
        data = _post(o["url"], "/api/chat", payload, timeout=float(o["timeout_s"]))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"ollama {e.code} for {model}: {body}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"ollama unreachable at {o['url']}: {e.reason}") from None
    msg = data.get("message") or {}
    text = msg.get("content") or ""
    # Ollama quirk, measured on qwen3-vl:2b: with think=false and a JSON
    # schema the answer lands in message.thinking and content is empty.
    if not text.strip() and msg.get("thinking"):
        text = msg["thinking"]
    stats = {"done_reason": data.get("done_reason"),"eval_count": data.get("eval_count", 0),
             "prompt_eval_count": data.get("prompt_eval_count", 0),
             "eval_duration_s": round((data.get("eval_duration") or 0) / 1e9, 3),
             "total_duration_s": round((data.get("total_duration") or 0) / 1e9, 3)}
    return text, stats


# ----------------------------------------------------------------------------
# prompts and parsing

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def substitute(text, scenario, events, objects):
    return (text.replace("{scenario}", scenario or "")
                .replace("{events}", ", ".join(events or []))
                .replace("{objects_of_interest}", ", ".join(objects or [])))


def build_caption_prompt(cfg, scenario, events, objects, structured, enable_audio):
    """The LVS _create_vlm_prompt: role, instruction, constraints, output format."""
    p = cfg["prompts"]
    if not structured:
        return p["caption"]
    parts = [p["vlm_role"], p["vlm_instruction"]]
    if enable_audio:
        parts.append(p["vlm_audio_instruction"])
    if p.get("vlm_constraints"):
        parts.append(p["vlm_constraints"])
    parts.append(p["vlm_structured_output"])
    return "\n\n".join(substitute(x.strip(), scenario, events, objects) for x in parts)


def chunk_context(chunk, times, cues, has_audio_text):
    lines = [f"This chunk covers {chunk['start']:.1f} s to {chunk['end']:.1f} s of the video. "
             f"The {len(times)} frames shown are sampled at, in order: "
             + ", ".join(f"{t:.1f} s" for t in times) + "."]
    if has_audio_text and cues:
        lines.append("Audio transcript for this chunk:")
        lines += [f"  [{c['start']:.1f}-{c['end']:.1f}] {c['text']}" for c in cues]
    elif has_audio_text:
        lines.append("Audio transcript for this chunk: (no speech)")
    return "\n".join(lines)


def strip_think(text):
    text = _THINK.sub("", text or "")
    # An unclosed block is a reply that ran out of tokens while thinking.
    i = text.find("<think>")
    if i >= 0:
        text = text[:i]
    return text.strip()


def extract_json(text):
    """The JSON a VLM meant to return, out of whatever it wrapped it in."""
    text = strip_think(text)
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        # Whichever bracket opens first is the outer value: a list of objects
        # must not be read as its first object.
        pairs = sorted((p for p in (("{", "}"), ("[", "]")) if cand.find(p[0]) >= 0),
                       key=lambda p: cand.find(p[0]))
        for opener, closer in pairs:
            i, j = cand.find(opener), cand.rfind(closer)
            if j <= i:
                continue
            blob = cand[i:j + 1]
            blob = re.sub(r"#[^\n]*", "", blob)          # the template's #(MANDATORY) comments
            blob = re.sub(r",\s*([\]}])", r"\1", blob)   # trailing commas
            try:
                return json.loads(blob)
            except json.JSONDecodeError:
                pass
            # A response cut off by num_predict ends mid-object. Keep every
            # object that did close and shut the containers around them.
            fixed = _close_truncated(blob)
            if fixed is not None:
                return fixed
    return None


def _close_truncated(blob):
    last = blob.rfind("}")
    if last < 0:
        return None
    head = blob[:last + 1]
    for tail in ("", "]", "}", "]}", "}]", "}]}"):
        try:
            return json.loads(head + tail)
        except json.JSONDecodeError:
            continue
    return None


def _num(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = _srt_seconds(v) if ":" in v else None
        if s is not None:
            return s
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if m:
            return float(m.group())
    return None


def normalize_events(parsed, chunk, allowed=None):
    """Event dicts with absolute times inside the chunk, whatever the VLM did.

    Small models answer with chunk-relative seconds as often as absolute ones;
    a time that fits the window only after adding the chunk start is treated
    as relative. Everything is then clamped into the window.
    """
    if parsed is None:
        return []
    events = parsed.get("events", parsed) if isinstance(parsed, dict) else parsed
    if isinstance(events, dict):
        events = [events]
    if not isinstance(events, list):
        return []
    start, end = chunk["start"], chunk["end"]
    span = end - start
    out = []
    for e in events:
        if not isinstance(e, dict):
            continue
        desc = str(e.get("description") or e.get("caption") or "").strip()
        if not desc:
            continue
        s, t = _num(e.get("start_time")), _num(e.get("end_time"))
        if s is None:
            s = start
        if t is None:
            t = end
        # Relative offsets: both inside [0, span] but not inside the window.
        if not (start - 0.5 <= s <= end + 0.5) and 0 <= s <= span + 0.5:
            s, t = s + start, t + start
        s = min(max(s, start), end)
        t = min(max(t, s), end)
        typ = str(e.get("type") or (allowed[0] if allowed else "notable activity")).strip()
        if allowed and typ not in allowed:
            # The model paraphrased the label; keep the closest one by overlap.
            low = typ.lower()
            match = [a for a in allowed if a.lower() in low or low in a.lower()]
            typ = match[0] if match else allowed[0]
        out.append({"start_time": round(s, 2), "end_time": round(t, 2), "type": typ,
                    "description": desc})
    out.sort(key=lambda e: (e["start_time"], e["end_time"]))
    return out


def events_to_text(events):
    return "\n".join(f"**{e['start_time']:.1f}-{e['end_time']:.1f}** {e['description']}"
                     for e in events)


# ----------------------------------------------------------------------------
# captions

def caption_file(video, cfg, scenario, events, objects, enable_audio, vlm_model,
                 chunk_duration, overlap, frames, resume=False, out_path=None, log=print):
    """Dense captions for every chunk -> <base>.captions.jsonl. Returns the records."""
    info = probe(video)
    if not info["has_video"]:
        raise RuntimeError("no video stream")
    out_path = out_path or sidecar(video, ".captions.jsonl")
    chunks = plan_chunks(info["duration"], chunk_duration, overlap)
    structured = bool(events) and bool(cfg["summarization"]["structured_output"])
    prompt = build_caption_prompt(cfg, scenario, events, objects, structured, enable_audio)

    cues = []
    srt = sidecar(video, ".srt")
    if enable_audio:
        if os.path.exists(srt):
            cues = parse_srt(srt)
            log(f"    audio: {len(cues)} cues from {os.path.basename(srt)}")
        else:
            log(f"    [!] --enable-audio but no {os.path.basename(srt)} beside the file; "
                "run Start-SubtitlesAudioCpp.ps1 first")
            enable_audio = False

    done = {}
    if resume and os.path.exists(out_path):
        for rec in read_captions(out_path):
            done[rec["chunk"]["chunkIdx"]] = rec
        if done:
            log(f"    resume: {len(done)} of {len(chunks)} chunks already captioned")

    records = []
    tmp = tempfile.mkdtemp(prefix="vss-frames-")
    t_all = time.time()
    try:
        with open(out_path, "a" if done else "w", encoding="utf-8") as out:
            for ch in chunks:
                if ch["chunkIdx"] in done:
                    records.append(done[ch["chunkIdx"]])
                    continue
                times = frame_times(ch["start"], ch["end"], frames)
                fdir = os.path.join(tmp, f"c{ch['chunkIdx']:04d}")
                paths = extract_frames(video, times, fdir, cfg["chunking"]["frame_max_side"],
                                       cfg["chunking"]["jpeg_quality"])
                if not paths:
                    log(f"    [!] chunk {ch['chunkIdx']}: no frames decoded, skipped")
                    continue
                window = cues_in_window(cues, ch["start"], ch["end"])
                full = prompt + "\n\n" + chunk_context(ch, times, window, enable_audio)
                t0 = time.time()
                text, stats = ollama_chat(cfg, vlm_model, full, images=paths,
                                          schema=EVENTS_SCHEMA if structured else None)
                dt = time.time() - t0
                parsed = extract_json(text) if structured else None
                evs = normalize_events(parsed, ch, allowed=events) if structured else []
                caption = events_to_text(evs) if evs else strip_think(text)
                if structured and not evs:
                    log(f"    [!] chunk {ch['chunkIdx']}: no parsable events, kept raw text")
                rec = {
                    "vlm_response": text,
                    "caption": caption,
                    "events": evs,
                    "frame_times": times,
                    "audio": " ".join(c["text"] for c in window) if window else "",
                    "chunk": {"chunkIdx": ch["chunkIdx"], "file": os.path.basename(video),
                              "start": ch["start"], "end": ch["end"],
                              "is_first": ch["is_first"], "is_last": ch["is_last"]},
                    "model": vlm_model,
                    "stats": stats,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                records.append(rec)
                shutil.rmtree(fdir, ignore_errors=True)
                first = caption.replace("\n", " ")
                log(f"    chunk {ch['chunkIdx'] + 1}/{len(chunks)} "
                    f"[{ch['start']:.0f}-{ch['end']:.0f}s] {dt:.1f}s  {first[:90]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    records.sort(key=lambda r: r["chunk"]["chunkIdx"])
    log(f"    captions: {len(records)} chunks in {time.time() - t_all:.1f}s -> "
        f"{os.path.basename(out_path)}")
    return records, info


def read_captions(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    recs.sort(key=lambda r: r.get("chunk", {}).get("chunkIdx", 0))
    return recs


# ----------------------------------------------------------------------------
# aggregation, the LVS half


def caption_lines(records):
    """Every caption as a **start-end** line, the input shape of the CA-RAG prompts."""
    lines = []
    for r in records:
        if r.get("events"):
            lines += [f"**{e['start_time']:.1f}-{e['end_time']:.1f}** {e['description']}"
                      for e in r["events"]]
        else:
            c = r["chunk"]
            text = (r.get("caption") or strip_think(r.get("vlm_response", ""))).replace("\n", " ")
            if text:
                lines.append(f"**{c['start']:.1f}-{c['end']:.1f}** {text}")
    return lines


def batched(items, n):
    n = max(1, int(n))
    return [items[i:i + n] for i in range(0, len(items), n)]


def aggregate(records, cfg, scenario, events, llm_model, log=print):
    """caption_summarization per batch -> summary_aggregation -> final JSON."""
    p = cfg["prompts"]
    lines = caption_lines(records)
    if not lines:
        return {"video_summary": "", "events": []}, {"llm_seconds": 0.0, "llm_calls": 0}
    t0 = time.time()
    calls = 0
    batches = batched(lines, cfg["summarization"]["max_events_per_batch"])
    duration = max((r["chunk"]["end"] for r in records), default=0.0)
    whole = {"start": 0.0, "end": duration}

    def merge(prompt, text):
        # Every call is schema-constrained. Free text from a small model that
        # reasons is its deliberation, not the merged list.
        out, _ = ollama_chat(cfg, llm_model, prompt.strip() + "\n\n" + text, schema=MERGED_SCHEMA)
        evs = normalize_events(extract_json(out), whole)
        return events_to_text(evs) if evs else text

    merged = []
    for i, b in enumerate(batches):
        merged.append(merge(p["caption_summarization"], "\n".join(b)))
        calls += 1
        log(f"    aggregate: batch {i + 1}/{len(batches)} ({len(b)} captions)")
    combined = "\n".join(merged)
    if len(batches) > 1:
        combined = merge(p["summary_aggregation"], combined)
        calls += 1
    final_prompt = (substitute(p["final_summary"].strip(), scenario, events, None)
                    + "\n\n" + combined)
    text, _ = ollama_chat(cfg, llm_model, final_prompt, schema=SUMMARY_SCHEMA)
    calls += 1
    parsed = extract_json(text) or {}
    summary = {
        "video_summary": str(parsed.get("video_summary") or strip_think(text)).strip(),
        "events": normalize_events(parsed.get("events", []), whole, allowed=events or None),
    }
    return summary, {"llm_seconds": round(time.time() - t0, 2), "llm_calls": calls}


def write_summary(video, summary, request, usage, model):
    """The LVS CompletionResponse: content is a JSON string, as VSS's CLI expects."""
    out = sidecar(video, ".summary.json")
    envelope = {
        "id": os.path.basename(video),
        "model": model,
        "created": time.strftime("%a, %d %b %Y %H:%M"),
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant",
                                 "content": json.dumps(summary, ensure_ascii=False)}}],
        "usage": usage,
        "request": request,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False, indent=1)
    md = sidecar(video, ".summary.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# {os.path.basename(video)}\n\n")
        f.write(f"Scenario: {request.get('scenario')}  \nEvents: {', '.join(request.get('events') or [])}  \n")
        f.write(f"Model: {model}  \nWritten: {envelope['created']}\n\n")
        f.write(summary["video_summary"].strip() + "\n\n")
        if summary["events"]:
            f.write("## Events\n\n")
            for e in summary["events"]:
                f.write(f"- **{e['start_time']:.1f}-{e['end_time']:.1f} s** [{e['type']}] {e['description']}\n")
    return out, md


def write_tags(video, records, summary, vlm_model, llm_model, duration):
    """`caption` and `event` tracks beside whatever the ASR passes wrote."""
    path = sidecar(video, ".tags.json")
    store = tagstore.load(path)
    tagstore.set_media(store, video, duration)
    caps = []
    for r in records:
        c = r["chunk"]
        text = (r.get("caption") or "").strip()
        if text:
            caps.append({"start": c["start"], "end": c["end"], "value": text})
    tagstore.put_track(store, "caption", caps, "vlm", model=vlm_model)
    evs = [{"start": e["start_time"], "end": e["end_time"], "value": e["type"],
            "description": e["description"]} for e in (summary or {}).get("events", [])]
    tagstore.put_track(store, "event", evs, "summarize", model=llm_model)
    tagstore.save(store, path)
    return path


# ----------------------------------------------------------------------------
# search, the Elasticsearch leg

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)
_STOP = set("a an the of to in on at is are was were be been and or for with this that it its "
            "as by from into over under there here what which who whom when where why how do does "
            "did doing has have had".split())


def tokenize(text):
    return [t for t in (w.lower() for w in _TOKEN.findall(text or "")) if t not in _STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        self.df = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1

    def score(self, query):
        q = tokenize(query)
        out = []
        for d in self.docs:
            if not d:
                out.append(0.0)
                continue
            tf = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                f = tf[t]
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


def collect_captions(targets):
    """Every .captions.jsonl under the targets, as (video, record) pairs."""
    files = []
    for t in targets:
        if os.path.isdir(t):
            files += glob.glob(os.path.join(glob.escape(t), "**", "*.captions.jsonl"), recursive=True)
        elif t.endswith(".captions.jsonl"):
            files.append(t)
        else:
            c = sidecar(t, ".captions.jsonl")
            if os.path.exists(c):
                files.append(c)
    pairs = []
    for f in sorted(set(files)):
        for r in read_captions(f):
            pairs.append((f[:-len(".captions.jsonl")], r))
    return pairs


def doc_text(rec):
    bits = [rec.get("caption") or strip_think(rec.get("vlm_response", ""))]
    bits += [e["description"] + " " + e["type"] for e in rec.get("events", [])]
    if rec.get("audio"):
        bits.append(rec["audio"])
    return " ".join(bits)


def search(targets, query, top_k=10):
    pairs = collect_captions(targets)
    if not pairs:
        return []
    scores = BM25([doc_text(r) for _, r in pairs]).score(query)
    ranked = sorted(zip(scores, range(len(pairs))), key=lambda x: -x[0])
    hits = []
    for s, i in ranked[:top_k]:
        if s <= 0:
            break
        video, r = pairs[i]
        c = r["chunk"]
        hits.append({"score": round(s, 3), "media": video, "start": c["start"], "end": c["end"],
                     "chunkIdx": c["chunkIdx"], "caption": (r.get("caption") or "").replace("\n", " ")})
    return hits


# ----------------------------------------------------------------------------
# Q&A, the VLM chat tool


def ask(video, question, cfg, vlm_model, top_k=2, frames=6, text_only=False, log=print):
    cap = sidecar(video, ".captions.jsonl")
    if not os.path.exists(cap):
        raise RuntimeError(f"no captions for {os.path.basename(video)}; run `caption` or `summarize` first")
    records = read_captions(cap)
    scores = BM25([doc_text(r) for r in records]).score(question)
    ranked = sorted(range(len(records)), key=lambda i: -scores[i])
    picked = [records[i] for i in ranked[:top_k] if scores[i] > 0] or records[:1]
    picked.sort(key=lambda r: r["chunk"]["chunkIdx"])
    context = "\n".join(f"[{r['chunk']['start']:.1f}-{r['chunk']['end']:.1f} s] "
                        + (r.get("caption") or "").replace("\n", " ") for r in records)
    prompt = (cfg["prompts"]["qa"].strip() + "\n\nTime-stamped descriptions of the whole video:\n"
              + context + "\n\nQuestion: " + question)
    images = []
    tmp = tempfile.mkdtemp(prefix="vss-ask-")
    try:
        if not text_only:
            per = max(1, frames // len(picked))
            for r in picked:
                c = r["chunk"]
                times = frame_times(c["start"], c["end"], per)
                images += extract_frames(video, times, os.path.join(tmp, f"c{c['chunkIdx']}"),
                                         cfg["chunking"]["frame_max_side"], cfg["chunking"]["jpeg_quality"])
            shown = ", ".join(f"{r['chunk']['start']:.0f}-{r['chunk']['end']:.0f} s" for r in picked)
            prompt = f"The frames shown are from {shown}.\n\n" + prompt
            log(f"    ask: {len(images)} frames from chunks {[r['chunk']['chunkIdx'] for r in picked]}")
        text, stats = ollama_chat(cfg, vlm_model, prompt, images=images or None,
                                  schema=ANSWER_SCHEMA)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    parsed = extract_json(text) or {}
    answer = str(parsed.get("answer") or strip_think(text)).strip()
    times = [t for t in (_num(x) for x in parsed.get("evidence_times") or []) if t is not None]
    if times:
        answer += "\n  at " + ", ".join(f"{t:.1f} s" for t in times[:8])
    return answer, picked, stats


# ----------------------------------------------------------------------------
# CLI


def _targets(path, recurse):
    if os.path.isdir(path):
        pattern = "**/*" if recurse else "*"
        files = [f for f in glob.glob(os.path.join(glob.escape(path), pattern), recursive=recurse)
                 if os.path.splitext(f)[1].lower() in VIDEO_EXT]
        return sorted(files)
    return [path]


def _common(ap):
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--ollama-url")
    ap.add_argument("--vlm-model")
    ap.add_argument("--llm-model")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--num-ctx", type=int)


def _caption_args(ap):
    ap.add_argument("target", help="video file or directory")
    ap.add_argument("--recurse", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip files that already have the output beside them")
    ap.add_argument("--resume", action="store_true", help="keep chunks already in .captions.jsonl")
    ap.add_argument("--scenario", default=None, help=f'default "{VSS_DEFAULT_SCENARIO}"')
    ap.add_argument("--event", action="append", default=None,
                    help="event type to detect; repeatable. None = free-form dense captions")
    ap.add_argument("--object-of-interest", action="append", default=[])
    ap.add_argument("--chunk-duration", type=float)
    ap.add_argument("--chunk-overlap-duration", type=float)
    ap.add_argument("--num-frames-per-chunk", type=int)
    ap.add_argument("--enable-audio", action="store_true",
                    help="fold the .srt beside the file into each chunk prompt")


def _apply_overrides(cfg, a):
    o = cfg["ollama"]
    for k in ("ollama_url", "vlm_model", "llm_model", "seed", "temperature", "num_ctx"):
        v = getattr(a, k, None)
        if v is not None:
            o[k.replace("ollama_", "")] = v
    c = cfg["chunking"]
    for k in ("chunk_duration", "chunk_overlap_duration", "num_frames_per_chunk"):
        v = getattr(a, k, None)
        if v is not None:
            c[k] = v
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe", help="duration, streams, chunk plan")
    s.add_argument("target")
    s.add_argument("--chunk-duration", type=float)
    s.add_argument("--chunk-overlap-duration", type=float)
    s.add_argument("--num-frames-per-chunk", type=int)
    s.add_argument("--config", default=DEFAULT_CONFIG)

    s = sub.add_parser("caption", help="dense captions per chunk -> .captions.jsonl")
    _caption_args(s)
    _common(s)

    s = sub.add_parser("summarize", help="caption if needed, then aggregate -> .summary.json")
    _caption_args(s)
    s.add_argument("--recaption", action="store_true", help="ignore an existing .captions.jsonl")
    _common(s)

    s = sub.add_parser("search", help="rank chunks across files by BM25 over their captions")
    s.add_argument("targets", nargs="+", help="videos, .captions.jsonl files or directories")
    s.add_argument("--query", "-q", required=True)
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("ask", help="answer a question with the frames of the matching chunks")
    s.add_argument("target")
    s.add_argument("--question", "-q", required=True)
    s.add_argument("--top-k", type=int, default=2)
    s.add_argument("--frames", type=int, default=6)
    s.add_argument("--text-only", action="store_true")
    _common(s)

    a = ap.parse_args(argv)
    cfg = load_config(getattr(a, "config", None))
    cfg = _apply_overrides(cfg, a)
    ch = cfg["chunking"]

    if a.cmd == "probe":
        info = probe(a.target)
        chunks = plan_chunks(info["duration"], ch["chunk_duration"], ch["chunk_overlap_duration"])
        print(json.dumps({"media": info, "chunks": len(chunks),
                          "frames": len(chunks) * int(ch["num_frames_per_chunk"]),
                          "plan": chunks[:3] + (["..."] if len(chunks) > 3 else [])}, indent=1))
        return 0

    if a.cmd == "search":
        hits = search(a.targets, a.query, a.top_k)
        if a.json:
            print(json.dumps({"count": len(hits), "hits": hits}, ensure_ascii=False, indent=1))
        elif not hits:
            print("no hits")
        else:
            for h in hits:
                print(f"{h['score']:6.2f}  {os.path.basename(h['media'])}  "
                      f"{h['start']:.0f}-{h['end']:.0f}s  {h['caption'][:110]}")
        return 0

    vlm = cfg["ollama"]["vlm_model"]
    llm = cfg["ollama"]["llm_model"] or vlm

    if a.cmd == "ask":
        answer, picked, stats = ask(a.target, a.question, cfg, vlm, a.top_k, a.frames, a.text_only)
        print(answer)
        print(f"\n  [{vlm}, {stats['eval_count']} tokens in {stats['total_duration_s']}s, "
              f"chunks {[r['chunk']['chunkIdx'] for r in picked]}]")
        return 0

    scenario = a.scenario or VSS_DEFAULT_SCENARIO
    events = a.event if a.event is not None else (VSS_DEFAULT_EVENTS if a.cmd == "summarize" else None)
    files = _targets(a.target, a.recurse)
    if not files:
        print(f"no video files under {a.target}")
        return 0
    failed = 0
    for video in files:
        want = sidecar(video, ".summary.json" if a.cmd == "summarize" else ".captions.jsonl")
        if a.skip_existing and os.path.exists(want):
            print(f"  = {os.path.basename(video)} (exists)")
            continue
        print(f"  > {os.path.basename(video)}")
        t0 = time.time()
        try:
            cap = sidecar(video, ".captions.jsonl")
            records, info = None, None
            reuse = a.cmd == "summarize" and not getattr(a, "recaption", False) and \
                not a.resume and os.path.exists(cap)
            if reuse:
                records, info = read_captions(cap), probe(video)
                print(f"    captions: {len(records)} chunks from {os.path.basename(cap)}")
            if not records:
                records, info = caption_file(
                    video, cfg, scenario, events, a.object_of_interest, a.enable_audio, vlm,
                    ch["chunk_duration"], ch["chunk_overlap_duration"], ch["num_frames_per_chunk"],
                    resume=a.resume)
            summary = None
            if a.cmd == "summarize":
                summary, lstats = aggregate(records, cfg, scenario, events, llm)
                usage = {"total_chunks_processed": len(records),
                         "vlm_seconds": round(sum(r.get("stats", {}).get("total_duration_s", 0) for r in records), 2),
                         **lstats}
                request = {"scenario": scenario, "events": events,
                           "objects_of_interest": a.object_of_interest,
                           "chunk_duration": ch["chunk_duration"],
                           "chunk_overlap_duration": ch["chunk_overlap_duration"],
                           "num_frames_per_chunk": ch["num_frames_per_chunk"],
                           "enable_audio": a.enable_audio, "vlm_model": vlm, "llm_model": llm}
                out, md = write_summary(video, summary, request, usage, llm)
                print(f"    summary: {len(summary['events'])} events -> {os.path.basename(md)}")
            tags = write_tags(video, records, summary, vlm, llm, info["duration"])
            store = tagstore.load(tags)
            print("    tags: " + ", ".join(f"{k}={len(v['spans'])} spans"
                                          for k, v in sorted(store.get("tracks", {}).items())))
            print(f"    done in {time.time() - t0:.1f}s")
        except Exception as e:  # one bad file must not stop a directory run
            failed += 1
            print(f"    [!] {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
