#!/usr/bin/env python3
"""One-time migration (operator, Thu 05 Nov 2026): every `<media>.tags.json` into its media's main sidecar.

For each `.tags.json` the Anything index (or --roots) finds:
  * a tag store with its media beside it: the label tracks (language, speaker, voice, face) go into the media's
    sidecar `tags` through tagstore.save (their original `source` kept); `caption` / `event` are dropped (VSS keeps
    that text in .captions.jsonl / .summary.json); the cues are dropped, except that an .srt is written from them
    when the media has none; then the .tags.json goes to the Recycle Bin, once the sidecar reads back right;
  * no media beside it, or the old list format: to the Recycle Bin (operator: delete them).
VSS outputs: an old-style `.summary.json` (LVS envelope, content a JSON string) is rewritten readable, and every
`.summary.md` goes to the Recycle Bin.

Dry run unless --execute. The report (with paths) goes to %LOCALAPPDATA%\\VaultWares\\raggae\\reports\\.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import json
import os
import subprocess
import sys
from ctypes import wintypes
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tagstore  # noqa: E402

ANYTHING = r"C:\Program Files\VaultWares\Anything\raggae-files.exe"
INDEX = r"C:\ProgramData\VaultWares\anything"
DROPPED_TRACKS = {"caption", "event"}
PREFERRED = [".mkv", ".mp4", ".webm", ".mov", ".m4v", ".avi", ".wmv", ".flv", ".ts", ".mpg", ".mpeg",
             ".wav", ".mp3", ".flac", ".m4a", ".aac", ".opus", ".ogg"]


def find(pattern, roots):
    """Paths of the files named like `pattern` (`*.tags.json`): from the Anything index, else a walk of `roots`."""
    suffix = pattern.lstrip("*")
    if roots:
        out = []
        for r in roots:
            for d, _, files in os.walk(r):
                out += [os.path.join(d, f) for f in files if f.lower().endswith(suffix)]
        return out
    res = subprocess.run([ANYTHING, "query", pattern, "--index", INDEX, "--limit", "100000"], capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    out = []
    for ln in res.stdout.splitlines():
        i = ln.lower().find(suffix)
        if i < 0 or "MiB  " not in ln:
            continue
        out.append(os.path.join(ln.rsplit("MiB  ", 1)[-1].strip(), ln[: i + len(suffix)]))
    return out


def media_beside(tags_file):
    base = tags_file[: -len(".tags.json")]
    folder, stem = os.path.split(base)
    try:
        names = {n.lower(): n for n in os.listdir(folder or ".")}
    except OSError:
        return None
    for ext in PREFERRED:
        n = names.get((stem + ext).lower())
        if n:
            return os.path.join(folder, n)
    return None


class SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]


def recycle(path):
    """To the Recycle Bin (FOF_ALLOWUNDO), silently. True when the file is gone."""
    op = SHFILEOPSTRUCTW(None, 3, os.path.abspath(path) + "\0\0", None, 0x4 | 0x10 | 0x40 | 0x400, False, None, None)
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    return rc == 0 and not op.fAnyOperationsAborted and not os.path.exists(path)


def migrate_one(path, execute, stats, report):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        stats["unreadable"] += 1
        report.append({"file": path, "action": "unreadable", "error": str(e)})
        return
    if not isinstance(data, dict) or data.get("schema") != 1:
        stats["old format: recycled"] += 1
        report.append({"file": path, "action": "old format"})
        if execute and not recycle(path):
            stats["recycle failed"] += 1
        return
    media = media_beside(path)
    if not media:
        stats["no media: recycled"] += 1
        report.append({"file": path, "action": "no media"})
        if execute and not recycle(path):
            stats["recycle failed"] += 1
        return
    tracks = {k: v for k, v in (data.get("tracks") or {}).items() if k not in DROPPED_TRACKS and v.get("spans")}
    dropped = sorted(set(data.get("tracks") or {}) - set(tracks))
    cues = data.get("cues") or []
    srt = tagstore.srt_path(media)
    write_srt = bool(cues) and not os.path.exists(srt)
    entry = {"file": path, "media": media, "tracks": sorted(tracks), "dropped": dropped, "srt_written": write_srt}
    report.append(entry)
    for t in tracks:
        stats[f"track {t}"] += 1
    for t in dropped:
        stats[f"dropped track {t}"] += 1
    stats["migrated"] += 1
    if write_srt:
        stats["srt written from cues"] += 1
    if not execute:
        return
    store = tagstore.load(media)
    duration = (data.get("media") or {}).get("duration")
    if duration:
        tagstore.set_media(store, media, duration)
    for name, track in tracks.items():
        if name in store["tracks"]:          # the new format already has it (a pass ran after the switch): keep that
            stats[f"kept newer {name}"] += 1
            continue
        store["tracks"][name] = track
        store["_dirty"].add(name)
    side = tagstore.save(store)
    check = tagstore.load(media)
    if not all(n in check["tracks"] for n in tracks):
        stats["verify failed (kept .tags.json)"] += 1
        entry["error"] = f"tracks missing after save in {side}"
        return
    if write_srt:
        tagstore.write_srt(srt, cues)
    entry["sidecar"] = side
    if not recycle(path):
        stats["recycle failed"] += 1


def vss_outputs(roots, execute, stats, report):
    for path in find("*.summary.md", roots):
        stats["summary.md recycled"] += 1
        report.append({"file": path, "action": "summary.md"})
        if execute and not recycle(path):
            stats["recycle failed"] += 1
    for path in find("*.summary.json", roots):
        try:
            with open(path, encoding="utf-8") as f:
                env = json.load(f)
            content = json.loads(env["choices"][0]["message"]["content"])
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue                                   # not an old LVS envelope (or already readable)
        doc = {"video": env.get("id"), "written": env.get("created"),
               "summary": str(content.get("video_summary", "")).strip(),
               "events": [{"start": e.get("start_time"), "end": e.get("end_time"), "type": e.get("type"),
                           "description": e.get("description")} for e in content.get("events", [])],
               "model": env.get("model"), "request": env.get("request"), "usage": env.get("usage")}
        stats["summary.json made readable"] += 1
        report.append({"file": path, "action": "summary.json readable"})
        if execute:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="One-time: <media>.tags.json into the media's sidecar `tags`")
    ap.add_argument("--execute", action="store_true", help="write sidecars and recycle the old files")
    ap.add_argument("--roots", nargs="*", help="walk these folders instead of asking the Anything index")
    args = ap.parse_args()
    stats, report = collections.Counter(), []
    files = find("*.tags.json", args.roots)
    stats["found"] = len(files)
    for p in files:
        migrate_one(p, args.execute, stats, report)
    vss_outputs(args.roots, args.execute, stats, report)
    out_dir = os.path.join(os.environ.get("LOCALAPPDATA", "."), "VaultWares", "raggae", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"tags-migration-{datetime.now():%Y%m%d-%H%M%S}{'' if args.execute else '-dry'}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"execute": args.execute, "stats": dict(stats), "files": report}, f, ensure_ascii=False, indent=1)
    print(("EXECUTED" if args.execute else "DRY RUN") + f" -> {out}")
    for k, v in sorted(stats.items()):
        print(f"  {k:36} {v}")
    return 1 if stats["recycle failed"] or stats["verify failed (kept .tags.json)"] else 0


if __name__ == "__main__":
    sys.exit(main())
