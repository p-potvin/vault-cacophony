#!/usr/bin/env python3
"""A conversation held in one resident PersonaPlex session.

Start-Conversation.ps1 spawns a process per turn, so every turn re-primes from
an empty cache: the agent hears one utterance and knows nothing of the
conversation. This drives a long-lived audiocpp_server instead, so the KV cache
survives from turn to turn and the model actually remembers what was said.

    audiocpp_server --config pp-server.json \\
        --model-spec-override model_specs/personaplex.json --no-ui
    python duplex_conversation.py --seed opener.wav --turns 6

Measured against the per-process driver: the first turn costs ~21 s and every
turn after it ~15 s, because the 3.5 s prompt preamble is replayed only once.

One thing this cannot do, and it is a property of the model rather than of the
script. The voice prompt and the system prompt are consumed inside
`start_conversation`, which continuing deliberately skips -- so a continued
session keeps the voice and persona it opened with. Shared audio context and
alternating personas are alternatives here, not a pair. With one card there is
no second instance to give the other side its own session, so this is one agent
speaking both parts: PersonaPlex talking to himself, which is what the repo was
named for.

`--restart-each-turn` takes the other branch, for comparison: a fresh session
every turn, so voice and persona can alternate, at the cost of the preamble and
of all memory.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tagstore
import trim_speech

RATE = 24000


def post(url, payload, timeout=1200):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"  server refused the turn ({exc.code}): {exc.read().decode(errors='replace')[:300]}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"  no server at {url}: {exc.reason}")


def write_wav_b64(path, b64):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def pad(src, dst, seconds):
    """Room for the agent to answer in; output length equals input length."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-i", src, "-af", f"apad=pad_dur={seconds}",
                    "-ac", "1", "-ar", str(RATE), dst], check=True)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Hold a conversation in one resident session")
    ap.add_argument("--seed", required=True, help="opening utterance wav")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--url", default="http://127.0.0.1:8131/v1/tasks/run")
    ap.add_argument("--model", default="personaplex")
    ap.add_argument("--voice", default="NATF2")
    ap.add_argument("--prompt",
                    default="You are having a natural conversation. Reply in at most two short "
                            "sentences, then stop talking.")
    ap.add_argument("--reply-window", type=float, default=25.0)
    ap.add_argument("--restart-each-turn", action="store_true",
                    help="fresh session per turn: persona and voice can change, memory cannot")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(os.environ.get("TEMP", "."), "cacophony-duplex")
    os.makedirs(out_dir, exist_ok=True)

    current = os.path.join(out_dir, "turn000.seed.wav")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-i", args.seed, "-ac", "1", "-ar", str(RATE), current], check=True)

    mode = "restarting each turn" if args.restart_each_turn else "one resident session"
    print(f"\n  {args.model} ({args.voice}), {mode}, {args.turns} turns")
    print(f"  seed: {os.path.basename(args.seed)}\n")

    clock = duration(current)
    record = [{"agent": "SEED", "voice": "-", "wav": current, "text": "",
               "start": 0.0, "end": round(clock, 3)}]

    for turn in range(args.turns):
        stem = os.path.join(out_dir, f"turn{turn + 1:03d}")
        padded, raw, wav = f"{stem}.in.wav", f"{stem}.raw.wav", f"{stem}.wav"
        heard = duration(current)
        pad(current, padded, args.reply_window)

        request = {"audio": padded, "text": args.prompt, "voice_id": args.voice}
        # The first turn must build the session; every one after it continues,
        # which is the whole point and also what skips the 3.5 s preamble.
        if turn > 0 and not args.restart_each_turn:
            request["options"] = {"continue_conversation": "true"}
        print(f"  {turn + 1:2d}. listening to {heard:.1f}s...", flush=True)
        reply = post(args.url, {"model": args.model, "request": request})
        if "audio" not in reply:
            raise SystemExit(f"  no audio in reply: {list(reply)}")
        write_wav_b64(raw, reply["audio"])

        audio, sr = trim_speech.read(raw)
        audio = audio[int(heard * sr):]
        bounds = trim_speech.speech_bounds(audio, sr)
        if bounds is None:
            print("      the agent said nothing; stopping")
            break
        start, end = bounds
        a = max(0.0, start - trim_speech.MARGIN_S)
        b = min(len(audio) / sr, end + trim_speech.MARGIN_S)
        trim_speech.write(wav, audio[int(a * sr):int(b * sr)], sr)

        tail = len(audio) / sr - end
        wall = reply.get("timing", {}).get("wall_ms", 0) / 1000.0
        flag = "  [!] cut off -- raise --reply-window" if tail <= 0.04 else ""
        print(f"      {b - a:.2f}s speech (waited {start:.2f}s, left {tail:.2f}s) "
              f"in {wall:.1f}s{flag}")

        record.append({"agent": f"TURN_{turn + 1:02d}", "voice": args.voice, "wav": wav,
                       "text": "", "start": round(clock, 3), "end": round(clock + (b - a), 3)})
        clock += b - a
        current = wav

    if len(record) <= 1:
        raise SystemExit("  no turns generated")

    listing = os.path.join(out_dir, "concat.txt")
    with open(listing, "w", encoding="utf-8") as f:
        for item in record:
            f.write("file '%s'\n" % os.path.abspath(item["wav"]).replace("\\", "/"))
    conversation = os.path.join(out_dir, "conversation.wav")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-f", "concat", "-safe", "0", "-i", listing,
                    "-ac", "1", "-ar", str(RATE), conversation], check=True)

    store = tagstore.load(os.path.join(out_dir, "conversation.tags.json"))
    tagstore.set_media(store, conversation, clock)
    tagstore.put_cues(store, [(t["start"], t["end"], t["text"]) for t in record])
    tagstore.put_track(store, "speaker",
                       [{"start": t["start"], "end": t["end"], "value": t["agent"]}
                        for t in record],
                       "conversation", model="personaplex-7b-v1")
    tagstore.save(store, os.path.join(out_dir, "conversation.tags.json"))

    print(f"\n  {clock:.1f}s over {len(record) - 1} turns")
    print(f"    {conversation}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
