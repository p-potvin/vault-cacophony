#!/usr/bin/env python3
"""Two agents talking, assembled from parts instead of one speech-to-speech model.

Every end-to-end S2S model tried here degenerates once its counterpart is not a
human: PersonaPlex collapses into repeating itself when fed its own voice, and
LFM2-Audio loops on the first turn against a recorded human. The part they all
fail at is holding a conversation, which is the one part a text LLM is good at.
So the loop is split, and each piece does only what it is good at:

    audio in -> SenseVoice  ASR, with <|emotion|> and <|event|> tags
             -> Ollama      the actual conversation, per-agent persona + history
             -> Qwen3-TTS   speech, in a voice cloned from 3 s of reference
             -> audio out, which is the other agent's input

The cost is three model hops per turn instead of one, and no possibility of
overlap -- a cascade is turn-based by construction. What it buys is agents that
answer each other, distinct clonable voices, and about 3.3 GB for the pair
against 10.9 GB for a single PersonaPlex.

Emotion survives the text hop, which a plain transcript would lose: SenseVoice
tags each utterance and the tag is handed to the listening agent as part of what
it heard.

    ollama pull llama3.2:3b
    python cascade_conversation.py --seed opener.wav --turns 8
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tagstore
import trim_speech

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIOCPP = os.path.join(REPO, "audio.cpp")
CLI = os.path.join(AUDIOCPP, "audiocpp_cli.exe")
ASR_MODEL = os.path.join(AUDIOCPP, "models", "SenseVoice-Small-GGUF")
TTS_MODEL = os.path.join(AUDIOCPP, "models", "Qwen3-TTS-12Hz-0.6B-Base-GGUF")
OLLAMA = "http://127.0.0.1:11434/api/chat"
RATE = 24000

# SenseVoice wraps everything in <|tag|>. The emotion and event ones are worth
# keeping and handing to the other agent; the rest is machinery.
TAG = re.compile(r"<\|([A-Za-z_]+)\|>")
CARRY = {"HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED",
         "Laughter", "Applause", "Crying", "Coughing", "Sneeze"}


def run(args, what, cwd=None):
    # Explicit UTF-8: the default is the Windows code page, which turns an
    # accented word from the LLM into mojibake on the way through.
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise SystemExit(f"  {what} failed ({proc.returncode}): {' | '.join(tail)}")
    return proc.stdout


def transcribe(wav, work):
    """Text plus whatever SenseVoice noticed about how it was said."""
    wav16 = os.path.join(work, "asr.16k.wav")
    out = os.path.join(work, "asr.txt")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-i", wav, "-ac", "1", "-ar", "16000", wav16], "resample")
    run([CLI, "--task", "asr", "--family", "sense_asr", "--model", ASR_MODEL,
         "--backend", "cuda", "--audio", wav16,
         "--request-option", "keep_tags=true", "--request-option", "language=auto",
         "--text-out", out], "asr", cwd=AUDIOCPP)
    if not os.path.exists(out):
        return "", []
    raw = open(out, encoding="utf-8").read()
    marks = [m for m in TAG.findall(raw) if m in CARRY]
    return re.sub(r"\s+", " ", TAG.sub(" ", raw)).strip(), sorted(set(marks))


def think(model, system, history, temperature, ctx=4096):
    """The conversation itself. History is this agent's view of who said what.

    Two settings keep the three stages out of each other's way on one card.
    Ollama defaults to a large context and held 8 GB of the 12 GB card, which
    left the TTS unable to allocate a 35 MB buffer -- so the window is capped at
    what a short conversation actually needs, and `keep_alive: 0` unloads the
    LLM the moment it has answered. The cascade is sequential anyway; paying a
    reload per turn is cheaper than not fitting.
    """
    payload = {"model": model, "stream": False, "keep_alive": 0,
               "options": {"temperature": temperature, "num_ctx": ctx},
               "messages": [{"role": "system", "content": system}] + history}
    req = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)["message"]["content"].strip()
    except Exception as exc:
        raise SystemExit(f"  ollama did not answer ({exc}); is it running, and is {model} pulled?")


def synthesize(text, agent, raw, seed=None):
    args = [CLI, "--task", "tts", "--family", "qwen3_tts", "--model", TTS_MODEL,
            "--backend", "cuda", "--text", text,
            "--voice-ref", agent["voice_ref"], "--reference-text", agent["voice_text"],
            "--out", raw]
    if seed is not None:
        args += ["--seed", str(seed)]
    run(args, "tts", cwd=AUDIOCPP)


def speak(text, agent, out_wav, work, attempts=2):
    """Qwen3-TTS wants --task tts *with* a reference, not --task clon.

    An autoregressive TTS can fail by not stopping: a 26-word sentence once came
    back as 655 seconds of audio, which the silence trim happily kept because it
    was not silent. Speech runs about 2.5 words a second, so anything past three
    times that is the model having run away rather than having a lot to say.
    Retry first -- it is usually a one-off -- and truncate if it happens twice,
    because a clipped turn is recoverable and an eleven-minute one is not.
    """
    words = max(1, len(text.split()))
    budget = max(6.0, words / 2.5 * 3.0)
    raw = os.path.join(work, "tts.raw.wav")

    for attempt in range(attempts):
        synthesize(text, agent, raw, seed=None if attempt == 0 else 1234 + attempt)
        audio, sr = trim_speech.read(raw)
        bounds = trim_speech.speech_bounds(audio, sr)
        if bounds is None:
            continue
        a = max(0.0, bounds[0] - trim_speech.MARGIN_S)
        b = min(len(audio) / sr, bounds[1] + trim_speech.MARGIN_S)
        if b - a <= budget:
            trim_speech.write(out_wav, audio[int(a * sr):int(b * sr)], sr)
            return True
        print(f"      [!] synthesis ran to {b - a:.0f}s for {words} words "
              f"(budget {budget:.0f}s)" + ("; retrying" if attempt + 1 < attempts else "; truncating"))

    if bounds is None:
        return False
    trim_speech.write(out_wav, audio[int(a * sr):int((a + budget) * sr)], sr)
    return True


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Two cascaded agents in conversation")
    ap.add_argument("--seed", required=True, help="opening utterance wav")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(os.environ.get("TEMP", "."), "cacophony-cascade"))
    ap.add_argument("--llm", default="llama3.2:3b")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--a-name", default="MARGOT")
    ap.add_argument("--a-persona", default="You are Margot, a blunt, impatient physicist who thinks "
                                           "most things are overcomplicated. Dry and a little sarcastic.")
    ap.add_argument("--a-voice", required=True, help="wav of the voice agent A speaks in")
    ap.add_argument("--a-voice-text", required=True, help="what that wav says")
    ap.add_argument("--b-name", default="DESMOND")
    ap.add_argument("--b-persona", default="You are Desmond, a warm, over-enthusiastic startup founder "
                                           "who turns everything into a business idea.")
    ap.add_argument("--b-voice", required=True)
    ap.add_argument("--b-voice-text", required=True)
    ap.add_argument("--play", action="store_true", help="play each turn as it is made")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "work")
    os.makedirs(work, exist_ok=True)

    style = ("Reply in at most two short sentences, as speech, with no stage directions, "
             "emoji or markdown. Do not repeat what the other person just said.")
    agents = [
        {"name": args.a_name, "system": f"{args.a_persona} {style}",
         "voice_ref": os.path.abspath(args.a_voice), "voice_text": args.a_voice_text},
        {"name": args.b_name, "system": f"{args.b_persona} {style}",
         "voice_ref": os.path.abspath(args.b_voice), "voice_text": args.b_voice_text},
    ]
    # A voice reference is a pair -- audio and the words in it -- and cloning
    # conditions on the two lining up. Given text for only part of the clip the
    # model does not fail, it produces confident gibberish: a 14.5 s reference
    # labelled with its first clause (13 words, 0.9 words/s) made three of four
    # turns unintelligible while the other agent, whose text matched, was fine.
    # Speech runs 2-3 words a second, so anything far outside that is a
    # mismatch rather than an unusual talker.
    for a in agents:
        if not os.path.exists(a["voice_ref"]):
            raise SystemExit(f"  missing voice reference: {a['voice_ref']}")
        seconds = duration(a["voice_ref"])
        rate = len(a["voice_text"].split()) / seconds if seconds else 0.0
        if seconds and not (1.2 <= rate <= 4.5):
            print(f"  [!] {a['name']}: reference is {seconds:.1f}s but its text is "
                  f"{len(a['voice_text'].split())} words ({rate:.1f} words/s). "
                  f"The text must be exactly what the audio says, or cloning degrades.")

    current = os.path.join(args.out, "turn000.seed.wav")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-i", args.seed, "-ac", "1", "-ar", str(RATE), current], "seed")

    print(f"\n  {agents[0]['name']} <-> {agents[1]['name']}  |  {args.llm}  |  {args.turns} turns")
    seed_text, seed_marks = transcribe(current, work)
    print(f"   0. opener: \"{seed_text}\"" + (f"  {seed_marks}" if seed_marks else ""))

    # One canonical record; each agent's messages are built from it, with its own
    # lines as assistant and the other's as user.
    said = [{"who": "SOMEONE", "text": seed_text, "marks": seed_marks}]
    clock = duration(current)
    record = [{"agent": "SEED", "wav": current, "text": seed_text,
               "start": 0.0, "end": round(clock, 3)}]

    for turn in range(args.turns):
        agent = agents[turn % 2]
        history = []
        for item in said:
            heard = item["text"]
            if item["marks"]:
                heard = f"[{', '.join(item['marks']).lower()}] {heard}"
            history.append({"role": "assistant" if item["who"] == agent["name"] else "user",
                            "content": heard})
        reply = think(args.llm, agent["system"], history, args.temperature)
        reply = re.sub(r"\s+", " ", reply).strip().strip('"')
        if not reply:
            print("      the agent had nothing to say; stopping")
            break
        print(f"  {turn + 1:2d}. {agent['name']}: \"{reply}\"")

        wav = os.path.join(args.out, f"turn{turn + 1:03d}.wav")
        if not speak(reply, agent, wav, work):
            print("      synthesis produced silence; stopping")
            break
        length = duration(wav)
        print(f"      {length:.2f}s")

        said.append({"who": agent["name"], "text": reply, "marks": []})
        record.append({"agent": agent["name"], "wav": wav, "text": reply,
                       "start": round(clock, 3), "end": round(clock + length, 3)})
        clock += length
        current = wav
        if args.play:
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"(New-Object System.Media.SoundPlayer '{wav}').PlaySync()"])

    if len(record) <= 1:
        raise SystemExit("  no turns generated")

    listing = os.path.join(args.out, "concat.txt")
    with open(listing, "w", encoding="utf-8") as f:
        for item in record:
            f.write("file '%s'\n" % os.path.abspath(item["wav"]).replace("\\", "/"))
    conversation = os.path.join(args.out, "conversation.wav")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-f", "concat", "-safe", "0", "-i", listing,
         "-ac", "1", "-ar", str(RATE), conversation], "stitch")

    tags = os.path.join(args.out, "conversation.tags.json")
    store = tagstore.load(tags)
    tagstore.set_media(store, conversation, clock)
    tagstore.put_cues(store, [(t["start"], t["end"], t["text"]) for t in record])
    tagstore.put_track(store, "speaker",
                       [{"start": t["start"], "end": t["end"], "value": t["agent"]}
                        for t in record], "conversation", model=args.llm)
    tagstore.save(store, tags)

    with open(os.path.join(args.out, "conversation.txt"), "w", encoding="utf-8") as f:
        for t in record:
            f.write(f"{t['agent']}: {t['text']}\n")

    print(f"\n  {clock:.1f}s over {len(record) - 1} turns")
    print(f"    {conversation}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
