#!/usr/bin/env python3
"""Turn PersonaPlex's packaged voices into clone references the cascade can use.

PersonaPlex sounds better than anything else here -- its 18 packaged voices are
hard to tell from real recordings -- but it only does speech-to-speech, so it
cannot be the cascade's synthesizer: there is no way to hand it text. What it
can do is *speak*, and a few seconds of it speaking is exactly what Qwen3-TTS
needs to clone a voice.

So each voice is minted once, offline:

    PersonaPlex speaks (voice_id=VARF1)  ->  trim to the reply
      -> SenseVoice transcribes it       ->  reference text that matches by
                                             construction rather than by hand

That last step matters more than it looks. A clone reference is a pair, and the
model conditions on the two agreeing; given text for only part of the clip it
does not fail, it produces confident gibberish. Transcribing the audio we just
made removes the chance of a mismatch entirely -- nobody types the text, so
nobody can get it wrong.

Minting needs PersonaPlex resident (10.9 GB of a 12 GB card), so it runs alone
and once. Afterwards the cascade only ever touches the resulting wav+txt pairs.

    python mint_voice_refs.py --seed opener.wav --voice VARF1 --voice VARM2
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trim_speech

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIOCPP = os.path.join(REPO, "audio.cpp")
PP_CLI = os.path.join(AUDIOCPP, "build", "windows-cuda-release", "bin", "audiocpp_cli.exe")
PP_MODEL = "models\\PersonaPlex-GGUF"
ASR_CLI = os.path.join(AUDIOCPP, "audiocpp_cli.exe")
ASR_MODEL = os.path.join(AUDIOCPP, "models", "SenseVoice-Small-GGUF")
RATE = 24000

# Long enough for the clone to have something to work with, short enough that
# the reference stays a reference. Qwen3-TTS clones from about three seconds.
MIN_REF_S, MAX_REF_S = 3.0, 8.0


def run(args, what, cwd=None):
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise SystemExit(f"  {what} failed ({proc.returncode}): {' | '.join(tail)}")
    return proc.stdout


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Mint clone references from PersonaPlex voices")
    ap.add_argument("--seed", required=True, help="something for PersonaPlex to answer")
    ap.add_argument("--voice", action="append", required=True,
                    help="packaged voice id, repeatable (NATF0-3, NATM0-3, VARF0-4, VARM0-4)")
    ap.add_argument("--out", default=os.path.join(REPO, "samples", "voice-refs"))
    ap.add_argument("--reply-window", type=float, default=20.0)
    ap.add_argument("--prompt", default="Say one short, ordinary sentence about the weather, "
                                        "then stop talking.")
    args = ap.parse_args()

    for path in (PP_CLI, ASR_CLI):
        if not os.path.exists(path):
            raise SystemExit(f"  missing: {path}")
    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "work")
    os.makedirs(work, exist_ok=True)

    # PersonaPlex answers *into* trailing silence, and its output is as long as
    # its input, so the room to reply has to be added up front.
    padded = os.path.join(work, "seed.padded.wav")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", args.seed,
         "-af", f"apad=pad_dur={args.reply_window}", "-ac", "1", "-ar", str(RATE), padded],
        "pad seed")
    heard = duration(args.seed)

    print(f"\n  minting {len(args.voice)} voice reference(s) into {args.out}\n")
    minted = []
    for voice in args.voice:
        raw = os.path.join(work, f"{voice}.raw.wav")
        wav = os.path.join(args.out, f"{voice}.wav")
        txt = os.path.join(args.out, f"{voice}.txt")
        print(f"  {voice}: speaking...", flush=True)
        run([PP_CLI, "--task", "s2s", "--family", "personaplex", "--model", PP_MODEL,
             "--backend", "cuda", "--audio", padded, "--text", args.prompt,
             "--request-option", f"voice_id={voice}", "--out", raw],
            f"personaplex {voice}", cwd=AUDIOCPP)

        # Drop the stretch where the other party was still talking: PersonaPlex
        # is full duplex and speaks over its input.
        audio, sr = trim_speech.read(raw)
        audio = audio[int(heard * sr):]
        bounds = trim_speech.speech_bounds(audio, sr)
        if bounds is None:
            print(f"      [!] {voice} said nothing; skipped")
            continue
        start = max(0.0, bounds[0] - trim_speech.MARGIN_S)
        end = min(len(audio) / sr, bounds[1] + trim_speech.MARGIN_S)
        if end - start < MIN_REF_S:
            print(f"      [!] only {end - start:.1f}s of speech; skipped")
            continue
        end = min(end, start + MAX_REF_S)
        trim_speech.write(wav, audio[int(start * sr):int(end * sr)], sr)

        # The text comes from the audio, so the pair cannot disagree.
        wav16 = os.path.join(work, f"{voice}.16k.wav")
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-i", wav, "-ac", "1", "-ar", "16000", wav16], "resample")
        out_txt = os.path.join(work, f"{voice}.asr.txt")
        run([ASR_CLI, "--task", "asr", "--family", "sense_asr", "--model", ASR_MODEL,
             "--backend", "cuda", "--audio", wav16, "--text-out", out_txt],
            "asr", cwd=AUDIOCPP)
        import re
        spoken = re.sub(r"<\|[A-Za-z_]+\|>", " ", open(out_txt, encoding="utf-8").read())
        spoken = re.sub(r"\s+", " ", spoken).strip()
        if not spoken:
            print(f"      [!] nothing transcribable; skipped")
            continue
        with open(txt, "w", encoding="utf-8") as f:
            f.write(spoken)

        seconds = end - start
        rate = len(spoken.split()) / seconds
        print(f"      {seconds:.2f}s, {len(spoken.split())} words ({rate:.1f} w/s)")
        print(f"      \"{spoken}\"")
        minted.append((voice, wav, txt))

    if not minted:
        raise SystemExit("\n  nothing minted\n")
    print(f"\n  {len(minted)} reference(s) ready. Use with cascade_conversation.py:")
    for voice, wav, txt in minted:
        print(f"    --a-voice \"{wav}\" --a-voice-text \"$(cat {os.path.basename(txt)})\"")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
