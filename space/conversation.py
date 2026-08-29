"""Two PersonaPlex agents in conversation, for the ZeroGPU Space.

Locally this is impossible: one PersonaPlex is 10.9 GB on a 12 GB card, so the
second agent has nowhere to live, and feeding a single model its own voice makes
it collapse into repeating itself. Here there is room, and each agent hears the
*other* voice -- which is the configuration the model was actually trained for.

Upload a seed utterance, pick two personas and two voices, and the agents take
turns until the conversation is as long as you asked for. What comes back is one
stitched wav plus a per-turn transcript, which is a demo and also a source of
natural, emotional, conversational speech for a voice store -- material a
dataset does not give you, because you chose the personas and the topic.

Each turn is primed rather than continued: voice and persona are set on `lm_gen`
before `step_system_prompts`, so one model can serve both agents by re-priming
between turns. That is deliberate. Continuing a session keeps the KV cache but
skips the voice and system prompts, which collapses both agents into one voice
and one persona -- measured, not assumed. Re-priming costs a fraction of a
second here and keeps the two agents distinct.
"""

from __future__ import annotations

import json
import os
import tempfile
import wave
import zipfile

import numpy as np
import torch


FRAME_RATE = 12.5
# Speech sits near -25 dB and PersonaPlex's silence near -70 dB, so the gate is
# nowhere near either. The margin keeps a soft onset from being clipped.
SILENCE_DB = -55.0
MARGIN_S = 0.25
# Room for the agent to answer in. PersonaPlex emits one output frame per input
# frame, so a reply is only as long as the silence it is given to speak into.
REPLY_WINDOW_S = 14.0
# The model greets before it has heard anything; that greeting is discarded.
LEAD_IN_S = 2.0


def trim_to_speech(pcm: np.ndarray, sample_rate: int,
                   threshold_db: float = SILENCE_DB, margin: float = MARGIN_S):
    """Cut a turn down to the part where the agent is talking.

    Without this the conversation grows without bound: every turn is as long as
    its input, so silence accumulates and each reply is longer than the last.
    """
    frame = max(1, int(0.02 * sample_rate))
    n = len(pcm) // frame
    if n == 0:
        return None
    frames = pcm[:n * frame].reshape(n, frame)
    rms = np.sqrt((frames * frames).mean(axis=1))
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(np.maximum(rms, 1e-9))
    loud = np.flatnonzero(db > threshold_db)
    if not len(loud):
        return None
    start = max(0.0, loud[0] * 0.02 - margin)
    end = min(len(pcm) / sample_rate, (loud[-1] + 1) * 0.02 + margin)
    if end - start < 0.3:
        return None
    return pcm[int(start * sample_rate):int(end * sample_rate)]


def _write_wav(path, pcm: np.ndarray, sample_rate: int):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def package_turns(turns_out, sample_rate: int):
    """One wav per turn plus a manifest, zipped.

    The stitched conversation is the demo; this is the material. A voice store
    enrols one speaker saying one thing, so each turn is its own file, and the
    manifest carries the transcript beside it -- which is what a clone reference
    needs, and the pair the model conditions on.
    """
    if not turns_out:
        return None
    out_dir = tempfile.mkdtemp(prefix="personaplex-conversation-")
    zip_path = os.path.join(out_dir, "conversation-turns.zip")
    manifest = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for item in turns_out:
            name = f"turn{item['index']:02d}_{item['speaker']}.wav"
            wav_path = os.path.join(out_dir, name)
            _write_wav(wav_path, item["pcm"], sample_rate)
            z.write(wav_path, name)
            seconds = len(item["pcm"]) / sample_rate
            words = len(item.get("text", "").split())
            manifest.append({
                "file": name,
                "index": item["index"],
                "speaker": item["speaker"],
                "voice": item.get("voice", ""),
                "persona": item.get("persona", ""),
                "text": item.get("text", ""),
                "seconds": round(seconds, 3),
                "sample_rate": sample_rate,
                # Cloning conditions on the text matching the audio; this is the
                # cheap check that they do, and it is the number to look at
                # before trusting a turn as a reference.
                "words_per_second": round(words / seconds, 2) if seconds else None,
            })
        z.writestr("manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False))
        z.writestr("transcript.txt",
                   "\n".join(f"{m['speaker']}: {m['text']}" for m in manifest if m["text"]))
    return zip_path


def run_turn(models, text_tokenizer, helpers, voice_path, persona,
             audio_in: np.ndarray, sample_rate: int):
    """One agent hears `audio_in` and replies. Returns (pcm, text).

    `helpers` carries the Space's own encode/decode utilities so this module
    does not duplicate them: wrap_with_system_tags, decode_tokens_to_pcm,
    _iterate_audio, encode_from_sphn.
    """
    mimi = models["mimi"]
    other_mimi = models["other_mimi"]
    lm_gen = models["lm_gen"]
    frame_size = models["frame_size"]

    lead_in = np.zeros(int(LEAD_IN_S * sample_rate), dtype=np.float32)
    tail = np.zeros(int(REPLY_WINDOW_S * sample_rate), dtype=np.float32)
    audio = np.concatenate([lead_in, audio_in.astype(np.float32), tail])[None, :]
    skip_frames = int(LEAD_IN_S * FRAME_RATE)

    lm_gen.load_voice_prompt_embeddings(voice_path)
    lm_gen.text_prompt_tokens = (
        text_tokenizer.encode(helpers["wrap_with_system_tags"](persona))
        if persona.strip() else None
    )

    frames, pieces = [], []
    with lm_gen.streaming(1):
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()

        index = 0
        for user_encoded in helpers["encode_from_sphn"](
            mimi,
            helpers["_iterate_audio"](audio, sample_interval_size=frame_size, pad=True),
            max_batch=1,
        ):
            for c in range(user_encoded.shape[-1]):
                tokens = lm_gen.step(user_encoded[:, :, c:c + 1])
                index += 1
                if tokens is None or index <= skip_frames:
                    continue
                frames.append(helpers["decode_tokens_to_pcm"](mimi, other_mimi, tokens))
                token = tokens[0, 0, 0].item()
                if token not in (0, 3):
                    pieces.append(text_tokenizer.id_to_piece(token).replace("▁", " "))

    if not frames:
        return None, ""
    return np.concatenate(frames, axis=-1), "".join(pieces).strip()


def converse(models, text_tokenizer, helpers, seed_audio: np.ndarray, sample_rate: int,
             agent_a: dict, agent_b: dict, turns: int, progress=None):
    """Alternate between two agents, returning (stitched_pcm, transcript rows).

    Each agent hears only the other's last utterance, trimmed to speech. That is
    the whole loop; everything else is bookkeeping.
    """
    agents = [agent_a, agent_b]
    current = seed_audio.astype(np.float32)
    pieces = [current]
    rows = [["SEED", "", f"{len(current) / sample_rate:.1f}s"]]
    # Per-turn segments kept separately: a voice store enrols one utterance by
    # one speaker at a time, so the stitched file is the demo and these are the
    # material.
    turns_out = [{"index": 0, "speaker": "SEED", "text": "", "voice": "",
                  "pcm": current}]

    for turn in range(turns):
        agent = agents[turn % 2]
        if progress is not None:
            progress((turn + 1) / (turns + 1),
                     desc=f"turn {turn + 1}/{turns}: {agent['name']} replying")

        pcm, text = run_turn(models, text_tokenizer, helpers,
                             agent["voice_path"], agent["persona"], current, sample_rate)
        if pcm is None:
            rows.append([agent["name"], "(silence -- stopping)", "0.0s"])
            break
        speech = trim_to_speech(pcm, sample_rate)
        if speech is None:
            rows.append([agent["name"], "(silence -- stopping)", "0.0s"])
            break

        pieces.append(speech)
        rows.append([agent["name"], text or "(no text decoded)",
                     f"{len(speech) / sample_rate:.1f}s"])
        turns_out.append({"index": turn + 1, "speaker": agent["name"],
                          "text": text, "voice": agent.get("voice", ""),
                          "persona": agent.get("persona", ""), "pcm": speech})
        current = speech
        torch.cuda.empty_cache()

    if len(pieces) <= 1:
        return None, rows, []
    return np.concatenate(pieces, axis=-1), rows, turns_out
