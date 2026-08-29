import gradio as gr
import spaces
import torch
import numpy as np
import os
import tarfile
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download
import sentencepiece

# Configuration
HF_REPO = "nvidia/personaplex-7b-v1"
DEVICE = "cuda"
SAMPLE_RATE = 24000

# Available voices in PersonaPlex
ALL_VOICES = [
    "NATF0", "NATF1", "NATF2", "NATF3",  # Natural Female
    "NATM0", "NATM1", "NATM2", "NATM3",  # Natural Male
    "VARF0", "VARF1", "VARF2", "VARF3", "VARF4",  # Variety Female
    "VARM0", "VARM1", "VARM2", "VARM3", "VARM4",  # Variety Male
]

# Example persona prompts from PersonaPlex paper
EXAMPLE_PERSONAS = [
    "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
    "You enjoy having a good conversation.",
    "You work for CitySan Services which is a waste management company and your name is Ayelen Lucero.",
    "You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex.",
]

# moshi-personaplex is installed here, at startup, rather than from
# requirements.txt -- and deliberately with --no-deps.
#
# Its pyproject pins are from the Moshi era and none of them survive contact
# with a current Space: huggingface-hub <0.25 against gradio's >=0.33.5 (no
# overlap at all, which is what failed the first build), plus torch <2.5,
# einops ==0.7 and numpy <2.2. The pins are conservative metadata rather than
# real incompatibilities, so the package is installed without them and its
# actual imports are satisfied by requirements.txt.
import subprocess
import sys

try:
    from moshi.models import loaders  # noqa: F401
except ImportError:
    print("installing moshi-personaplex (--no-deps)...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir",
         "git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi"],
        check=True,
    )

# Import moshi after spaces to allow interception
from moshi.models import loaders, LMGen
from moshi.models.lm import load_audio, _iterate_audio, encode_from_sphn

# Pre-download model weights at startup (cached by huggingface_hub)
print("Downloading model weights...")
MIMI_WEIGHT = hf_hub_download(HF_REPO, loaders.MIMI_NAME)
MOSHI_WEIGHT = hf_hub_download(HF_REPO, loaders.MOSHI_NAME)
TOKENIZER_PATH = hf_hub_download(HF_REPO, loaders.TEXT_TOKENIZER_NAME)
VOICES_TGZ = hf_hub_download(HF_REPO, "voices.tgz")

# Extract voices archive
VOICES_DIR = Path(VOICES_TGZ).parent / "voices"
if not VOICES_DIR.exists():
    print("Extracting voice embeddings...")
    with tarfile.open(VOICES_TGZ, "r:gz") as tar:
        tar.extractall(path=Path(VOICES_TGZ).parent)
print("Model weights ready.")

# Load text tokenizer (CPU only, no CUDA needed)
text_tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)

# Global model cache - models loaded lazily inside @spaces.GPU
_model_cache = {}


def get_models():
    """Lazy load models on first GPU call."""
    global _model_cache
    if "initialized" not in _model_cache:
        print("Loading models to GPU...")

        # Load Mimi encoder/decoder
        mimi = loaders.get_mimi(MIMI_WEIGHT, DEVICE)
        other_mimi = loaders.get_mimi(MIMI_WEIGHT, DEVICE)

        # Load Moshi LM
        lm = loaders.get_moshi_lm(MOSHI_WEIGHT, device=DEVICE)
        lm.eval()

        # Create LMGen wrapper
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
            sample_rate=mimi.sample_rate,
            device=DEVICE,
            frame_rate=mimi.frame_rate,
            temp=0.8,
            temp_text=0.7,
            top_k=250,
            top_k_text=25,
        )

        # Enable streaming mode
        mimi.streaming_forever(1)
        other_mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)

        # Run warmup to initialize CUDA graphs (improves performance)
        print("Running warmup...")
        _warmup_models(mimi, other_mimi, lm_gen, frame_size)
        print("Warmup complete.")

        _model_cache.update({
            "mimi": mimi,
            "other_mimi": other_mimi,
            "lm_gen": lm_gen,
            "frame_size": frame_size,
            "initialized": True,
        })
        print("Models loaded successfully.")

    return _model_cache


def _warmup_models(mimi, other_mimi, lm_gen, frame_size):
    """Run warmup passes to initialize CUDA graphs."""
    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=DEVICE)
        codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c:c+1])
            if tokens is not None:
                _ = mimi.decode(tokens[:, 1:9])
                _ = other_mimi.decode(tokens[:, 1:9])
    torch.cuda.synchronize()
    # Reset after warmup
    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as PersonaPlex expects."""
    text = text.strip()
    if text.startswith("<system>") and text.endswith("<system>"):
        return text
    return f"<system> {text} <system>"


def decode_tokens_to_pcm(mimi, other_mimi, tokens: torch.Tensor) -> np.ndarray:
    """Decode audio tokens to PCM waveform."""
    # tokens shape: [B, num_codebooks, 1]
    # Agent audio is in codebooks 1:9
    agent_audio_tokens = tokens[:, 1:9, :]
    pcm = other_mimi.decode(agent_audio_tokens)
    return pcm[0, 0].detach().cpu().numpy()


@spaces.GPU(duration=120)
def generate_response(audio_input, persona: str, voice: str):
    """Process audio input and generate PersonaPlex response."""
    if audio_input is None:
        return None, "Please record audio first."

    # Get lazily loaded models
    models = get_models()
    mimi = models["mimi"]
    other_mimi = models["other_mimi"]
    lm_gen = models["lm_gen"]
    frame_size = models["frame_size"]

    # Process input audio
    sr, audio = audio_input
    audio = audio.astype(np.float32)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Normalize to [-1, 1]
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / 32768.0 if audio.dtype == np.int16 else audio / np.abs(audio).max()

    # Resample to model's sample rate if needed
    if sr != mimi.sample_rate:
        import sphn
        audio = sphn.resample(audio, sr, mimi.sample_rate)

    # PREPEND SILENCE: Let model say its default greeting during this time (we'll discard this output)
    prepend_silence_duration = 2  # seconds
    prepend_silence = np.zeros(int(prepend_silence_duration * mimi.sample_rate), dtype=np.float32)

    # APPEND SILENCE: Give model time to complete its response after user finishes speaking
    append_silence_duration = 8  # seconds
    append_silence = np.zeros(int(append_silence_duration * mimi.sample_rate), dtype=np.float32)

    # Final audio: [prepend_silence] + [user_audio] + [append_silence]
    audio = np.concatenate([prepend_silence, audio, append_silence])

    # Calculate how many output frames to skip (corresponds to prepend silence)
    # frame_rate is 12.5 Hz, so frames_to_skip = prepend_silence_duration * frame_rate
    frames_to_skip = int(prepend_silence_duration * 12.5)

    # Add channel dimension: (T,) -> (1, T)
    if audio.ndim == 1:
        audio = audio[None, :]

    # Load voice prompt
    voice_path = str(VOICES_DIR / f"{voice}.pt")
    if not os.path.exists(voice_path):
        return None, f"Voice '{voice}' not found."
    lm_gen.load_voice_prompt_embeddings(voice_path)

    # Set text prompt
    if persona.strip():
        lm_gen.text_prompt_tokens = text_tokenizer.encode(wrap_with_system_tags(persona))
    else:
        lm_gen.text_prompt_tokens = None

    # Run system prompts (voice + text conditioning)
    with lm_gen.streaming(1):
        # Reset streaming state inside the context
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()

        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()

        # Process user audio frames
        generated_frames = []
        generated_text = []
        frame_count = 0  # Track frame index to skip prepend silence output

        for user_encoded in encode_from_sphn(
            mimi,
            _iterate_audio(audio, sample_interval_size=frame_size, pad=True),
            max_batch=1,
        ):
            for c in range(user_encoded.shape[-1]):
                step_in = user_encoded[:, :, c:c+1]
                tokens = lm_gen.step(step_in)
                frame_count += 1

                if tokens is None:
                    continue

                # Skip frames generated during prepend silence (model's default greeting)
                if frame_count <= frames_to_skip:
                    continue

                # Decode agent audio
                pcm = decode_tokens_to_pcm(mimi, other_mimi, tokens)
                generated_frames.append(pcm)

                # Decode text token
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):  # Skip special tokens
                    text_piece = text_tokenizer.id_to_piece(text_token).replace("▁", " ")
                    generated_text.append(text_piece)

    if not generated_frames:
        return None, "No audio generated. Try speaking more clearly."

    # Concatenate output audio
    output_audio = np.concatenate(generated_frames, axis=-1)
    output_text = "".join(generated_text).strip()

    return (mimi.sample_rate, output_audio), output_text


import conversation as convo


# Each turn is its own full pass, so the budget scales with the turn count. The
# ceiling keeps a runaway request from eating the whole ZeroGPU quota at once.
@spaces.GPU(duration=600)
def run_conversation(seed_audio, persona_a, voice_a, persona_b, voice_b, turns,
                     progress=gr.Progress()):
    if seed_audio is None:
        return None, [["", "Upload an opening utterance first.", ""]], None

    models = get_models()
    sr_in, audio = seed_audio
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / (32768.0 if peak > 2.0 else peak)
    if sr_in != models["mimi"].sample_rate:
        import sphn
        audio = sphn.resample(audio, sr_in, models["mimi"].sample_rate)

    for name, voice in (("A", voice_a), ("B", voice_b)):
        if not os.path.exists(str(VOICES_DIR / f"{voice}.pt")):
            return None, [["", f"Voice '{voice}' not found for agent {name}.", ""]], None

    helpers = {
        "wrap_with_system_tags": wrap_with_system_tags,
        "decode_tokens_to_pcm": decode_tokens_to_pcm,
        "_iterate_audio": _iterate_audio,
        "encode_from_sphn": encode_from_sphn,
    }
    agent_a = {"name": "AGENT_A", "persona": persona_a, "voice": voice_a,
               "voice_path": str(VOICES_DIR / f"{voice_a}.pt")}
    agent_b = {"name": "AGENT_B", "persona": persona_b, "voice": voice_b,
               "voice_path": str(VOICES_DIR / f"{voice_b}.pt")}

    sample_rate = models["mimi"].sample_rate
    pcm, rows, turns_out = convo.converse(models, text_tokenizer, helpers, audio,
                                          sample_rate, agent_a, agent_b,
                                          int(turns), progress)
    if pcm is None:
        return None, rows, None
    return (sample_rate, pcm), rows, convo.package_turns(turns_out, sample_rate)


# Build Gradio interface
with gr.Blocks(title="PersonaPlex Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎭 PersonaPlex
        **Voice and Role Control for Full Duplex Conversational Speech Models**

        [Paper](https://arxiv.org/abs/2503.04721) | [GitHub](https://github.com/NVIDIA/personaplex) | [Model](https://huggingface.co/nvidia/personaplex-7b-v1)

        ---

        Record your message, and PersonaPlex will respond with the configured persona and voice.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            persona = gr.Textbox(
                label="Persona Description",
                placeholder="Describe the assistant's persona...",
                value=EXAMPLE_PERSONAS[0],
                lines=4,
            )
            voice = gr.Dropdown(
                choices=ALL_VOICES,
                value="NATF2",
                label="Voice"
            )
            gr.Examples(
                examples=[[p] for p in EXAMPLE_PERSONAS],
                inputs=[persona],
                label="Example Personas"
            )

        with gr.Column(scale=2):
            audio_input = gr.Audio(
                label="🎤 Record your message",
                sources=["microphone", "upload"],
                type="numpy",
            )
            generate_btn = gr.Button("Generate Response", variant="primary", size="lg")

            audio_output = gr.Audio(
                label="🔊 PersonaPlex Response",
                type="numpy",
                autoplay=True,
            )
            text_output = gr.Textbox(
                label="📝 Response Text",
                interactive=False,
            )

    generate_btn.click(
        fn=generate_response,
        inputs=[audio_input, persona, voice],
        outputs=[audio_output, text_output],
    )

    gr.Markdown("---")
    with gr.Accordion("🗣️ Two agents in conversation", open=False):
        gr.Markdown(
            """
            Upload one opening utterance and two PersonaPlex agents talk to each
            other, taking turns until the conversation is as long as you asked
            for. You get back a single stitched wav and a per-turn transcript.

            Each agent hears the *other* voice, which is what the model was
            trained for — handing it its own voice instead makes it repeat
            itself. Turns are re-primed rather than continued, because
            continuing a session skips the voice and system prompts and
            collapses both agents into one voice and one persona.

            Long conversations cost ZeroGPU quota: each turn is a full pass over
            its input plus a reply window, so start with 4 turns.
            """
        )
        with gr.Row():
            with gr.Column():
                conv_seed = gr.Audio(
                    label="🎤 Opening utterance",
                    sources=["upload", "microphone"],
                    type="numpy",
                )
                conv_turns = gr.Number(
                    value=4, precision=0, minimum=1, maximum=60,
                    label="Turns",
                    info="Each turn is a full pass plus a reply window. The GPU "
                         "budget is 600 s per run; start at 4 and see what it costs.",
                )
            with gr.Column():
                conv_persona_a = gr.Textbox(
                    label="Agent A persona", lines=3,
                    value="You are Margot, a blunt, impatient physicist who thinks "
                          "most things are overcomplicated.",
                )
                conv_voice_a = gr.Dropdown(choices=ALL_VOICES, value="VARF1", label="Agent A voice")
            with gr.Column():
                conv_persona_b = gr.Textbox(
                    label="Agent B persona", lines=3,
                    value="You are Desmond, a warm, over-enthusiastic startup founder "
                          "who turns everything into a business idea.",
                )
                conv_voice_b = gr.Dropdown(choices=ALL_VOICES, value="VARM2", label="Agent B voice")

        conv_btn = gr.Button("Run the conversation", variant="primary", size="lg")
        conv_audio = gr.Audio(label="🔊 Full conversation", type="numpy")
        conv_table = gr.Dataframe(
            headers=["speaker", "said", "length"],
            label="📝 Turns",
            wrap=True,
        )
        conv_zip = gr.File(
            label="📦 Per-turn wavs + manifest.json + transcript.txt",
        )

        conv_btn.click(
            fn=run_conversation,
            inputs=[conv_seed, conv_persona_a, conv_voice_a,
                    conv_persona_b, conv_voice_b, conv_turns],
            outputs=[conv_audio, conv_table, conv_zip],
        )


if __name__ == "__main__":
    demo.launch()
