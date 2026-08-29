"""
Nemotron-3.5-ASR-Streaming reference backend for crispasr-diff.

model_dir: HF id "nvidia/nemotron-3.5-asr-streaming-0.6b" or local .nemo path.
audio: 16 kHz mono WAV.

Stages:
  mel_spectrogram   — (n_mels, T_mel) log-mel filterbank
  pre_encode_output — (T_enc, d_model) after 4× conv subsampling
  encoder_output    — (T_enc, d_model) after 24L FastConformer + prompt kernel

Requires: nemo_toolkit[asr], torch
"""

import os
import sys
import numpy as np

DEFAULT_STAGES = [
    "mel_spectrogram",
    "pre_encode_output",
    "encoder_output",
]


def dump(model_dir: str, audio_path: str, output_path: str, stages=None,
         verbose: bool = False, **kwargs):
    """Run NeMo nemotron inference and dump intermediate activations."""
    import torch
    import nemo.collections.asr as nemo_asr
    import soundfile as sf

    if stages is None:
        stages = DEFAULT_STAGES

    # Load audio
    pcm, sr = sf.read(audio_path, dtype="float32")
    if sr != 16000:
        import torchaudio
        pcm_t = torch.from_numpy(pcm).unsqueeze(0)
        pcm_t = torchaudio.functional.resample(pcm_t, sr, 16000)
        pcm = pcm_t.squeeze(0).numpy()
        sr = 16000

    # Load model
    if model_dir.endswith(".nemo") or os.path.isfile(model_dir):
        model = nemo_asr.models.ASRModel.restore_from(model_dir, map_location="cpu")
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_dir, map_location="cpu")
    model.eval()

    tensors = {}
    lengths_tensor = torch.tensor([len(pcm)], dtype=torch.int64)

    # Mel spectrogram
    if "mel_spectrogram" in stages:
        with torch.no_grad():
            processed, processed_len = model.preprocessor(
                input_signal=torch.from_numpy(pcm).unsqueeze(0),
                length=lengths_tensor,
            )
        mel = processed.squeeze(0).cpu().numpy()  # (n_mels, T_mel)
        tensors["mel_spectrogram"] = mel
        if verbose:
            print(f"  mel_spectrogram: {mel.shape} min={mel.min():.4f} max={mel.max():.4f}")

    # Encoder output (includes pre-encode internally)
    if "encoder_output" in stages or "pre_encode_output" in stages:
        with torch.no_grad():
            # Run full encoder
            encoded, encoded_len = model.encoder(
                audio_signal=torch.from_numpy(pcm).unsqueeze(0),
                length=lengths_tensor,
            )
        enc = encoded.squeeze(0).cpu().numpy()  # (T_enc, d_model)
        # NeMo encoder output is (B, d, T) — transpose to (T, d)
        if enc.shape[0] == model.encoder._feat_out:
            enc = enc.T
        tensors["encoder_output"] = enc
        if verbose:
            print(f"  encoder_output: {enc.shape} min={enc.min():.4f} max={enc.max():.4f}")

    # Write GGUF
    from gguf import GGUFWriter
    writer = GGUFWriter(output_path, arch="nemotron-ref")
    writer.add_string("nemotron.audio_path", os.path.basename(audio_path))
    writer.add_string("nemotron.model_dir", model_dir)

    for name, data in tensors.items():
        arr = np.ascontiguousarray(data, dtype=np.float32)
        writer.add_tensor(name, arr)
        if verbose:
            print(f"  wrote {name}: shape={arr.shape}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {output_path} ({len(tensors)} tensors)")
