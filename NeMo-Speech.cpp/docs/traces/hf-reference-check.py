"""Run the HF reference implementation on the black-hole clips.

The point of this script is to separate two explanations that every test so far
has been unable to tell apart:

  * our pipeline (GGUF conversion, C++ inference, patched CUDA kernels) diverges
    from the reference, or
  * the model itself goes silent across a language switch.

A clean upstream C++ build already reproduced the hole character-for-character,
so the build stack is out. This runs the same audio through PyTorch + the
stock transformers implementation, which shares nothing with our C++ path.
"""

import sys
import glob
import os

import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoConfig

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
CLIPS = sys.argv[1:]

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
import transformers

print(f"transformers {transformers.__version__}")

cfg = AutoConfig.from_pretrained(MODEL)
print(f"config: {type(cfg).__name__}")

processor = AutoProcessor.from_pretrained(MODEL)
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, dtype=torch.float32)
except Exception as e:  # the auto class may not be registered for this arch
    print(f"AutoModelForSpeechSeq2Seq failed ({e}); trying the explicit class")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32)
print(f"model: {type(model).__name__}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

for path in CLIPS:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    print(f"\n=== {os.path.basename(path)}  {len(audio)/sr:.2f}s @ {sr} Hz ===")
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs)
    # generate() may hand back already-decoded strings on this arch, or ids.
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], str):
        text = out
    elif hasattr(out, "sequences"):
        text = processor.batch_decode(out.sequences, skip_special_tokens=False)
    else:
        try:
            text = processor.batch_decode(out, skip_special_tokens=False)
        except TypeError:
            text = out
    print("HF REFERENCE:", text)
