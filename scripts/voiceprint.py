#!/usr/bin/env python3
"""CAM++ speaker embeddings — a fixed-length vector per stretch of speech.

The vector is the whole point: two vectors from the same person land close
together under cosine distance and two from different people do not, so
"is this the same voice as before" becomes arithmetic rather than a model call.
That is what lets a speaker keep one label across a diarization window boundary,
and what lets a name learned on Tuesday still attach on Friday.

Model: WeSpeaker's CAM++ trained on VoxCeleb with large-margin fine-tuning.
29 MB, ONNX, CPU -- an order of magnitude smaller than anything else in this
pipeline and fast enough that it never becomes the reason a pass is slow.

Features have to match what the model was trained on or the embeddings are
quietly worse: 80-bin Kaldi fbank, 25 ms window, 10 ms shift, 16 kHz, mean
normalised over time. kaldi-native-fbank gives the exact Kaldi semantics in a
1 MB wheel; torchaudio's compliance module would too, at the price of importing
torch into a pipeline built to avoid it.
"""

from __future__ import annotations

import os
import sys
import wave

import numpy as np

SR = 16000
EMB_DIM = 512
# The export's training crop: 200 frames of 10 ms. Half-window hop so a short
# span still yields more than one vector to average.
WINDOW_FRAMES = 200
HOP_FRAMES = 100
MIN_FRAMES = 50
MODEL_FILE = "voxceleb_CAM++_LM.onnx"
HF_REPO = "Wespeaker/wespeaker-voxceleb-campplus-LM"


def find_model(explicit=None):
    store = os.path.join(os.environ.get("LOCALAPPDATA", ""), "VaultWares", "models")
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (explicit,
              os.environ.get("VW_CAMPPLUS"),
              os.path.join(store, "campplus-voxceleb-lm", MODEL_FILE),
              os.path.join(here, "models", "campplus-voxceleb-lm", MODEL_FILE),
              os.path.join(here, "..", "models", MODEL_FILE)):
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        f"{MODEL_FILE} not found. Fetch it with:\n"
        f"    hf download {HF_REPO} {MODEL_FILE} --local-dir "
        f"\"%LOCALAPPDATA%\\VaultWares\\models\\campplus-voxceleb-lm\"")


def read_wav(path, start=None, end=None):
    """16 kHz mono float32 from a wav, optionally just one span of it."""
    with wave.open(path, "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit")
        total = w.getnframes()
        first = int(max(0.0, start or 0.0) * SR)
        last = min(total, int(end * SR)) if end else total
        if last <= first:
            return np.zeros(0, dtype=np.float32)
        w.setpos(first)
        raw = w.readframes(last - first)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


class Embedder:
    """One resident ONNX session; embed as many spans as you like."""

    def __init__(self, model=None, threads=4):
        import onnxruntime as ort  # imported here so --help costs nothing
        self.path = find_model(model)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            self.path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    @staticmethod
    def fbank(audio):
        import kaldi_native_fbank as knf
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = float(SR)
        opts.frame_opts.frame_length_ms = 25.0
        opts.frame_opts.frame_shift_ms = 10.0
        # Dither adds noise for training robustness; at inference it only makes
        # the same audio embed differently twice.
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = True
        # Hamming, not Kaldi's default Povey window: WeSpeaker trains and runs
        # with window_type='hamming'. It is one line and it is not cosmetic --
        # with Povey the features differ from the reference by up to 7.4 and the
        # embeddings stop separating speakers at all (two different voices
        # scored 0.90 while two clips of the same voice scored 0.57). With
        # Hamming they match torchaudio's Kaldi fbank to 5e-4.
        opts.frame_opts.window_type = "hamming"
        opts.mel_opts.num_bins = 80
        fb = knf.OnlineFbank(opts)
        fb.accept_waveform(float(SR), (audio * 32768.0).tolist())
        fb.input_finished()
        frames = [fb.get_frame(i) for i in range(fb.num_frames_ready)]
        if not frames:
            return np.zeros((0, 80), dtype=np.float32)
        feats = np.asarray(frames, dtype=np.float32)
        # Cepstral mean normalisation over the segment, as WeSpeaker trains it:
        # removes the channel, keeps the voice.
        return feats - feats.mean(axis=0, keepdims=True)

    def embed(self, audio, min_seconds=0.5):
        """A unit-length embedding, or None if there is not enough audio.

        The window length is not a tuning knob -- it is the contract. This
        export was trained on 200-frame (2 s) crops and its pooling does not
        generalise to arbitrary lengths: handed a whole utterance it returns
        vectors that barely separate anyone. Measured over 24 clips from 6
        speakers, same-speaker against different-speaker cosine:

            whole utterance      0.458 / 0.403   (a gap of 0.055 -- useless)
            one 200-frame window 0.600 / 0.095
            averaged windows     0.803 / 0.122

        So the audio is cut into 200-frame windows, each embedded and
        normalised, and the mean is renormalised. Averaging also buys
        robustness: a window that lands on a cough or a breath is outvoted.

        Below about half a second the vector is dominated by whichever phonemes
        happened to be in it rather than by the speaker, so it is refused rather
        than returned as something that looks usable.
        """
        vectors = self.embed_windows(audio, min_seconds)
        if not len(vectors):
            return None
        mean = np.mean(vectors, axis=0)
        norm = np.linalg.norm(mean)
        return (mean / norm).astype(np.float32) if norm > 0 else None

    def embed_windows(self, audio, min_seconds=0.5):
        """The per-window vectors `embed` averages, as an (n, 512) array.

        Kept separately because the windows are the evidence and the mean is
        only the conclusion: the voice store records both, so a speaker whose
        centroid later looks wrong can be examined rather than guessed at.
        """
        if audio is None or len(audio) < int(min_seconds * SR):
            return np.zeros((0, EMB_DIM), dtype=np.float32)
        feats = self.fbank(audio)
        if feats.shape[0] < MIN_FRAMES:
            return np.zeros((0, EMB_DIM), dtype=np.float32)
        vectors = []
        for start in range(0, max(1, feats.shape[0] - WINDOW_FRAMES + 1), HOP_FRAMES):
            window = feats[start:start + WINDOW_FRAMES]
            if window.shape[0] < MIN_FRAMES:
                break
            out = self.session.run(None, {self.input: window[None, :, :]})[0][0]
            norm = np.linalg.norm(out)
            if norm > 0:
                vectors.append(out / norm)
        return (np.stack(vectors).astype(np.float32) if vectors
                else np.zeros((0, EMB_DIM), dtype=np.float32))

    def embed_file(self, path, start=None, end=None, min_seconds=0.5):
        return self.embed(read_wav(path, start, end), min_seconds)


def similarity(a, b):
    """Cosine similarity of two unit vectors: 1.0 identical, ~0 unrelated."""
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Embed spans of a 16 kHz mono wav with CAM++")
    ap.add_argument("wav")
    ap.add_argument("--span", action="append", default=[], metavar="START:END",
                    help="seconds, repeatable; omit to embed the whole file")
    ap.add_argument("--model")
    args = ap.parse_args()

    emb = Embedder(args.model)
    spans = [tuple(float(x) for x in s.split(":")) for s in args.span] or [(None, None)]
    vecs = [emb.embed_file(args.wav, s, e) for s, e in spans]
    for (s, e), v in zip(spans, vecs):
        label = "whole file" if s is None else f"{s:.2f}-{e:.2f}s"
        print(f"  {label:16} {'no embedding' if v is None else f'dim {len(v)}, |v|={np.linalg.norm(v):.3f}'}")
    if len(vecs) > 1:
        print("\n  cosine similarity")
        for i in range(len(vecs)):
            row = " ".join(f"{similarity(vecs[i], vecs[j]):6.3f}" for j in range(len(vecs)))
            print(f"   {i}: {row}")
