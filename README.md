# vault-cacophony [![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/p-potvin/vault-cacophony)

Local speech analysis on consumer hardware. Point it at a video and it writes
subtitles beside the file, then keeps going: which language each line was spoken
in, who spoke it, and — as passes get added — whatever else can be read off the
audio.

Nothing leaves the machine and nothing imports PyTorch. Every model runs through
[audio.cpp](https://github.com/0xShug0/audio.cpp) on ggml/CUDA, plus one 29 MB
ONNX model on the CPU for speaker embeddings. The only network call in the whole
repo is the optional subtitle translation.

The long-term target is the thing the repo is named after: local speech-to-speech
models talking over each other, which needs every one of these passes running
live rather than as a batch job over a file.

Picking this up mid-stream: [HANDOFF.md](HANDOFF.md) has the current state, the
approaches that failed and why, and the gotchas that cost real time.
[ROADMAP.md](ROADMAP.md) has where it is going.

## The pipeline

```
   video/audio file
        |
   ffmpeg              44.1 kHz stereo for separation, 16 kHz mono for ASR
        |
   htdemucs            vocals stem, CUDA, ~10.6x realtime, 6-minute chunks
   (optional)          crossfaded back together
        |
   nemotron-3.5-asr    streaming mode, ~31x realtime, sub-word timings
        |               + the language it decoded each segment in
        |
   words_to_srt.py     cues from gaps between words --> <base>.srt
        |                                           --> <base>.tags.json
        |
   speakers.py         CAM++ embeddings of 2 s windows, clustered
   (-Speakers)         --> speaker track in the tag store
        |               --> one observation per speaker in the voice store,
        |                   named from it when a cluster matches
        |
   translate_srt.py    deep_translator --> <base>.<lang>.srt
   (-Langs)
```

Two files land beside the media: a plain `.srt` any player will pick up, and a
`.tags.json` holding everything that is not text.

## The stack

| Piece | What it is | Where it runs |
|---|---|---|
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | ggml runtime for NVIDIA speech models. Vendored as a submodule with prebuilt `audiocpp_cli.exe` and its CUDA DLLs. | CUDA |
| htdemucs | Music/voice separation, 81 MB f16 GGUF. Strips score and effects before the ASR sees the audio. | CUDA, ~0.8 GB VRAM |
| Nemotron-3.5-ASR-Streaming-0.6B | Default ASR. Per-token timings, and it announces the language per segment. | CUDA, ~5.5 GB VRAM |
| Parakeet-TDT-0.6B-v3 | Previous ASR, kept for the live path (~110x realtime, 90 ms/window). `-Engine parakeet`. | CUDA, ~2.5 GB VRAM |
| CAM++ (WeSpeaker, VoxCeleb LM) | 512-d speaker embeddings, 29 MB ONNX. | CPU, onnxruntime |
| kaldi-native-fbank | The 80-bin Kaldi filterbank CAM++ was trained on. 1 MB wheel. | CPU |
| Sortformer-Diar-4spk-v1 | Downloaded, not wired up yet. Real diarization: per-frame speaker activity, handles overlap. | CUDA |
| deep_translator | Subtitle translation. The only thing here that touches the network. | — |
| [CrispASR](https://github.com/p-potvin/CrispASR) | The older pure-ggml path, `Start-SubtitlesGgml.ps1`. Superseded — its htdemucs CUDA path aborts, leaving separation on a scalar CPU route at ~9x realtime. | — |

Hardware this is developed against: one RTX 3060 12 GB, Windows 11, Python 3.12.
A second card (RTX 2060) was pulled after it interfered with nsys, nvcc, CUDA,
PyTorch and WDDM alike; single-GPU is the configuration everything here is
measured on.

## Getting started

Clone with submodules — `audio.cpp` carries the prebuilt CLI:

```bash
git clone --recurse-submodules https://github.com/p-potvin/vault-cacophony
```

`ffmpeg` and `ffprobe` must be on `PATH`. Then the Python side, which is
deliberately small:

```bash
uv venv --python 3.12 && uv pip install numpy onnxruntime kaldi-native-fbank deep-translator
```

Fetch the models. The ASR models and htdemucs live under `audio.cpp/models`;
CAM++ lives in the shared VaultWares model store:

```bash
hf download nvidia/Nemotron-3.5-ASR-Streaming-0.6B-GGUF --local-dir audio.cpp/models/Nemotron-3.5-ASR-Streaming-0.6B-GGUF
```

```bash
hf download Wespeaker/wespeaker-voxceleb-campplus-LM voxceleb_CAM++_LM.onnx --local-dir "%LOCALAPPDATA%\VaultWares\models\campplus-voxceleb-lm"
```

`voiceprint.py` also accepts `--model`, or `VW_CAMPPLUS` pointing at the `.onnx`.

## Commands

Subtitle one file or a whole tree. `-TargetDir` takes either:

```powershell
.\scripts\Start-SubtitlesAudioCpp.ps1 -TargetDir "D:\Media\episode.mkv"
```

```powershell
.\scripts\Start-SubtitlesAudioCpp.ps1 -TargetDir "D:\Media" -Recurse -SkipExisting
```

Add the speaker pass, and translations:

```powershell
.\scripts\Start-SubtitlesAudioCpp.ps1 -TargetDir "D:\Media\episode.mkv" -Speakers -Langs "fr,es"
```

Skip separation when the audio is already clean dialogue — it is the expensive
stage:

```powershell
.\scripts\Start-SubtitlesAudioCpp.ps1 -TargetDir "D:\podcasts" -NoSeparate
```

Read a tag store, and ask what every track says at one moment:

```bash
.venv/Scripts/python.exe scripts/tagstore.py "D:\Media\episode.tags.json" --at 412.5
```

Run the speaker pass on its own, against a tag store that already has cues:

```bash
.venv/Scripts/python.exe scripts/speakers.py --audio episode.16k.wav --tags episode.tags.json
```

Teach it a name, then check that it sticks. The audio must be 16 kHz mono:

```bash
.venv/Scripts/python.exe scripts/voices.py enroll --name "Clopeux" --audio clopeux.16k.wav
```

```bash
.venv/Scripts/python.exe scripts/voices.py identify --audio unknown.16k.wav
```

```bash
.venv/Scripts/python.exe scripts/voices.py list
```

Everything the passes have recorded about one person, and everything recorded
about one file:

```bash
.venv/Scripts/python.exe scripts/voices.py show --name "Clopeux"
```

```bash
.venv/Scripts/python.exe scripts/voices.py observations --media episode.mkv
```

Fix a wrong attribution — point the observation at the right person, which
rebuilds that voice's centroid from its evidence:

```bash
.venv/Scripts/python.exe scripts/voices.py attribute --id 42 --name "Clopeux"
```

Describe a stretch of audio without identifying it — pitch, spread, loudness:

```bash
.venv/Scripts/python.exe scripts/prosody.py episode.16k.wav --span 30:90
```

Time every pass and capture a targeted CUDA trace beside the timing:

```powershell
.\scripts\Measure-Passes.ps1 -Audio "D:\Media\episode.mkv" -Seconds 30 -Repeat 2
```

Compare two stretches of a file directly, without the store in the way:

```bash
.venv/Scripts/python.exe scripts/voiceprint.py episode.16k.wav --span 12.0:20.0 --span 95.0:103.0
```

Make a 16 kHz mono wav out of anything:

```bash
ffmpeg -i input.m4a -vn -ac 1 -ar 16000 output.16k.wav
```

## What it writes

`<base>.srt` is ordinary SubRip. `<base>.tags.json` is the tag store: one file
per media file, holding the cues and one independent *track* per analysis pass.

```json
{
  "schema": 1,
  "media": {"path": "episode.mkv", "duration": 1356.85},
  "tracks": {
    "language": {"source": {"pass": "asr", "model": "nemotron-3.5-asr-streaming-0.6b"},
                 "spans": [{"start": 0.88, "end": 4.40, "value": "en-US"}]},
    "speaker":  {"source": {"pass": "speaker", "model": "campplus-voxceleb-lm"},
                 "spans": [{"start": 0.88, "end": 12.10, "value": "Clopeux"}]}
  },
  "cues": [{"start": 0.88, "end": 4.40, "text": "Mister Quilter is..."}]
}
```

Tracks are independent spans rather than fields on a cue, because the passes do
not agree on boundaries and never will: a speaker turn runs across cue breaks, a
language tag covers a sentence, a cue is sized to be read in the time it is on
screen. Whoever writes first would otherwise fix the segmentation for everyone
after. Consumers intersect instead, and a pass can run late, run again, or never
run at all without invalidating the file.

## The voice store

A tag store is about one file. The voice store is what carries identity between
files, so "speaker 0" in tonight's episode can be the same person as a name
learned last week. It is SQLite, permanent, and lives at
`%LOCALAPPDATA%\VaultWares\voices.db`.

Two tables, and the distinction between them is the whole design.

`voice` is the answer: one row per person, holding the centroid that naming is
done against. It is small, it is what `identify` reads, and it is derived.

`observation` is the evidence: one row per stretch of speech any pass has ever
attributed. Nothing is discarded, because the things worth asking of a corpus
are not knowable in advance — *which voices appear in more than one film, did
this person's pitch move between seasons, what did the pass get wrong and what
did the windows look like when it did* — and every one of them needs the
evidence rather than the summary. A centroid cannot be un-averaged.

Each observation keeps:

| | |
|---|---|
| `embedding` | the 512-d CAM++ vector for this stretch |
| `windows` | *every* 2 s window that fed it, float16, `(n, 512)` |
| `cohesion` | mean cosine of those windows to their own centroid — how much one person it is |
| `separation` | best cosine to any *other* speaker in the same file — how distinct it was from the competition it was actually judged against |
| `match_score` | what the voice store scored it at, named or not |
| `metrics` | pitch median/p10/p90, pitch spread in semitones, voiced ratio, RMS and crest in dB, words, characters, words per minute, cue count, languages, first line |
| `source` | model, pass, window and hop, both thresholds, how many speakers were in that file |
| | plus media, path, duration, span, and speech seconds |

Those last two columns are why this is stored rather than recomputed. A
centroid at 0.85 cohesion against a 0.10 nearest neighbour is a clean answer;
the same 0.85 against a 0.55 neighbour is a coin toss that happened to land, and
only the file it came from ever knew that.

Keeping the evidence also makes a mistake recoverable. Under a store that only
kept centroids, attributing a stretch to the wrong person permanently poisoned
that centroid. Here the link is a column:

```bash
.venv/Scripts/python.exe scripts/voices.py attribute --id 42 --name "Clopeux"
```

points the observation at the right voice and rebuilds both centroids from their
observations, weighted by how much audio each was built from. `forget` drops a
name and leaves its observations behind, unlinked — deleting the evidence with
the label would mean a slip costs the recordings too, and the evidence is the
expensive half.

An old `voices.json` beside the database is folded in once, automatically. Those
voices arrive with a centroid and no evidence, because the observations behind
them were never kept, and they are marked so that is visible rather than
surprising.

Cost: 8 observations with 174 window embeddings came to 244 KB. A feature film
with four speakers is a few MB.

## Numbers

Everything here is measured on this box rather than quoted from a model card.

**ASR.** Nemotron replaced Parakeet on word recovery, not on WER. Across two
LibriSpeech sets Parakeet scored 3.41% and 6.77% against Nemotron's 3.71% and
2.61% — but the second set is the tell: Parakeet returned 3998 of 4212 words and
Nemotron 4207. Parakeet-TDT predicts a duration per token and a bad one jumps the
decoder forward, so whole sentences vanish. Nemotron costs ~31x realtime against
Parakeet's ~110x, still thirty times faster than the audio.

**Separation.** ~10.6x realtime on CUDA. A 16-minute video costs ~90 s, against
~2.5 hours on CrispASR's CPU path and ~35 minutes for PyTorch demucs at
`shifts=1`. VRAM is flat at ~0.8 GB; host RAM grows at 0.28 GB per minute of
audio, which is why it runs in 6-minute chunks.

**Speaker embeddings.** Over 24 clips from 6 LibriSpeech speakers: 0.808 mean
cosine within a speaker, 0.124 between, one error in 276 pairs at a 0.52
threshold. On a two-speaker file, 98.1% of speech landed on the right person.

The two things that had to be right, neither obvious:

- **The window is a contract, not a knob.** The CAM++ export was trained on
  200-frame (2 s) crops and its pooling does not generalise. Handed a whole
  utterance: 0.458 same-speaker against 0.403 different — a gap of 0.055, which
  is useless. One 2 s window: 0.600 / 0.095. Averaged 2 s windows: 0.803 / 0.122.
  Same model, same audio, same code path; only the framing changed.
- **The window function is Hamming, not Kaldi's default Povey.** WeSpeaker
  trains with `window_type='hamming'`. With Povey the features differ from the
  reference by up to 7.4 and the embeddings stop separating speakers at all —
  two different voices scored 0.90 while two clips of the same voice scored 0.57.
  With Hamming they match torchaudio's Kaldi fbank to 5e-4.

## Profiling

`Measure-Passes.ps1` times each stage and puts an Nsight Systems trace beside
the timing, because wall clock answers "is it fast enough" and never answers
"why not". On the 113-second two-speaker file, RTX 3060:

| stage | wall | realtime | GPU memory | GPU kernels | launches |
|---|---|---|---|---|---|
| separation | 10.8 s | 10.4x | 886 MB | 418 ms (3.9% of wall) | 1 326 |
| ASR | 4.4–6.9 s | 16–26x | 1 521 MB | 107 ms (1.6% of wall) | 28 629 |
| speakers | 2.8 s | 40x | — (CPU) | — | — |

The traces say something the timings do not: **neither GPU stage is
compute-bound.** The ASR spends 107 ms of GPU kernel time inside a ~5 second
run — 28,629 launches averaging 3.7 µs each, against 218 ms of memcpy and
2.3 s of the host sitting in `cudaStreamSynchronize`. The GPU is idle almost
all of the time; what costs is model load, per-launch overhead, and host-device
copies. Separation is the opposite shape and the same conclusion: three
`conv2d_transpose` kernels account for 60% of its GPU time in 1,326 launches,
and the other 96% of the wall clock is elsewhere.

That matters for what to do next. Quantising further or reaching for a smaller
model would attack 1.6% of the runtime.

Traces are kept small on purpose, three ways: a bounded clip (`-Seconds`, since
trace volume scales with work done, not wall time), `--sample=none
--cpuctxsw=none` to drop CPU sampling and context-switch tracking, and
`--trace=cuda` alone. Measured that way a full-file ASR trace is 4.3 MB and a
separation trace 0.3 MB, and both still resolve every kernel launch. A 30-second
clip halves that again.

Two things the script has to do to be honest, both learned the hard way. Timing
and tracing are separate runs — under nsys the stage is measurably slower, so a
profiled run's clock is not worth reporting. And `nsys stats` caches a `.sqlite`
beside the report and, finding a stale one, refuses the command rather than
re-exporting: the second run of the script silently lost every GPU number while
still printing a confident wall clock. The export is forced now.

The speaker pass runs CAM++ on the CPU through onnxruntime, so it is timed and
reported without a trace rather than given an empty one.

## Why kaldi-native-fbank

CAM++ does not take audio. It takes an 80-bin log-mel filterbank computed exactly
the way Kaldi computes one — 25 ms window, 10 ms shift, Hamming window, no
dither, snip-edges, then cepstral mean normalisation over the segment. That is
what its training data looked like, and a model is only as good as the match
between the features it sees now and the features it saw then. Get any of it
wrong and nothing errors; the embeddings just quietly stop separating people,
which is the failure mode above.

Kaldi is the old C++ speech toolkit whose feature extraction became the de facto
standard — WeSpeaker, and most of the speaker-ID world, trains on it. Two ways to
reproduce it from Python: `torchaudio.compliance.kaldi`, which drags PyTorch back
into a pipeline built to avoid it, or `kaldi-native-fbank`, a 1 MB wheel that is
the same C++ lifted out of Kaldi. Same numbers, none of the weight. That is the
whole reason it is in here.

## Layout

```
scripts/
  Start-SubtitlesAudioCpp.ps1   the pipeline; everything else is called from here
  Start-SubtitlesGgml.ps1       the superseded CrispASR path
  Measure-Passes.ps1            per-stage timings + targeted nsys traces
  words_to_srt.py               word timings --> readable cues, + the tag store
  translate_srt.py              sentence-aware batched translation
  tagstore.py                   the per-media tag store (also a CLI: inspect one)
  voiceprint.py                 CAM++ embeddings (also a CLI: compare two spans)
  prosody.py                    pitch, spread, loudness -- description, not identity
  voices.py                     the voice store: voices, observations, repair
  speakers.py                   the speaker pass: window, cluster, name, record
audio.cpp/                      submodule: ggml runtime, prebuilt CLI, models
CrispASR/                       submodule: the older ASR path
NeMo-Speech.cpp/                submodule: upstream reference
cacophony-ui/                   Vite front-end, early
samples/                        audio fixtures — gitignored, 364 MB, local only
```

`samples/` is not in the repo: it is 364 MB and includes a commercial track, and
this repo is public. Regenerate derived clips from a source with ffmpeg.

## Next

**Sortformer** replaces the speaker pass's weakest part. Clustering fixed 2 s
windows locates a speaker change to within one hop, and cannot represent two
people talking at once at all — Sortformer predicts per-frame, per-speaker
activity and does both. Its labels are local to a 20-second window, which is
exactly what the voice store is for: linking them into one identity across a
film. The GGUF is already sitting in `audio.cpp/models`.

**Per-word confidence** never reaches the voice store: the ASR emits it, the cue
builder drops it, and the pipeline deletes the words JSON on its way out. It is
the one thing the models offer that is currently thrown away.

**The live path**, which is the point of the name — see [ROADMAP.md](ROADMAP.md).
