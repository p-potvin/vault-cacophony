# VSS on one card: the summarization profile, ggml-shaped

Sun, 13 Sep 2026

NVIDIA's [Video Search and Summarization blueprint](../video-search-and-summarization/)
is copied into the repo root. This note is what came of trying to run it
here, and what was built instead: the same pipeline, one script, GGUF models
through Ollama, outputs beside the file. Every number is measured on this box
(RTX 3060 12 GB, Windows 11) with `nemo-speech.exe` from another session
holding 7.8 GB of the card at the time, so the timings are under contention.

## The stock stack does not fit, on three counts

The blueprint's `vss-build-vision-ai` skill routes every request to one of
four developer profiles (`base`, `alerts`, `lvs`, `search`) or the warehouse
profile, all Docker Compose, all built on NIMs:

| What the profile needs | What is here |
|---|---|
| Cosmos3 Nano Reasoner BF16 for rt-vlm, about 20 GB; FP8 to share a card | 12 GB total |
| Nemotron 3.5 Lightning 30B-A3B as the LLM NIM, 45 GB observed at INT4, 66 GB minimum for the BF16 profile | 12 GB total |
| NGC entitlement to pull `nvcr.io` NIM images, NVIDIA AI Enterprise for local hosting | none |
| Linux Docker with the NVIDIA container toolkit; `HARDWARE_PROFILE` in H100 / GB300 / L40S / RTXPRO4500BW / RTXPRO6000BW | Docker Desktop on Windows |

The sizing reference is explicit that nothing may be silently substituted
(`references/sizing.md`: "Never silently substitute a smaller model, lower
precision, or remote endpoint"), so the skill itself would stop at the
preflight. Its edge recipe (DGX Spark, Thor) is the smallest supported shape
and still assumes 128 GB of unified memory.

There is no PyTorch route worth taking either. The blueprint's rt-vlm is vLLM
with seven patches applied at image build time and Cosmos3 Nano is served
from an NGC-only checkpoint; the public Cosmos-Reason models are 7B and 8B
and would fit only at 4-bit, at which point they are GGUFs anyway.

## What was built

[`scripts/vss_pipeline.py`](../scripts/vss_pipeline.py), driven by
[`scripts/Start-VideoSummary.ps1`](../scripts/Start-VideoSummary.ps1), with
prompts and defaults in [`config/vss/lvs-local.yaml`](../config/vss/lvs-local.yaml).

| VSS service | Here |
|---|---|
| VIOS (upload, storage, clip URLs) | the file on disk; `ffprobe` for duration and streams |
| rt-vlm chunking: `chunk_duration`, `chunk_overlap_duration`, fixed frames per chunk | `plan_chunks` and `frame_times`, `ffmpeg -ss` per frame, JPEG at 512 px |
| rt-vlm VLM inference, Cosmos3 Nano | Ollama `/api/chat` with images, `qwen3-vl:2b` by default |
| LVS `_create_vlm_prompt`: role, instruction, constraints, structured output, `{scenario}` `{events}` `{objects_of_interest}` | the same four sections, same template variables, in the YAML |
| LVS `ENABLE_AUDIO` (its own ASR into the prompt) | `--enable-audio` folds the `.srt` from `Start-SubtitlesAudioCpp.ps1` into each chunk |
| CA-RAG `caption_summarization` and `summary_aggregation` prompts, `max_events_per_batch: 50` | the same prompts verbatim, same batch size, then one JSON-schema pass for `{video_summary, events}` |
| LLM NIM | Ollama `llama3.2:3b` |
| `DenseCaptionSerializer` JSON lines | `<base>.captions.jsonl` |
| `POST /v1/summarize` response, content as a JSON string | `<base>.summary.json`, plus `.summary.md` |
| Elasticsearch over captions, the search leg | BM25 in the script over every `.captions.jsonl` under a folder |
| VLM Q&A (`chat_completion`) | `ask`: BM25 picks the chunks, their frames go back to the VLM with the question |
| Kafka, Logstash, Redis, HAProxy, Phoenix, the agent and its UI | nothing; the outputs are files beside the media |

Request vocabulary is VSS's so a caller written against LVS reads the same:
`--scenario`, `--event` (repeatable), `--object-of-interest`,
`--chunk-duration`, `--chunk-overlap-duration`, `--num-frames-per-chunk`,
`--enable-audio`. Defaults are the HITL defaults from the operations skill
(`activity monitoring`, `notable activity`).

The tag store gets two tracks, `caption` and `event`, next to `language` and
`speaker`, which is the point of putting this here: one file answers "what
was on screen while speaker 2 was talking French".

## Measured

Two clips cut from a 166 s screen recording of a browser session, 950x640 at 30 fps.

| | 30 s clip | 166 s clip |
|---|---|---|
| chunks (10 s, 8 frames) | 3 | 17 |
| captioning, qwen3-vl:2b | 19.3 s | 124.1 s |
| per chunk | 4.4 to 8.6 s | 2.1 to 9.0 s |
| captioning rate | 1.6x realtime | 1.3x realtime |
| aggregation, llama3.2:3b | 16.4 s (1 batch + final, captions reused) | 97 s (2 batches + merge + final) |
| end to end | 35.7 s | 221 s |
| events in the summary | 3 | 20 |
| `ask`, one question, 6 frames | | 1.4 s after retrieval |

Prompt cost: 8 frames at 512 px on qwen3-vl:2b through Ollama is 8952 prompt
tokens, about 1100 per frame. VRAM with both models resident: 3.5 GB for the
VLM at 16k context and 2 GB for llama3.2:3b.

## What broke on the way, so it is not re-discovered

Each of these cost a full run.

**`num_ctx: 8192` rejects every chunk.** Ollama answers
`exceed_context_size_error` at 8952 tokens. 16k is the floor for 8 frames.

**`think: false` plus a JSON schema empties `content`.** On qwen3-vl:2b the
JSON arrives in `message.thinking` and `content` is `""`. The client reads
whichever is non-empty. Without `think: false` the same request thinks for
1800 tokens first, so it stays off for schema calls.

**Free text from qwen3-vl:2b is its deliberation.** With `think: false` it
writes a literal `<think>` block into `content`; with thinking on it writes
"Got it, let's look at the frames..." into `content` instead, either way
running into `num_predict` before an answer. So every call in the script is
schema-constrained: events for the captions and the two merge passes,
`{video_summary, events}` for the final pass, `{answer, evidence_times}` for
`ask`. The same question that ran out of tokens at 1024 then answered in 106
tokens and 1.4 s, with the timestamps it took the answer from.

**A 2B VLM shown a static screen loops.** Without `num_predict` it repeated
"The screen is static and does not change." until the context was full: 7423
tokens, 72 s per chunk, and a JSON array that never closed. `num_predict:
1024`, `repeat_penalty: 1.1`, and a parser that closes a truncated array
around the objects that did complete. Per chunk went from 72 s to 4 to 9 s.

**The VLM as its own summarizer invents.** On the 30 s clip the final pass
described typing "google" into a search bar and opening tabs, none of which is
in any caption. llama3.2:3b on the 166 s clip stayed inside the captions. So
the LLM defaults to llama3.2:3b, at the cost of a second resident model.

**Ollama serialises.** Two callers on the same server queue behind each
other; there is no per-file parallelism to be had from running two wrappers.

## Models

Pulled and working: `qwen3-vl:2b` (Q4_K_M, 1.9 GB, vision). Also pulled,
untried here: `gemma4:e2b-it-qat` (4.6B, vision), `minicpm-v4.6:1b`.

The NVIDIA-native option is Cosmos-Reason2-2B, which is the Qwen3-VL
architecture and has GGUF builds with an mmproj
([apolo13x/Cosmos-Reason2-2B-GGUF](https://hf.co/apolo13x/Cosmos-Reason2-2B-GGUF));
Cosmos-Reason2-8B likewise
([mradermacher/Cosmos-Reason2-8B-i1-GGUF](https://hf.co/mradermacher/Cosmos-Reason2-8B-i1-GGUF)),
about 5 GB at Q4_K_M. Either loads into Ollama with a two-line Modelfile
(`FROM` the weights, `FROM` the mmproj) and then `--vlm-model` selects it.
Not pulled: the card was full with another session's server.

## Not built

- Live RTSP, the alerts profile, the object-detection and 3D tracking legs,
  and the embeddings search (`rt-embed`). Search here is lexical over
  captions, which is the part of VSS's search that runs without a second
  vision model.
- The `vss` CLI and the operations skills assume a running LVS server on an
  HTTP origin. Nothing here listens; wrapping the script in the LVS REST
  surface (`/v1/summarize`, `/v1/ready`, `/models`) is the step that would
  let those skills drive it unchanged.
- No `Sortformer` or speaker fusion in the prompt yet; `--enable-audio` hands
  the VLM the cue text only.
