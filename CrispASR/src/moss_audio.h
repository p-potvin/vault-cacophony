// moss_audio.h — public C API for MOSS-Audio-4B-Instruct ggml runtime
//
// Audio understanding (ASR + audio QA + scene description) using a 32-layer
// Whisper-style encoder with DeepStack 3-tap cross-layer injection +
// 36-layer Qwen3 LLM. Models loaded from GGUF files produced by:
//   `python models/convert-moss-audio-to-gguf.py --input <hf_dir> --output X.gguf`
//
// Architecture: OpenMOSS-Team/MOSS-Audio-4B-Instruct (Apache-2.0)
//   Audio encoder: 128-mel → 3×Conv2d(stride 2) → stem_proj → 32 WhisperEncoderLayer
//   DeepStack: taps at L8/L16/L24 → 3× GatedMLP → residual inject at LM L0/L1/L2
//   Audio adapter: GatedMLP(1280→8192→2560) for final encoder output
//   LM: 36-layer Qwen3 (2560d, 32Q/8KV, head_dim=128, QK-norm, SwiGLU, RoPE θ=1M)

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct moss_audio_context;

struct moss_audio_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    bool flash_attn;
};

struct moss_audio_context_params moss_audio_context_default_params(void);

// Load model from GGUF.
struct moss_audio_context* moss_audio_init_from_file(const char* path_model, struct moss_audio_context_params params);

void moss_audio_free(struct moss_audio_context* ctx);

// Transcribe / understand audio. prompt is the text instruction (e.g.
// "Transcribe this audio." or "Describe the sounds in this clip.").
// Returns malloc'd UTF-8 string (caller owns).
char* moss_audio_process(struct moss_audio_context* ctx, const float* samples, int n_samples, const char* prompt);

// Convenience wrapper: transcribe with default prompt.
char* moss_audio_transcribe(struct moss_audio_context* ctx, const float* samples, int n_samples);

// ---- Stage helpers for differential testing ----

// Compute 128-bin log-mel spectrogram (Whisper-style).
// Output: malloc'd (n_mels, T_mel) F32 row-major. Caller frees.
float* moss_audio_compute_mel(struct moss_audio_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                              int* out_T_mel);

// Run audio encoder only. Returns (T_enc, d_model=1280) F32 row-major.
// Also fills deepstack taps if ds_tap_0/1/2 are non-null (each T_enc × 1280).
float* moss_audio_run_encoder(struct moss_audio_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                              int* out_d, float** ds_tap_0, float** ds_tap_1, float** ds_tap_2);

// Run audio adapter on encoder output. Returns (T_enc, llm_dim=2560) F32.
float* moss_audio_run_adapter(struct moss_audio_context* ctx, const float* encoder_out, int T_enc, int d_enc,
                              int* out_T, int* out_d);

// Embed tokens. Returns (n_tokens, llm_dim) F32.
float* moss_audio_embed_tokens(struct moss_audio_context* ctx, const int32_t* token_ids, int n_tokens);

// Initialize / reset KV cache for LLM decode.
bool moss_audio_kv_init(struct moss_audio_context* ctx, int max_ctx);
void moss_audio_kv_reset(struct moss_audio_context* ctx);

// Run LLM with KV cache. Returns logits (vocab_size,) F32 for last token.
float* moss_audio_run_llm_kv(struct moss_audio_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                             int* out_n_tokens, int* out_vocab_size);

// Tokenize text using BPE.
int moss_audio_tokenize(struct moss_audio_context* ctx, const char* text, int32_t* out_tokens, int max_tokens);

// Token ID → string.
const char* moss_audio_token_text(struct moss_audio_context* ctx, int token_id);

// Seed for sampling.
void moss_audio_set_seed(struct moss_audio_context* ctx, uint32_t seed);

// Beam search. 1 = greedy (default). >1 = beam search via
// core_beam_decode replay-from-prefix (§167g).
void moss_audio_set_beam_size(struct moss_audio_context* ctx, int beam_size);

// Per-token streaming callback. Fires once per generated token (id, softmax prob, userdata).
typedef void (*moss_audio_token_cb)(int tok_id, float prob, void* userdata);

// Like moss_audio_process() but fires cb(tok_id, prob, userdata) for each generated token.
// The final assembled text is NOT returned; all output is via the callback.
// Falls back to greedy decode (beam_size ignored).
void moss_audio_process_cb(struct moss_audio_context* ctx, const float* samples, int n_samples, const char* prompt,
                           moss_audio_token_cb cb, void* userdata);

#ifdef __cplusplus
}
#endif
