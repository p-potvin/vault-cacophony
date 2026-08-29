// nemotron.h — public C API for nvidia/nemotron-3.5-asr-streaming-0.6b ggml runtime
//
// Cache-aware streaming FastConformer + RNN-T (pure RNNT, no TDT durations).
// Multilingual ASR using the same encoder architecture as parakeet-tdt but with
// causal downsampling and cache-aware self-attention + depthwise conv modules
// for streaming inference.
//
// Models are loaded from GGUF files produced by:
//   python models/convert-nemotron-to-gguf.py --nemo X.nemo --output X.gguf

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nemotron_context;

struct nemotron_context_params {
    int n_threads;
    bool use_flash; // flash attention in encoder (default: false)
    int verbosity;  // 0=silent 1=normal 2=verbose
    bool use_gpu;   // false => force CPU backend
};

struct nemotron_context_params nemotron_context_default_params(void);

// Load model from GGUF (produced by convert-nemotron-to-gguf.py)
struct nemotron_context* nemotron_init_from_file(const char* path_model, struct nemotron_context_params params);

void nemotron_free(struct nemotron_context* ctx);

// ---- Per-token data ----

struct nemotron_token_data {
    int id;        // token id (0 .. vocab_size-1)
    char text[48]; // decoded text (SentencePiece '▁' converted to ' ')
    int64_t t0;    // start time, centiseconds
    int64_t t1;    // end time, centiseconds
    float p;       // softmax probability [0,1]
};

struct nemotron_word_data {
    char text[64]; // word text
    int64_t t0;    // start time, centiseconds
    int64_t t1;    // end time, centiseconds
    float p;       // mean probability
};

struct nemotron_result {
    char* text;                         // full transcript (malloc'd, caller owns)
    struct nemotron_token_data* tokens; // per-token timing (malloc'd)
    int n_tokens;
    struct nemotron_word_data* words; // grouped word timings (malloc'd)
    int n_words;
};

void nemotron_result_free(struct nemotron_result* r);

// Transcribe raw 16 kHz mono PCM, returning a malloc'd UTF-8 string.
char* nemotron_transcribe(struct nemotron_context* ctx, const float* samples, int n_samples);

// Like nemotron_transcribe but returns per-token timestamps.
struct nemotron_result* nemotron_transcribe_ex(struct nemotron_context* ctx, const float* samples, int n_samples,
                                               int64_t t_offset_cs);

// Set the streaming context preset (controls att_context_size).
// preset=0: [56,3]  (best accuracy, ~1.04s lookahead)
// preset=1: [56,0]  (no lookahead, lowest latency)
// preset=2: [56,6]  (moderate lookahead)
// preset=3: [56,13] (more lookahead)
// Default is preset 0.
void nemotron_set_context_preset(struct nemotron_context* ctx, int preset);

// Set the language for prompt features. Pass ISO-639-1 code (e.g. "en", "de").
// Default is "en" (prompt_id=0).
void nemotron_set_language(struct nemotron_context* ctx, const char* lang_code);

// Sampling controls
void nemotron_set_temperature(struct nemotron_context* ctx, float temperature, uint64_t seed);
void nemotron_set_beam_size(struct nemotron_context* ctx, int beam_size);

// MAES (Modified Adaptive Expansion Search) beam decoding. Requires
// beam_size > 1. enable=false disables (reverts to standard beam search).
void nemotron_set_maes(struct nemotron_context* ctx, bool enable, int num_steps, float gamma, int beta);

// Vocabulary helpers
int nemotron_n_vocab(struct nemotron_context* ctx);
int nemotron_blank_id(struct nemotron_context* ctx);
const char* nemotron_token_to_str(struct nemotron_context* ctx, int token_id);

// Hyper-parameters
int nemotron_frame_dur_cs(struct nemotron_context* ctx);
int nemotron_n_mels(struct nemotron_context* ctx);
int nemotron_sample_rate(struct nemotron_context* ctx);

// Per-token streaming callback. For RNN-T, fires once per emitted (non-blank) token.
typedef void (*nemotron_token_cb)(int tok_id, float prob, void* userdata);

// Like nemotron_transcribe() but fires cb for each non-blank token emitted by RNN-T.
// Uses greedy RNN-T decode regardless of beam_size setting.
void nemotron_transcribe_cb(struct nemotron_context* ctx, const float* samples, int n_samples, nemotron_token_cb cb,
                            void* userdata);

#ifdef __cplusplus
}
#endif
