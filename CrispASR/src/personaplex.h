// personaplex.h — C API for NVIDIA PersonaPlex (personaplex-7b-v1).
//
// Full-duplex speech-to-speech on the Moshi architecture: a 4096d/32L temporal
// transformer over 17 delayed streams (1 text + 8 own audio + 8 other audio),
// plus a 1024d/6L depformer with per-step weights that fills the 16 audio
// codebooks each frame. Mimi codec at 24 kHz / 12.5 Hz (1920 samples/frame).
//
// See docs/personaplex/BLUEPRINT.md for the hyperparameters, the full tensor
// map, and the traps this implementation is written against.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct personaplex_context;

struct personaplex_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;  // false => force CPU backend
    float temperature;      // audio streams; 0 = greedy argmax
    float temperature_text; // text stream, sampled separately upstream
    // KV cache depth in frames. The checkpoint's context is 3000 (240 s at
    // 12.5 Hz), which costs 32L x 3000 x 4096 x 2 x 2 B ~= 1.6 GB. 750 frames
    // is 60 s of conversation for ~400 MB. 0 = use the model's own context.
    int context_frames;
};

struct personaplex_context_params personaplex_context_default_params(void);

struct personaplex_context* personaplex_init_from_file(const char* path_model,
                                                       struct personaplex_context_params params);

void personaplex_free(struct personaplex_context* ctx);

// Geometry, for callers and for the load-time acceptance test.
struct personaplex_model_info {
    int dim;
    int num_layers;
    int num_heads;
    int ffn_hidden;
    int depformer_dim;
    int depformer_num_layers;
    int depformer_num_heads;
    int depformer_ffn_hidden;
    int n_q;    // audio streams total (own + other)
    int dep_q;  // codebooks the depformer predicts
    int card;   // audio codebook size
    int text_card;
    int context;
    int frame_size; // samples per frame at the model's sample rate
    int n_tensors_loaded;
};

void personaplex_model_info_get(const struct personaplex_context* ctx, struct personaplex_model_info* out);

// Per-stream delay pattern, length n_q + 1 (text first). Borrowed pointer,
// owned by the context.
const int32_t* personaplex_delays(const struct personaplex_context* ctx, int* out_n);

#ifdef __cplusplus
}
#endif
