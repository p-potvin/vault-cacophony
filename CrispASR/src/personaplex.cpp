// personaplex.cpp — NVIDIA PersonaPlex runtime (personaplex-7b-v1).
//
// Moshi-architecture full-duplex S2S. Two transformers:
//
//   temporal  4096d, 32L, 32 heads, SwiGLU hidden 11264, RoPE, RMSNorm,
//             causal, context 3000. One step per 12.5 Hz frame over 17
//             delayed streams: 1 text + 8 own audio + 8 other audio.
//   depformer 1024d, 6L, 16 heads, SwiGLU hidden 2816, no positional
//             embedding, context 8. Runs dep_q=16 sequential steps per frame,
//             filling the audio codebooks from the temporal hidden state.
//
// The temporal layer shape is identical to kyutai_stt's `lm_layer` — Kyutai
// STT is the same architecture family — so the tensor names, the packed QKV
// and the SwiGLU pair are shared with that backend by construction (see
// models/convert-personaplex-to-gguf.py).
//
// Reference: moshi (MIT, Kyutai) as adapted by NVIDIA in the personaplex repo.
// See docs/personaplex/BLUEPRINT.md.

#include "personaplex.h"

#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (#214)

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct personaplex_hparams {
    // Temporal transformer
    int dim = 4096;
    int num_layers = 32;
    int num_heads = 32;
    int context = 3000;
    float max_period = 10000.0f;
    float hidden_scale = 4.125f;
    // Written explicitly by the converter rather than re-derived. The rule is
    // hidden = (2 * dim_feedforward) // 3, with a separate (21 * dim) // 8
    // branch when dim_feedforward == 4 * dim (moshi/modules/gating.py). A port
    // that assumes the 4x case mis-sizes every FFN, so this is not recomputed
    // here — BLUEPRINT.md trap 1.
    int ffn_hidden = 11264;

    // Depformer
    int depformer_dim = 1024;
    int depformer_num_layers = 6;
    int depformer_num_heads = 16;
    int depformer_context = 8;
    int depformer_ffn_hidden = 2816;

    // Streams
    int n_q = 16;   // audio streams: 8 own + 8 other
    int dep_q = 16; // codebooks the depformer predicts
    int card = 2048;
    int text_card = 32000;
    int existing_text_padding_id = 3;

    // Mimi
    int sample_rate = 24000;
    float frame_rate = 12.5f;
    int frame_size = 1920;

    // Derived
    int head_dim = 128;           // dim / num_heads
    int depformer_head_dim = 64;  // depformer_dim / depformer_num_heads
};

// --- Temporal transformer layer. Field-for-field the kyutai_stt lm_layer. ---
struct pplx_layer {
    ggml_tensor* norm1_alpha = nullptr;  // RMSNorm, [1,1,dim]
    ggml_tensor* attn_qkv_w = nullptr;   // packed QKV, [dim, 3*dim]
    ggml_tensor* attn_out_w = nullptr;   // [dim, dim]
    ggml_tensor* norm2_alpha = nullptr;  // RMSNorm
    ggml_tensor* gating_in_w = nullptr;  // SwiGLU, [dim, 2*ffn_hidden]
    ggml_tensor* gating_out_w = nullptr; // [ffn_hidden, dim]
};

// --- Depformer layer.
//
// Asymmetric per-step layout, and getting it backwards yields a model that
// loads cleanly and generates noise (BLUEPRINT.md trap 2):
//   attention  per-step but PACKED into one tensor — qkv is
//              [3 * depformer_dim * dep_q, depformer_dim], out is
//              [depformer_dim * dep_q, depformer_dim]. Slice by step.
//   gating     per-step as SEPARATE tensors — dep_q of each, per layer.
//   norms      SHARED across steps — one alpha per layer, not dep_q.
struct pplx_dep_layer {
    ggml_tensor* norm1_alpha = nullptr; // shared across steps
    ggml_tensor* norm2_alpha = nullptr; // shared across steps
    ggml_tensor* attn_qkv_w = nullptr;  // packed over steps
    ggml_tensor* attn_out_w = nullptr;  // packed over steps
    std::vector<ggml_tensor*> gating_in_w;  // [dep_q]
    std::vector<ggml_tensor*> gating_out_w; // [dep_q]
};

struct personaplex_model {
    personaplex_hparams hp;

    // Temporal
    std::vector<pplx_layer> layers;
    ggml_tensor* out_norm_alpha = nullptr;

    // Embeddings and heads
    std::vector<ggml_tensor*> audio_emb; // [n_q]   each [card+1, dim]
    ggml_tensor* text_emb = nullptr;     // [text_card+1, dim]
    ggml_tensor* text_linear_w = nullptr;// [text_card, dim]

    // Depformer
    std::vector<pplx_dep_layer> dep_layers;
    std::vector<ggml_tensor*> depformer_in;  // [dep_q]   each [depformer_dim, dim]
    std::vector<ggml_tensor*> depformer_emb; // [dep_q-1] each [card+1, depformer_dim]
    ggml_tensor* depformer_text_emb = nullptr;
    std::vector<ggml_tensor*> linears;       // [dep_q]   each [card, depformer_dim]

    std::vector<int32_t> delays; // n_q + 1, text first
    std::vector<std::string> vocab;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int n_tensors_loaded = 0;
};

struct personaplex_context {
    personaplex_context_params params;
    personaplex_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
};

struct personaplex_context_params personaplex_context_default_params(void) {
    struct personaplex_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.temperature = 0.8f;      // moshi LMGen default
    p.temperature_text = 0.7f; // moshi LMGen default
    p.context_frames = 0;      // 0 = the model's own context
    return p;
}

// ===========================================================================
// Load
// ===========================================================================

struct personaplex_context* personaplex_init_from_file(const char* path_model,
                                                       struct personaplex_context_params params) {
    if (!path_model || !*path_model)
        return nullptr;

    auto* pctx = new personaplex_context();
    pctx->params = params;
    auto& m = pctx->model;
    auto& hp = m.hp;

    // ---- pass 1: hparams + delays + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path_model);
        if (!gctx) {
            fprintf(stderr, "personaplex: failed to open '%s'\n", path_model);
            delete pctx;
            return nullptr;
        }

        hp.dim = core_gguf::kv_u32(gctx, "personaplex.dim", hp.dim);
        hp.num_layers = core_gguf::kv_u32(gctx, "personaplex.num_layers", hp.num_layers);
        hp.num_heads = core_gguf::kv_u32(gctx, "personaplex.num_heads", hp.num_heads);
        hp.context = core_gguf::kv_u32(gctx, "personaplex.context", hp.context);
        hp.max_period = core_gguf::kv_f32(gctx, "personaplex.max_period", hp.max_period);
        hp.hidden_scale = core_gguf::kv_f32(gctx, "personaplex.hidden_scale", hp.hidden_scale);
        hp.ffn_hidden = core_gguf::kv_u32(gctx, "personaplex.ffn_hidden", hp.ffn_hidden);

        hp.depformer_dim = core_gguf::kv_u32(gctx, "personaplex.depformer_dim", hp.depformer_dim);
        hp.depformer_num_layers =
            core_gguf::kv_u32(gctx, "personaplex.depformer_num_layers", hp.depformer_num_layers);
        hp.depformer_num_heads =
            core_gguf::kv_u32(gctx, "personaplex.depformer_num_heads", hp.depformer_num_heads);
        hp.depformer_context = core_gguf::kv_u32(gctx, "personaplex.depformer_context", hp.depformer_context);
        hp.depformer_ffn_hidden =
            core_gguf::kv_u32(gctx, "personaplex.depformer_ffn_hidden", hp.depformer_ffn_hidden);

        hp.n_q = core_gguf::kv_u32(gctx, "personaplex.n_q", hp.n_q);
        hp.dep_q = core_gguf::kv_u32(gctx, "personaplex.dep_q", hp.dep_q);
        hp.card = core_gguf::kv_u32(gctx, "personaplex.card", hp.card);
        hp.text_card = core_gguf::kv_u32(gctx, "personaplex.text_card", hp.text_card);
        hp.existing_text_padding_id =
            core_gguf::kv_u32(gctx, "personaplex.existing_text_padding_id", hp.existing_text_padding_id);

        hp.sample_rate = core_gguf::kv_u32(gctx, "personaplex.mimi.sample_rate", hp.sample_rate);
        hp.frame_rate = core_gguf::kv_f32(gctx, "personaplex.mimi.frame_rate", hp.frame_rate);
        hp.frame_size = core_gguf::kv_u32(gctx, "personaplex.mimi.frame_size", hp.frame_size);

        hp.head_dim = hp.dim / hp.num_heads;
        hp.depformer_head_dim = hp.depformer_dim / hp.depformer_num_heads;

        // Delay pattern: [0, 0,1x7, 0,1x7] — text, then own audio, then the
        // user's stream. The zero at index 1 + n_q/2 is where the second
        // stream starts; it is the full-duplex mechanism, not a typo.
        const int delay_key = gguf_find_key(gctx, "personaplex.delays");
        if (delay_key >= 0) {
            const int n = gguf_get_arr_n(gctx, delay_key);
            m.delays.resize(n);
            const void* data = gguf_get_arr_data(gctx, delay_key);
            for (int i = 0; i < n; i++)
                m.delays[i] = ((const int32_t*)data)[i];
        }
        if ((int)m.delays.size() != hp.n_q + 1) {
            fprintf(stderr, "personaplex: expected %d delays (1 text + %d audio), got %d\n", hp.n_q + 1, hp.n_q,
                    (int)m.delays.size());
            gguf_free(gctx);
            delete pctx;
            return nullptr;
        }

        m.vocab.resize(hp.text_card);
        const int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (tok_key >= 0) {
            const int n = gguf_get_arr_n(gctx, tok_key);
            for (int i = 0; i < n && i < (int)m.vocab.size(); i++) {
                const char* s = gguf_get_arr_str(gctx, tok_key, i);
                if (s)
                    m.vocab[i] = s;
            }
        }

        gguf_free(gctx);
    }

    const int n_threads = params.n_threads > 0 ? params.n_threads : 4;
    pctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : ggml_backend_cpu_init();
    if (!pctx->backend)
        pctx->backend = ggml_backend_cpu_init();
    pctx->backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(pctx->backend_cpu, n_threads);
    if (ggml_backend_is_cpu(pctx->backend))
        ggml_backend_cpu_set_n_threads(pctx->backend, n_threads);

    // ---- pass 2: tensors ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, pctx->backend, "personaplex", wl)) {
        fprintf(stderr, "personaplex: failed to load weights from '%s'\n", path_model);
        delete pctx;
        return nullptr;
    }
    m.ctx = wl.ctx;
    m.buf = wl.buf;
    auto& ts = wl.tensors;

    bool ok = true;
    auto get = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        if (it == ts.end()) {
            fprintf(stderr, "personaplex: tensor '%s' not found\n", name.c_str());
            ok = false;
            return nullptr;
        }
        m.n_tensors_loaded++;
        return it->second;
    };

    char buf[128];

    // --- temporal transformer ---
    m.layers.resize(hp.num_layers);
    for (int i = 0; i < hp.num_layers; i++) {
        auto& L = m.layers[i];
        snprintf(buf, sizeof(buf), "lm.transformer.layers.%d.norm1.alpha", i);
        L.norm1_alpha = get(buf);
        snprintf(buf, sizeof(buf), "lm.transformer.layers.%d.attn.qkv_w", i);
        L.attn_qkv_w = get(buf);
        snprintf(buf, sizeof(buf), "lm.transformer.layers.%d.attn.out_w", i);
        L.attn_out_w = get(buf);
        snprintf(buf, sizeof(buf), "lm.transformer.layers.%d.norm2.alpha", i);
        L.norm2_alpha = get(buf);
        snprintf(buf, sizeof(buf), "lm.transformer.layers.%d.gating_in_w", i);
        L.gating_in_w = get(buf);
        snprintf(buf, sizeof(buf), "lm.transformer.layers.%d.gating_out_w", i);
        L.gating_out_w = get(buf);
    }
    m.out_norm_alpha = get("lm.out_norm.alpha");

    // --- embeddings and the text head ---
    m.audio_emb.resize(hp.n_q);
    for (int i = 0; i < hp.n_q; i++) {
        snprintf(buf, sizeof(buf), "lm.emb.%d.weight", i);
        m.audio_emb[i] = get(buf);
    }
    m.text_emb = get("lm.text_emb.weight");
    // existing_text_padding_id is set, so extra_text == 0 and this is
    // [text_card, dim] rather than [text_card + 1, dim].
    m.text_linear_w = get("lm.text_linear.weight");

    // --- depformer ---
    m.dep_layers.resize(hp.depformer_num_layers);
    for (int i = 0; i < hp.depformer_num_layers; i++) {
        auto& L = m.dep_layers[i];
        snprintf(buf, sizeof(buf), "lm.depformer.layers.%d.norm1.alpha", i);
        L.norm1_alpha = get(buf);
        snprintf(buf, sizeof(buf), "lm.depformer.layers.%d.norm2.alpha", i);
        L.norm2_alpha = get(buf);
        snprintf(buf, sizeof(buf), "lm.depformer.layers.%d.attn.qkv_w", i);
        L.attn_qkv_w = get(buf);
        snprintf(buf, sizeof(buf), "lm.depformer.layers.%d.attn.out_w", i);
        L.attn_out_w = get(buf);
        L.gating_in_w.resize(hp.dep_q);
        L.gating_out_w.resize(hp.dep_q);
        for (int s = 0; s < hp.dep_q; s++) {
            snprintf(buf, sizeof(buf), "lm.depformer.layers.%d.gating.%d.in_w", i, s);
            L.gating_in_w[s] = get(buf);
            snprintf(buf, sizeof(buf), "lm.depformer.layers.%d.gating.%d.out_w", i, s);
            L.gating_out_w[s] = get(buf);
        }
    }

    m.depformer_in.resize(hp.dep_q);
    m.linears.resize(hp.dep_q);
    for (int s = 0; s < hp.dep_q; s++) {
        snprintf(buf, sizeof(buf), "lm.depformer_in.%d.weight", s);
        m.depformer_in[s] = get(buf);
        snprintf(buf, sizeof(buf), "lm.linears.%d.weight", s);
        m.linears[s] = get(buf);
    }
    // dep_q - 1: the last codebook is never fed back as a depformer input.
    m.depformer_emb.resize(hp.dep_q - 1);
    for (int s = 0; s < hp.dep_q - 1; s++) {
        snprintf(buf, sizeof(buf), "lm.depformer_emb.%d.weight", s);
        m.depformer_emb[s] = get(buf);
    }
    m.depformer_text_emb = get("lm.depformer_text_emb.weight");

    if (!ok) {
        fprintf(stderr, "personaplex: model is missing required tensors\n");
        personaplex_free(pctx);
        return nullptr;
    }

    // Shape assertions against the blueprint. These are cheap and they catch a
    // converter regression at load rather than as noise 3 phases later.
    auto want = [&](ggml_tensor* t, const char* name, int64_t d0, int64_t d1) {
        if (!t)
            return;
        if (t->ne[0] != d0 || t->ne[1] != d1) {
            fprintf(stderr, "personaplex: %s has shape [%lld,%lld], expected [%lld,%lld]\n", name,
                    (long long)t->ne[0], (long long)t->ne[1], (long long)d0, (long long)d1);
            ok = false;
        }
    };
    want(m.layers[0].attn_qkv_w, "temporal qkv", hp.dim, 3LL * hp.dim);
    want(m.layers[0].gating_in_w, "temporal gating_in", hp.dim, 2LL * hp.ffn_hidden);
    want(m.layers[0].gating_out_w, "temporal gating_out", hp.ffn_hidden, hp.dim);
    // Packed over steps — this is the assertion that catches trap 2.
    want(m.dep_layers[0].attn_qkv_w, "depformer qkv", hp.depformer_dim, 3LL * hp.depformer_dim * hp.dep_q);
    want(m.dep_layers[0].attn_out_w, "depformer out", hp.depformer_dim, (int64_t)hp.depformer_dim * hp.dep_q);
    want(m.dep_layers[0].gating_in_w[0], "depformer gating_in", hp.depformer_dim, 2LL * hp.depformer_ffn_hidden);
    want(m.linears[0], "depformer head", hp.depformer_dim, hp.card);
    want(m.text_linear_w, "text head", hp.dim, hp.text_card);

    if (!ok) {
        fprintf(stderr, "personaplex: geometry does not match the expected model\n");
        personaplex_free(pctx);
        return nullptr;
    }

    if (params.verbosity > 0) {
        fprintf(stderr,
                "personaplex: %dd/%dL/%dH temporal + %dd/%dL/%dH depformer, "
                "n_q=%d dep_q=%d card=%d, %d tensors\n",
                hp.dim, hp.num_layers, hp.num_heads, hp.depformer_dim, hp.depformer_num_layers,
                hp.depformer_num_heads, hp.n_q, hp.dep_q, hp.card, m.n_tensors_loaded);
    }

    return pctx;
}

void personaplex_free(struct personaplex_context* ctx) {
    if (!ctx)
        return;
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

void personaplex_model_info_get(const struct personaplex_context* ctx, struct personaplex_model_info* out) {
    if (!ctx || !out)
        return;
    const auto& hp = ctx->model.hp;
    out->dim = hp.dim;
    out->num_layers = hp.num_layers;
    out->num_heads = hp.num_heads;
    out->ffn_hidden = hp.ffn_hidden;
    out->depformer_dim = hp.depformer_dim;
    out->depformer_num_layers = hp.depformer_num_layers;
    out->depformer_num_heads = hp.depformer_num_heads;
    out->depformer_ffn_hidden = hp.depformer_ffn_hidden;
    out->n_q = hp.n_q;
    out->dep_q = hp.dep_q;
    out->card = hp.card;
    out->text_card = hp.text_card;
    out->context = hp.context;
    out->frame_size = hp.frame_size;
    out->n_tensors_loaded = ctx->model.n_tensors_loaded;
}

const int32_t* personaplex_delays(const struct personaplex_context* ctx, int* out_n) {
    if (!ctx)
        return nullptr;
    if (out_n)
        *out_n = (int)ctx->model.delays.size();
    return ctx->model.delays.data();
}
