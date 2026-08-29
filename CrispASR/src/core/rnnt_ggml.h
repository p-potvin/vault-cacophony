// src/core/rnnt_ggml.h — RNNT/TDT transducer decode as ggml graphs (§232).
//
// Shared by every NeMo-style transducer backend (parakeet, nemotron, …) whose
// predictor is a 2-layer LSTM and whose joint is pred/enc projections + ReLU +
// output projection. The default runtimes step these on the CPU via
// cblas_sgemv, leaving the GPU idle (the §232 P100 decode bottleneck). These
// helpers run the SAME math as ggml graphs on the sched's backend so the whole
// per-step decode executes on the GPU (opt-in per backend).
//
// Correctness-first: per-step sched dispatch, LSTM state carried via CPU
// readback. Transcript-validated identical to the cblas path on M1 (parakeet,
// jfk + multispeaker, CPU & Metal). The perf win is P100-only — Apple Accelerate
// cblas is already fast, so M1 is neutral (see LEARNINGS 29-30). A persistent-
// graph / in-graph-argmax variant is the follow-up if per-step is launch-bound.
//
// LSTM recurrence (PyTorch convention, concatenated gate order [i,f,g,o]):
//   gates = w_ih·x + b_ih + w_hh·h + b_hh
//   c' = sigmoid(f)·c + sigmoid(i)·tanh(g);   h' = sigmoid(o)·tanh(c')
// Joint (ReLU — NOT tanh):
//   mid = pred_w·pred_u + pred_b;   logits = out_w·relu(proj_e + mid) + out_b
#pragma once

#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace core_rnnt_ggml {

// One LSTM layer as ggml ops. x/h_in/c_in are [H]; returns h' ([H]), writes c'.
static inline ggml_tensor* lstm_layer(ggml_context* c0, ggml_tensor* x, ggml_tensor* w_ih, ggml_tensor* b_ih,
                                      ggml_tensor* w_hh, ggml_tensor* b_hh, ggml_tensor* h_in, ggml_tensor* c_in, int H,
                                      ggml_tensor** c_out) {
    ggml_tensor* g = ggml_add(c0, ggml_mul_mat(c0, w_ih, x), b_ih);
    g = ggml_add(c0, g, ggml_add(c0, ggml_mul_mat(c0, w_hh, h_in), b_hh));
    const size_t fs = sizeof(float);
    ggml_tensor* i_ = ggml_sigmoid(c0, ggml_view_1d(c0, g, H, 0 * (size_t)H * fs));
    ggml_tensor* f_ = ggml_sigmoid(c0, ggml_view_1d(c0, g, H, 1 * (size_t)H * fs));
    ggml_tensor* g_ = ggml_tanh(c0, ggml_view_1d(c0, g, H, 2 * (size_t)H * fs));
    ggml_tensor* o_ = ggml_sigmoid(c0, ggml_view_1d(c0, g, H, 3 * (size_t)H * fs));
    ggml_tensor* c_new = ggml_add(c0, ggml_mul(c0, f_, c_in), ggml_mul(c0, i_, g_)); // f·c + i·g
    ggml_tensor* h_new = ggml_mul(c0, o_, ggml_tanh(c0, c_new));                     // o·tanh(c')
    *c_out = c_new;
    return h_new;
}

// One predictor step on the sched's backend. Reads/writes the CPU LSTM state
// (h0/c0/h1/c1, each [H]); pred_out (= top-layer hidden) is filled on return.
// All weight tensors are backend-resident (any dtype: F16/F32/quantized).
static inline void predictor_step(ggml_backend_sched_t sched, ggml_tensor* embed_w, ggml_tensor* l0_wih,
                                  ggml_tensor* l0_bih, ggml_tensor* l0_whh, ggml_tensor* l0_bhh, ggml_tensor* l1_wih,
                                  ggml_tensor* l1_bih, ggml_tensor* l1_whh, ggml_tensor* l1_bhh, int token_id, int H,
                                  std::vector<float>& h0, std::vector<float>& c0, std::vector<float>& h1,
                                  std::vector<float>& c1, std::vector<float>& pred_out) {
    const size_t mem = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    ggml_init_params ip = {mem, nullptr, true};
    ggml_context* cx = ggml_init(ip);

    ggml_tensor* tok = ggml_new_tensor_1d(cx, GGML_TYPE_I32, 1);
    ggml_set_input(tok);
    ggml_tensor* h0i = ggml_new_tensor_1d(cx, GGML_TYPE_F32, H);
    ggml_tensor* c0i = ggml_new_tensor_1d(cx, GGML_TYPE_F32, H);
    ggml_tensor* h1i = ggml_new_tensor_1d(cx, GGML_TYPE_F32, H);
    ggml_tensor* c1i = ggml_new_tensor_1d(cx, GGML_TYPE_F32, H);
    for (ggml_tensor* t : {h0i, c0i, h1i, c1i})
        ggml_set_input(t);

    ggml_tensor* emb = ggml_reshape_1d(cx, ggml_get_rows(cx, embed_w, tok), H);
    if (emb->type != GGML_TYPE_F32)
        emb = ggml_cast(cx, emb, GGML_TYPE_F32);

    ggml_tensor *c0o, *c1o;
    ggml_tensor* h0o = lstm_layer(cx, emb, l0_wih, l0_bih, l0_whh, l0_bhh, h0i, c0i, H, &c0o);
    ggml_tensor* h1o = lstm_layer(cx, h0o, l1_wih, l1_bih, l1_whh, l1_bhh, h1i, c1i, H, &c1o);
    for (ggml_tensor* t : {h0o, c0o, h1o, c1o})
        ggml_set_output(t);

    ggml_cgraph* gf = ggml_new_graph(cx);
    for (ggml_tensor* t : {h0o, c0o, h1o, c1o})
        ggml_build_forward_expand(gf, t);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    int32_t tid = token_id;
    ggml_backend_tensor_set(tok, &tid, 0, sizeof(int32_t));
    ggml_backend_tensor_set(h0i, h0.data(), 0, H * sizeof(float));
    ggml_backend_tensor_set(c0i, c0.data(), 0, H * sizeof(float));
    ggml_backend_tensor_set(h1i, h1.data(), 0, H * sizeof(float));
    ggml_backend_tensor_set(c1i, c1.data(), 0, H * sizeof(float));
    ggml_backend_sched_graph_compute(sched, gf);

    h0.resize(H);
    c0.resize(H);
    h1.resize(H);
    c1.resize(H);
    ggml_backend_tensor_get(h0o, h0.data(), 0, H * sizeof(float));
    ggml_backend_tensor_get(c0o, c0.data(), 0, H * sizeof(float));
    ggml_backend_tensor_get(h1o, h1.data(), 0, H * sizeof(float));
    ggml_backend_tensor_get(c1o, c1.data(), 0, H * sizeof(float));
    pred_out = h1;
    ggml_free(cx);
}

// One joint step: logits = out_w·relu(proj_e + pred_w·pred_u + pred_b) + out_b.
// proj_e is [Jh] (precomputed encoder projection), pred_u is [H]. Vocab size is
// read from out_w->ne[1]. Weight tensors are backend-resident.
static inline void joint_step(ggml_backend_sched_t sched, ggml_tensor* pred_w, ggml_tensor* pred_b, ggml_tensor* out_w,
                              ggml_tensor* out_b, const float* proj_e, const float* pred_u, int Jh, int H,
                              std::vector<float>& logits) {
    const int Vt = (int)out_w->ne[1];
    const size_t mem = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    ggml_init_params ip = {mem, nullptr, true};
    ggml_context* cx = ggml_init(ip);

    ggml_tensor* pe = ggml_new_tensor_1d(cx, GGML_TYPE_F32, Jh);
    ggml_set_input(pe);
    ggml_tensor* pu = ggml_new_tensor_1d(cx, GGML_TYPE_F32, H);
    ggml_set_input(pu);

    ggml_tensor* mid = ggml_add(cx, ggml_mul_mat(cx, pred_w, pu), pred_b); // pred_w·pred_u + pred_b
    mid = ggml_relu(cx, ggml_add(cx, mid, pe));                            // relu(proj_e + mid)
    ggml_tensor* lg = ggml_add(cx, ggml_mul_mat(cx, out_w, mid), out_b);   // out_w·mid + out_b
    ggml_set_output(lg);

    ggml_cgraph* gf = ggml_new_graph(cx);
    ggml_build_forward_expand(gf, lg);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(pe, proj_e, 0, Jh * sizeof(float));
    ggml_backend_tensor_set(pu, pred_u, 0, H * sizeof(float));
    ggml_backend_sched_graph_compute(sched, gf);
    logits.resize(Vt);
    ggml_backend_tensor_get(lg, logits.data(), 0, Vt * sizeof(float));
    ggml_free(cx);
}

// ── Persistent-graph decoder ───────────────────────────────────────────────
// The step_ functions above rebuild the graph + realloc every step; that fixed
// per-step cost dominates long decodes (nemotron: ggml 2x SLOWER than cblas on
// M1 — LEARNINGS 31). Decoder builds the predictor + joint graphs ONCE, gallocr-
// allocates each ONCE on `backend`, and dispatches sched-free per step
// (tensor_set inputs → ggml_backend_graph_compute → read). RAII; non-copyable.
//
// §234 gotcha: gallocr may alias input slots with intermediates, so ALL inputs
// are re-set before every compute (they are — state + token/proj each step).
struct Decoder {
    ggml_backend_t backend = nullptr;
    int H = 0, Jh = 0, Vt = 0;
    // predictor persistent graph
    ggml_context* pctx = nullptr;
    ggml_cgraph* pgf = nullptr;
    ggml_gallocr_t palloc = nullptr;
    ggml_tensor *p_tok = nullptr, *p_h0i = nullptr, *p_c0i = nullptr, *p_h1i = nullptr, *p_c1i = nullptr;
    ggml_tensor *p_h0o = nullptr, *p_c0o = nullptr, *p_h1o = nullptr, *p_c1o = nullptr;
    // joint persistent graph
    ggml_context* jctx = nullptr;
    ggml_cgraph* jgf = nullptr;
    ggml_gallocr_t jalloc = nullptr;
    ggml_tensor *j_pe = nullptr, *j_pu = nullptr, *j_lg = nullptr;

    Decoder() = default;
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    ~Decoder() {
        if (palloc)
            ggml_gallocr_free(palloc);
        if (pctx)
            ggml_free(pctx);
        if (jalloc)
            ggml_gallocr_free(jalloc);
        if (jctx)
            ggml_free(jctx);
    }
    bool active() const { return pgf != nullptr; }
};

// Build both persistent graphs. Returns true on success (Decoder::active()).
static inline bool decoder_init(Decoder& d, ggml_backend_t backend, ggml_tensor* embed_w, ggml_tensor* l0_wih,
                                ggml_tensor* l0_bih, ggml_tensor* l0_whh, ggml_tensor* l0_bhh, ggml_tensor* l1_wih,
                                ggml_tensor* l1_bih, ggml_tensor* l1_whh, ggml_tensor* l1_bhh, ggml_tensor* pred_w,
                                ggml_tensor* pred_b, ggml_tensor* out_w, ggml_tensor* out_b, int H, int Jh) {
    d.backend = backend;
    d.H = H;
    d.Jh = Jh;
    d.Vt = (int)out_w->ne[1];
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // predictor graph
    d.pctx = ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true});
    d.p_tok = ggml_new_tensor_1d(d.pctx, GGML_TYPE_I32, 1);
    d.p_h0i = ggml_new_tensor_1d(d.pctx, GGML_TYPE_F32, H);
    d.p_c0i = ggml_new_tensor_1d(d.pctx, GGML_TYPE_F32, H);
    d.p_h1i = ggml_new_tensor_1d(d.pctx, GGML_TYPE_F32, H);
    d.p_c1i = ggml_new_tensor_1d(d.pctx, GGML_TYPE_F32, H);
    for (ggml_tensor* t : {d.p_tok, d.p_h0i, d.p_c0i, d.p_h1i, d.p_c1i})
        ggml_set_input(t);
    ggml_tensor* emb = ggml_reshape_1d(d.pctx, ggml_get_rows(d.pctx, embed_w, d.p_tok), H);
    if (emb->type != GGML_TYPE_F32)
        emb = ggml_cast(d.pctx, emb, GGML_TYPE_F32);
    d.p_h0o = lstm_layer(d.pctx, emb, l0_wih, l0_bih, l0_whh, l0_bhh, d.p_h0i, d.p_c0i, H, &d.p_c0o);
    d.p_h1o = lstm_layer(d.pctx, d.p_h0o, l1_wih, l1_bih, l1_whh, l1_bhh, d.p_h1i, d.p_c1i, H, &d.p_c1o);
    for (ggml_tensor* t : {d.p_h0o, d.p_c0o, d.p_h1o, d.p_c1o})
        ggml_set_output(t);
    d.pgf = ggml_new_graph(d.pctx);
    for (ggml_tensor* t : {d.p_h0o, d.p_c0o, d.p_h1o, d.p_c1o})
        ggml_build_forward_expand(d.pgf, t);
    d.palloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(d.palloc, d.pgf))
        return false;

    // joint graph
    d.jctx = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true});
    d.j_pe = ggml_new_tensor_1d(d.jctx, GGML_TYPE_F32, Jh);
    d.j_pu = ggml_new_tensor_1d(d.jctx, GGML_TYPE_F32, H);
    ggml_set_input(d.j_pe);
    ggml_set_input(d.j_pu);
    ggml_tensor* mid = ggml_add(d.jctx, ggml_mul_mat(d.jctx, pred_w, d.j_pu), pred_b);
    mid = ggml_relu(d.jctx, ggml_add(d.jctx, mid, d.j_pe));
    d.j_lg = ggml_add(d.jctx, ggml_mul_mat(d.jctx, out_w, mid), out_b);
    ggml_set_output(d.j_lg);
    d.jgf = ggml_new_graph(d.jctx);
    ggml_build_forward_expand(d.jgf, d.j_lg);
    d.jalloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(d.jalloc, d.jgf))
        return false;
    return true;
}

// One predictor step on the persistent graph. Reads/writes CPU state; fills pred_out.
static inline void decoder_predictor(Decoder& d, int token_id, std::vector<float>& h0, std::vector<float>& c0,
                                     std::vector<float>& h1, std::vector<float>& c1, std::vector<float>& pred_out) {
    const size_t nb = (size_t)d.H * sizeof(float);
    int32_t tid = token_id;
    ggml_backend_tensor_set(d.p_tok, &tid, 0, sizeof(int32_t));
    ggml_backend_tensor_set(d.p_h0i, h0.data(), 0, nb);
    ggml_backend_tensor_set(d.p_c0i, c0.data(), 0, nb);
    ggml_backend_tensor_set(d.p_h1i, h1.data(), 0, nb);
    ggml_backend_tensor_set(d.p_c1i, c1.data(), 0, nb);
    ggml_backend_graph_compute(d.backend, d.pgf);
    h0.resize(d.H);
    c0.resize(d.H);
    h1.resize(d.H);
    c1.resize(d.H);
    ggml_backend_tensor_get(d.p_h0o, h0.data(), 0, nb);
    ggml_backend_tensor_get(d.p_c0o, c0.data(), 0, nb);
    ggml_backend_tensor_get(d.p_h1o, h1.data(), 0, nb);
    ggml_backend_tensor_get(d.p_c1o, c1.data(), 0, nb);
    pred_out = h1;
}

// One joint step on the persistent graph.
static inline void decoder_joint(Decoder& d, const float* proj_e, const float* pred_u, std::vector<float>& logits) {
    ggml_backend_tensor_set(d.j_pe, proj_e, 0, (size_t)d.Jh * sizeof(float));
    ggml_backend_tensor_set(d.j_pu, pred_u, 0, (size_t)d.H * sizeof(float));
    ggml_backend_graph_compute(d.backend, d.jgf);
    logits.resize(d.Vt);
    ggml_backend_tensor_get(d.j_lg, logits.data(), 0, (size_t)d.Vt * sizeof(float));
}

} // namespace core_rnnt_ggml
