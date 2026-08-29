// src/parakeet_orchestrate.cpp — shared parakeet transcription orchestration
// (improvements Phase 1). Faithful hoist of the CLI backend adapter's
// transcribe() path selection + long-audio + segmentation into the library so
// the session C-ABI can call the SAME code. Byte-identical to the pre-hoist
// adapter path (verified by the surface-parity harness + CLI A/B).

#include "parakeet_orchestrate.h"

#include "core/asr_segment_group.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// parakeet_result → one neutral segment (mirrors the adapter's result_to_segment).
parakeet_seg result_to_seg(const parakeet_result* r, int64_t fallback_t0_cs) {
    parakeet_seg seg;
    seg.t0 = fallback_t0_cs;
    seg.t1 = fallback_t0_cs;
    seg.text = r->text ? r->text : "";

    seg.words.reserve(r->n_words);
    for (int i = 0; i < r->n_words; i++) {
        const auto& w = r->words[i];
        parakeet_seg::word cw;
        cw.text = w.text;
        cw.t0 = w.t0;
        cw.t1 = w.t1;
        cw.p = w.p;
        seg.words.push_back(std::move(cw));
    }
    seg.tokens.reserve(r->n_tokens);
    for (int i = 0; i < r->n_tokens; i++) {
        const auto& t = r->tokens[i];
        parakeet_seg::token ct;
        ct.text = t.text;
        ct.id = t.id;
        ct.t0 = t.t0;
        ct.t1 = t.t1;
        ct.p = t.p;
        seg.tokens.push_back(std::move(ct));
    }
    if (!seg.words.empty()) {
        seg.t0 = seg.words.front().t0;
        seg.t1 = seg.words.back().t1;
    } else if (!seg.tokens.empty()) {
        seg.t0 = seg.tokens.front().t0;
        seg.t1 = seg.tokens.back().t1;
    }
    return seg;
}

// Split ONE coherent decode into ~seg_seconds segments (mirrors the adapter's
// split_result_into_segments; issue #257).
std::vector<parakeet_seg> split_result_to_segs(const parakeet_result* r, int64_t fallback_t0_cs, int seg_seconds) {
    std::vector<parakeet_seg> out;
    if (!r)
        return out;
    if (r->n_words <= 0) {
        parakeet_seg seg = result_to_seg(r, fallback_t0_cs);
        if (!seg.text.empty() || !seg.tokens.empty())
            out.push_back(std::move(seg));
        return out;
    }
    std::vector<int64_t> word_t0;
    word_t0.reserve(r->n_words);
    for (int i = 0; i < r->n_words; i++)
        word_t0.push_back(r->words[i].t0);
    const std::vector<int> starts = core_segment::group_by_window(word_t0, (int64_t)seg_seconds * 100);

    int ti = 0;
    for (size_t si = 0; si < starts.size(); si++) {
        const int w_begin = starts[si];
        const int w_end = (si + 1 < starts.size()) ? starts[si + 1] : r->n_words;
        parakeet_seg seg;
        std::string text;
        for (int wi = w_begin; wi < w_end; wi++) {
            const auto& w = r->words[wi];
            if (!text.empty())
                text += ' ';
            text += w.text;
            parakeet_seg::word cw;
            cw.text = w.text;
            cw.t0 = w.t0;
            cw.t1 = w.t1;
            cw.p = w.p;
            seg.words.push_back(std::move(cw));
        }
        seg.text = std::move(text);
        seg.t0 = seg.words.front().t0;
        seg.t1 = seg.words.back().t1;
        const int64_t next_start = (w_end < r->n_words) ? r->words[w_end].t0 : INT64_MAX;
        for (; ti < r->n_tokens && r->tokens[ti].t0 < next_start; ti++) {
            const auto& t = r->tokens[ti];
            parakeet_seg::token ct;
            ct.text = t.text;
            ct.id = t.id;
            ct.t0 = t.t0;
            ct.t1 = t.t1;
            ct.p = t.p;
            seg.tokens.push_back(std::move(ct));
        }
        out.push_back(std::move(seg));
    }
    for (; ti < r->n_tokens && !out.empty(); ti++) {
        const auto& t = r->tokens[ti];
        parakeet_seg::token ct;
        ct.text = t.text;
        ct.id = t.id;
        ct.t0 = t.t0;
        ct.t1 = t.t1;
        ct.p = t.p;
        out.back().tokens.push_back(std::move(ct));
    }
    return out;
}

// Lowest-RMS 100 ms frame in [target-window, target] (mirrors the adapter's
// find_silence_cut) — keep chunk boundaries off mid-word.
int find_silence_cut(const float* s, int n, int target, int window, int sr) {
    const int lo = std::max(1, target - window);
    const int hi = std::min(n - 1, target);
    if (hi <= lo)
        return std::min(std::max(target, 1), n);
    const int win = std::max(1, sr / 10);
    double best = 1e30;
    int best_pos = target;
    for (int c = lo; c <= hi; c += win / 2) {
        const int a = std::max(0, c - win / 2);
        const int b = std::min(n, c + win / 2);
        double e = 0.0;
        for (int i = a; i < b; i++)
            e += (double)s[i] * (double)s[i];
        e /= std::max(1, b - a);
        if (e < best) {
            best = e;
            best_pos = c;
        }
    }
    return best_pos;
}

// Silence-split single-pass longform (mirrors the adapter's transcribe_longform).
std::vector<parakeet_seg> transcribe_longform(parakeet_context* ctx, const float* samples, int n_samples,
                                              int64_t t_offset_cs, int cap_samples) {
    std::vector<parakeet_seg> out;
    const int SR = 16000;
    const int search = 5 * SR;
    const int ctxs = 2 * SR;
    int pos = 0;
    while (pos < n_samples) {
        int end;
        if (n_samples - pos <= cap_samples) {
            end = n_samples;
        } else {
            end = find_silence_cut(samples, n_samples, pos + cap_samples, search, SR);
            if (end <= pos)
                end = std::min(n_samples, pos + cap_samples);
        }
        const int ext_s = std::max(0, pos - ctxs);
        const int ext_e = std::min(n_samples, end + ctxs);
        const int64_t ext_t0 = t_offset_cs + (int64_t)((double)ext_s / SR * 100.0);
        const int64_t left_cs = (pos == 0) ? INT64_MIN : t_offset_cs + (int64_t)((double)pos / SR * 100.0);
        const int64_t right_cs = (end == n_samples) ? INT64_MAX : t_offset_cs + (int64_t)((double)end / SR * 100.0);

        parakeet_result* r = parakeet_transcribe_ex(ctx, samples + ext_s, ext_e - ext_s, ext_t0);
        if (r) {
            parakeet_seg full = result_to_seg(r, ext_t0);
            parakeet_result_free(r);

            parakeet_seg seg;
            seg.t0 = left_cs == INT64_MIN ? full.t0 : left_cs;
            seg.t1 = seg.t0;
            std::string text;
            for (auto& w : full.words) {
                if (w.t0 >= left_cs && w.t0 < right_cs) {
                    if (!text.empty())
                        text += ' ';
                    text += w.text;
                    seg.words.push_back(std::move(w));
                }
            }
            for (auto& tk : full.tokens) {
                if (tk.t0 >= left_cs && tk.t0 < right_cs)
                    seg.tokens.push_back(std::move(tk));
            }
            seg.text = std::move(text);
            if (!seg.words.empty()) {
                seg.t0 = seg.words.front().t0;
                seg.t1 = seg.words.back().t1;
            }
            if (!seg.text.empty() || !seg.words.empty())
                out.push_back(std::move(seg));
        }
        pos = end;
    }
    return out;
}

} // namespace

std::vector<parakeet_seg> parakeet_transcribe_segments(parakeet_context* ctx, const float* samples, int n_samples,
                                                       int64_t t_offset_cs, bool is_ja,
                                                       const parakeet_orchestrate_opts& opts) {
    std::vector<parakeet_seg> out;
    if (!ctx || !samples || n_samples <= 0)
        return out;

    const int SR = 16000;

    // Issue #257: explicit --chunk-seconds (non-JA) → coherent quality-window
    // decode grouped into ~N-second segments (see parakeet_orchestrate.h).
    if (!is_ja && opts.chunk_seconds_explicit && opts.chunk_seconds > 0) {
        const int seg_seconds = std::max(2, opts.chunk_seconds);
        int enc_window = 0;
        if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_CHUNK"))
            enc_window = std::max(2, atoi(e));
        const int ov = std::max(0, (int)(opts.chunk_overlap_seconds + 0.5f));
        parakeet_result* rc = parakeet_transcribe_streamed(ctx, samples, n_samples, t_offset_cs, enc_window, ov);
        if (!rc)
            return out;
        out = split_result_to_segs(rc, t_offset_cs, seg_seconds);
        parakeet_result_free(rc);
        return out;
    }

    int stream_threshold_s = is_ja ? 12 : 300;
    bool longform_enabled = !is_ja;
    int stream_chunk_s = 0;
    int stream_overlap_s = 2;
    if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_THRESHOLD"))
        stream_threshold_s = std::max(0, atoi(e));
    if (const char* e = getenv("CRISPASR_PARAKEET_LONGFORM"))
        longform_enabled = atoi(e) != 0;
    if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_CHUNK"))
        stream_chunk_s = std::max(2, atoi(e));
    if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_OVERLAP"))
        stream_overlap_s = std::max(0, atoi(e));

    parakeet_strategy_in sin;
    sin.n_samples = n_samples;
    sin.sample_rate = SR;
    sin.is_ja = is_ja;
    sin.chunk_seconds_explicit = opts.chunk_seconds_explicit;
    sin.chunk_seconds = opts.chunk_seconds;
    sin.stream_threshold_s = stream_threshold_s;
    sin.longform_enabled = longform_enabled;
    parakeet_strategy strat = parakeet_pick_strategy(sin);

    // Phase 2: proactive encoder memory policy. When single-pass is chosen but
    // its estimated O(T^2) rel-pos bias would exceed a user-set VRAM budget,
    // switch to the streamed (bounded-window) encoder BEFORE allocating —
    // instead of allocate → OOM → reactive fallback. Opt-in: default budget 0
    // = disabled → single-pass as before (the reactive fallback still backstops).
    //   CRISPASR_PARAKEET_MEM_POLICY = auto (default) | off | single | streamed
    //   CRISPASR_PARAKEET_VRAM_BUDGET_MB : budget (MiB); 0/unset = disabled
    //   CRISPASR_PARAKEET_MEM_COEFF : O(T^2) estimate coefficient (default 8.0)
    if (strat == parakeet_strategy::SINGLE_PASS) {
        const char* pol = getenv("CRISPASR_PARAKEET_MEM_POLICY");
        const bool mode_off = pol && strcmp(pol, "off") == 0;
        const bool mode_force_single = pol && strcmp(pol, "single") == 0;
        const bool mode_force_streamed = pol && strcmp(pol, "streamed") == 0;
        if (mode_force_streamed) {
            strat = parakeet_strategy::STREAMED;
        } else if (!mode_off && !mode_force_single) {
            double budget = 0.0, coeff = 8.0;
            if (const char* e = getenv("CRISPASR_PARAKEET_VRAM_BUDGET_MB"))
                budget = atof(e);
            if (const char* e = getenv("CRISPASR_PARAKEET_MEM_COEFF"))
                coeff = atof(e);
            const int T_enc = parakeet_est_enc_frames(ctx, n_samples);
            const int H = parakeet_n_heads(ctx);
            if (!parakeet_singlepass_fits_budget(T_enc, H, budget, coeff)) {
                if (!opts.no_prints)
                    fprintf(stderr,
                            "crispasr[parakeet]: single-pass est %.0f MiB > budget %.0f MiB (T=%d, H=%d); "
                            "using streamed encoding (set CRISPASR_PARAKEET_MEM_POLICY=single to force)\n",
                            parakeet_est_singlepass_peak_mb(T_enc, H, coeff), budget, T_enc, H);
                strat = parakeet_strategy::STREAMED;
            }
        }
    }

    if (strat == parakeet_strategy::LONGFORM)
        return transcribe_longform(ctx, samples, n_samples, t_offset_cs, stream_threshold_s * SR);

    parakeet_result* r = nullptr;
    if (strat == parakeet_strategy::SINGLE_PASS) {
        // Issue #257: single-pass full attention is O(T^2); a VRAM-limited GPU
        // can fail the encode alloc → null → empty transcript. Fall back to the
        // streamed (bounded-window) encoder. Simulate with
        // CRISPASR_PARAKEET_SIMULATE_ENCODE_OOM=1.
        const bool simulate_oom = getenv("CRISPASR_PARAKEET_SIMULATE_ENCODE_OOM") != nullptr;
        r = simulate_oom ? nullptr : parakeet_transcribe_ex(ctx, samples, n_samples, t_offset_cs);
        if (!r) {
            if (!opts.no_prints)
                fprintf(stderr,
                        "crispasr[parakeet]: single-pass encode failed (likely VRAM OOM at %.0fs); "
                        "falling back to streamed encoding — pass --chunk-seconds N for segmented "
                        "output or --att-context L,R for bounded-memory single-pass\n",
                        (double)n_samples / SR);
            r = parakeet_transcribe_streamed(ctx, samples, n_samples, t_offset_cs, stream_chunk_s, stream_overlap_s);
        }
    } else { // STREAMED
        r = parakeet_transcribe_streamed(ctx, samples, n_samples, t_offset_cs, stream_chunk_s, stream_overlap_s);
    }
    if (!r)
        return out;
    out.push_back(result_to_seg(r, t_offset_cs));
    parakeet_result_free(r);
    return out;
}
