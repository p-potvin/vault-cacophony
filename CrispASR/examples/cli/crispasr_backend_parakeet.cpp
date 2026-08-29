// crispasr_backend_parakeet.cpp — adapter for nvidia/parakeet-tdt-0.6b-v3.
//
// Wraps parakeet_init_from_file + parakeet_transcribe_ex and converts the
// native parakeet_result into a std::vector<crispasr_segment>. One segment
// per transcribe() call, with word-level data attached (parakeet emits word
// timestamps for free via its TDT duration head).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"
#include "core/asr_context_bias.h"
#include "parakeet.h"
#include "parakeet_orchestrate.h" // improvements Phase 1: shared transcribe orchestration

#include <cstdio>
#include <cstring>

namespace {

class ParakeetBackend : public CrispasrBackend {
public:
    ParakeetBackend() = default;
    ~ParakeetBackend() override { ParakeetBackend::shutdown(); }

    const char* name() const override { return "parakeet"; }

    uint32_t capabilities() const override {
        // CAP_LANGUAGE_DETECT intentionally NOT declared: the parakeet
        // backend has no native LID code path. Declaring the cap would
        // disable the framework's pre-step LID gate
        // (crispasr_run.cpp:`!has_native_lid`), so users wanting LID
        // get nothing. With the cap absent, `-dl` correctly routes
        // through the whisper-tiny pre-step.
        //
        uint32_t caps = CAP_TIMESTAMPS_NATIVE | CAP_WORD_TIMESTAMPS | CAP_TOKEN_CONFIDENCE | CAP_FLASH_ATTN |
                        CAP_PUNCTUATION_TOGGLE | CAP_TEMPERATURE | CAP_BEAM_SEARCH | CAP_DIARIZE |
                        CAP_PARALLEL_PROCESSORS | CAP_AUTO_DOWNLOAD | CAP_UNBOUNDED_INPUT;
        // CAP_INTERNAL_CHUNKING for non-JA models (2026-06-21): a single
        // full-attention pass is byte-for-byte NeMo-exact and far better
        // than the dispatcher's chunk-30 + overlap-save + LCS-merge, which
        // leaves a duplicated phrase at every 30 s boundary because the LCS
        // dedup is token-id-exact and adjacent chunks transcribe the overlap
        // differently. Verified 100% word match vs nvidia/parakeet-tdt-0.6b-v3
        // (30 s→5 min). With this flag the dispatcher hands us the whole clip;
        // transcribe() runs single-pass up to a memory-safe cap and a
        // silence-split single-pass longform beyond it. The JA model collapses
        // on single-pass (#89), so it keeps the dispatcher's VAD/chunk path.
        // Default: non-JA models advertise internal chunking so the dispatcher
        // hands us the whole clip (transcribe() runs single-pass / longform).
        // Override with CRISPASR_PARAKEET_INTERNAL_CHUNKING=0 to fall back to
        // the dispatcher's chunk-30 + overlap-save + LCS-merge path (e.g. for
        // A/B comparison), or =1 to force it on even for JA.
        bool internal_chunking = !is_ja_model_;
        if (const char* e = getenv("CRISPASR_PARAKEET_INTERNAL_CHUNKING"))
            internal_chunking = atoi(e) != 0;
        if (internal_chunking)
            caps |= CAP_INTERNAL_CHUNKING;
        return caps;
    }

    bool init(const whisper_params& p) override {
        parakeet_context_params cp = parakeet_context_default_params();
        cp.n_threads = p.n_threads;
        cp.use_flash = p.flash_attn;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = parakeet_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[parakeet]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        // Issue #89: JA-only models (vocab=3072) collapse past ~12 s on
        // real audio. Auto-chunk at 10 s instead of the global 30 s default.
        // Issue #257: detect JA by vocab content, not size — small-vocab ENGLISH
        // models (parakeet-tdt-1.1b, vocab ~1024) were misclassified as Japanese
        // and forced onto the JA short-chunk path, corrupting long/chunked output.
        is_ja_model_ = parakeet_vocab_is_japanese(ctx_) != 0;
        // CTC decode mode (hybrid TDT+CTC models).
        if (p.parakeet_decoder == "ctc") {
            if (parakeet_has_ctc(ctx_)) {
                parakeet_set_ctc_mode(ctx_, true);
                if (!p.no_prints)
                    fprintf(stderr, "crispasr[parakeet]: using CTC decoder\n");
            } else {
                fprintf(stderr, "crispasr[parakeet]: --parakeet-decoder ctc requested but model has no CTC head\n");
            }
        }
        return true;
    }

    void warmup() override {
        if (!ctx_)
            return;
        // 0.5 s of silence at 16 kHz — touches mel, encoder, and decoder
        // graphs once so subsequent calls hit pre-allocated buffers.
        std::vector<float> silence(8000, 0.0f);
        parakeet_result* r = parakeet_transcribe_ex(ctx_, silence.data(), (int)silence.size(), 0);
        if (r)
            parakeet_result_free(r);
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        // Sticky per-call sampling state. The setter just stores the
        // value on the parakeet_context, so subsequent transcribe calls
        // re-pick it up. We zero it on the first temp==0 call so a user
        // who toggles --temperature back off doesn't keep the previous
        // sampling state from a prior file.
        parakeet_set_temperature(ctx_, params.temperature, params.seed);
        parakeet_set_beam_size(ctx_, params.beam_size > 0 ? params.beam_size : 1);

        // Issue #257: local-attention window (--att-context "L,R") — NeMo
        // rel_pos_local_attn, bounds long-audio encoder VRAM. INT_MIN = unset
        // (keep the model default loaded from the GGUF / env).
        if (params.att_context_left != INT_MIN && params.att_context_right != INT_MIN) {
            parakeet_set_att_context(ctx_, params.att_context_left, params.att_context_right);
        }

        // MAES beam search (env: CRISPASR_PARAKEET_MAES=1, or --decode maes).
        // Requires beam_size > 1. Configurable via env vars.
        {
            const char* maes_env = std::getenv("CRISPASR_PARAKEET_MAES");
            bool use_maes = (maes_env && atoi(maes_env) > 0) || params.parakeet_decoder == "maes";
            if (use_maes && params.beam_size > 1) {
                int num_steps = 2;
                float gamma = 2.3f;
                int beta = 2;
                if (const char* v = std::getenv("CRISPASR_MAES_NUM_STEPS"))
                    num_steps = atoi(v);
                if (const char* v = std::getenv("CRISPASR_MAES_GAMMA"))
                    gamma = (float)atof(v);
                if (const char* v = std::getenv("CRISPASR_MAES_BETA"))
                    beta = atoi(v);
                parakeet_set_maes(ctx_, true, num_steps, gamma, beta);
            }
        }

        // PLAN #98: CTC-WS hotword phrase boost
        if (!params.hotwords.empty()) {
            auto hw = core_context_bias::parse_hotwords(params.hotwords);
            std::vector<const char*> ptrs;
            for (auto& s : hw)
                ptrs.push_back(s.c_str());
            parakeet_set_hotwords(ctx_, ptrs.data(), (int)ptrs.size(), params.hotwords_boost);
        }

        // Issue #89 / #257: long-audio path selection, the single-pass OOM
        // fallback, and --chunk-seconds output segmentation are hoisted into the
        // library (parakeet_transcribe_segments) so the session C-ABI runs the
        // SAME orchestration instead of a divergent inline copy (improvements
        // Phase 1). Sticky per-call state (temperature/beam/att-context/hotwords/
        // ctc) was set on ctx_ above; the orchestration reads the
        // CRISPASR_PARAKEET_* env knobs and takes JA-ness from is_ja_model_.
        parakeet_orchestrate_opts oo;
        oo.chunk_seconds_explicit = params.chunk_seconds_explicit;
        oo.chunk_seconds = params.chunk_seconds;
        oo.chunk_overlap_seconds = params.chunk_overlap_seconds;
        oo.no_prints = params.no_prints;
        for (const auto& ps : parakeet_transcribe_segments(ctx_, samples, n_samples, t_offset_cs, is_ja_model_, oo))
            out.push_back(seg_from_parakeet_seg(ps));
        return out;
    }

    // Convert a neutral parakeet_seg (from the shared orchestration) into the
    // CLI crispasr_segment type.
    static crispasr_segment seg_from_parakeet_seg(const parakeet_seg& ps) {
        crispasr_segment seg;
        seg.text = ps.text;
        seg.t0 = ps.t0;
        seg.t1 = ps.t1;
        seg.words.reserve(ps.words.size());
        for (const auto& w : ps.words) {
            crispasr_word cw;
            cw.text = w.text;
            cw.t0 = w.t0;
            cw.t1 = w.t1;
            seg.words.push_back(std::move(cw));
        }
        seg.tokens.reserve(ps.tokens.size());
        for (const auto& t : ps.tokens) {
            crispasr_token ct;
            ct.text = t.text;
            ct.id = t.id;
            ct.t0 = t.t0;
            ct.t1 = t.t1;
            ct.confidence = t.p;
            seg.tokens.push_back(std::move(ct));
        }
        return seg;
    }

    bool prefers_vad() const override {
        // Issue #89: parakeet-ja's encoder degenerates on arbitrary chunks
        // (repetition loops). VAD gives silence-bounded segments matching
        // the ~10-15 s utterances the model was trained on.
        return is_ja_model_;
    }

    int vad_slice_cap_seconds() const override {
        // Issue #89: on continuous speech (podcasts) VAD merges slices far
        // past the JA encoder's ~12 s safe single-pass window and the
        // decode goes sparse (56 % content recall on the reporter's clip).
        // Capping slices at 12 s + single-pass per slice measured best
        // (73-81 % vs whisper-large-v3 reference, NeMo's own long-form
        // paths score 15-46 % on the same audio). Non-JA models are exact
        // in single-pass and don't need a cap.
        int cap = is_ja_model_ ? 12 : 0;
        if (const char* e = getenv("CRISPASR_PARAKEET_VAD_SLICE_CAP"))
            cap = std::max(0, atoi(e));
        return cap;
    }

    void shutdown() override {
        if (ctx_) {
            parakeet_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    parakeet_context* ctx_ = nullptr;
    bool is_ja_model_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_parakeet_backend() {
    return std::unique_ptr<CrispasrBackend>(new ParakeetBackend());
}
