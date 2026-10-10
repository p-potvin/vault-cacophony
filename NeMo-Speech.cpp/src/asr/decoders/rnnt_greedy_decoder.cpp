// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Host-side RNNT/TDT greedy decoding over the staged predictor and joint graphs.
#include "rnnt_greedy_decoder.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime.h"

namespace nemo_speech::asr {

namespace {
class DecodeStepScope {
   public:
    explicit DecodeStepScope(RnntEngine* engine) : engine_(engine) { engine_->begin_decode_step(); }
    ~DecodeStepScope() { engine_->end_decode_step(); }

   private:
    RnntEngine* engine_;
};
}  // namespace

namespace {
// Riva treats phrases matching ^\$[A-Z_]+$ as class hints (e.g. "$NAME"), not
// literal boost phrases; skip them like parse_riva_speech_contexts does.
bool
is_class_hint(const std::string& s) {
    if (s.size() < 2 || s[0] != '$')
        return false;
    for (size_t i = 1; i < s.size(); ++i) {
        const char c = s[i];
        if (!((c >= 'A' && c <= 'Z') || c == '_'))
            return false;
    }
    return true;
}
}  // namespace

// ===========================================================================
// RnntGreedyDecoder
// ===========================================================================
RnntGreedyDecoder::RnntGreedyDecoder(RnntEngine* engine, const DecoderConfig& cfg)
    : engine_(engine), bias_alpha_cfg_(static_cast<float>(cfg.boosting_tree_alpha)),
      bias_depth_scaling_(static_cast<float>(cfg.boosting_depth_scaling)),
      bias_max_boost_(cfg.boosting_max_boost) {
    build_punct_bias();
    reset();
}

// Identify the sentence-terminator tokens and their end-of-utterance floor.
// riva's PunctBiasHelper uses bump 1.5 for '.'/'?' (and ▁-prefixed) with an EOU
// floor of PUNCT_BIAS_EOS_STALL_FLOOR (5) * bump = 7.5; we reuse that floor.
// Comma is mid-sentence (the model emits it on its own) and is not floored.
void
RnntGreedyDecoder::build_punct_bias() {
    const auto& vocab = engine_->vocab();
    // The vocabulary vector may omit the terminal blank entry even though the
    // joint output includes it, so size the graph input from RNNT metadata.
    punct_bias_.assign(static_cast<size_t>(engine_->rnnt_config().vocab_size), 0.0f);
    has_punct_bias_ = false;
    constexpr float kFloor = 5.0f * 1.5f;
    for (int id = 0; id < static_cast<int>(vocab.size()); ++id) {
        const std::string& p = vocab[id];
        // Sentence terminators get the EOU floor so the marginal trailing one
        // wins over blank on the flush. Multilingual models also terminate with
        // '!' and the Devanagari danda U+0964 (Hindi etc.); floor those too so
        // the model's natural terminal wins instead of a spurious '.'/'?'.
        if (p == "." || p == "?" || p == "!" || p == "\xE0\xA5\xA4" || p == "\xE2\x96\x81." ||
            p == "\xE2\x96\x81?") {
            punct_bias_[static_cast<size_t>(id)] = kFloor;
            has_punct_bias_ = true;
        }
    }
}

// Everything reset() does to the LANGUAGE state, and nothing it does to the
// TRANSCRIPT state. The recurrent predictor state lives in stream_state_, so
// clearing prev_token_ alone would not be enough: the LSTM would still carry
// the previous segment's context. The in-progress word is flushed first so a
// word is never split across the boundary.
void
RnntGreedyDecoder::reset_context() {
    flush_word();
    stream_state_.reset();
    stream_state_ = engine_->make_rnnt_stream_state();
    if (!stream_state_)
        throw std::runtime_error("RnntEngine returned a null stream state");
    prev_token_ = engine_->rnnt_config().blank_id;
    active_bank_ = 0;
    predictor_valid_ = false;
    token_ids_.clear();
    blank_run_ = 0;
    bias_node_ = ContextBiasingTree::kRoot;
}

void
RnntGreedyDecoder::set_request_options(const AsrRequestOptions& opts) {
    set_compute_timestamps(opts.needs_word_timings());
    bias_on_ = false;
    bias_node_ = ContextBiasingTree::kRoot;
    bias_eff_alpha_ = 1.0f;
    bias_tree_.clear();
    if (opts.speech_contexts.empty() || bias_alpha_cfg_ <= 0.0f)
        return;

    // Mirror riva's parse_riva_speech_contexts: collect every phrase (skipping
    // class hints like "$NAME"), tokenize with the model's tokenizer, build the
    // tree with a fixed per-token context_score of 1.0, and fold the request's
    // boost into the fusion alpha (clamped to [0.1, max_boost]). The single
    // max boost across the request drives one alpha for all phrases.
    std::vector<std::vector<int>> phrases;
    float max_boost = 0.0f;
    for (const auto& sc : opts.speech_contexts) {
        if (sc.boost > 0.0f)
            max_boost = std::max(max_boost, sc.boost);
        for (const auto& phrase : sc.phrases) {
            if (is_class_hint(phrase))
                continue;
            std::vector<int> ids = engine_->encode_phrase(phrase);
            if (ids.empty())
                continue;
            phrases.push_back(std::move(ids));
        }
    }
    if (phrases.empty()) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::fprintf(
                stderr,
                "[asr] RNNT word boosting requested but no phrases could be tokenized "
                "(no embedded SentencePiece tokenizer in the GGUF?); boosting disabled.\n");
        return;
    }

    const float req_alpha =
        (max_boost > 0.0f)
            ? std::max(0.1f, std::min(max_boost, static_cast<float>(bias_max_boost_)))
            : 1.0f;
    bias_eff_alpha_ = bias_alpha_cfg_ * req_alpha;

    const std::vector<float> scores(phrases.size(), 1.0f);  // context_score = 1.0, as in riva
    bias_tree_.build(phrases, scores, bias_depth_scaling_);
    bias_on_ = !bias_tree_.empty();
}

void
RnntGreedyDecoder::reset() {
    const auto& cfg = engine_->rnnt_config();
    // Release the old slot before acquiring the new one; plain assignment would
    // transiently hold two slots per stream and exhaust the arena under churn.
    stream_state_.reset();
    stream_state_ = engine_->make_rnnt_stream_state();
    if (!stream_state_)
        throw std::runtime_error("RnntEngine returned a null stream state");
    prev_token_ = cfg.blank_id;
    active_bank_ = 0;
    predictor_valid_ = false;
    token_ids_.clear();
    stats_ = {};
    last_emit_frame_ = -1;
    words_.clear();
    cur_open_ = false;
    cur_ = WordTiming{};
    finalizing_ = false;
    bias_node_ = ContextBiasingTree::kRoot;
}

void
RnntGreedyDecoder::flush_word() {
    if (cur_open_ && !cur_.word.empty())
        words_.push_back(cur_);
    cur_open_ = false;
    cur_ = WordTiming{};
}

void
RnntGreedyDecoder::finalize() {
    flush_word();
}

static bool rnnt_trace_enabled();
static int rnnt_max_blank_run();
static int rnnt_blank_shadow_threshold();

std::vector<int>
RnntGreedyDecoder::step(const float* enc_out, int d_model, int T, int64_t frame_offset) {
    return step_impl(enc_out, nullptr, d_model, T, frame_offset);
}

std::vector<int>
RnntGreedyDecoder::step_device(
    const ggml_runtime::DeviceTensor& enc_out, int d_model, int T, int64_t frame_offset) {
    if (!enc_out.valid())
        throw std::runtime_error("RnntGreedyDecoder: invalid device encoder output");
    return step_impl(nullptr, &enc_out, d_model, T, frame_offset);
}

std::vector<int>
RnntGreedyDecoder::step_impl(
    const float* enc_out, const ggml_runtime::DeviceTensor* device_enc_out, int d_model, int T,
    int64_t frame_offset) {
    std::vector<int> emitted;
    const auto& vocab = engine_->vocab();
    const auto& cfg = engine_->rnnt_config();
    const int blank_id = cfg.blank_id;
    const int max_sym = cfg.max_symbols_per_step;
    if (d_model != cfg.joint_dim) {
        throw std::runtime_error(
            "RnntGreedyDecoder: encoder input is not joint-projected (got " +
            std::to_string(d_model) + ", expected " + std::to_string(cfg.joint_dim) + ")");
    }
    if (T <= 0)
        return emitted;
    DecodeStepScope decode_scope(engine_);
    stats_.encoder_frames += static_cast<uint64_t>(T);
    if (rnnt_trace_enabled())
        std::fprintf(
            stderr,
            "[rnnt] CHUNK enter frame_offset=%lld T=%d frames (%.0f ms) blank_id=%d"
            " max_symbols_per_step=%d prev_token=%d boosting=%s\n",
            static_cast<long long>(frame_offset), T, T * 80.0, blank_id, max_sym,
            prev_token_, bias_on_ ? "on" : "off");

    // With a fixed predictor output, joint evaluations for all remaining
    // encoder frames are independent. Evaluate them in one graph and consume
    // the leading blank run. If a non-blank appears, later results are ignored:
    // committing that token changes the predictor and requires a fresh round
    // beginning at the same frame.
    token_ids_.resize(static_cast<size_t>(T));
    int t = 0;
    int symbols_at_frame = 0;
    while (t < T) {
        const int remaining = T - t;
        const float* logit_bias = (finalizing_ && has_punct_bias_) ? punct_bias_.data() : nullptr;
        const float* boost_bias = nullptr;
        if (bias_on_) {
            // NeMo/riva semantics (rnnt_label_looping.py): blank vs non-blank is
            // decided on the unboosted logits; boosting only re-ranks which
            // non-blank label wins. Build a second bias for a blank-excluded
            // argmax pass. A uniform backoff base on every non-blank token
            // cannot change that ranking, so only the per-token deltas from the
            // current match state are applied. Valid for the whole speculative
            // frame run: the match state only changes on a non-blank commit,
            // which starts a fresh round.
            const ContextBiasingTree::StepBoost sb = bias_tree_.step_boost(bias_node_);
            if (logit_bias != nullptr)
                bias_scratch_.assign(punct_bias_.begin(), punct_bias_.end());
            else
                bias_scratch_.assign(static_cast<size_t>(cfg.vocab_size), 0.0f);
            for (const auto& sp : sb.specials) {
                if (sp.first != blank_id && sp.first >= 0 && sp.first < cfg.vocab_size)
                    bias_scratch_[static_cast<size_t>(sp.first)] +=
                        bias_eff_alpha_ * (sp.second - sb.base);
            }
            bias_scratch_[static_cast<size_t>(blank_id)] = -1e30f;  // exclude blank
            boost_bias = bias_scratch_.data();
        }
        bool joint_complete = false;
        if (!predictor_valid_ && device_enc_out != nullptr && boost_bias == nullptr) {
            engine_->predict_and_joint_rnnt_argmax_device(
                *stream_state_, prev_token_, active_bank_, *device_enc_out, t, d_model, remaining,
                token_ids_.data(), logit_bias);
            ++stats_.predictor_calls;
            ++stats_.joint_calls;
            stats_.joint_frames += static_cast<uint64_t>(remaining);
            predictor_valid_ = true;
            joint_complete = true;
        } else if (!predictor_valid_) {
            engine_->predict_rnnt(*stream_state_, prev_token_, active_bank_);
            ++stats_.predictor_calls;
            predictor_valid_ = true;
        }

        // Apply the release branch's end-of-utterance punctuation floor in the
        // staged joint graph, before its device-side argmax. Keeping this as an
        // optional dense bias preserves the vectorized blank-run path and avoids
        // copying full logits back to the host.
        if (!joint_complete) {
            if (device_enc_out != nullptr) {
                engine_->joint_argmax_device(
                    *stream_state_, *device_enc_out, t, d_model, remaining, token_ids_.data(),
                    logit_bias);
            } else {
                engine_->joint_argmax(
                    *stream_state_, enc_out + static_cast<size_t>(t) * d_model, d_model, remaining,
                    token_ids_.data(), logit_bias);
            }
            ++stats_.joint_calls;
            stats_.joint_frames += static_cast<uint64_t>(remaining);
        }
        if (boost_bias != nullptr) {
            boost_token_ids_.resize(static_cast<size_t>(remaining));
            if (device_enc_out != nullptr) {
                engine_->joint_argmax_device(
                    *stream_state_, *device_enc_out, t, d_model, remaining, boost_token_ids_.data(),
                    boost_bias);
            } else {
                engine_->joint_argmax(
                    *stream_state_, enc_out + static_cast<size_t>(t) * d_model, d_model, remaining,
                    boost_token_ids_.data(), boost_bias);
            }
            ++stats_.joint_calls;
            stats_.joint_frames += static_cast<uint64_t>(remaining);
            for (int i = 0; i < remaining; ++i) {
                if (token_ids_[static_cast<size_t>(i)] != blank_id)
                    token_ids_[static_cast<size_t>(i)] = boost_token_ids_[static_cast<size_t>(i)];
            }
        }

        int first_emit = -1;
        for (int i = 0; i < remaining; ++i) {
            if (token_ids_[static_cast<size_t>(i)] != blank_id) {
                first_emit = i;
                break;
            }
        }
        // Blank frames this scan is about to consume before any emission.
        // first_emit < 0 means the whole rest of the chunk is blank.
        const int blanks_here = (first_emit < 0) ? remaining : first_emit;

        // Shadow decode: what was the model ranking under the blank? Read-only.
        const int shadow_n = rnnt_blank_shadow_threshold();
        if (shadow_n > 0 && blanks_here >= shadow_n) {
            const char* prev_piece = "";
            if (prev_token_ >= 0 && prev_token_ < static_cast<int>(vocab.size()))
                prev_piece = vocab[static_cast<size_t>(prev_token_)].c_str();
            std::fprintf(
                stderr,
                "[rnnt]   ~~ SHADOW: %d blank frames from t=%d, predictor FROZEN on"
                " token=%d %s (every frame below is scored against that one state)\n",
                blanks_here, t, prev_token_, prev_piece);
            if (finalizing_ && has_punct_bias_)
                forced_scratch_.assign(punct_bias_.begin(), punct_bias_.end());
            else
                forced_scratch_.assign(static_cast<size_t>(cfg.vocab_size), 0.0f);
            forced_scratch_[static_cast<size_t>(blank_id)] = -1e30f;
            forced_ids_.resize(static_cast<size_t>(blanks_here));
            if (device_enc_out != nullptr) {
                engine_->joint_argmax_device(
                    *stream_state_, *device_enc_out, t, d_model, blanks_here,
                    forced_ids_.data(), forced_scratch_.data());
            } else {
                engine_->joint_argmax(
                    *stream_state_, enc_out + static_cast<size_t>(t) * d_model, d_model,
                    blanks_here, forced_ids_.data(), forced_scratch_.data());
            }
            ++stats_.joint_calls;
            stats_.joint_frames += static_cast<uint64_t>(blanks_here);
            std::string line;
            for (int i = 0; i < blanks_here; ++i) {
                const int sid = forced_ids_[static_cast<size_t>(i)];
                const char* piece = "?";
                if (sid >= 0 && sid < static_cast<int>(vocab.size()))
                    piece = vocab[static_cast<size_t>(sid)].c_str();
                char buf[128];
                std::snprintf(
                    buf, sizeof(buf), "%s%d:%s", line.empty() ? "" : "  ", t + i, piece);
                line += buf;
                // Wrap so a 50-frame run stays readable.
                if ((i + 1) % 8 == 0 || i + 1 == blanks_here) {
                    std::fprintf(stderr, "[rnnt]      %s\n", line.c_str());
                    line.clear();
                }
            }
        }

        const int max_blank = rnnt_max_blank_run();
        if (max_blank > 0 && blanks_here > 0 && blank_run_ + blanks_here >= max_blank) {
            // Force a non-blank at the frame where the blank budget runs out.
            const int k = std::max(0, max_blank - blank_run_);
            if (k < blanks_here) {
                if (finalizing_ && has_punct_bias_)
                    forced_scratch_.assign(punct_bias_.begin(), punct_bias_.end());
                else
                    forced_scratch_.assign(static_cast<size_t>(cfg.vocab_size), 0.0f);
                forced_scratch_[static_cast<size_t>(blank_id)] = -1e30f;
                const int forced_remaining = remaining - k;
                forced_ids_.resize(static_cast<size_t>(forced_remaining));
                if (device_enc_out != nullptr) {
                    engine_->joint_argmax_device(
                        *stream_state_, *device_enc_out, t + k, d_model, forced_remaining,
                        forced_ids_.data(), forced_scratch_.data());
                } else {
                    engine_->joint_argmax(
                        *stream_state_, enc_out + static_cast<size_t>(t + k) * d_model, d_model,
                        forced_remaining, forced_ids_.data(), forced_scratch_.data());
                }
                ++stats_.joint_calls;
                stats_.joint_frames += static_cast<uint64_t>(forced_remaining);
                const int forced = forced_ids_[0];
                if (forced != blank_id) {
                    if (rnnt_trace_enabled()) {
                        const char* piece = "";
                        if (forced >= 0 && forced < static_cast<int>(vocab.size()))
                            piece = vocab[static_cast<size_t>(forced)].c_str();
                        std::fprintf(
                            stderr,
                            "[rnnt]   !! BLANK CAP: %d consecutive blank frames"
                            " (NEMO_SPEECH_RNNT_MAX_BLANK=%d, run would have been %d)."
                            " Forcing best non-blank token=%d %s at t=%d to unfreeze"
                            " the predictor.\n",
                            blank_run_ + k, max_blank, blank_run_ + blanks_here, forced, piece,
                            t + k);
                    }
                    t += k;
                    symbols_at_frame = 0;
                    blank_run_ = 0;
                    emitted.push_back(forced);
                    ++stats_.emitted_tokens;
                    last_emit_frame_ = frame_offset + t;
                    prev_token_ = forced;
                    active_bank_ ^= 1;
                    predictor_valid_ = false;
                    if (bias_on_)
                        bias_node_ = bias_tree_.advance(bias_node_, forced);
                    continue;
                }
            }
        }
        if (first_emit < 0) {
            // Every remaining frame is blank under this predictor state, and
            // the state cannot change without an emission, so the rest of the
            // chunk is already decided.
            blank_run_ += remaining;
            break;
        }

        // A blank advances the encoder clock and ends any symbol run at the
        // previous frame. The first non-blank is handled at its own frame.
        if (first_emit > 0)
            symbols_at_frame = 0;
        blank_run_ = 0;  // a non-blank commits a new predictor state
        t += first_emit;
        const int token = token_ids_[static_cast<size_t>(first_emit)];

        if (rnnt_trace_enabled()) {
            const char* piece = "";
            if (token >= 0 && token < static_cast<int>(vocab.size()))
                piece = vocab[static_cast<size_t>(token)].c_str();
            std::fprintf(
                stderr,
                "[rnnt]   t=%2d/%d EMIT token=%-6d %-14s (blank_run=%d before it,"
                " symbols_at_frame=%d/%d)\n",
                t, T, token, piece, first_emit, symbols_at_frame, max_sym);
        }
        emitted.push_back(token);
        ++stats_.emitted_tokens;
        last_emit_frame_ = frame_offset + t;
        if (compute_ts_ && token >= 0 && token < (int)vocab.size()) {
            const int64_t f = frame_offset + t;
            const std::string& piece = vocab[token];
            if (sp_starts_new_word(piece) || !cur_open_) {
                flush_word();
                cur_open_ = true;
                cur_.start_frame = f;
                cur_.confidence = 1.0f;  // RNNT greedy emits no per-token posterior
            }
            cur_.word += sp_piece_text(piece);
            cur_.end_frame = f + 1;
        }

        // A non-blank commits the candidate prediction state. The newly
        // emitted token becomes the next predictor input, invalidating every
        // cached predictor-side value. Blank paths never reach this block and
        // retain the cache across frames and step() calls.
        prev_token_ = token;
        active_bank_ ^= 1;
        predictor_valid_ = false;
        // Advance the context-biasing match on the emitted token.
        if (bias_on_)
            bias_node_ = bias_tree_.advance(bias_node_, token);

        if (++symbols_at_frame >= max_sym) {
            if (rnnt_trace_enabled())
                std::fprintf(
                    stderr,
                    "[rnnt]   !! SYMBOL CAP: %d symbols emitted on frame t=%d"
                    " (max_symbols_per_step=%d). Force-advancing to t=%d; anything the"
                    " model still wanted on this frame is DISCARDED.\n",
                    symbols_at_frame, t, max_sym, t + 1);
            ++t;
            symbols_at_frame = 0;
        }
    }
    if (rnnt_trace_enabled())
        std::fprintf(
            stderr,
            "[rnnt] CHUNK exit  t=%d/%d emitted=%zu token(s)\n",
            t, T, emitted.size());
    return emitted;
}

TdtGreedyDecoder::TdtGreedyDecoder(RnntEngine* engine) : engine_(engine) {
    if (!engine_ || !engine_->rnnt_config().is_tdt())
        throw std::invalid_argument("TdtGreedyDecoder requires TDT durations");
    reset();
}

void
TdtGreedyDecoder::reset() {
    // Release the old slot before acquiring the new one; plain assignment would
    // transiently hold two slots per stream and exhaust the arena under churn.
    stream_state_.reset();
    stream_state_ = engine_->make_rnnt_stream_state();
    if (!stream_state_)
        throw std::runtime_error("RnntEngine returned a null stream state");
    prev_token_ = engine_->rnnt_config().blank_id;
    active_bank_ = 0;
    predictor_valid_ = false;
    pending_skip_ = 0;
    stats_ = {};
    last_emit_frame_ = -1;
    words_.clear();
    cur_open_ = false;
    cur_ = WordTiming{};
}

// See RnntGreedyDecoder::reset_context(). pending_skip_ is dropped too: a
// duration that overshot the previous segment must not skip frames at the
// start of the next one.
void
TdtGreedyDecoder::reset_context() {
    flush_word();
    stream_state_.reset();
    stream_state_ = engine_->make_rnnt_stream_state();
    if (!stream_state_)
        throw std::runtime_error("RnntEngine returned a null stream state");
    prev_token_ = engine_->rnnt_config().blank_id;
    active_bank_ = 0;
    predictor_valid_ = false;
    pending_skip_ = 0;
}

void
TdtGreedyDecoder::reset_utterance() {
    pending_skip_ = 0;
    words_.clear();
    cur_open_ = false;
    cur_ = WordTiming{};
}

void
TdtGreedyDecoder::flush_word() {
    if (cur_open_ && !cur_.word.empty())
        words_.push_back(cur_);
    cur_open_ = false;
    cur_ = WordTiming{};
}

void
TdtGreedyDecoder::finalize() {
    flush_word();
}

// ---------------------------------------------------------------------------
// Optional TDT decode tracing. Set NEMO_SPEECH_DEBUG_RNNT=1 to dump the decode
// loop to stderr. Off by default; costs one cached bool test per frame.
//
// What the loop does, since the trace only makes sense alongside it:
//
//   * The encoder hands the decoder a CHUNK of T frames. T is the cache-aware
//     chunk, cache_chunk_frames = 1 + cache_right_ctx, so rnnt_right_context=1
//     gives T=2 (160 ms) and =3 gives T=4 (320 ms). Right context is therefore
//     not just lookahead: it sets how much audio the joint sees per chunk.
//   * At each frame the joint predicts BOTH a token and a DURATION index. The
//     duration says how many frames to jump: skip = durations[idx]. That is
//     what makes this TDT (token-and-duration transducer) rather than plain
//     RNN-T, where every step advances exactly one frame.
//   * The inner loop repeats only while skip == 0, i.e. while the model wants
//     several tokens on the SAME frame. max_symbols_per_step bounds that, so
//     the cap can only bite on chains of zero-duration tokens.
//   * A duration may jump PAST the end of the chunk. The overshoot is stored in
//     pending_skip_ and consumed at the start of the next chunk, where those
//     frames are skipped outright and never reach the joint. Audio inside them
//     cannot be transcribed, and nothing reports an error. That is the most
//     plausible route for speech to disappear silently.
static bool
rnnt_trace_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NEMO_SPEECH_DEBUG_RNNT");
        return v != nullptr && *v != 0 && *v != '0';
    }();
    return on;
}

// Optional escape hatch from a self-sustaining blank run, gated on
// NEMO_SPEECH_RNNT_MAX_BLANK (0 or unset = off, the shipped behaviour).
//
// Why a blank run can sustain itself. In RNN-T the predictor is conditioned on
// the last EMITTED token, and a blank is not an emission: the blank path never
// touches prev_token_ and never invalidates the predictor cache. So for the
// entire length of a blank run the joint is evaluated against one frozen
// predictor state. If that state is a poor prior for the audio now arriving --
// the clearest case being the last token of one language followed by speech in
// another -- every frame can score blank, and nothing in greedy decoding
// perturbs the state to let it recover. The run ends only when the acoustics
// alone outvote the stale prior.
//
// max_symbols_per_step cannot help here, and it is worth being precise about
// why: it counts EMITTED symbols at a single frame and is incremented only on
// the non-blank commit path. During a blank run the counter is pinned at zero,
// so the cap can never fire. It bounds runaway emission, not silence.
//
// This knob bounds silence instead. After N consecutive blank frames it runs
// one blank-excluded argmax and forces the best non-blank token, which commits
// a new predictor state and breaks the lock. That is a deliberate departure
// from greedy decoding -- it emits a token the model ranked below blank -- so
// it is a diagnostic for whether a run is a stuck prior or a genuine judgement
// about the audio, not a default.
// Read-only companion to the blank cap, gated on NEMO_SPEECH_DEBUG_BLANK_SHADOW
// (unset = off; set to a number to use it as the threshold, or to any other
// non-empty value for the default of 3).
//
// When a blank run of at least that many frames is about to be consumed, this
// runs ONE extra blank-excluded argmax across the run and prints, frame by
// frame, the token the model would have emitted if blank were not on the table.
// It commits nothing: no token is emitted, prev_token_ is untouched, the
// predictor cache is not invalidated, and the transcript is bit-identical to a
// run with this unset. It is purely a window into what the joint was ranking
// second while it stayed silent.
//
// Read the output against the frozen predictor state named in the header line.
// Every shadow frame in a run is scored against that ONE state, because a blank
// does not advance it -- which is the whole reason a run can sustain itself.
static int
rnnt_blank_shadow_threshold() {
    static const int n = [] {
        const char* v = std::getenv("NEMO_SPEECH_DEBUG_BLANK_SHADOW");
        if (v == nullptr || *v == 0 || (*v == '0' && v[1] == 0))
            return 0;
        const int parsed = std::atoi(v);
        return parsed > 0 ? parsed : 3;
    }();
    return n;
}

static int
rnnt_max_blank_run() {
    static const int n = [] {
        const char* v = std::getenv("NEMO_SPEECH_RNNT_MAX_BLANK");
        if (v == nullptr || *v == 0)
            return 0;
        const int parsed = std::atoi(v);
        return parsed > 0 ? parsed : 0;
    }();
    return n;
}

std::vector<int>
TdtGreedyDecoder::step(const float* enc_out, int d_model, int T, int64_t frame_offset) {
    const auto& cfg = engine_->rnnt_config();
    if (d_model != cfg.joint_dim)
        throw std::runtime_error("TdtGreedyDecoder: invalid encoder projection width");
    std::vector<int> emitted;
    if (T <= 0)
        return emitted;

    DecodeStepScope decode_scope(engine_);

    stats_.encoder_frames += static_cast<uint64_t>(T);
    int t = pending_skip_;
    if (rnnt_trace_enabled()) {
        std::fprintf(
            stderr,
            "[rnnt] CHUNK enter frame_offset=%lld T=%d frames carried_pending_skip=%d"
            " max_symbols_per_step=%d blank_id=%d\n",
            static_cast<long long>(frame_offset), T, pending_skip_,
            cfg.max_symbols_per_step, cfg.blank_id);
        if (pending_skip_ > 0)
            std::fprintf(
                stderr,
                "[rnnt]   ^ previous chunk overshot its end by %d frame(s); they are"
                " SKIPPED here and never reach the joint (%.0f ms of audio lost)\n",
                pending_skip_, pending_skip_ * 80.0);
    }
    pending_skip_ = 0;
    while (t < T) {
        int symbols_added = 0;
        bool need_loop = true;
        int skip = 0;
        while (need_loop && symbols_added < cfg.max_symbols_per_step) {
            int32_t token = cfg.blank_id;
            int32_t duration_index = 0;
            if (!predictor_valid_) {
                engine_->predict_and_joint_tdt_argmax(
                    *stream_state_, prev_token_, active_bank_,
                    enc_out + static_cast<size_t>(t) * d_model, d_model, &token, &duration_index);
                ++stats_.predictor_calls;
                ++stats_.joint_calls;
                ++stats_.joint_frames;
                predictor_valid_ = true;
            } else {
                engine_->joint_tdt_argmax(
                    *stream_state_, enc_out + static_cast<size_t>(t) * d_model, d_model, 1, &token,
                    &duration_index);
                ++stats_.joint_calls;
                ++stats_.joint_frames;
            }
            if (duration_index < 0 || duration_index >= static_cast<int>(cfg.durations.size()))
                throw std::runtime_error("TDT joint returned an invalid duration index");
            skip = cfg.durations[static_cast<size_t>(duration_index)];
            if (skip < 0)
                throw std::runtime_error("TDT duration values must be non-negative");
            if (rnnt_trace_enabled()) {
                const bool is_blank = (token == cfg.blank_id);
                const char* piece = "";
                if (!is_blank && token >= 0 &&
                    token < static_cast<int>(engine_->vocab().size()))
                    piece = engine_->vocab()[static_cast<size_t>(token)].c_str();
                std::fprintf(
                    stderr,
                    "[rnnt]   t=%2d/%d sym#%d token=%-6d %-14s dur_idx=%d skip=%d"
                    " (%.0f ms)%s\n",
                    t, T, symbols_added, token, is_blank ? "<blank>" : piece,
                    duration_index, skip, skip * 80.0, is_blank ? "" : "  EMIT");
            }

            // NeMo special-cases blank + duration zero: no decoder state has
            // changed and revisiting the same encoder frame could only repeat
            // that decision, so advance one frame immediately.
            if (token == cfg.blank_id && skip == 0) {
                if (rnnt_trace_enabled())
                    std::fprintf(
                        stderr,
                        "[rnnt]     blank with duration 0 -> forcing skip=1 so this"
                        " frame cannot be revisited forever\n");
                skip = 1;
            }

            if (token != cfg.blank_id) {
                emitted.push_back(token);
                ++stats_.emitted_tokens;
                last_emit_frame_ = frame_offset + t;
                if (compute_ts_ && token >= 0 &&
                    token < static_cast<int>(engine_->vocab().size())) {
                    const int64_t f = frame_offset + t;
                    const auto& piece = engine_->vocab()[static_cast<size_t>(token)];
                    if (sp_is_word_boundary(piece) || !cur_open_) {
                        flush_word();
                        cur_open_ = true;
                        cur_.start_frame = f;
                        cur_.confidence = 1.0f;
                    }
                    cur_.word += sp_piece_text(piece);
                    cur_.end_frame = f + std::max(1, skip);
                }
                prev_token_ = token;
                active_bank_ ^= 1;
                predictor_valid_ = false;
            }

            ++symbols_added;
            t += skip;
            need_loop = skip == 0;
        }

        if (rnnt_trace_enabled() && symbols_added >= cfg.max_symbols_per_step)
            std::fprintf(
                stderr,
                "[rnnt]   !! symbol cap reached at t=%d: %d symbols on one frame"
                " (max_symbols_per_step=%d), skip=%d\n",
                t, symbols_added, cfg.max_symbols_per_step, skip);
        // NeMo's infinite-loop guard: duration zero can otherwise revisit the
        // same frame forever (blank or a chain of zero-duration tokens).
        if (symbols_added == cfg.max_symbols_per_step && skip == 0) {
            if (rnnt_trace_enabled())
                std::fprintf(
                    stderr,
                    "[rnnt]   !! GUARD FIRED: cap hit with duration 0, force-advancing"
                    " t %d -> %d. Anything still pending on this frame is DISCARDED.\n",
                    t, t + 1);
            ++t;
        }
    }
    pending_skip_ = std::max(0, t - T);
    if (rnnt_trace_enabled())
        std::fprintf(
            stderr,
            "[rnnt] CHUNK exit  t=%d T=%d emitted=%zu token(s) new_pending_skip=%d%s\n",
            t, T, emitted.size(), pending_skip_,
            pending_skip_ > 0 ? "  <-- will skip the next chunk's opening frames" : "");
    return emitted;
}

}  // namespace nemo_speech::asr
