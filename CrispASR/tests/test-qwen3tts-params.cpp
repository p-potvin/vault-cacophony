// test-qwen3tts-params.cpp — unit tests for qwen3_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "qwen3_tts.h"

TEST_CASE("qwen3_tts_params: default values are sensible", "[unit][qwen3_tts]") {
    struct qwen3_tts_context_params p = qwen3_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). qwen3-tts ships
// greedy talker decode (temperature 0) with GPU + flash-attn on by default; pin
// them so a drift fails CI.
TEST_CASE("qwen3_tts_params: value knobs match the shipped defaults", "[unit][qwen3_tts]") {
    struct qwen3_tts_context_params p = qwen3_tts_context_default_params();

    REQUIRE(p.temperature == Catch::Approx(0.0f)); // 0 = greedy
    REQUIRE(p.max_codec_steps == 0);               // 0 = model default
    REQUIRE(p.seed == 0);
    REQUIRE(p.use_gpu == true);
    REQUIRE(p.flash_attn == true);
}

TEST_CASE("qwen3_tts_init_from_file: null path returns nullptr", "[unit][qwen3_tts]") {
    struct qwen3_tts_context_params p = qwen3_tts_context_default_params();
    struct qwen3_tts_context* ctx = qwen3_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("qwen3_tts_init_from_file: empty path returns nullptr", "[unit][qwen3_tts]") {
    struct qwen3_tts_context_params p = qwen3_tts_context_default_params();
    struct qwen3_tts_context* ctx = qwen3_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("qwen3_tts_free: NULL context is a no-op", "[unit][qwen3_tts]") {
    qwen3_tts_free(nullptr);
    SUCCEED("qwen3_tts_free tolerated a NULL ctx.");
}
