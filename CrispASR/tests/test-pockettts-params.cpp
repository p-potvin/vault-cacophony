// test-pockettts-params.cpp — unit tests for pocket_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "pocket_tts.h"

TEST_CASE("pocket_tts_params: default values are sensible", "[unit][pocket_tts]") {
    struct pocket_tts_context_params p = pocket_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). pocket-tts ships
// a full set of load-bearing generation knobs; pin them so a silent drift fails
// CI rather than changing every synthesis.
TEST_CASE("pocket_tts_params: value knobs match the shipped defaults", "[unit][pocket_tts]") {
    struct pocket_tts_context_params p = pocket_tts_context_default_params();

    REQUIRE(p.temperature == Catch::Approx(0.7f));
    REQUIRE(p.lsd_decode_steps == 1);
    REQUIRE(p.noise_clamp == Catch::Approx(3.0f));
    REQUIRE(p.eos_threshold == Catch::Approx(0.5f));
    REQUIRE(p.seed == 0);
    REQUIRE(p.max_audio_frames == 0); // 0 = model default
    REQUIRE(p.use_gpu == true);
}

TEST_CASE("pocket_tts_init_from_file: null path returns nullptr", "[unit][pocket_tts]") {
    struct pocket_tts_context_params p = pocket_tts_context_default_params();
    struct pocket_tts_context* ctx = pocket_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("pocket_tts_init_from_file: empty path returns nullptr", "[unit][pocket_tts]") {
    struct pocket_tts_context_params p = pocket_tts_context_default_params();
    struct pocket_tts_context* ctx = pocket_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("pocket_tts_free: NULL context is a no-op", "[unit][pocket_tts]") {
    pocket_tts_free(nullptr);
    SUCCEED("pocket_tts_free tolerated a NULL ctx.");
}
