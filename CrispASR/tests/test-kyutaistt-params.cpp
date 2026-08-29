// test-kyutaistt-params.cpp — unit tests for kyutai_stt_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "kyutai_stt.h"

TEST_CASE("kyutai_stt_params: default values are sensible", "[unit][kyutai_stt]") {
    struct kyutai_stt_context_params p = kyutai_stt_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). ASR ships greedy
// deterministic decode (temperature 0, beam 1); pin them so a silent drift fails
// CI.
TEST_CASE("kyutai_stt_params: decode knobs match the shipped defaults", "[unit][kyutai_stt]") {
    struct kyutai_stt_context_params p = kyutai_stt_context_default_params();

    REQUIRE(p.temperature == Catch::Approx(0.0f)); // greedy
    REQUIRE(p.beam_size == 1);
    REQUIRE(p.use_gpu == true);
}

TEST_CASE("kyutai_stt_init_from_file: null path returns nullptr", "[unit][kyutai_stt]") {
    struct kyutai_stt_context_params p = kyutai_stt_context_default_params();
    struct kyutai_stt_context* ctx = kyutai_stt_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("kyutai_stt_init_from_file: empty path returns nullptr", "[unit][kyutai_stt]") {
    struct kyutai_stt_context_params p = kyutai_stt_context_default_params();
    struct kyutai_stt_context* ctx = kyutai_stt_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("kyutai_stt_free: NULL context is a no-op", "[unit][kyutai_stt]") {
    kyutai_stt_free(nullptr);
    SUCCEED("kyutai_stt_free tolerated a NULL ctx.");
}
