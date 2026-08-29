// test-kokoro-params.cpp — unit tests for kokoro_context_params defaults
// and null-guard coverage. No GGUF required.

#include <cstring>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "kokoro.h"

TEST_CASE("kokoro_params: default values are sensible", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). Kokoro's one
// perceptual knob is speed, carried here as length_scale (1.0 = upstream default
// speed; >1 slower, <1 faster). Pin it plus the shipped defaults so a drift fails
// CI rather than silently changing every synthesis. espeak_lang seeds the G2P
// front-end and defaults to en-us upstream.
TEST_CASE("kokoro_params: value knobs match the shipped/upstream contract", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();

    REQUIRE(p.length_scale == Catch::Approx(1.0f)); // upstream Kokoro speed default 1.0
    REQUIRE(std::strcmp(p.espeak_lang, "en-us") == 0);
    REQUIRE(p.use_gpu == true);
    REQUIRE(p.flash_attn == true); // PLAN #89 plumbing — must stay on
}

TEST_CASE("kokoro_init_from_file: null path returns nullptr", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();
    struct kokoro_context* ctx = kokoro_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("kokoro_init_from_file: empty path returns nullptr", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();
    struct kokoro_context* ctx = kokoro_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("kokoro_free: NULL context is a no-op", "[unit][kokoro]") {
    kokoro_free(nullptr);
    SUCCEED("kokoro_free tolerated a NULL ctx.");
}
