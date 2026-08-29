// test-dia-params.cpp — unit tests for dia_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "dia_tts.h"

TEST_CASE("dia_params: default values are sensible", "[unit][dia]") {
    struct dia_tts_context_params p = dia_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). Pin the shipped
// Dia sampling defaults (cfg_scale + nucleus/top-k gate) so a silent drift fails
// CI. Dia needs CFG for meaningful output, so cfg_scale is the load-bearing knob.
TEST_CASE("dia_params: value knobs match the shipped defaults", "[unit][dia]") {
    struct dia_tts_context_params p = dia_tts_context_default_params();

    REQUIRE(p.temperature == Catch::Approx(1.2f));
    REQUIRE(p.cfg_scale == Catch::Approx(3.0f)); // CFG required for coherent output
    REQUIRE(p.top_p == Catch::Approx(0.95f));
    REQUIRE(p.top_k == 45);
    REQUIRE(p.seed == 0);       // 0 = non-deterministic
    REQUIRE(p.max_tokens == 0); // 0 = model default
    REQUIRE(p.flash_attn == false);
}

TEST_CASE("dia_init_from_file: null path returns nullptr", "[unit][dia]") {
    struct dia_tts_context_params p = dia_tts_context_default_params();
    struct dia_tts_context* ctx = dia_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("dia_init_from_file: empty path returns nullptr", "[unit][dia]") {
    struct dia_tts_context_params p = dia_tts_context_default_params();
    struct dia_tts_context* ctx = dia_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("dia_free: NULL context is a no-op", "[unit][dia]") {
    dia_tts_free(nullptr);
    SUCCEED("dia_tts_free tolerated a NULL ctx.");
}
