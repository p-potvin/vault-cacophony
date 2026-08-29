// test-nemotron-params.cpp — unit tests for nemotron_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "nemotron.h"

TEST_CASE("nemotron_params: default values are sensible", "[unit][nemotron]") {
    struct nemotron_context_params p = nemotron_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197 + PLAN #89). Pin
// the shipped use_flash/use_gpu defaults so a silent flip fails CI.
TEST_CASE("nemotron_params: gpu/flash defaults are pinned", "[unit][nemotron]") {
    struct nemotron_context_params p = nemotron_context_default_params();
    REQUIRE(p.use_gpu == false);
    REQUIRE(p.use_flash == false);
}

TEST_CASE("nemotron_init_from_file: null path returns nullptr", "[unit][nemotron]") {
    struct nemotron_context_params p = nemotron_context_default_params();
    struct nemotron_context* ctx = nemotron_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("nemotron_init_from_file: empty path returns nullptr", "[unit][nemotron]") {
    struct nemotron_context_params p = nemotron_context_default_params();
    struct nemotron_context* ctx = nemotron_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("nemotron_free: NULL context is a no-op", "[unit][nemotron]") {
    nemotron_free(nullptr);
    SUCCEED("nemotron_free tolerated a NULL ctx.");
}
