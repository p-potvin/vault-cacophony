// test-t5translate-params.cpp — unit tests for t5_translate_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "t5_translate.h"

TEST_CASE("t5_translate_params: default values are sensible", "[unit][t5_translate]") {
    struct t5_translate_context_params p = t5_translate_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("t5_translate_init_from_file: null path returns nullptr", "[unit][t5_translate]") {
    struct t5_translate_context_params p = t5_translate_context_default_params();
    struct t5_translate_context* ctx = t5_translate_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("t5_translate_init_from_file: empty path returns nullptr", "[unit][t5_translate]") {
    struct t5_translate_context_params p = t5_translate_context_default_params();
    struct t5_translate_context* ctx = t5_translate_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("t5_translate_free: NULL context is a no-op", "[unit][t5_translate]") {
    t5_translate_free(nullptr);
    SUCCEED("t5_translate_free tolerated a NULL ctx.");
}
