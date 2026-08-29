// test-personaplex-params.cpp — unit tests for personaplex_context_params
// defaults and null-guard coverage, plus the geometry contract the loader is
// written against. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "personaplex.h"

#include <cstdlib>

TEST_CASE("personaplex_params: default values are sensible", "[unit][personaplex]") {
    struct personaplex_context_params p = personaplex_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    // moshi LMGen defaults: audio 0.8, text 0.7.
    REQUIRE(p.temperature > 0.0f);
    REQUIRE(p.temperature_text > 0.0f);
    // 0 means "use the model's own context" rather than a silent cap.
    REQUIRE(p.context_frames == 0);
}

TEST_CASE("personaplex_init_from_file: null path returns nullptr", "[unit][personaplex]") {
    struct personaplex_context_params p = personaplex_context_default_params();
    REQUIRE(personaplex_init_from_file(nullptr, p) == nullptr);
}

TEST_CASE("personaplex_init_from_file: empty path returns nullptr", "[unit][personaplex]") {
    struct personaplex_context_params p = personaplex_context_default_params();
    REQUIRE(personaplex_init_from_file("", p) == nullptr);
}

TEST_CASE("personaplex_init_from_file: missing file returns nullptr", "[unit][personaplex]") {
    struct personaplex_context_params p = personaplex_context_default_params();
    p.verbosity = 0;
    REQUIRE(personaplex_init_from_file("./no-such-personaplex.gguf", p) == nullptr);
}

TEST_CASE("personaplex_free: NULL context is a no-op", "[unit][personaplex]") {
    personaplex_free(nullptr);
    SUCCEED("personaplex_free tolerated a NULL ctx.");
}

TEST_CASE("personaplex accessors: NULL context is a no-op", "[unit][personaplex]") {
    struct personaplex_model_info info;
    personaplex_model_info_get(nullptr, &info);

    int n = -1;
    REQUIRE(personaplex_delays(nullptr, &n) == nullptr);
    SUCCEED("accessors tolerated a NULL ctx.");
}

// ─── live ────────────────────────────────────────────────────────────────────
// Needs the converted model: CRISPASR_PERSONAPLEX_MODEL=<path to the GGUF>.

TEST_CASE("personaplex loads personaplex-7b-v1 with the blueprint geometry", "[live][personaplex]") {
    const char* path = std::getenv("CRISPASR_PERSONAPLEX_MODEL");
    if (!path || !*path) {
        SUCCEED("CRISPASR_PERSONAPLEX_MODEL not set — skipping.");
        return;
    }

    struct personaplex_context_params p = personaplex_context_default_params();
    p.use_gpu = false; // weights only; no graph is built yet
    struct personaplex_context* ctx = personaplex_init_from_file(path, p);
    REQUIRE(ctx != nullptr);

    struct personaplex_model_info info;
    personaplex_model_info_get(ctx, &info);

    // docs/personaplex/BLUEPRINT.md §1.
    REQUIRE(info.dim == 4096);
    REQUIRE(info.num_layers == 32);
    REQUIRE(info.num_heads == 32);
    REQUIRE(info.ffn_hidden == 11264);  // (2 * 16896) / 3, NOT 16896 — trap 1
    REQUIRE(info.depformer_dim == 1024);
    REQUIRE(info.depformer_num_layers == 6);
    REQUIRE(info.depformer_num_heads == 16);
    REQUIRE(info.depformer_ffn_hidden == 2816); // (2 * 4224) / 3
    REQUIRE(info.n_q == 16);
    REQUIRE(info.dep_q == 16); // overridden 8 -> 16 by get_moshi_lm
    REQUIRE(info.card == 2048);
    REQUIRE(info.text_card == 32000);
    REQUIRE(info.frame_size == 1920); // 24000 / 12.5

    // Every LM tensor in the checkpoint is mapped, none silently dropped:
    //   temporal 32*6 + out_norm 1            = 193
    //   audio_emb 16 + text_emb + text_linear =  18
    //   depformer 6*(2 norm + 2 attn + 32 gating) = 216
    //   depformer_in 16 + linears 16 + emb 15 + text_emb 1 = 48
    REQUIRE(info.n_tensors_loaded == 475);

    // The full-duplex delay pattern: 1 text + 8 own + 8 other.
    int n_delays = 0;
    const int32_t* delays = personaplex_delays(ctx, &n_delays);
    REQUIRE(delays != nullptr);
    REQUIRE(n_delays == 17);
    REQUIRE(delays[0] == 0); // text
    REQUIRE(delays[1] == 0); // own stream starts
    REQUIRE(delays[9] == 0); // other stream starts — the duplex mechanism
    for (int i = 2; i < 9; i++)
        REQUIRE(delays[i] == 1);
    for (int i = 10; i < 17; i++)
        REQUIRE(delays[i] == 1);

    personaplex_free(ctx);
}
