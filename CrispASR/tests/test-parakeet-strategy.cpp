// test-parakeet-strategy.cpp — unit tests for parakeet_pick_strategy, the pure
// long-audio routing decision hoisted into the shared orchestration
// (improvements Phase 1). Pinning it here dedups the decision that used to be
// written twice (CLI adapter + session C-ABI) and was the source of the JA /
// #257 routing bugs.

#include <catch2/catch_test_macros.hpp>

#include "parakeet_orchestrate.h"

namespace {
constexpr int SR = 16000;
parakeet_strategy_in mk(int secs, bool is_ja, bool chunk_explicit, int chunk_s, int thr, bool longform) {
    parakeet_strategy_in in;
    in.n_samples = secs * SR;
    in.sample_rate = SR;
    in.is_ja = is_ja;
    in.chunk_seconds_explicit = chunk_explicit;
    in.chunk_seconds = chunk_s;
    in.stream_threshold_s = thr;
    in.longform_enabled = longform;
    return in;
}
} // namespace

TEST_CASE("non-JA explicit --chunk-seconds → CHUNK_SEGMENTED at any length",
          "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(20, false, true, 7, 300, true)) == parakeet_strategy::CHUNK_SEGMENTED);
    REQUIRE(parakeet_pick_strategy(mk(600, false, true, 7, 300, true)) == parakeet_strategy::CHUNK_SEGMENTED);
}

TEST_CASE("non-JA short (<= cap) → SINGLE_PASS", "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(11, false, false, 0, 300, true)) == parakeet_strategy::SINGLE_PASS);
    REQUIRE(parakeet_pick_strategy(mk(225, false, false, 0, 300, true)) == parakeet_strategy::SINGLE_PASS);
    REQUIRE(parakeet_pick_strategy(mk(300, false, false, 0, 300, true)) == parakeet_strategy::SINGLE_PASS);
}

TEST_CASE("non-JA long (> cap) + longform → LONGFORM", "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(301, false, false, 0, 300, true)) == parakeet_strategy::LONGFORM);
}

TEST_CASE("non-JA long + longform disabled → STREAMED", "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(600, false, false, 0, 300, false)) == parakeet_strategy::STREAMED);
}

TEST_CASE("threshold 0 (always-streamed) → STREAMED", "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(11, false, false, 0, 0, false)) == parakeet_strategy::STREAMED);
    REQUIRE(parakeet_pick_strategy(mk(600, false, false, 0, 0, false)) == parakeet_strategy::STREAMED);
}

TEST_CASE("JA never takes the CHUNK_SEGMENTED branch", "[unit][parakeet-strategy][improvements]") {
    // JA caller passes threshold=12, longform=false. Explicit chunk is ignored
    // (is_ja gate) → falls to the length logic.
    REQUIRE(parakeet_pick_strategy(mk(8, true, true, 7, 12, false)) == parakeet_strategy::SINGLE_PASS);
    REQUIRE(parakeet_pick_strategy(mk(30, true, true, 7, 12, false)) == parakeet_strategy::STREAMED);
}

// ---- Phase 2: memory policy ----

TEST_CASE("est_singlepass_peak_mb matches the reporter's ~1.9 GiB data point",
          "[unit][parakeet-strategy][improvements]") {
    // ~4 min clip → T_enc≈2800, 8 heads, default coeff 8.0 → ~1914 MiB.
    const double est = parakeet_est_singlepass_peak_mb(2800, 8, 8.0);
    REQUIRE(est > 1800.0);
    REQUIRE(est < 2100.0);
    // O(T^2): doubling T ~quadruples the estimate.
    REQUIRE(parakeet_est_singlepass_peak_mb(5600, 8, 8.0) > 3.9 * est);
    // degenerate inputs → 0
    REQUIRE(parakeet_est_singlepass_peak_mb(0, 8, 8.0) == 0.0);
    REQUIRE(parakeet_est_singlepass_peak_mb(2800, 0, 8.0) == 0.0);
}

TEST_CASE("singlepass_fits_budget gating", "[unit][parakeet-strategy][improvements]") {
    // Disabled: budget 0 or coeff 0 → always fits (historical behaviour).
    REQUIRE(parakeet_singlepass_fits_budget(9999, 8, 0.0, 8.0));
    REQUIRE(parakeet_singlepass_fits_budget(9999, 8, 100.0, 0.0));
    // ~4 min clip fits a 3.7 GiB card but not a 1.5 GiB budget.
    REQUIRE(parakeet_singlepass_fits_budget(2800, 8, 3700.0, 8.0));
    REQUIRE_FALSE(parakeet_singlepass_fits_budget(2800, 8, 1500.0, 8.0));
    // A short clip (T≈250, 20 s) fits even a tiny budget.
    REQUIRE(parakeet_singlepass_fits_budget(250, 8, 256.0, 8.0));
}
