// test-marking-policy.cpp — unit tests for the spoken-AI-disclaimer opt-out gate.
//
// Pins the policy #312 fixed: a voice-clone request that sets
// "spoken_disclaimer": false without an attestation must be DENIED THE OPT-OUT,
// not refused outright — v0.8.22/v0.8.23 returned 400 there, which took out
// voice cloning for every Subtitle Edit build up to v5.1.0-rc16.
//
// Header-only decision logic — no server, no socket, no models.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_marking_policy.h"

using crispasr_marking::decide;

TEST_CASE("non-clone output is not policed", "[unit][marking]") {
    // Nothing prepends a disclaimer to non-clone output, so the field is a
    // no-op rather than an opt-out — never a denial, never a disclaimer.
    for (bool asked : {true, false}) {
        auto d = decide(/*is_voice_clone=*/false, /*requested_spoken_disclaimer=*/asked, "", false, "");
        REQUIRE_FALSE(d.apply_spoken_disclaimer);
        REQUIRE_FALSE(d.optout_denied);
        REQUIRE_FALSE(d.optout_honored);
    }
}

TEST_CASE("a clone that does not ask to skip gets the disclaimer", "[unit][marking]") {
    auto d = decide(true, /*requested_spoken_disclaimer=*/true, "", false, "");
    REQUIRE(d.apply_spoken_disclaimer);
    REQUIRE_FALSE(d.optout_denied);
    REQUIRE_FALSE(d.optout_honored);
}

TEST_CASE("an attested opt-out is honored", "[unit][marking]") {
    auto d = decide(true, false, "I will disclose this is AI-generated", false, "");
    REQUIRE(d.optout_honored);
    REQUIRE_FALSE(d.apply_spoken_disclaimer);
    REQUIRE_FALSE(d.optout_denied);
    REQUIRE(d.attestation == "I will disclose this is AI-generated");
    REQUIRE(d.scope == "request");
}

TEST_CASE("#312: an unattested opt-out is denied, not refused", "[unit][marking]") {
    auto d = decide(true, /*requested_spoken_disclaimer=*/false, "", false, "");
    // The load-bearing assertion: the request still produces audio.
    REQUIRE(d.optout_denied);
    REQUIRE(d.apply_spoken_disclaimer); // served with the default, not rejected
    REQUIRE_FALSE(d.optout_honored);
    REQUIRE(d.attestation.empty());
    REQUIRE(d.scope.empty());
}

TEST_CASE("the server's --accept-marking-responsibility satisfies the gate", "[unit][marking]") {
    // The operator accepted the duty for every response the process serves,
    // which subsumes the per-request field.
    auto d = decide(true, false, "", /*server_accepted=*/true, "CLI --accept-marking-responsibility flag");
    REQUIRE(d.optout_honored);
    REQUIRE_FALSE(d.apply_spoken_disclaimer);
    REQUIRE(d.scope == "server");
    REQUIRE(d.attestation == "CLI --accept-marking-responsibility flag");
}

TEST_CASE("an operator attestation with no prose still names its source", "[unit][marking]") {
    auto d = decide(true, false, "", true, "");
    REQUIRE(d.optout_honored);
    REQUIRE(d.scope == "server");
    REQUIRE(d.attestation == "(unspecified)"); // audit line always has a source
}

TEST_CASE("the request attestation wins over the server one", "[unit][marking]") {
    auto d = decide(true, false, "per-request words", true, "server words");
    REQUIRE(d.optout_honored);
    REQUIRE(d.scope == "request");
    REQUIRE(d.attestation == "per-request words");
}

TEST_CASE("an attestation on a request that keeps the disclaimer changes nothing", "[unit][marking]") {
    // Sending marking_attestation without spoken_disclaimer:false is not an
    // opt-out; the disclaimer stays on and nothing is logged as honored.
    auto d = decide(true, /*requested_spoken_disclaimer=*/true, "I will disclose this is AI-generated", true, "x");
    REQUIRE(d.apply_spoken_disclaimer);
    REQUIRE_FALSE(d.optout_honored);
    REQUIRE_FALSE(d.optout_denied);
}
