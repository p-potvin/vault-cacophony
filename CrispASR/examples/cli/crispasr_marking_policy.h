// crispasr_marking_policy.h — who may opt out of the spoken AI-disclaimer.
//
// The rule the server applies to a /v1/audio/speech request that sets
// "spoken_disclaimer": false, in one weight-free place so it can be unit-tested
// without a model, a socket, or a server process (tests/test-marking-policy.cpp).
// It lived inline in the request handler until #312, where a policy change went
// four days before anyone noticed it had broken every released client — a gate
// nobody can test on the tier CI actually runs is a gate that ships wrong.
//
// The policy itself:
//   - Only voice clones get a spoken disclaimer at all, so only they can opt out.
//   - Opting out requires an attestation that the caller accepts the AI-content
//     disclosure duty: the request's own `marking_attestation`, or a server
//     launched with --accept-marking-responsibility (the operator accepted that
//     duty for EVERY response the process serves, which subsumes the per-request
//     field — checking only the body refuses requests the operator has covered).
//   - #312: an unattested opt-out is DENIED, not refused. The caller doesn't get
//     the opt-out; it still gets its audio, with the disclaimer the default
//     prescribes. Serving the STRONGER default cannot emit weaker-than-default
//     output — which is the entire point of the gate — while refusing outright
//     only turns a client one field behind into a client with no TTS at all.
//     The caller is told (headers + audit log); the denial is never silent.

#pragma once

#include <string>

namespace crispasr_marking {

// What actually happens to this response, as opposed to what was asked for.
struct Decision {
    // Prepend the spoken AI-disclosure to the audio. True for a clone that
    // either didn't ask to skip it, or asked without attesting.
    bool apply_spoken_disclaimer = false;
    // The caller asked to skip it and attested — honored.
    bool optout_honored = false;
    // The caller asked to skip it without attesting — denied, served with the
    // disclaimer. The caller learns this from the response headers.
    bool optout_denied = false;
    // The attestation recorded in the audit log; set only when honored.
    std::string attestation;
    // Where that attestation came from: "request" or "server". Empty unless
    // honored.
    std::string scope;
};

// `server_accepted` / `server_attestation` are the launch-time
// --accept-marking-responsibility state (whisper_params::
// tts_marking_responsibility_accepted / tts_marking_attestation).
inline Decision decide(bool is_voice_clone, bool requested_spoken_disclaimer, const std::string& request_attestation,
                       bool server_accepted, const std::string& server_attestation) {
    Decision d;
    // Non-clone output carries no spoken disclaimer in the first place, so
    // "spoken_disclaimer": false is a no-op rather than an opt-out to police.
    if (!is_voice_clone)
        return d;
    if (requested_spoken_disclaimer) {
        d.apply_spoken_disclaimer = true;
        return d;
    }
    if (!request_attestation.empty()) {
        d.optout_honored = true;
        d.attestation = request_attestation;
        d.scope = "request";
        return d;
    }
    if (server_accepted) {
        d.optout_honored = true;
        // The flag can be passed without prose; record that rather than an
        // empty string, so the audit line always names a source.
        d.attestation = server_attestation.empty() ? "(unspecified)" : server_attestation;
        d.scope = "server";
        return d;
    }
    d.optout_denied = true;
    d.apply_spoken_disclaimer = true;
    return d;
}

} // namespace crispasr_marking
