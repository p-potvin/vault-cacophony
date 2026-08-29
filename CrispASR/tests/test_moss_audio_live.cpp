// MOSS-Audio-4B integration test — exercises ASR transcription and
// audio-understanding (custom prompt).
//
// Requires CRISPASR_MODEL_MOSS_AUDIO env var pointing to the GGUF.
// SKIPs cleanly when not set.

#include <catch2/catch_test_macros.hpp>

#include "moss_audio.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> load_wav_16k(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f) - 44;
    fseek(f, 44, SEEK_SET);
    std::vector<int16_t> raw(sz / 2);
    size_t n = fread(raw.data(), 2, raw.size(), f);
    (void)n;
    fclose(f);
    std::vector<float> pcm(raw.size());
    for (size_t i = 0; i < raw.size(); i++)
        pcm[i] = raw[i] / 32768.0f;
    return pcm;
}

TEST_CASE("moss-audio transcribe JFK", "[integration][moss-audio]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_MOSS_AUDIO");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_MOSS_AUDIO not set");
    }

    auto params = moss_audio_context_default_params();
    params.verbosity = 0;
    auto* ctx = moss_audio_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    auto pcm = load_wav_16k("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    char* text = moss_audio_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(text != nullptr);

    std::string result(text);
    free(text);

    INFO("transcript: " << result);
    CHECK(result.find("country") != std::string::npos);
    CHECK(result.size() > 20);

    moss_audio_free(ctx);
}

TEST_CASE("moss-audio custom prompt", "[integration][moss-audio]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_MOSS_AUDIO");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_MOSS_AUDIO not set");
    }

    auto params = moss_audio_context_default_params();
    params.verbosity = 0;
    auto* ctx = moss_audio_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    auto pcm = load_wav_16k("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    // Audio understanding: ask a question about the audio
    char* text = moss_audio_process(ctx, pcm.data(), (int)pcm.size(), "What language is spoken in this audio?");
    REQUIRE(text != nullptr);

    std::string result(text);
    free(text);

    INFO("response: " << result);
    // Should mention English in some form
    CHECK(result.size() > 5);

    moss_audio_free(ctx);
}
