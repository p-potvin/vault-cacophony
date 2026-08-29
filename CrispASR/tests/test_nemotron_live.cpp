// tests/test_nemotron_live.cpp — integration tests for nemotron backend.
//
// Live tests: require model + audio files on disk. Gated behind [.live].
// Run: ctest -R test-nemotron --output-on-failure
//
// Env vars:
//   CRISPASR_MODEL_NEMOTRON    — Q4_K GGUF path
//   CRISPASR_MODEL_NEMOTRON_F16 — F16 GGUF path (optional)

#include <catch2/catch_test_macros.hpp>
#include "nemotron.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string get_env(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

// Chunk-skipping 16-bit PCM WAV loader.
static std::vector<float> load_wav_16k_mono(const std::string& path) {
    std::vector<float> pcm;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return pcm;

    char riff[12];
    if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
        fclose(f);
        return pcm;
    }

    int channels = 0, sample_rate = 0, bits = 0;
    int32_t data_size = 0;
    bool found_fmt = false, found_data = false;

    while (!found_data) {
        char chunk_id[4];
        int32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4)
            break;
        if (fread(&chunk_size, 4, 1, f) != 1)
            break;
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            char fmt[16];
            if (chunk_size < 16 || fread(fmt, 1, 16, f) != 16)
                break;
            channels = *(int16_t*)(fmt + 2);
            sample_rate = *(int32_t*)(fmt + 4);
            bits = *(int16_t*)(fmt + 14);
            found_fmt = true;
            if (chunk_size > 16)
                fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data || bits != 16 || channels < 1) {
        fclose(f);
        return pcm;
    }

    int n_samples = data_size / (channels * (bits / 8));
    std::vector<int16_t> raw(n_samples * channels);
    fread(raw.data(), sizeof(int16_t), raw.size(), f);
    fclose(f);

    pcm.resize(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float sum = 0;
        for (int c = 0; c < channels; c++)
            sum += raw[i * channels + c];
        pcm[i] = sum / (channels * 32768.0f);
    }
    return pcm;
}

TEST_CASE("nemotron: init and free", "[nemotron][.live]") {
    std::string model = get_env("CRISPASR_MODEL_NEMOTRON");
    if (model.empty()) {
        SKIP("CRISPASR_MODEL_NEMOTRON not set");
    }

    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    nemotron_context* ctx = nemotron_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);
    nemotron_free(ctx);
}

TEST_CASE("nemotron: JFK English transcription", "[nemotron][.live]") {
    std::string model = get_env("CRISPASR_MODEL_NEMOTRON");
    if (model.empty()) {
        SKIP("CRISPASR_MODEL_NEMOTRON not set");
    }

    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    nemotron_context* ctx = nemotron_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);

    char* text = nemotron_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(text != nullptr);

    std::string got(text);
    std::free(text);
    INFO("Transcription: " << got);

    // Check key phrases are present (case-insensitive substring match)
    auto contains_ci = [](const std::string& hay, const char* needle) {
        std::string h = hay, n = needle;
        for (auto& c : h)
            c = (char)tolower((unsigned char)c);
        for (auto& c : n)
            c = (char)tolower((unsigned char)c);
        return h.find(n) != std::string::npos;
    };

    CHECK(contains_ci(got, "fellow americans"));
    CHECK(contains_ci(got, "country"));
    CHECK(got.size() > 30);

    nemotron_free(ctx);
}

TEST_CASE("nemotron: F16 produces same text as Q4_K", "[nemotron][.live]") {
    std::string model_q4k = get_env("CRISPASR_MODEL_NEMOTRON");
    std::string model_f16 = get_env("CRISPASR_MODEL_NEMOTRON_F16");
    if (model_q4k.empty() || model_f16.empty()) {
        SKIP("CRISPASR_MODEL_NEMOTRON or CRISPASR_MODEL_NEMOTRON_F16 not set");
    }

    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;

    // Q4_K
    nemotron_context* ctx_q = nemotron_init_from_file(model_q4k.c_str(), cp);
    REQUIRE(ctx_q != nullptr);
    char* t_q = nemotron_transcribe(ctx_q, pcm.data(), (int)pcm.size());
    REQUIRE(t_q != nullptr);
    std::string text_q(t_q);
    std::free(t_q);
    nemotron_free(ctx_q);

    // F16
    nemotron_context* ctx_f = nemotron_init_from_file(model_f16.c_str(), cp);
    REQUIRE(ctx_f != nullptr);
    char* t_f = nemotron_transcribe(ctx_f, pcm.data(), (int)pcm.size());
    REQUIRE(t_f != nullptr);
    std::string text_f(t_f);
    std::free(t_f);
    nemotron_free(ctx_f);

    INFO("Q4_K: " << text_q);
    INFO("F16:  " << text_f);
    CHECK(text_q == text_f);
}
