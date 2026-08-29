#include "components/audio_processing.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"

#include "components/component_weights.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace engine::models::chatterbox::components {

namespace {

int64_t reflect_index(int64_t index, int64_t length) {
    while (index < 0 || index >= length) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * length - index - 2;
        }
    }
    return index;
}

std::vector<float> resample_component_mono_impl(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate) {
    engine::audio::SoxrResampleOptions options;
    options.profile = engine::audio::SoxrResampleProfile::QualityOnly;
    options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ActualOutput;
    options.output_padding = 256;
    options.reject_empty_output = true;
    options.warning_context = "Chatterbox S3 prompt mel";
    options.fallback_description = "linear resampling";
    return engine::audio::resample_mono_soxr_or_linear(input, input_sample_rate, output_sample_rate, options);
}

}  // namespace

std::vector<float> resample_component_mono(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate) {
    return resample_component_mono_impl(input, input_sample_rate, output_sample_rate);
}

S3PromptMelOutputs compute_s3_prompt_mel(const runtime::AudioBuffer & audio) {
    if (audio.channels != 1) {
        throw std::runtime_error("S3 prompt mel expects mono audio");
    }

    std::vector<float> mono = audio.sample_rate == 24000
        ? audio.samples
        : resample_component_mono(audio.samples, audio.sample_rate, 24000);
    constexpr int64_t n_fft = 1920;
    constexpr int64_t hop = 480;
    constexpr int64_t win = 1920;
    constexpr int64_t n_mels = 80;
    constexpr float kEps = 1.0e-9f;
    constexpr float kLogClamp = 1.0e-5f;

    const int64_t pad = (n_fft - hop) / 2;
    const int64_t padded_samples = static_cast<int64_t>(mono.size()) + 2 * pad;
    const int64_t freq_bins = (n_fft / 2) + 1;
    const int64_t frames = 1 + (padded_samples - n_fft) / hop;
    std::vector<float> padded(static_cast<size_t>(padded_samples), 0.0f);
    for (int64_t i = 0; i < padded_samples; ++i) {
        padded[static_cast<size_t>(i)] = mono[static_cast<size_t>(reflect_index(i - pad, static_cast<int64_t>(mono.size())))];
    }
    const engine::audio::STFTConfig stft_config{
        n_fft,
        hop,
        win,
        false,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const auto filterbank = engine::audio::MelFilterbank().build(
        engine::audio::MelFilterbankConfig{24000, n_fft, n_mels, 0.0f, 8000.0f, true});
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        padded,
        window,
        1,
        padded_samples,
        stft_config);

    S3PromptMelOutputs out;
    out.n_mels = n_mels;
    out.frames = frames;
    out.mel.assign(static_cast<size_t>(n_mels * frames), 0.0f);
#ifdef _OPENMP
#pragma omp parallel for if (frames > 8)
#endif
    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        for (int64_t mel_bin = 0; mel_bin < n_mels; ++mel_bin) {
            double sum = 0.0;
            for (int64_t freq = 0; freq < freq_bins; ++freq) {
                const float mag = magnitude.values[static_cast<size_t>(freq * frames + frame_index)];
                const float stabilized_mag = std::sqrt(mag * mag + kEps);
                sum += static_cast<double>(filterbank.values[static_cast<size_t>(mel_bin * freq_bins + freq)]) *
                       static_cast<double>(stabilized_mag);
            }
            out.mel[static_cast<size_t>(mel_bin * frames + frame_index)] =
                static_cast<float>(std::log(std::max(sum, static_cast<double>(kLogClamp))));
        }
    }
    return out;
}

}  // namespace engine::models::chatterbox::components
