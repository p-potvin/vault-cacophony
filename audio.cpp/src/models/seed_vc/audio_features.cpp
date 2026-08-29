#include "engine/models/seed_vc/audio_features.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/waveform_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace engine::models::seed_vc {
namespace {

struct KaldiFilterbankKey {
    int64_t sample_rate = 0;
    int64_t padded_window_size = 0;
    int64_t num_mels = 0;
    float low_freq = 0.0F;
    float high_freq = 0.0F;

    bool operator==(const KaldiFilterbankKey & other) const noexcept {
        return sample_rate == other.sample_rate &&
            padded_window_size == other.padded_window_size &&
            num_mels == other.num_mels &&
            low_freq == other.low_freq &&
            high_freq == other.high_freq;
    }
};

struct KaldiFilterbankKeyHash {
    size_t operator()(const KaldiFilterbankKey & key) const noexcept {
        size_t seed = 0;
        seed ^= std::hash<int64_t>{}(key.sample_rate) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.padded_window_size) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.num_mels) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<float>{}(key.low_freq) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<float>{}(key.high_freq) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct SeedVcMelFilterbankKey {
    int64_t sample_rate = 0;
    int64_t n_fft = 0;
    int64_t num_mels = 0;
    float fmin = 0.0F;
    float fmax = 0.0F;

    bool operator==(const SeedVcMelFilterbankKey & other) const noexcept {
        return sample_rate == other.sample_rate && n_fft == other.n_fft && num_mels == other.num_mels &&
            fmin == other.fmin && fmax == other.fmax;
    }
};

struct SeedVcMelFilterbankKeyHash {
    size_t operator()(const SeedVcMelFilterbankKey & key) const noexcept {
        size_t seed = std::hash<int64_t>{}(key.sample_rate);
        seed ^= std::hash<int64_t>{}(key.n_fft) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.num_mels) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<float>{}(key.fmin) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<float>{}(key.fmax) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

std::vector<float> make_povey_window(int64_t window_size) {
    std::vector<float> window(static_cast<size_t>(window_size), 0.0F);
    constexpr float kPi = 3.14159265358979323846F;
    for (int64_t i = 0; i < window_size; ++i) {
        const float hann =
            0.5F - 0.5F * std::cos(2.0F * kPi * static_cast<float>(i) / static_cast<float>(window_size - 1));
        window[static_cast<size_t>(i)] = std::pow(hann, 0.85F);
    }
    return window;
}

std::vector<float> make_kaldi_mel_filterbank(
    int64_t sample_rate,
    int64_t n_fft,
    int64_t n_mels,
    float low_freq,
    float high_freq) {
    const int64_t num_fft_bins = n_fft / 2;
    const float nyquist = 0.5F * static_cast<float>(sample_rate);
    if (high_freq <= 0.0F) {
        high_freq += nyquist;
    }
    const float fft_bin_width = static_cast<float>(sample_rate) / static_cast<float>(n_fft);
    const float mel_low = 1127.0F * std::log(1.0F + low_freq / 700.0F);
    const float mel_high = 1127.0F * std::log(1.0F + high_freq / 700.0F);
    const float mel_delta = (mel_high - mel_low) / static_cast<float>(n_mels + 1);

    std::vector<float> filterbank(static_cast<size_t>(n_mels * (num_fft_bins + 1)), 0.0F);
    for (int64_t mel_bin = 0; mel_bin < n_mels; ++mel_bin) {
        const float left_mel = mel_low + static_cast<float>(mel_bin) * mel_delta;
        const float center_mel = mel_low + static_cast<float>(mel_bin + 1) * mel_delta;
        const float right_mel = mel_low + static_cast<float>(mel_bin + 2) * mel_delta;
        for (int64_t fft_bin = 0; fft_bin < num_fft_bins; ++fft_bin) {
            const float freq = fft_bin_width * static_cast<float>(fft_bin);
            const float mel = 1127.0F * std::log(1.0F + freq / 700.0F);
            const float up_slope = (mel - left_mel) / std::max(center_mel - left_mel, 1.0e-12F);
            const float down_slope = (right_mel - mel) / std::max(right_mel - center_mel, 1.0e-12F);
            filterbank[static_cast<size_t>(mel_bin * (num_fft_bins + 1) + fft_bin)] =
                std::max(0.0F, std::min(up_slope, down_slope));
        }
    }
    return filterbank;
}

const std::vector<float> & get_cached_povey_window(int64_t window_size) {
    static std::mutex mutex;
    static std::unordered_map<int64_t, std::vector<float>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(window_size);
    if (it != cache.end()) {
        return it->second;
    }
    return cache.emplace(window_size, make_povey_window(window_size)).first->second;
}

const std::vector<float> & get_cached_kaldi_mel_filterbank(
    int64_t sample_rate,
    int64_t padded_window_size,
    int64_t num_mels,
    float low_freq,
    float high_freq) {
    static std::mutex mutex;
    static std::unordered_map<KaldiFilterbankKey, std::vector<float>, KaldiFilterbankKeyHash> cache;
    const KaldiFilterbankKey key{sample_rate, padded_window_size, num_mels, low_freq, high_freq};
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    return cache.emplace(
        key,
        make_kaldi_mel_filterbank(sample_rate, padded_window_size, num_mels, low_freq, high_freq)).first->second;
}

const engine::audio::SparseMelFilterbank & get_cached_seed_vc_mel_filterbank(const SeedVcMelConfig & config) {
    static std::mutex mutex;
    static std::unordered_map<SeedVcMelFilterbankKey, engine::audio::SparseMelFilterbank, SeedVcMelFilterbankKeyHash>
        cache;
    const SeedVcMelFilterbankKey key{
        config.sample_rate,
        config.n_fft,
        config.num_mels,
        config.fmin,
        config.fmax,
    };
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    return cache
        .emplace(
            key,
            engine::audio::MelFilterbank().prepare_sparse(engine::audio::MelFilterbank().build({
                config.sample_rate,
                config.n_fft,
                config.num_mels,
                config.fmin,
                config.fmax,
                true,
            })))
        .first->second;
}

}  // namespace

namespace {

std::vector<float> validated_seed_vc_mono_samples(const std::vector<float> & samples, int channels) {
    if (channels <= 0) {
        throw std::runtime_error("Seed-VC audio channel count must be positive");
    }
    if (samples.empty()) {
        throw std::runtime_error("Seed-VC audio must not be empty");
    }
    if (samples.size() % static_cast<size_t>(channels) != 0) {
        throw std::runtime_error("Seed-VC audio sample count must be divisible by channels");
    }
    if (channels == 1) {
        return samples;
    }
    return engine::audio::mixdown_interleaved_to_mono_average(samples, channels);
}

}  // namespace

namespace {

std::vector<float> seed_vc_resample_mono(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate) {
    if (input_sample_rate <= 0 || output_sample_rate <= 0) {
        throw std::runtime_error("Seed-VC resampling requires positive sample rates");
    }
    if (input_sample_rate == output_sample_rate || input.empty()) {
        return input;
    }
    engine::audio::SoxrResampleOptions options;
    options.profile = engine::audio::SoxrResampleProfile::ExplicitFloat32Runtime;
    options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ExactExpected;
    options.require_full_input = true;
    options.warning_context = "Seed-VC";
    options.fallback_description = "torchaudio-compatible resampling";
    if (auto output = engine::audio::try_resample_mono_soxr(input, input_sample_rate, output_sample_rate, options)) {
        return *output;
    }
    return engine::audio::resample_mono_torchaudio_sinc_hann(input, input_sample_rate, output_sample_rate);
}

}  // namespace

SeedVcPreparedAudio seed_vc_prepare_audio(
    const std::vector<float> & samples,
    int sample_rate,
    int channels,
    int64_t max_22k_samples) {
    auto wave_22k = seed_vc_resample_mono(validated_seed_vc_mono_samples(samples, channels), sample_rate, 22050);
    if (max_22k_samples > 0 && static_cast<int64_t>(wave_22k.size()) > max_22k_samples) {
        engine::audio::truncate_samples_to_count(wave_22k, static_cast<size_t>(max_22k_samples));
    }
    SeedVcPreparedAudio out;
    out.waveform_16k = seed_vc_resample_mono(wave_22k, 22050, 16000);
    out.waveform_22k = std::move(wave_22k);
    return out;
}

SeedVcPreparedAudioForSampleRate seed_vc_prepare_audio_for_sample_rate(
    const std::vector<float> & samples,
    int sample_rate,
    int channels,
    int output_sample_rate,
    int64_t max_output_samples) {
    auto waveform = seed_vc_resample_mono(validated_seed_vc_mono_samples(samples, channels), sample_rate, output_sample_rate);
    if (max_output_samples > 0 && static_cast<int64_t>(waveform.size()) > max_output_samples) {
        engine::audio::truncate_samples_to_count(waveform, static_cast<size_t>(max_output_samples));
    }
    SeedVcPreparedAudioForSampleRate out;
    out.waveform_16k = engine::audio::resample_mono_torchaudio_sinc_hann(waveform, output_sample_rate, 16000);
    out.waveform = std::move(waveform);
    return out;
}

SeedVcMelSpectrogramOutput compute_seed_vc_mel_spectrogram(
    const std::vector<float> & waveform,
    const SeedVcMelConfig & config,
    size_t threads) {
    if (config.sample_rate <= 0 || config.n_fft <= 0 || config.win_size <= 0 ||
        config.hop_size <= 0 || config.num_mels <= 0) {
        throw std::runtime_error("Seed-VC mel spectrogram config is invalid");
    }
    const int64_t pad = (config.n_fft - config.hop_size) / 2;
    const auto padded = engine::audio::reflect_pad_samples(waveform, pad, pad);
    const engine::audio::STFTConfig stft_config{
        config.n_fft,
        config.hop_size,
        config.win_size,
        false,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        padded,
        window,
        1,
        static_cast<int64_t>(padded.size()),
        stft_config,
        threads);
    auto magnitude_values = magnitude.values;
    for (float & value : magnitude_values) {
        value = std::sqrt(value * value + 1.0e-9F);
    }
    const auto & filterbank = get_cached_seed_vc_mel_filterbank(config);
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t frames = magnitude.shape[2];
    if (filterbank.dense.shape.size() != 2 || filterbank.dense.shape[0] != config.num_mels ||
        filterbank.dense.shape[1] != freq_bins) {
        throw std::runtime_error("Seed-VC mel filterbank shape mismatch");
    }
    engine::audio::AudioTensor mel;
    mel.shape = {1, config.num_mels, frames};
    mel.values.assign(static_cast<size_t>(config.num_mels * frames), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(config.num_mels * frames >= 4096)
#endif
    for (int64_t mel_bin = 0; mel_bin < config.num_mels; ++mel_bin) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            float sum = 0.0F;
            const int64_t start = filterbank.starts[static_cast<size_t>(mel_bin)];
            const int64_t end = filterbank.ends[static_cast<size_t>(mel_bin)];
            for (int64_t freq = start; freq < end; ++freq) {
                sum += filterbank.dense.values[static_cast<size_t>(mel_bin * freq_bins + freq)] *
                    magnitude_values[static_cast<size_t>(freq * frames + frame)];
            }
            mel.values[static_cast<size_t>(mel_bin * frames + frame)] = std::log(std::max(sum, 1.0e-5F));
        }
    }
    SeedVcMelSpectrogramOutput out;
    out.mel = std::move(mel.values);
    out.channels = config.num_mels;
    out.frames = mel.shape[2];
    return out;
}

SeedVcCampplusFbankOutput compute_seed_vc_campplus_fbank_16k(const std::vector<float> & waveform_16k) {
    constexpr int64_t kSampleRate = 16000;
    constexpr int64_t kWindowSize = 400;
    constexpr int64_t kWindowShift = 160;
    constexpr int64_t kPaddedWindowSize = 512;
    constexpr int64_t kNumMels = 80;
    constexpr float kLowFreq = 20.0F;
    constexpr float kHighFreq = 0.0F;
    constexpr float kPreemphasis = 0.97F;
    constexpr float kEpsilon = std::numeric_limits<float>::epsilon();

    if (static_cast<int64_t>(waveform_16k.size()) < kWindowSize) {
        return {};
    }

    const int64_t frames = 1 + (static_cast<int64_t>(waveform_16k.size()) - kWindowSize) / kWindowShift;
    const auto & window = get_cached_povey_window(kWindowSize);
    const auto & mel_filterbank = get_cached_kaldi_mel_filterbank(
        kSampleRate,
        kPaddedWindowSize,
        kNumMels,
        kLowFreq,
        kHighFreq);

    const int64_t freq_bins = (kPaddedWindowSize / 2) + 1;
    std::vector<float> frame(static_cast<size_t>(kWindowSize), 0.0F);
    std::vector<float> stft_batch(static_cast<size_t>(frames * kPaddedWindowSize), 0.0F);
    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        const int64_t start = frame_index * kWindowShift;
        float mean = 0.0F;
        for (int64_t i = 0; i < kWindowSize; ++i) {
            const float sample = waveform_16k[static_cast<size_t>(start + i)];
            frame[static_cast<size_t>(i)] = sample;
            mean += sample;
        }
        mean /= static_cast<float>(kWindowSize);
        for (int64_t i = 0; i < kWindowSize; ++i) {
            frame[static_cast<size_t>(i)] -= mean;
        }
        for (int64_t i = kWindowSize - 1; i > 0; --i) {
            frame[static_cast<size_t>(i)] -= kPreemphasis * frame[static_cast<size_t>(i - 1)];
        }
        frame[0] -= kPreemphasis * frame[0];
        for (int64_t i = 0; i < kWindowSize; ++i) {
            stft_batch[static_cast<size_t>(frame_index * kPaddedWindowSize + i)] =
                frame[static_cast<size_t>(i)] * window[static_cast<size_t>(i)];
        }
    }

    std::vector<float> stft_window(static_cast<size_t>(kPaddedWindowSize), 0.0F);
    std::fill(stft_window.begin(), stft_window.end(), 1.0F);
    const engine::audio::STFTConfig stft_config{
        kPaddedWindowSize,
        kPaddedWindowSize,
        kPaddedWindowSize,
        false,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Default,
    };
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        stft_batch,
        stft_window,
        frames,
        kPaddedWindowSize,
        stft_config);

    SeedVcCampplusFbankOutput outputs;
    outputs.frames = frames;
    outputs.dims = kNumMels;
    outputs.features.assign(static_cast<size_t>(frames * kNumMels), 0.0F);
    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        for (int64_t mel_bin = 0; mel_bin < kNumMels; ++mel_bin) {
            float energy = 0.0F;
            for (int64_t freq = 0; freq < freq_bins; ++freq) {
                const float mag = magnitude.values[static_cast<size_t>(frame_index * freq_bins + freq)];
                energy += (mag * mag) *
                    mel_filterbank[static_cast<size_t>(mel_bin * freq_bins + freq)];
            }
            outputs.features[static_cast<size_t>(frame_index * kNumMels + mel_bin)] =
                std::log(std::max(energy, kEpsilon));
        }
    }

    for (int64_t mel_bin = 0; mel_bin < kNumMels; ++mel_bin) {
        float mean = 0.0F;
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            mean += outputs.features[static_cast<size_t>(frame_index * kNumMels + mel_bin)];
        }
        mean /= static_cast<float>(frames);
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            outputs.features[static_cast<size_t>(frame_index * kNumMels + mel_bin)] -= mean;
        }
    }
    return outputs;
}

}  // namespace engine::models::seed_vc
