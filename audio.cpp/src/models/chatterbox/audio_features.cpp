#include "components/component_weights.h"
#include "components/audio_processing.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace engine::models::chatterbox::components {
namespace {

struct KaldiFilterbankKey {
    int64_t sample_rate = 0;
    int64_t padded_window_size = 0;
    int64_t num_mels = 0;
    float low_freq = 0.0f;
    float high_freq = 0.0f;

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

std::vector<float> make_povey_window(int64_t window_size) {
    std::vector<float> window(static_cast<size_t>(window_size), 0.0f);
    constexpr float kPi = 3.14159265358979323846f;
    for (int64_t i = 0; i < window_size; ++i) {
        const float hann =
            0.5f - 0.5f * std::cos(2.0f * kPi * static_cast<float>(i) / static_cast<float>(window_size - 1));
        window[static_cast<size_t>(i)] = std::pow(hann, 0.85f);
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
    const float nyquist = 0.5f * static_cast<float>(sample_rate);
    if (high_freq <= 0.0f) {
        high_freq += nyquist;
    }
    const float fft_bin_width = static_cast<float>(sample_rate) / static_cast<float>(n_fft);
    const float mel_low = 1127.0f * std::log(1.0f + low_freq / 700.0f);
    const float mel_high = 1127.0f * std::log(1.0f + high_freq / 700.0f);
    const float mel_delta = (mel_high - mel_low) / static_cast<float>(n_mels + 1);

    std::vector<float> filterbank(static_cast<size_t>(n_mels * (num_fft_bins + 1)), 0.0f);
    for (int64_t mel_bin = 0; mel_bin < n_mels; ++mel_bin) {
        const float left_mel = mel_low + static_cast<float>(mel_bin) * mel_delta;
        const float center_mel = mel_low + static_cast<float>(mel_bin + 1) * mel_delta;
        const float right_mel = mel_low + static_cast<float>(mel_bin + 2) * mel_delta;
        for (int64_t fft_bin = 0; fft_bin < num_fft_bins; ++fft_bin) {
            const float freq = fft_bin_width * static_cast<float>(fft_bin);
            const float mel = 1127.0f * std::log(1.0f + freq / 700.0f);
            const float up_slope = (mel - left_mel) / std::max(center_mel - left_mel, 1.0e-12f);
            const float down_slope = (right_mel - mel) / std::max(right_mel - center_mel, 1.0e-12f);
            filterbank[static_cast<size_t>(mel_bin * (num_fft_bins + 1) + fft_bin)] =
                std::max(0.0f, std::min(up_slope, down_slope));
        }
    }
    return filterbank;
}

const std::vector<float> & get_cached_povey_window(int64_t window_size) {
    static std::mutex mutex;
    static std::unordered_map<int64_t, std::vector<float>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(window_size);
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
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    return cache.emplace(
        key,
        make_kaldi_mel_filterbank(sample_rate, padded_window_size, num_mels, low_freq, high_freq)).first->second;
}

}  // namespace

std::vector<float> resample_component_torchaudio_hann_mono(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate) {
    engine::audio::TorchaudioSincHannResampleOptions options;
    options.kernel_mode = engine::audio::TorchaudioSincHannKernelMode::Float64ComputationStoredAsFloat64;
    return engine::audio::resample_mono_torchaudio_sinc_hann(
        input,
        input_sample_rate,
        output_sample_rate,
        options);
}

S3TokenizerLogMelOutputs compute_s3tokenizer_log_mel(const runtime::AudioBuffer & audio) {
    if (audio.channels != 1) {
        throw std::runtime_error("S3Tokenizer log-mel expects mono audio");
    }
    std::vector<float> mono = audio.sample_rate == 16000
        ? audio.samples
        : resample_component_mono(audio.samples, audio.sample_rate, 16000);
    constexpr int64_t n_fft = 400;
    constexpr int64_t hop = 160;
    constexpr int64_t n_mels = 128;
    const int64_t pad = n_fft / 2;
    const int64_t padded_samples = static_cast<int64_t>(mono.size()) + 2 * pad;
    const int64_t total_frames = 1 + (padded_samples - n_fft) / hop;
    const int64_t freq_bins = (n_fft / 2) + 1;
    const int64_t frames = total_frames - 1;
    const auto filterbank = engine::audio::MelFilterbank().build(
        engine::audio::MelFilterbankConfig{16000, n_fft, n_mels, 0.0f, 8000.0f, true});
    const engine::audio::STFTConfig window_config{
        n_fft,
        hop,
        n_fft,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(window_config);
    const engine::audio::STFTConfig stft_config{
        n_fft,
        hop,
        n_fft,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Default,
    };
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        mono,
        window,
        1,
        static_cast<int64_t>(mono.size()),
        stft_config);

    S3TokenizerLogMelOutputs out;
    out.n_mels = n_mels;
    out.frames = frames;
    out.log_mel.assign(static_cast<size_t>(n_mels * frames), 0.0f);

#ifdef _OPENMP
#pragma omp parallel for if (frames > 8)
#endif
    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        for (int64_t mel_bin = 0; mel_bin < n_mels; ++mel_bin) {
            float sum = 0.0f;
            for (int64_t freq = 0; freq < freq_bins; ++freq) {
                const float mag = magnitude.values[static_cast<size_t>(freq * total_frames + frame_index)];
                sum += filterbank.values[static_cast<size_t>(mel_bin * freq_bins + freq)] * (mag * mag);
            }
            out.log_mel[static_cast<size_t>(mel_bin * frames + frame_index)] =
                std::log10(std::max(sum, 1.0e-10f));
        }
    }
    float max_value = -std::numeric_limits<float>::infinity();
    for (float value : out.log_mel) {
        max_value = std::max(max_value, value);
    }
    for (float & value : out.log_mel) {
        value = std::max(value, max_value - 8.0f);
        value = (value + 4.0f) / 4.0f;
    }
    return out;
}

CampplusFbankOutputs compute_campplus_fbank(const runtime::AudioBuffer & audio) {
    if (audio.channels != 1) {
        throw std::runtime_error("S3 speaker fbank expects mono audio");
    }

    std::vector<float> mono = audio.sample_rate == 16000
        ? audio.samples
        : resample_component_torchaudio_hann_mono(audio.samples, audio.sample_rate, 16000);

    constexpr int64_t sample_rate = 16000;
    constexpr int64_t window_size = 400;
    constexpr int64_t window_shift = 160;
    constexpr int64_t padded_window_size = 512;
    constexpr int64_t num_mels = 80;
    constexpr float low_freq = 20.0f;
    constexpr float high_freq = 0.0f;
    constexpr float preemphasis = 0.97f;
    constexpr float epsilon = std::numeric_limits<float>::epsilon();

    if (static_cast<int64_t>(mono.size()) < window_size) {
        return {};
    }

    const int64_t frames = 1 + (static_cast<int64_t>(mono.size()) - window_size) / window_shift;
    const auto & window = get_cached_povey_window(window_size);
    const auto & mel_filterbank = get_cached_kaldi_mel_filterbank(
        sample_rate,
        padded_window_size,
        num_mels,
        low_freq,
        high_freq);

    CampplusFbankOutputs outputs;
    outputs.frames = frames;
    outputs.dims = num_mels;
    outputs.features.assign(static_cast<size_t>(frames * num_mels), 0.0f);

    const int64_t freq_bins = (padded_window_size / 2) + 1;
    std::vector<float> frame(static_cast<size_t>(window_size), 0.0f);
    std::vector<float> stft_batch(static_cast<size_t>(frames * padded_window_size), 0.0f);
    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        const int64_t start = frame_index * window_shift;
        float mean = 0.0f;
        for (int64_t i = 0; i < window_size; ++i) {
            const float sample = mono[static_cast<size_t>(start + i)];
            frame[static_cast<size_t>(i)] = sample;
            mean += sample;
        }
        mean /= static_cast<float>(window_size);

        for (int64_t i = 0; i < window_size; ++i) {
            frame[static_cast<size_t>(i)] -= mean;
        }
        for (int64_t i = window_size - 1; i > 0; --i) {
            frame[static_cast<size_t>(i)] -= preemphasis * frame[static_cast<size_t>(i - 1)];
        }
        frame[0] -= preemphasis * frame[0];

        for (int64_t i = 0; i < window_size; ++i) {
            stft_batch[static_cast<size_t>(frame_index * padded_window_size + i)] =
                frame[static_cast<size_t>(i)];
        }
    }

    std::vector<float> stft_window(static_cast<size_t>(padded_window_size), 0.0f);
    std::copy(window.begin(), window.end(), stft_window.begin());
    const engine::audio::STFTConfig stft_config{
        padded_window_size,
        padded_window_size,
        padded_window_size,
        false,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Default,
    };
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        stft_batch,
        stft_window,
        frames,
        padded_window_size,
        stft_config);

    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        for (int64_t mel_bin = 0; mel_bin < num_mels; ++mel_bin) {
            float energy = 0.0f;
            for (int64_t freq = 0; freq <= padded_window_size / 2; ++freq) {
                const float mag = magnitude.values[static_cast<size_t>((frame_index * freq_bins) + freq)];
                energy += (mag * mag) *
                    mel_filterbank[static_cast<size_t>(mel_bin * ((padded_window_size / 2) + 1) + freq)];
            }
            outputs.features[static_cast<size_t>(frame_index * num_mels + mel_bin)] =
                std::log(std::max(energy, epsilon));
        }
    }

    for (int64_t mel_bin = 0; mel_bin < num_mels; ++mel_bin) {
        float mean = 0.0f;
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            mean += outputs.features[static_cast<size_t>(frame_index * num_mels + mel_bin)];
        }
        mean /= static_cast<float>(frames);
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            outputs.features[static_cast<size_t>(frame_index * num_mels + mel_bin)] -= mean;
        }
    }

    return outputs;
}


}  // namespace engine::models::chatterbox::components
