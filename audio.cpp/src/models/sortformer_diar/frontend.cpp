#include "engine/models/sortformer_diar/frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::sortformer_diar {

namespace {
using engine::debug::measure_ms;
}  // namespace

SortformerFeatureBatch compute_sortformer_features(
    const runtime::AudioBuffer & audio,
    const SortformerAssets & assets,
    int64_t threads,
    SortformerRunTimings * timings) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("Sortformer diar requires positive sample rate");
    }
    if (audio.sample_rate != assets.feature_config.sample_rate) {
        throw std::runtime_error("Sortformer diar currently requires 16 kHz input audio");
    }
    auto mono = audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (!mono.empty()) {
        const auto max_it = std::max_element(mono.begin(), mono.end());
        const float scale = 1.0f / (*max_it + 1.0e-3f);
        for (float & sample : mono) {
            sample *= scale;
        }
    }
    mono = audio::apply_preemphasis(std::move(mono), assets.feature_config.preemphasis);
    audio::AudioTensor mel;
    const auto log_mel_compute = [&]() {
        const audio::STFTConfig stft_config{
            assets.feature_config.n_fft,
            assets.feature_config.hop_length,
            assets.feature_config.win_length,
            true,
            // Caution: this is Based on NeMo Sortformer preprocessing, which uses
            // torch.stft(..., center=True, pad_mode="constant"). Other model
            // families may legitimately require a different STFT pad mode.
            audio::STFTPadMode::Constant,
        };
        mel = audio::LogMelSpectrogram().compute(
            mono,
            1,
            static_cast<int64_t>(mono.size()),
            audio.sample_rate,
            stft_config,
            assets.feature_config.num_mel_bins,
            static_cast<size_t>(std::max<int64_t>(1, threads)));
    };
    if (timings != nullptr) {
        timings->log_mel_ms += measure_ms(log_mel_compute);
    } else {
        log_mel_compute();
    }
    const int64_t raw_frames = mel.shape[2];
    const int64_t pad_amount = assets.feature_config.n_fft;
    const int64_t valid_frames = std::max<int64_t>(
        0,
        (static_cast<int64_t>(mono.size()) + pad_amount - assets.feature_config.n_fft)
            / assets.feature_config.hop_length);
    std::vector<int64_t> lengths = {std::min(valid_frames, raw_frames)};
    // Caution: use the valid frame count, not the raw centered-STFT frame count,
    // when sizing the padded feature buffer. Based on NeMo, the graph contract
    // tracks the semantically valid frame length, while the centered STFT can
    // expose one extra raw frame at exact boundaries.
    const int64_t frames = ((lengths.front() + 15) / 16) * 16;
    audio::FeatureNormalizeOutput normalized;
    const auto normalize_compute = [&]() {
        normalized = audio::FeatureNormalizer().compute(
            mel.values,
            lengths,
            1,
            assets.feature_config.num_mel_bins,
            raw_frames,
            audio::FeatureNormalizeType::PerFeature);
    };
    if (timings != nullptr) {
        timings->feature_normalizer_ms += measure_ms(normalize_compute);
    } else {
        normalize_compute();
    }

    SortformerFeatureBatch batch;
    batch.frames = frames;
    batch.valid_frames = lengths.front();
    batch.time_major.resize(static_cast<size_t>(frames * assets.feature_config.num_mel_bins), 0.0f);
    for (int64_t t = 0; t < batch.valid_frames; ++t) {
        for (int64_t m = 0; m < assets.feature_config.num_mel_bins; ++m) {
            const size_t dst = static_cast<size_t>((t * assets.feature_config.num_mel_bins) + m);
            batch.time_major[dst] = normalized.normalized.values[static_cast<size_t>((m * raw_frames) + t)];
        }
    }
    return batch;
}

}  // namespace engine::models::sortformer_diar
