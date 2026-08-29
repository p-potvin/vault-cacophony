#include "engine/framework/audio/wav_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace engine::audio {

void write_pcm16_wav(
    const std::filesystem::path & path,
    int sample_rate,
    int channel_count,
    const std::vector<float> & audio) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("could not open WAV output: " + path.string());
    }
    if (sample_rate <= 0) {
        throw std::runtime_error("sample rate must be positive");
    }
    if (channel_count <= 0) {
        throw std::runtime_error("channel count must be positive");
    }
    if (audio.size() % static_cast<size_t>(channel_count) != 0) {
        throw std::runtime_error("audio sample count must be divisible by channel count");
    }
    const uint16_t channels = static_cast<uint16_t>(channel_count);
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t data_bytes = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_bytes;
    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char *>(&riff_size), 4);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    const uint32_t fmt_size = 16;
    const uint16_t audio_format = 1;
    out.write(reinterpret_cast<const char *>(&fmt_size), 4);
    out.write(reinterpret_cast<const char *>(&audio_format), 2);
    out.write(reinterpret_cast<const char *>(&channels), 2);
    out.write(reinterpret_cast<const char *>(&sample_rate), 4);
    out.write(reinterpret_cast<const char *>(&byte_rate), 4);
    out.write(reinterpret_cast<const char *>(&block_align), 2);
    out.write(reinterpret_cast<const char *>(&bits_per_sample), 2);
    out.write("data", 4);
    out.write(reinterpret_cast<const char *>(&data_bytes), 4);
    for (float sample : audio) {
        sample = std::max(-1.0F, std::min(1.0F, sample));
        const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
        out.write(reinterpret_cast<const char *>(&pcm), sizeof(pcm));
    }
}

}  // namespace engine::audio
