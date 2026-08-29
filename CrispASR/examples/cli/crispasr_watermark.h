// crispasr_watermark.h — spread-spectrum audio watermark for AI-generated speech.
//
// Embeds an imperceptible pseudorandom pattern into the frequency domain
// of synthesized PCM audio. The watermark survives common transformations
// (re-encoding, moderate compression, volume normalization) because it
// is spread across many frequency bins and frames.
//
// Header-only so the unit tests can exercise it without linking the
// server translation unit.
//
// Algorithm:
//   1. Divide audio into overlapping frames (hop = frame/2).
//   2. For each frame, compute a real DFT via the Danielson-Lanczos
//      radix-2 FFT (no external dependency).
//   3. A PRNG seeded with CRISPASR_WATERMARK_KEY selects which frequency
//      bins to modulate and the sign (+/-) of each nudge.
//   4. Each selected bin's magnitude is nudged by `alpha` (default 0.005,
//      ~-46 dB below full scale — inaudible for speech).
//   5. Inverse FFT + overlap-add reconstructs the watermarked signal.
//
// Detection:
//   Same PRNG sequence → same bin selection. Correlate the sign pattern
//   against actual magnitude deltas across frames. High positive
//   correlation → watermark present.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// Watermark key — a fixed 64-bit seed that defines the pseudorandom bin
// selection. Changing this key produces a different (incompatible)
// watermark. In a production deployment you'd want this to be
// configurable; for now it's compiled-in.
#ifndef CRISPASR_WATERMARK_KEY
#define CRISPASR_WATERMARK_KEY 0x437269737041535FULL // "CrispASR" in hex-ish
#endif

// Number of frequency bins to modulate per frame. More bins = more
// robust detection but marginally higher distortion. 32 is a good
// balance for 1024-sample frames at 24 kHz.
#ifndef CRISPASR_WATERMARK_NBINS
#define CRISPASR_WATERMARK_NBINS 32
#endif

namespace crispasr_wm {

// Simple xoshiro128+ PRNG — deterministic, fast, no external deps.
struct prng {
    uint64_t s[2];

    explicit prng(uint64_t seed) {
        // SplitMix64 to initialize state from a single seed
        s[0] = splitmix(seed);
        s[1] = splitmix(s[0]);
    }

    uint64_t next() {
        const uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        const uint64_t result = s0 + s1;
        s1 ^= s0;
        s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
        s[1] = (s1 << 36) | (s1 >> 28);
        return result;
    }

    // Uniform in [0, bound)
    uint32_t next_u32(uint32_t bound) { return (uint32_t)(next() % bound); }

private:
    static uint64_t splitmix(uint64_t& x) {
        x += 0x9e3779b97f4a7c15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

// In-place radix-2 Cooley-Tukey FFT. `re` and `im` are arrays of
// length `n` (must be a power of 2). `inverse` = true for IFFT.
inline void fft_radix2(float* re, float* im, int n, bool inverse) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    const float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= n; len <<= 1) {
        const float angle = sign * 2.0f * 3.14159265358979323846f / (float)len;
        const float wre = std::cos(angle);
        const float wim = std::sin(angle);
        for (int i = 0; i < n; i += len) {
            float ure = 1.0f, uim = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                const float tre = re[i + j + len / 2] * ure - im[i + j + len / 2] * uim;
                const float tim = re[i + j + len / 2] * uim + im[i + j + len / 2] * ure;
                re[i + j + len / 2] = re[i + j] - tre;
                im[i + j + len / 2] = im[i + j] - tim;
                re[i + j] += tre;
                im[i + j] += tim;
                const float new_ure = ure * wre - uim * wim;
                uim = ure * wim + uim * wre;
                ure = new_ure;
            }
        }
    }
    if (inverse) {
        const float inv_n = 1.0f / (float)n;
        for (int i = 0; i < n; i++) {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}

// Generate the watermark bin indices and signs for one frame.
// Returns (bin_index, sign) pairs. `n_fft` is the FFT size.
// Only bins in [lo_bin, hi_bin) are candidates (avoid DC, Nyquist,
// and very low frequencies that would cause audible artifacts).
struct wm_bin {
    int index;
    int sign; // +1 or -1
};

// Watermark band + strength.
//
// v2 (issue #260): the comb lives in the SPEECH band (~1.5–4 kHz) where speech
// formants mask it. The original v1 spread the same 32-bin comb across
// ~1.5–11.7 kHz; ~20 of those bins landed at 4–11.8 kHz where clean TTS speech
// is near-silent, so the comb painted an audible "tinny" tone (a fixed set of
// horizontal lines on the spectrogram) over the high-frequency near-silence.
// Keeping the comb inside the speech band + lowering alpha makes it inaudible
// while detection stays well above threshold on real speech.
//
// CRISPASR_WATERMARK_LEGACY=1 restores the pre-#260 wideband/loud behavior so
// the two paths can be A/B'd and old watermarks re-detected. Both embed and
// detect read the same env, so they always agree on the band.
inline void wm_params(int n_fft, int& lo_bin, int& hi_bin, float& default_alpha) {
    lo_bin = n_fft / 16; // skip lowest ~6% (sub-bass) — ~1.5 kHz @ 24 kHz
    if (std::getenv("CRISPASR_WATERMARK_LEGACY")) {
        hi_bin = n_fft / 2 - 1; // ~11.7 kHz — audible comb (legacy A/B only)
        default_alpha = 0.08f;
    } else {
        // ~4.8 kHz: caps the comb below 5 kHz, removing the entire 5–12 kHz
        // "tinny" region that was the audible complaint, while reaching just
        // into the fricative/sibilance band so detection stays robust on tonal
        // voiced speech (~0.88 on real qwen3-tts vs 0.94 legacy).
        hi_bin = n_fft / 5;
        default_alpha = 0.05f;
    }
}

inline std::vector<wm_bin> generate_bin_pattern(uint64_t key, int n_fft, int n_bins, int lo_bin, int hi_bin) {
    prng rng(key);
    const int range = hi_bin - lo_bin;
    if (range <= 0 || n_bins <= 0)
        return {};

    std::vector<wm_bin> bins(n_bins);
    for (int i = 0; i < n_bins; i++) {
        bins[i].index = lo_bin + (int)rng.next_u32((uint32_t)range);
        bins[i].sign = (rng.next() & 1) ? 1 : -1;
    }
    return bins;
}

} // namespace crispasr_wm

// ---------------------------------------------------------------------------
// Inline implementation: embed watermark into float32 PCM
// ---------------------------------------------------------------------------

// Embed the CrispASR watermark into `pcm` (float32 mono, any sample rate).
// Modifies the samples in-place. `alpha` controls watermark strength:
//   0.08  = reliable detection on speech (~38 dB SNR, imperceptible)
//   0.05  = conservative (lower confidence on tonal speech)
//   0.005 = legacy (too faint for reliable detection on real speech)
//
// Industry standard: AudioSeal/WavMark use 38-42 dB SNR. Human perception
// threshold for speech masking is ~20 dB; 38 dB is 18 dB below perception.
//
// The function is a no-op for very short audio (< 1 FFT frame).
// alpha < 0 selects the band-appropriate default (0.05 speech-band, 0.08 legacy).
// alpha == 0 is an explicit zero-strength no-op (leaves the signal unchanged);
// alpha > 0 uses that exact strength.
inline void crispasr_watermark_embed_impl(float* pcm, int n_samples, float alpha = -1.0f) {
    const int n_fft = 1024;
    const int hop = n_fft / 2; // 50% overlap
    if (n_samples < n_fft)
        return;

    int lo_bin, hi_bin;
    float default_alpha;
    crispasr_wm::wm_params(n_fft, lo_bin, hi_bin, default_alpha);
    if (alpha < 0.0f)
        alpha = default_alpha;

    const auto bins =
        crispasr_wm::generate_bin_pattern(CRISPASR_WATERMARK_KEY, n_fft, CRISPASR_WATERMARK_NBINS, lo_bin, hi_bin);
    if (bins.empty())
        return;

    // Hann window
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; i++)
        window[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f * (float)i / (float)(n_fft - 1)));

    // Overlap-add buffers
    std::vector<float> out(n_samples, 0.0f);
    std::vector<float> norm(n_samples, 0.0f);
    std::vector<float> re(n_fft), im(n_fft);

    for (int start = 0; start + n_fft <= n_samples; start += hop) {
        // Window the frame
        for (int i = 0; i < n_fft; i++) {
            re[i] = pcm[start + i] * window[i];
            im[i] = 0.0f;
        }

        // Forward FFT
        crispasr_wm::fft_radix2(re.data(), im.data(), n_fft, false);

        // Compute RMS magnitude across the spectrum to set an absolute
        // watermark energy level. This way even bins with near-zero
        // natural energy (common for tonal signals) receive a detectable
        // nudge proportional to the overall frame energy.
        double rms_mag = 0.0;
        for (int k = 1; k < n_fft / 2; k++)
            rms_mag += (double)re[k] * re[k] + (double)im[k] * im[k];
        rms_mag = std::sqrt(rms_mag / (double)(n_fft / 2 - 1));
        const float nudge = alpha * (float)rms_mag;

        // Modulate the *magnitude* of selected bins while preserving
        // phase. This ensures the sign pattern is detectable via
        // magnitude measurement — adding to re alone would lose the
        // sign for near-zero bins.
        for (const auto& b : bins) {
            float mag = std::sqrt(re[b.index] * re[b.index] + im[b.index] * im[b.index]);
            float new_mag = mag + nudge * (float)b.sign;
            if (new_mag < 0.0f)
                new_mag = 0.0f;
            float scale = (mag > 1e-15f) ? (new_mag / mag) : 0.0f;
            // For bins with no energy, inject at 0-phase so detection
            // can still see the magnitude
            if (mag < 1e-15f && b.sign > 0) {
                re[b.index] = nudge;
                im[b.index] = 0.0f;
            } else {
                re[b.index] *= scale;
                im[b.index] *= scale;
            }
            // Mirror the conjugate half for real-valued signal
            int mirror = n_fft - b.index;
            if (mirror != b.index && mirror > 0 && mirror < n_fft) {
                if (mag < 1e-15f && b.sign > 0) {
                    re[mirror] = nudge;
                    im[mirror] = 0.0f;
                } else {
                    re[mirror] *= scale;
                    im[mirror] *= scale;
                }
            }
        }

        // Inverse FFT
        crispasr_wm::fft_radix2(re.data(), im.data(), n_fft, true);

        // Overlap-add with window
        for (int i = 0; i < n_fft; i++) {
            out[start + i] += re[i] * window[i];
            norm[start + i] += window[i] * window[i];
        }
    }

    // Normalize and write back only the watermark delta. Near the first and
    // last Hann-window edges the overlap-add normalization can be tiny; using
    // a short boundary ramp avoids turning that into an audible impulse.
    for (int i = 0; i < n_samples; i++) {
        if (norm[i] > 1e-4f) {
            float watermarked = out[i] / norm[i];
            float delta = watermarked - pcm[i];
            float ramp_in = std::min(1.0f, (float)i / (float)n_fft);
            float ramp_out = std::min(1.0f, (float)(n_samples - 1 - i) / (float)n_fft);
            pcm[i] += delta * std::min(ramp_in, ramp_out);
        }
    }
}

// ---------------------------------------------------------------------------
// Inline implementation: detect watermark in float32 PCM
// ---------------------------------------------------------------------------

// Returns a confidence score in [0, 1]. Values above 0.65 strongly
// indicate the CrispASR watermark is present. Values below 0.4 indicate
// no watermark (or a different key).
//
// Uses averaged-spectrum detection: computes the mean magnitude spectrum
// across all frames, then correlates the watermark bin pattern against
// the averaged spectrum. This is significantly more robust on tonal/speech
// signals than per-frame detection because frame-level noise averages out.
inline float crispasr_watermark_detect_impl(const float* pcm, int n_samples) {
    const int n_fft = 1024;
    const int hop = n_fft / 2;
    if (n_samples < n_fft)
        return 0.0f;

    int lo_bin, hi_bin;
    float default_alpha_unused;
    crispasr_wm::wm_params(n_fft, lo_bin, hi_bin, default_alpha_unused);
    const auto bins =
        crispasr_wm::generate_bin_pattern(CRISPASR_WATERMARK_KEY, n_fft, CRISPASR_WATERMARK_NBINS, lo_bin, hi_bin);
    if (bins.empty())
        return 0.0f;

    // Hann window
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; i++)
        window[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f * (float)i / (float)(n_fft - 1)));

    const int n_fft_half = n_fft / 2;
    std::vector<float> re(n_fft), im(n_fft);

    // Phase 1: Accumulate magnitude spectra across all frames
    std::vector<double> avg_mags(n_fft_half, 0.0);
    int n_frames = 0;

    for (int start = 0; start + n_fft <= n_samples; start += hop) {
        for (int i = 0; i < n_fft; i++) {
            re[i] = pcm[start + i] * window[i];
            im[i] = 0.0f;
        }
        crispasr_wm::fft_radix2(re.data(), im.data(), n_fft, false);

        for (int b = 0; b < n_fft_half; b++)
            avg_mags[b] += std::sqrt((double)re[b] * re[b] + (double)im[b] * im[b]);
        n_frames++;
    }

    if (n_frames == 0)
        return 0.0f;

    // Phase 2: Average (cancels per-frame noise, preserves watermark)
    for (int b = 0; b < n_fft_half; b++)
        avg_mags[b] /= (double)n_frames;

    // Phase 3: Correlate watermark pattern against averaged spectrum
    double correlation = 0.0;
    int valid_bins = 0;

    for (const auto& b : bins) {
        if (b.index >= n_fft_half)
            continue;
        double local_mean = 0.0;
        int count = 0;
        for (int d = -2; d <= 2; d++) {
            int nb = b.index + d;
            if (nb >= 1 && nb < n_fft_half && d != 0) {
                local_mean += avg_mags[nb];
                count++;
            }
        }
        if (count == 0)
            continue;
        local_mean /= (double)count;
        if (local_mean < 1e-12 && avg_mags[b.index] < 1e-12)
            continue;
        double ref = std::max(local_mean, (double)1e-12);
        double delta = (avg_mags[b.index] - local_mean) / ref;
        correlation += (delta > 0 ? 1.0 : -1.0) * (double)b.sign;
        valid_bins++;
    }

    if (valid_bins == 0)
        return 0.0f;

    double score = (correlation / (double)valid_bins + 1.0) / 2.0;
    if (score < 0.0)
        score = 0.0;
    if (score > 1.0)
        score = 1.0;
    return (float)score;
}
