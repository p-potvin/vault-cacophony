// crispasr_gap_fill.h — issue #89 gap-fill second pass, shared between the
// CLI dispatcher (crispasr_run.cpp) and the HTTP server (crispasr_server.cpp).
//
// Bounded-window backends (parakeet-ja) sometimes emit nothing for a
// multi-second span *inside* a slice — the encoder blanks an utterance
// whenever enough context follows it, even though the same span transcribes
// verbatim in isolation (measured: the issue #89 reporter's 60 s clip's
// first 4.6 s transcribe perfectly alone but are skipped inside any ≥8 s
// window). Second pass: find spans ≥ min_gap with no emitted words, run
// each through the backend in isolation, keep words that land inside the
// gap (and don't restate covered content), and merge them back. Silence
// gaps re-transcribe to nothing and cost little.
//
// Gate at the call site on backend.vad_slice_cap_seconds() > 0;
// CRISPASR_GAP_FILL=0 disables, CRISPASR_GAP_FILL_MIN_CS tunes the trigger
// (60 measured worse than the default 100: more variant noise, slower).

#pragma once

#include "crispasr_backend.h"
#include "crispasr_vad.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

// Rebuild display text from a word list. Two word conventions coexist:
// whisper/parakeet carry a leading space in word.text (" on") while
// granite's [T:N]-parsed words do not; CJK boundaries never get a space
// (#205 / 617cd02).
inline std::string crispasr_rebuild_text_from_words(const std::vector<crispasr_word>& words) {
    std::string rebuilt;
    for (const auto& w : words) {
        if (w.text.empty())
            continue;
        if (!rebuilt.empty()) {
            const unsigned char prev_last = (unsigned char)rebuilt.back();
            const unsigned char cur_first = (unsigned char)w.text[0];
            const bool already_spaced = (cur_first == ' ');
            const bool cjk_boundary = (prev_last >= 0xE0) || (cur_first >= 0xE0);
            if (!already_spaced && !cjk_boundary)
                rebuilt += ' ';
        }
        rebuilt += w.text;
    }
    if (!rebuilt.empty() && rebuilt[0] == ' ')
        rebuilt = rebuilt.substr(1);
    return rebuilt;
}

inline void crispasr_gap_fill_slice(CrispasrBackend& be, const whisper_params& params, const float* samples,
                                    int n_samples_total, int sample_rate, const crispasr_audio_slice& sl,
                                    std::vector<crispasr_segment>& segs) {
    int64_t min_gap_cs = 100; // 1.0 s of missing speech triggers a refill
    if (const char* e = getenv("CRISPASR_GAP_FILL_MIN_CS"))
        min_gap_cs = std::max((int64_t)30, (int64_t)atoi(e));
    constexpr int64_t kEdgePadCs = 20;   // extend the refill window slightly
    constexpr int64_t kCoverSlopCs = 30; // words this close bridge a gap

    const int max_rounds = 2;
    for (int round = 0; round < max_rounds; round++) {
        // Covered intervals from emitted words (fallback: segment spans).
        std::vector<std::pair<int64_t, int64_t>> covered;
        for (const auto& seg : segs) {
            if (seg.words.empty()) {
                if (seg.t1 > seg.t0)
                    covered.push_back({seg.t0, seg.t1});
                continue;
            }
            for (const auto& w : seg.words)
                covered.push_back({w.t0, std::max(w.t1, w.t0 + 1)});
        }
        std::sort(covered.begin(), covered.end());
        std::vector<std::pair<int64_t, int64_t>> merged;
        for (auto& iv : covered) {
            if (!merged.empty() && iv.first <= merged.back().second + kCoverSlopCs)
                merged.back().second = std::max(merged.back().second, iv.second);
            else
                merged.push_back(iv);
        }
        // Uncovered gaps within the slice.
        std::vector<std::pair<int64_t, int64_t>> gaps;
        int64_t cursor = sl.t0_cs;
        for (auto& iv : merged) {
            if (iv.first - cursor >= min_gap_cs)
                gaps.push_back({cursor, iv.first});
            cursor = std::max(cursor, iv.second);
        }
        if (sl.t1_cs - cursor >= min_gap_cs)
            gaps.push_back({cursor, sl.t1_cs});
        if (gaps.empty())
            return;

        bool recovered_any = false;
        for (auto& g : gaps) {
            const int64_t win0_cs = std::max(sl.t0_cs, g.first - kEdgePadCs);
            const int64_t win1_cs = std::min(sl.t1_cs, g.second + kEdgePadCs);
            const int s0 = std::max(0, (int)(win0_cs * sample_rate / 100));
            const int s1 = std::min(n_samples_total, (int)(win1_cs * sample_rate / 100));
            if (s1 - s0 < sample_rate / 4)
                continue;
            auto fill = be.transcribe(samples + s0, s1 - s0, win0_cs, params);
            crispasr_segment rec;
            rec.t0 = g.first;
            rec.t1 = g.second;
            // Keep only words inside the gap that don't restate content the
            // first pass (or an earlier refill) already emitted — refill
            // windows overlap covered speech at their edges and would
            // otherwise duplicate the boundary words.
            auto mid_is_covered = [&merged](int64_t mid) {
                for (const auto& iv : merged)
                    if (mid >= iv.first && mid < iv.second)
                        return true;
                return false;
            };
            for (auto& fseg : fill) {
                for (auto& w : fseg.words) {
                    const int64_t mid = (w.t0 + w.t1) / 2;
                    if (mid >= g.first - kCoverSlopCs && mid < g.second + kCoverSlopCs && !mid_is_covered(mid))
                        rec.words.push_back(std::move(w));
                }
            }
            if (rec.words.empty())
                continue;
            rec.text = crispasr_rebuild_text_from_words(rec.words);
            if (rec.text.empty())
                continue;
            rec.t0 = rec.words.front().t0;
            rec.t1 = rec.words.back().t1;
            segs.push_back(std::move(rec));
            recovered_any = true;
        }
        if (!recovered_any)
            return;
        std::sort(segs.begin(), segs.end(),
                  [](const crispasr_segment& a, const crispasr_segment& b) { return a.t0 < b.t0; });
    }
}
