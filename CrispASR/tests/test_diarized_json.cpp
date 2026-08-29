// test_diarized_json.cpp — Catch2 unit tests for the diarized_json formatter
// and the --max-len + word-level display segment splitting (issues #205, #206).
//
// Pure CPU, no model load. Exercises crispasr_segments_to_diarized_json() and
// crispasr_make_disp_segments() with synthetic segments.

#include "crispasr_output.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: build a segment with optional speaker and words.
// ---------------------------------------------------------------------------
static crispasr_segment make_seg(const std::string& text, int64_t t0, int64_t t1, const std::string& speaker = {},
                                 const std::vector<crispasr_word>& words = {}) {
    crispasr_segment s;
    s.text = text;
    s.t0 = t0;
    s.t1 = t1;
    s.speaker = speaker;
    s.words = words;
    return s;
}

// =========================================================================
// Issue #206: diarized_json formatter
// =========================================================================

TEST_CASE("diarized_json: empty segments produce valid JSON", "[unit][diarized_json]") {
    std::vector<crispasr_segment> segs;
    std::string out = crispasr_segments_to_diarized_json(segs, 0.0, "en", "transcribe", 0.0f);
    REQUIRE(out.find("\"segments\": [") != std::string::npos);
    REQUIRE(out.find("\"task\": \"transcribe\"") != std::string::npos);
    REQUIRE(out.find("\"duration\":") != std::string::npos);
}

TEST_CASE("diarized_json: speaker labels normalised to letters", "[unit][diarized_json]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg("Hello", 0, 100, "(speaker 0) "));
    segs.push_back(make_seg("World", 100, 200, "(speaker 1) "));
    segs.push_back(make_seg("Third", 200, 300, "(speaker 2) "));

    std::string out = crispasr_segments_to_diarized_json(segs, 3.0, "en", "transcribe", 0.0f);
    REQUIRE(out.find("\"speaker\": \"A\"") != std::string::npos);
    REQUIRE(out.find("\"speaker\": \"B\"") != std::string::npos);
    REQUIRE(out.find("\"speaker\": \"C\"") != std::string::npos);
}

TEST_CASE("diarized_json: empty speaker defaults to A", "[unit][diarized_json]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg("No diarization", 0, 500, ""));

    std::string out = crispasr_segments_to_diarized_json(segs, 5.0, "en", "transcribe", 0.0f);
    REQUIRE(out.find("\"speaker\": \"A\"") != std::string::npos);
}

TEST_CASE("diarized_json: timestamps in seconds", "[unit][diarized_json]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg("Test", 0, 261)); // 0 cs → 0.00 s, 261 cs → 2.61 s

    std::string out = crispasr_segments_to_diarized_json(segs, 2.61, "en", "transcribe", 0.0f);
    REQUIRE(out.find("\"start\": 0.00") != std::string::npos);
    REQUIRE(out.find("\"end\": 2.61") != std::string::npos);
}

TEST_CASE("diarized_json: type field present", "[unit][diarized_json]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg("Test", 0, 100));

    std::string out = crispasr_segments_to_diarized_json(segs, 1.0, "en", "transcribe", 0.0f);
    REQUIRE(out.find("\"type\": \"transcript.text.segment\"") != std::string::npos);
}

TEST_CASE("diarized_json: full text field present", "[unit][diarized_json]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg("Hello there", 0, 100, "(speaker 0) "));
    segs.push_back(make_seg("General Kenobi", 100, 200, "(speaker 1) "));

    std::string out = crispasr_segments_to_diarized_json(segs, 2.0, "en", "transcribe", 0.0f);
    REQUIRE(out.find("\"text\": \"Hello there General Kenobi\"") != std::string::npos);
}

TEST_CASE("diarized_json: word-level timestamps included when present", "[unit][diarized_json]") {
    std::vector<crispasr_word> words = {{"Hello", 0, 50}, {"there", 50, 100}};
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg("Hello there", 0, 100, "(speaker 0) ", words));

    std::string out = crispasr_segments_to_diarized_json(segs, 1.0, "en", "transcribe", 0.0f);
    REQUIRE(out.find("\"words\"") != std::string::npos);
    REQUIRE(out.find("\"word\": \"Hello\"") != std::string::npos);
    REQUIRE(out.find("\"word\": \"there\"") != std::string::npos);
}

// =========================================================================
// Issue #205: max_len splitting requires word data
// =========================================================================

TEST_CASE("make_disp_segments: max_len splits when words present", "[unit][max-len]") {
    // One long segment with 4 words, max_len=20 should split.
    std::vector<crispasr_word> words = {
        {"The", 0, 25},
        {"quick", 25, 50},
        {"brown", 50, 75},
        {"fox", 75, 100},
    };
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg("The quick brown fox", 0, 100, "", words));

    // max_len=10: "The quick" (9 chars) fits, next word overflows → split.
    // The second segment gets a leading space from the word-packing logic.
    auto disp = crispasr_make_disp_segments(segs, 10);
    REQUIRE(disp.size() >= 2);
    // First segment should be at most 10 chars.
    REQUIRE(disp[0].text.size() <= 10);
}

TEST_CASE("make_disp_segments: splits text by max_len without words (#205)", "[unit][max-len]") {
    // #205: text-only segments (granite/qwen3 non-plus, plus model in plain
    // mode) carry no word timings, but --max-len must still split the text on
    // word boundaries and interpolate timestamps.
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg("The quick brown fox jumps over the lazy dog", 0, 100));

    auto disp = crispasr_make_disp_segments(segs, 10);
    REQUIRE(disp.size() >= 2);
    // Every piece is at most max_len bytes and on word boundaries.
    std::string joined;
    for (const auto& d : disp) {
        REQUIRE(d.text.size() <= 10);
        REQUIRE(!d.text.empty());
        if (!joined.empty())
            joined += " ";
        joined += d.text;
    }
    REQUIRE(joined == "The quick brown fox jumps over the lazy dog");
    // Timestamps stay within the segment and advance monotonically.
    REQUIRE(disp.front().t0 == 0);
    REQUIRE(disp.back().t1 == 100);
    for (size_t i = 1; i < disp.size(); ++i)
        REQUIRE(disp[i].t0 >= disp[i - 1].t0);
}

TEST_CASE("make_disp_segments: max_len=0 does not split", "[unit][max-len]") {
    std::vector<crispasr_word> words = {{"Hello", 0, 50}, {"world", 50, 100}};
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg("Hello world", 0, 100, "", words));

    auto disp = crispasr_make_disp_segments(segs, 0);
    REQUIRE(disp.size() == 1);
}

TEST_CASE("make_disp_segments: max_len=1 gives one segment per word", "[unit][max-len]") {
    std::vector<crispasr_word> words = {{"Hello", 0, 50}, {"world", 50, 100}};
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg("Hello world", 0, 100, "", words));

    auto disp = crispasr_make_disp_segments(segs, 1);
    REQUIRE(disp.size() == 2);
    REQUIRE(disp[0].text == "Hello");
    REQUIRE(disp[1].text == "world");
}

TEST_CASE("make_disp_segments: speaker preserved through split", "[unit][max-len]") {
    std::vector<crispasr_word> words = {{"Hello", 0, 50}, {"world", 50, 100}};
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg("Hello world", 0, 100, "(speaker 0) ", words));

    auto disp = crispasr_make_disp_segments(segs, 1);
    REQUIRE(disp.size() == 2);
    REQUIRE(disp[0].speaker == "(speaker 0) ");
    REQUIRE(disp[1].speaker == "(speaker 0) ");
}

// =========================================================================
// Issue #292: chunk-local speaker scope (chunk_id)
// =========================================================================

TEST_CASE("diarized_json: chunk_id emitted only when >= 0", "[unit][diarized_json][issue-292]") {
    // Two segments in different chunks with the SAME "(speaker 0)" label — the
    // whole point of chunk_id is that these are NOT necessarily the same person.
    auto a = make_seg("hello", 0, 100, "(speaker 0) ");
    a.chunk_id = 0;
    auto b = make_seg("goodbye", 12000, 12100, "(speaker 0) ");
    b.chunk_id = 2;
    std::vector<crispasr_segment> segs = {a, b};
    std::string out = crispasr_segments_to_diarized_json(segs, 130.0, "en", "transcribe", 0.0f);
    // Both chunk ids present, so a consumer can distinguish chunk 0 from chunk 2.
    REQUIRE(out.find("\"chunk_id\": 0") != std::string::npos);
    REQUIRE(out.find("\"chunk_id\": 2") != std::string::npos);
}

TEST_CASE("diarized_json: single-pass run emits NO chunk_id", "[unit][diarized_json][issue-292]") {
    // Default chunk_id is -1 (single pass / unchunked). It must be absent so a
    // consumer never sees a spurious scope on a run that had exactly one chunk.
    auto s = make_seg("one pass", 0, 500, "(speaker 1) ");
    REQUIRE(s.chunk_id == -1); // the struct default
    std::vector<crispasr_segment> segs = {s};
    std::string out = crispasr_segments_to_diarized_json(segs, 5.0, "en", "transcribe", 0.0f);
    REQUIRE(out.find("chunk_id") == std::string::npos);
}
