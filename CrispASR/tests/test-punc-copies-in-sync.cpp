// test-punc-copies-in-sync.cpp — the two fireredpunc implementations must not drift.
//
// `crisp_punc/src/fireredpunc.cpp` is the shared library CrispASR normally links
// (and CrispEmbed consumes via add_subdirectory). `src/fireredpunc.cpp` is the
// fallback src/CMakeLists.txt builds when the crisp_punc/ directory is absent
// from a checkout. They are the same implementation twice.
//
// That duplication silently ate a bug fix. #308 fixed a capitalisation defect —
// a pending capitalisation was not cleared by an ALREADY-uppercase letter, so
// "And" became "ANd" on every backend whose model emits sentence-cased text —
// but the fix landed in the FALLBACK copy only. The shipping copy kept the bug
// for months, and every symptom pointed at the file that was already correct.
//
// So: assert the two are byte-identical apart from their one legitimate
// difference, the header they include. A future fix applied to one copy now
// fails here instead of appearing to work.
//
// Pure file comparison — no model, no audio, no linkage against either copy.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <sstream>
#include <string>

#ifndef CRISPASR_SOURCE_DIR
#error "CRISPASR_SOURCE_DIR must be defined by the build"
#endif

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// The sole sanctioned difference: each copy includes its own public header.
std::string normalize(std::string s) {
    const std::string from = "#include \"crisp_punc.h\"";
    const std::string to = "#include \"fireredpunc.h\"";
    const size_t p = s.find(from);
    if (p != std::string::npos)
        s.replace(p, from.size(), to);
    return s;
}

} // namespace

TEST_CASE("the two fireredpunc copies are identical", "[unit][punc]") {
    const std::string root = CRISPASR_SOURCE_DIR;
    const std::string shared = normalize(read_file(root + "/crisp_punc/src/fireredpunc.cpp"));
    const std::string fallback = normalize(read_file(root + "/src/fireredpunc.cpp"));

    if (shared != fallback) {
        // Point at the first differing line — a bare "not equal" on a 900-line
        // file is not actionable.
        size_t line = 1, i = 0;
        for (; i < shared.size() && i < fallback.size() && shared[i] == fallback[i]; i++)
            if (shared[i] == '\n')
                line++;
        FAIL("crisp_punc/src/fireredpunc.cpp and src/fireredpunc.cpp diverge at line "
             << line
             << ". They are the same implementation twice (shared library vs partial-checkout "
                "fallback) — apply the change to BOTH. #308's capitalisation fix went to the "
                "fallback only and was dead code for months.");
    }
    SUCCEED();
}

TEST_CASE("both copies carry the #308 capitalisation guard", "[unit][punc]") {
    // A targeted check on top of the identity test: if someone "fixes" the
    // divergence by deleting the guard from both, identity still passes. This
    // pins the behaviour itself — an already-uppercase letter must satisfy the
    // pending capitalisation.
    const std::string root = CRISPASR_SOURCE_DIR;
    for (const char* rel : {"/crisp_punc/src/fireredpunc.cpp", "/src/fireredpunc.cpp"}) {
        const std::string src = read_file(root + rel);
        INFO("file: " << rel);
        REQUIRE(src.find("if (c >= 'A' && c <= 'Z') {") != std::string::npos);
    }
}
