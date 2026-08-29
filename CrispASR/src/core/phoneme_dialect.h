// core/phoneme_dialect.h — one G2P, several models, and they do NOT agree on
// how to spell a phoneme.
//
// Kokoro's phoneme embedding was trained on the output of misaki, its own G2P,
// which does NOT emit textbook IPA. It uses single-codepoint stand-ins for the
// English diphthongs and affricates and omits length marks entirely:
//
//     misaki   A    I    O    W    Y    ʧ    ʤ    T        ᵊ
//     IPA      eɪ   aɪ   oʊ   aʊ   ɔɪ   tʃ   dʒ   ɾ(flap)  ə(reduced)
//
// Our builtin G2P and espeak both produce the textbook forms, "tuned to match
// espeak-ng output for piper compatibility" (see core/g2p_en.h) — correct for
// piper, wrong for Kokoro. Every one of those symbols IS in Kokoro's vocab, so
// nothing is dropped and nothing errors; the model simply receives a token
// sequence from outside its training distribution and drifts. That is the
// "sounds British / old English" in #316: `ː` is the RP length mark, and
// `spˈiːtʃ` reaches the model as s,p,ˈ,i,ː,t,ʃ where it was trained on
// s,p,ˈ,i,ʧ.
//
// This is deliberately Kokoro-scoped rather than a change to the shared G2P:
// piper wants the espeak spelling and would regress. Gate:
// CRISPASR_KOKORO_MISAKI_IPA=0 restores the raw G2P output for A/B.
//
// Header-only and weight-free — tests/test-kokoro-misaki-ipa.cpp pins it, with
// misaki's own output for the issue's sentence as the expectation.

#pragma once

#include <string>
#include <vector>

namespace core_phoneme {

// Which spelling of IPA a TTS backend was trained on.
//
//   EspeakIpa  textbook / espeak-ng: `tʃ`, `oʊ`, `ɜː`, length marks. What piper
//              expects — core/g2p_en.h's ARPAbet table is explicitly "tuned to
//              match espeak-ng output for piper compatibility". THE DEFAULT:
//              changing what the G2P itself emits would regress piper.
//   Misaki     Kokoro's own G2P: `ʧ`, `O`, `ɜɹ`, and no length marks at all.
//
// New backends add a dialect here rather than teaching g2p_en a second
// spelling. The G2P has one output; the conversion belongs to the consumer
// that needs something else.
enum class Dialect {
    EspeakIpa, // identity — what the G2P already produces
    Misaki,    // Kokoro
};


// Multi-codepoint sequences first: longest match wins, so `tʃ` is consumed
// before a bare `t`. Order matters here, not just content.
struct Rule {
    const char* from;
    const char* to;
};

inline const std::vector<Rule>& rules() {
    static const std::vector<Rule> table = {
        // Affricates → misaki's single codepoints.
        {"tʃ", "ʧ"},
        {"dʒ", "ʤ"},
        // Diphthongs → misaki's capitals.
        {"eɪ", "A"},
        {"aɪ", "I"},
        {"oʊ", "O"},
        {"əʊ", "O"}, // en-gb spelling of the same vowel
        {"aʊ", "W"},
        {"ɔɪ", "Y"},
        // Rhotic schwa: misaki spells it out as schwa + r.
        {"ɚ", "əɹ"},
        {"ɝ", "ɜɹ"},
        // NURSE. CMUdict's ER1 reaches us as the RP `ɜː` (non-rhotic!), so
        // "world" was wˈɜːld and, once the length mark went, wˈɜld — an r
        // short of misaki's wˈɜɹld. Measured at 56 missing ɹ across a 1508-word
        // corpus, the single largest remaining divergence. Must precede the
        // length-mark rule below, which would otherwise eat the `ː` first.
        {"ɜː", "ɜɹ"},
        // The alveolar flap.
        {"ɾ", "T"},
        // t-glottalization. Our ARPAbet conversion emits a glottal stop before a
        // syllabic nasal ("certainly" -> sˈɜɹʔnli); misaki keeps the /t/ and was
        // never trained on `ʔ` in that position, so restore it.
        {"ʔ", "t"},
        // espeak's r is ɹ in misaki's inventory.
        {"r", "ɹ"},
        // Length marks do not exist in misaki's US output. Drop them LAST so
        // the vowel rules above still see `iː`/`uː` intact if they need to.
        {"ː", ""},
    };
    return table;
}

// Apply the rewrite. Single left-to-right pass with longest-match-first, so a
// replacement can never be re-examined and cascade (`tʃ`→`ʧ` must not then hit
// a `t` rule).
inline std::string to_misaki(const std::string& ipa) {
    std::string out;
    out.reserve(ipa.size());
    const auto& tbl = rules();
    size_t i = 0;
    while (i < ipa.size()) {
        bool hit = false;
        for (const auto& r : tbl) {
            const size_t len = std::char_traits<char>::length(r.from);
            if (len && ipa.compare(i, len, r.from) == 0) {
                out += r.to;
                i += len;
                hit = true;
                break;
            }
        }
        if (!hit)
            out += ipa[i++];
    }
    // ɚ→əɹ can meet an ɹ the G2P already emitted ("parameters" arrives as
    // pɚɹ…), leaving a doubled rhotic that misaki never produces. Collapse it.
    std::string collapsed;
    collapsed.reserve(out.size());
    for (size_t j = 0; j < out.size();) {
        if (out.compare(j, 4, "ɹɹ") == 0) {
            collapsed += "ɹ";
            j += 4;
            continue;
        }
        collapsed += out[j++];
    }
    return collapsed;
}

// Convert G2P output into the dialect a backend expects. EspeakIpa is the
// identity, so a backend on the default pays nothing.
inline std::string convert(const std::string& ipa, Dialect d) {
    return d == Dialect::Misaki ? to_misaki(ipa) : ipa;
}

} // namespace core_phoneme
