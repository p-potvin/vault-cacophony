// core/g2p_ctxwords.h — English function words whose pronunciation depends on
// what comes NEXT.
//
// A pronunciation lexicon can only store one form per word, so it stores the
// citation form: "the"→`ði`, "to"→`tu`, "a"→`A`. In running speech those reduce,
// and which reduction you get depends on whether the FOLLOWING word starts with
// a vowel:
//
//     the apple   ði ˈæpᵊl        the box   ðə bˈɑks
//     to open     tʊ ˈOpᵊn        to walk   tə wˈɔk
//     a cat       ɐ kˈæt          (the article is never `A` = "eɪ")
//
// Reading the citation form aloud in every position is exactly the "old English"
// diction reported in #316 — "ðiː bɑks" instead of "ðə bɑks". Measured on a
// 400-sentence corpus of real prose, `the`/`to`/`a` alone accounted for 91 of
// 198 sampled word errors against misaki.
//
// The rules are misaki's own, read out of misaki/en.py `get_special_case`:
//
//     the   ->  ði  when a vowel follows, else ðə
//     to    ->  tʊ  when a vowel follows, tə before a consonant,
//               tu  in isolation (no following word)
//     a/an  ->  ɐ / ɐn as an article; `ˈA` only as the letter
//     in    ->  ɪn in running text; ˈɪn in isolation
//
// misaki also special-cases `am`, `by` and `used`, all of which need a
// part-of-speech tag to disambiguate ("by" as adverb vs preposition). We have no
// tagger, so those are deliberately left to the lexicon rather than guessed —
// see the follow-ups in PLAN.md.
//
// Weight-free and header-only.

#pragma once

#include <string>

namespace core_g2p_ctxwords {

// Does the next word begin with a vowel? Unknown means "no next word", which
// misaki treats as its own case rather than as `false`.
enum class NextVowel { Unknown, No, Yes };

inline bool starts_with_vowel(const std::string& ipa) {
    // First CODEPOINT, skipping any leading stress mark.
    size_t i = 0;
    while (i < ipa.size()) {
        if (ipa.compare(i, 2, "ˈ") == 0 || ipa.compare(i, 2, "ˌ") == 0) {
            i += 2;
            continue;
        }
        break;
    }
    if (i >= ipa.size())
        return false;
    static const char* kVowels[] = {"ɑ", "æ", "ɛ", "ɪ", "i", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɜ",
                                    "A", "I", "O", "W", "Y", "ᵊ", "ᵻ", "ɚ", "e", "a", "ɐ"};
    for (const char* v : kVowels) {
        const size_t n = std::char_traits<char>::length(v);
        if (ipa.compare(i, n, v) == 0)
            return true;
    }
    return false;
}

// Returns the contextual pronunciation for `word`, or "" when the word is not
// context-dependent and the lexicon's entry should stand.
//
// `word` must already be lowercased. `lexical` is what the lexicon returned,
// used for the isolation case where misaki falls back to it.
inline std::string lookup(const std::string& word, NextVowel next, const std::string& lexical) {
    if (word == "the")
        return next == NextVowel::Yes ? "ði" : "ðə";
    if (word == "to") {
        if (next == NextVowel::Unknown)
            return lexical.empty() ? "tu" : lexical;
        return next == NextVowel::Yes ? "tʊ" : "tə";
    }
    if (word == "a")
        return "ɐ"; // the article; the LETTER "A" is `ˈA`, but a lone lowercase
                    // "a" in running text is the determiner in ~every case
    if (word == "an")
        return "ɐn";
    if (word == "in")
        return next == NextVowel::Unknown ? "ˈɪn" : "ɪn";
    if (word == "am")
        return next == NextVowel::Unknown ? "ˈæm" : "ɐm";
    if (word == "i")
        return "ˌI"; // misaki special-cases the pronoun to SECONDARY stress
    return "";
}

// ── capitalisation stress ───────────────────────────────────────────────────
//
// misaki derives a token's stress from its CAPITALISATION, not from a
// part-of-speech tag (Lexicon.__call__):
//
//     lowercase   ->  None  — the lexicon form stands
//     Titlecase   ->  0.5   — insert SECONDARY stress if the form has none
//     ALLCAPS     ->  2     — insert PRIMARY stress if the form has none
//
// That single rule is what produced "And"->`ˌænd`, "It"->`ˌɪt`, "She"->`ʃˌi`
// in the corpus — sentence-initial capitals, not grammar. It accounted for the
// large majority of the residual stress disagreements (#316), and it needs no
// tagger.
//
// The mark goes immediately BEFORE the first vowel, matching misaki's
// `restress`, which relocates a leading mark to just ahead of the vowel it
// belongs to: "she" is `ʃˌi`, not `ˌʃi`.
enum class Caps { Lower, Title, Upper };

inline Caps classify_caps(const std::string& word) {
    bool has_upper = false, has_lower = false;
    for (unsigned char c : word) {
        if (c >= 'A' && c <= 'Z')
            has_upper = true;
        else if (c >= 'a' && c <= 'z')
            has_lower = true;
    }
    if (!has_upper)
        return Caps::Lower;
    return has_lower ? Caps::Title : Caps::Upper;
}

inline bool has_stress_mark(const std::string& ps) {
    return ps.find("ˈ") != std::string::npos || ps.find("ˌ") != std::string::npos;
}

// Insert `mark` directly before the first vowel of `ps`.
inline std::string insert_stress(const std::string& ps, const char* mark) {
    static const char* kVowels[] = {"ɑ", "æ", "ɛ", "ɪ", "i", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɜ",
                                    "A", "I", "O", "W", "Y", "ᵊ", "ᵻ", "ɚ", "e", "a", "ɐ"};
    for (size_t i = 0; i < ps.size();) {
        size_t n = 1;
        const unsigned char c = (unsigned char)ps[i];
        if ((c & 0xF8) == 0xF0)
            n = 4;
        else if ((c & 0xF0) == 0xE0)
            n = 3;
        else if ((c & 0xE0) == 0xC0)
            n = 2;
        for (const char* v : kVowels) {
            if (ps.compare(i, std::char_traits<char>::length(v), v) == 0)
                return ps.substr(0, i) + mark + ps.substr(i);
        }
        i += n;
    }
    return ps; // no vowel — misaki leaves these alone
}

// Apply the capitalisation rule to one already-phonemized word.
inline std::string apply_caps_stress(const std::string& word, const std::string& ps) {
    const Caps c = classify_caps(word);
    if (c == Caps::Lower || ps.empty())
        return ps;
    if (has_stress_mark(ps))
        return ps; // an existing mark wins; misaki only INSERTS when there is none
    return insert_stress(ps, c == Caps::Upper ? "ˈ" : "ˌ");
}

} // namespace core_g2p_ctxwords
