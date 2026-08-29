// test-uroman.cpp — unit tests for core/uroman.h romanization.
// Validates Arabic, Cyrillic, and mixed-script romanization for CTC alignment.

#include <catch2/catch_test_macros.hpp>
#include "core/uroman.h"

using namespace core_uroman;

TEST_CASE("uroman: ASCII passes through unchanged", "[unit][uroman]") {
    REQUIRE(romanize("Hello world") == "Hello world");
    REQUIRE(romanize("test 123 !@#") == "test 123 !@#");
}

TEST_CASE("uroman: Arabic consonants romanize", "[unit][uroman]") {
    // بسم = b-s-m
    REQUIRE(romanize("\xd8\xa8\xd8\xb3\xd9\x85") == "bsm");
    // كتاب = k-t-a-b
    REQUIRE(romanize("\xd9\x83\xd8\xaa\xd8\xa7\xd8\xa8") == "ktab");
}

TEST_CASE("uroman: Arabic sentence romanizes", "[unit][uroman]") {
    // سأرى = s-a-r-a (alef hamza above + ra + alef maksura)
    std::string arabic = "\xd8\xb3\xd8\xa3\xd8\xb1\xd9\x89";
    std::string rom = romanize(arabic);
    REQUIRE(!rom.empty());
    REQUIRE(rom.find_first_of("sr") != std::string::npos);
}

TEST_CASE("uroman: Cyrillic romanizes", "[unit][uroman]") {
    // Привет = Privet
    std::string cyrillic = "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82";
    REQUIRE(romanize(cyrillic) == "privet");
}

TEST_CASE("uroman: mixed script preserves spaces", "[unit][uroman]") {
    // "مرحبا world" → "mrhba world"
    std::string mixed = "\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 world";
    std::string rom = romanize(mixed);
    REQUIRE(rom.find(" world") != std::string::npos);
    REQUIRE(rom[0] == 'm'); // mim
}

TEST_CASE("uroman: needs_romanization detects Arabic", "[unit][uroman]") {
    REQUIRE(needs_romanization("\xd8\xa8\xd8\xb3\xd9\x85"));
    REQUIRE_FALSE(needs_romanization("Hello world"));
    REQUIRE_FALSE(needs_romanization("123 test"));
}

TEST_CASE("uroman: needs_romanization detects Cyrillic", "[unit][uroman]") {
    REQUIRE(needs_romanization("\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82"));
}

TEST_CASE("uroman: empty string", "[unit][uroman]") {
    REQUIRE(romanize("") == "");
    REQUIRE_FALSE(needs_romanization(""));
}
