// tests/test-g2p-es.cpp — unit tests for Spanish G2P.

#include <catch2/catch_test_macros.hpp>
#include "core/g2p_es.h"
#include <string>

TEST_CASE("Spanish LTS: ch/ll/rr/ñ", "[g2p_es][lts]") {
    SECTION("ch → tʃ") {
        std::string ipa = g2p_es::lts_word_to_ipa("chico");
        CHECK(ipa.find("t\xCA\x83") != std::string::npos);
    }
    SECTION("ll → ʝ (yeísmo)") {
        std::string ipa = g2p_es::lts_word_to_ipa("calle");
        CHECK(ipa.find("\xCA\x9D") != std::string::npos);
    }
    SECTION("rr → trill") {
        std::string ipa = g2p_es::lts_word_to_ipa("perro");
        CHECK(ipa.find("r") != std::string::npos);
    }
    SECTION("ñ → ɲ") {
        std::string ipa = g2p_es::lts_word_to_ipa("espa\xC3\xB1""a");
        CHECK(ipa.find("\xC9\xB2") != std::string::npos);
    }
}

TEST_CASE("Spanish LTS: b/d/g lenition", "[g2p_es][lts]") {
    SECTION("initial b → stop") {
        std::string ipa = g2p_es::lts_word_to_ipa("bueno");
        CHECK(ipa[0] == 'b');
    }
    SECTION("intervocalic b → β") {
        std::string ipa = g2p_es::lts_word_to_ipa("haba");
        CHECK(ipa.find("\xCE\xB2") != std::string::npos);
    }
    SECTION("initial d → stop") {
        std::string ipa = g2p_es::lts_word_to_ipa("dia");
        CHECK(ipa[0] == 'd');
    }
    SECTION("intervocalic d → ð") {
        std::string ipa = g2p_es::lts_word_to_ipa("nada");
        CHECK(ipa.find("\xC3\xB0") != std::string::npos);
    }
}

TEST_CASE("Spanish LTS: c/z seseo, g/j jota", "[g2p_es][lts]") {
    SECTION("ce → s (seseo)") {
        std::string ipa = g2p_es::lts_word_to_ipa("cena");
        CHECK(ipa[0] == 's');
    }
    SECTION("z → s") {
        std::string ipa = g2p_es::lts_word_to_ipa("zapato");
        CHECK(ipa[0] == 's');
    }
    SECTION("j → x (jota)") {
        std::string ipa = g2p_es::lts_word_to_ipa("jugar");
        CHECK(ipa[0] == 'x');
    }
    SECTION("ge → x") {
        std::string ipa = g2p_es::lts_word_to_ipa("gente");
        CHECK(ipa[0] == 'x');
    }
    SECTION("ga → ɡ") {
        std::string ipa = g2p_es::lts_word_to_ipa("gato");
        CHECK(ipa.find("\xC9\xA1") != std::string::npos);
    }
}

TEST_CASE("Spanish LTS: silent h, qu", "[g2p_es][lts]") {
    SECTION("h is silent") {
        std::string ipa = g2p_es::lts_word_to_ipa("hola");
        CHECK(ipa.find("h") == std::string::npos);
    }
    SECTION("qu → k") {
        std::string ipa = g2p_es::lts_word_to_ipa("que");
        CHECK(ipa[0] == 'k');
    }
}

TEST_CASE("Spanish LTS: vowels and r", "[g2p_es][lts]") {
    SECTION("initial r → trill") {
        std::string ipa = g2p_es::lts_word_to_ipa("rojo");
        CHECK(ipa[0] == 'r');
    }
    SECTION("intervocalic r → tap ɾ") {
        std::string ipa = g2p_es::lts_word_to_ipa("pero");
        CHECK(ipa.find("\xC9\xBE") != std::string::npos);
    }
    SECTION("word-final y → i") {
        std::string ipa = g2p_es::lts_word_to_ipa("hoy");
        bool has_i = ipa.back() == 'i' || ipa.find("i") != std::string::npos;
        CHECK(has_i);
    }
}

TEST_CASE("Spanish text_to_ipa", "[g2p_es][sentence]") {
    g2p_es::context ctx;
    SECTION("hola mundo") {
        std::string ipa = g2p_es::text_to_ipa(ctx, "hola mundo");
        CHECK(!ipa.empty());
        CHECK(ipa.find(' ') != std::string::npos);
        // h silent, so first sound should be 'o'
        CHECK(ipa[0] == 'o');
    }
}
