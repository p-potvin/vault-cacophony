#include "engine/models/pocket_tts/text_conditioner.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/pocket_tts/assets.h"
#include "graph_common.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace engine::models::pocket_tts {
namespace {

std::string prepare_pocket_tts_text(
    std::string text,
    bool pad_with_spaces_for_short_inputs,
    bool remove_semicolons) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    if (text.empty()) {
        throw std::runtime_error("PocketTTS text prompt cannot be empty");
    }

    for (char & ch : text) {
        if (ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }
    for (size_t i = 1; i < text.size();) {
        if (text[i] == ' ' && text[i - 1] == ' ') {
            text.erase(text.begin() + static_cast<ptrdiff_t>(i));
        } else {
            ++i;
        }
    }
    if (remove_semicolons) {
        std::replace(text.begin(), text.end(), ';', ',');
    }
    if (!text.empty() && std::islower(static_cast<unsigned char>(text.front()))) {
        text.front() = static_cast<char>(std::toupper(static_cast<unsigned char>(text.front())));
    }
    if (std::isalnum(static_cast<unsigned char>(text.back()))) {
        text.push_back('.');
    }
    if (pad_with_spaces_for_short_inputs) {
        int words = 0;
        bool in_word = false;
        for (const char ch : text) {
            if (std::isspace(static_cast<unsigned char>(ch))) {
                if (in_word) {
                    ++words;
                    in_word = false;
                }
            } else {
                in_word = true;
            }
        }
        if (in_word) {
            ++words;
        }
        if (words < 5) {
            text = "        " + text;
        }
    }
    return text;
}

std::vector<float> gather_token_embeddings(
    const assets::TensorDataF32 & embedding_table,
    const std::vector<int32_t> & tokens,
    int64_t hidden_size) {
    if (embedding_table.shape.rank != 2 || embedding_table.shape.dims[1] != hidden_size) {
        throw std::runtime_error("PocketTTS embedding table must have shape [vocab, hidden_size]");
    }
    const int64_t vocab = embedding_table.shape.dims[0];
    const int64_t dim = embedding_table.shape.dims[1];

    std::vector<float> output(static_cast<size_t>(tokens.size() * dim));
    for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
        const int32_t token = tokens[token_index];
        if (token < 0 || token >= vocab) {
            throw std::runtime_error("PocketTTS token id is out of range for embedding table");
        }
        const size_t src_offset = static_cast<size_t>(token) * static_cast<size_t>(dim);
        const size_t dst_offset = token_index * static_cast<size_t>(dim);
        std::copy_n(
            embedding_table.values.begin() + static_cast<ptrdiff_t>(src_offset),
            static_cast<size_t>(dim),
            output.begin() + static_cast<ptrdiff_t>(dst_offset));
    }
    return output;
}

}  // namespace

TextConditioner::TextConditioner(TextConditionerConfig config) : config_(config) {}

TextConditioningResult TextConditioner::prepare(
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSHostWeights & weights,
    const std::string & text) const {
    TextConditioningResult result;
    const double prepare_ms = engine::debug::measure_ms([&]() {
        result.prepared_text = prepare_pocket_tts_text(
            text,
            manifest.model_config.pad_with_spaces_for_short_inputs,
            manifest.model_config.remove_semicolons);
        if (manifest.tokenizer_pieces.empty()) {
            throw std::runtime_error("PocketTTS tokenizer pieces are not loaded");
        }
        result.tokens = tokenizers::tokenize_sentencepiece(manifest.tokenizer_pieces, result.prepared_text);
        const auto & embedding_table = weights.conditioner_embedding_table;
        result.text_embeddings = gather_token_embeddings(embedding_table, result.tokens, config_.hidden_size);
    });
    engine::debug::timing_log_scalar("pocket_tts.text.prepare_ms", prepare_ms);
    return result;
}

}  // namespace engine::models::pocket_tts
