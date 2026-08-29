// test-gguf-bounds.cpp — deterministic regression guard for the GGUF
// load_weights out-of-bounds hardening (untrusted-input security work).
//
// A GGUF model file is untrusted (users download/convert models). A crafted or
// truncated GGUF can declare a tensor whose data range exceeds the file — the
// mmap zero-copy loader would then dereference past the mapping → SIGBUS. The
// fix added subtractive, overflow-safe bounds checks in
// core_gguf::load_weights (gguf_loader.cpp): `data_off > size || off > size -
// data_off || nbytes > size - data_off - off` → reject, don't crash.
//
// This test builds a VALID one-tensor GGUF, then truncates it to the start of
// the tensor-data section so metadata still parses but the declared tensor
// overruns the file, and asserts load_weights returns false (safe reject) — and
// that the intact file still loads. It runs with no model, deterministically.

#include <catch2/catch_test_macros.hpp>

#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <string>
#include <unistd.h> // truncate

namespace {

// Write a minimal valid GGUF with one F32 tensor.
void write_valid_gguf(const std::string& path, int n) {
    ggml_init_params ip = {/*mem_size=*/(size_t)n * sizeof(float) + ggml_tensor_overhead() + 1024,
                           /*mem_buffer=*/nullptr, /*no_alloc=*/false};
    ggml_context* ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(t, "test.weight");
    for (int i = 0; i < n; i++)
        ((float*)t->data)[i] = (float)i;

    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "test");
    gguf_add_tensor(g, t);
    REQUIRE(gguf_write_to_file(g, path.c_str(), /*only_meta=*/false));
    gguf_free(g);
    ggml_free(ctx);
}

// Re-read the written file to get the true data-section offset (the write
// context's gguf_get_data_offset is not populated until write).
size_t read_data_offset(const std::string& path) {
    gguf_init_params gp = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    REQUIRE(g != nullptr);
    size_t off = gguf_get_data_offset(g);
    gguf_free(g);
    return off;
}

} // namespace

TEST_CASE("core_gguf::load_weights rejects a truncated GGUF without crashing", "[unit][gguf-bounds]") {
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);

    const std::string good = "crispasr_test_gguf_bounds_ok.gguf";
    const std::string bad = "crispasr_test_gguf_bounds_trunc.gguf";

    const int n = 64; // 64 f32 = 256 tensor bytes
    write_valid_gguf(good, n);
    const size_t data_off = read_data_offset(good);
    const size_t nbytes = (size_t)n * sizeof(float);

    // Positive control: the intact file must load with the tensor present.
    {
        core_gguf::WeightLoad wl;
        REQUIRE(core_gguf::load_weights(good.c_str(), backend, "test-ok", wl));
        REQUIRE(wl.tensors.count("test.weight") == 1);
    }

    // Craft the malicious file: copy the good one, then truncate to just SHORT
    // of the full tensor data (metadata fully parses; the declared 256-byte
    // tensor overruns the file by 8 bytes). This forces control into the
    // subtractive bounds check in load_weights (nbytes > size - data_off - off),
    // not the earlier magic/metadata rejection — i.e. the exact hardened path.
    {
        FILE* src = std::fopen(good.c_str(), "rb");
        FILE* dst = std::fopen(bad.c_str(), "wb");
        REQUIRE(src);
        REQUIRE(dst);
        char buf[4096];
        size_t r;
        while ((r = std::fread(buf, 1, sizeof(buf), src)) > 0)
            std::fwrite(buf, 1, r, dst);
        std::fclose(src);
        std::fclose(dst);
        REQUIRE(::truncate(bad.c_str(), (off_t)(data_off + nbytes - 8)) == 0);
    }

    // The load must fail gracefully (false), not SIGBUS. Reaching this REQUIRE at
    // all means no crash; the value check confirms it was rejected.
    {
        core_gguf::WeightLoad wl;
        REQUIRE_FALSE(core_gguf::load_weights(bad.c_str(), backend, "test-trunc", wl));
    }

    std::remove(good.c_str());
    std::remove(bad.c_str());
    ggml_backend_free(backend);
}
