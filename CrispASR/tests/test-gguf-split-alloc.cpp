// test-gguf-split-alloc.cpp — unit tests for load_weights_split() chunked
// allocation (issue #276).
//
// Verifies that:
// 1. load_weights_split partitions tensors by the is_gpu predicate correctly.
// 2. Both GPU and CPU partitions are independently addressable.
// 3. Tensor data is correctly loaded into the right partition.
// 4. free_weights cleans up split_bufs without leaks.
// 5. The split_bufs field holds overflow buffers when the GPU partition
//    exceeds the 1.5 GiB chunk limit (structural — we verify the field
//    exists and is managed; actual multi-GiB allocation is only testable
//    on real Vulkan hardware).

#include <catch2/catch_test_macros.hpp>

#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Write a GGUF with multiple named F32 tensors.
void write_multi_tensor_gguf(const std::string& path, int n_tensors, int elems_per_tensor) {
    const size_t mem = (size_t)n_tensors * ((size_t)elems_per_tensor * sizeof(float) + ggml_tensor_overhead()) + 4096;
    ggml_init_params ip = {/*mem_size=*/mem, /*mem_buffer=*/nullptr, /*no_alloc=*/false};
    ggml_context* ctx = ggml_init(ip);
    REQUIRE(ctx != nullptr);

    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "test_split");

    for (int i = 0; i < n_tensors; i++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.weight", i);
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elems_per_tensor);
        ggml_set_name(t, name);
        // Fill with recognizable data: each tensor gets value = (float)(i+1).
        float* d = (float*)t->data;
        for (int j = 0; j < elems_per_tensor; j++)
            d[j] = (float)(i + 1);
        gguf_add_tensor(g, t);
    }

    REQUIRE(gguf_write_to_file(g, path.c_str(), /*only_meta=*/false));
    gguf_free(g);
    ggml_free(ctx);
}

// Predicate: layers 0..threshold-1 go to GPU, rest to CPU.
struct SplitCtx {
    int threshold;
};
bool test_is_gpu(const char* name, void* user) {
    auto* sc = static_cast<SplitCtx*>(user);
    // Parse layer index from "blk.<N>.weight"
    int il = core_gguf::blk_layer_of(name);
    if (il < 0)
        return true; // non-layered → GPU
    return il < sc->threshold;
}

} // namespace

TEST_CASE("load_weights_split partitions tensors by predicate", "[unit][gguf-split]") {
    const std::string path = "crispasr_test_split_alloc.gguf";
    const int n_tensors = 8;
    const int elems = 32; // small tensors, 128 bytes each

    write_multi_tensor_gguf(path, n_tensors, elems);

    ggml_backend_t gpu_be = ggml_backend_cpu_init(); // use CPU as "GPU" for test
    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    REQUIRE(gpu_be);
    REQUIRE(cpu_be);

    SECTION("all tensors to GPU (threshold = n_tensors)") {
        SplitCtx sc{n_tensors};
        core_gguf::WeightLoad wl;
        REQUIRE(core_gguf::load_weights_split(path.c_str(), gpu_be, cpu_be, test_is_gpu, &sc, "test-split", wl));

        // All 8 tensors should be in the map.
        REQUIRE(wl.tensors.size() == (size_t)n_tensors);
        // GPU buffer must exist (all tensors there), CPU buffer should be null.
        REQUIRE(wl.buf != nullptr);
        REQUIRE(wl.buf_cpu == nullptr);

        // Verify data integrity: blk.3.weight should contain all 4.0f.
        ggml_tensor* t3 = core_gguf::try_get(wl.tensors, "blk.3.weight");
        REQUIRE(t3 != nullptr);
        std::vector<float> data(elems);
        ggml_backend_tensor_get(t3, data.data(), 0, elems * sizeof(float));
        for (int j = 0; j < elems; j++)
            REQUIRE(data[j] == 4.0f);

        core_gguf::free_weights(wl);
    }

    SECTION("all tensors to CPU (threshold = 0)") {
        SplitCtx sc{0};
        core_gguf::WeightLoad wl;
        REQUIRE(core_gguf::load_weights_split(path.c_str(), gpu_be, cpu_be, test_is_gpu, &sc, "test-split", wl));

        REQUIRE(wl.tensors.size() == (size_t)n_tensors);
        REQUIRE(wl.buf == nullptr);
        REQUIRE(wl.buf_cpu != nullptr);

        // Verify data: blk.7.weight should contain all 8.0f.
        ggml_tensor* t7 = core_gguf::try_get(wl.tensors, "blk.7.weight");
        REQUIRE(t7 != nullptr);
        std::vector<float> data(elems);
        ggml_backend_tensor_get(t7, data.data(), 0, elems * sizeof(float));
        for (int j = 0; j < elems; j++)
            REQUIRE(data[j] == 8.0f);

        core_gguf::free_weights(wl);
    }

    SECTION("mixed split: 4 GPU + 4 CPU") {
        SplitCtx sc{4}; // blk.0..3 → GPU, blk.4..7 → CPU
        core_gguf::WeightLoad wl;
        REQUIRE(core_gguf::load_weights_split(path.c_str(), gpu_be, cpu_be, test_is_gpu, &sc, "test-split", wl));

        REQUIRE(wl.tensors.size() == (size_t)n_tensors);
        REQUIRE(wl.buf != nullptr);
        REQUIRE(wl.buf_cpu != nullptr);

        // GPU tensor: blk.1.weight → value 2.0f
        ggml_tensor* t1 = core_gguf::try_get(wl.tensors, "blk.1.weight");
        REQUIRE(t1 != nullptr);
        REQUIRE(t1->buffer == wl.buf);
        std::vector<float> d1(elems);
        ggml_backend_tensor_get(t1, d1.data(), 0, elems * sizeof(float));
        for (int j = 0; j < elems; j++)
            REQUIRE(d1[j] == 2.0f);

        // CPU tensor: blk.5.weight → value 6.0f
        ggml_tensor* t5 = core_gguf::try_get(wl.tensors, "blk.5.weight");
        REQUIRE(t5 != nullptr);
        REQUIRE(t5->buffer == wl.buf_cpu);
        std::vector<float> d5(elems);
        ggml_backend_tensor_get(t5, d5.data(), 0, elems * sizeof(float));
        for (int j = 0; j < elems; j++)
            REQUIRE(d5[j] == 6.0f);

        core_gguf::free_weights(wl);
    }

    SECTION("split_bufs is empty for small models (no chunking needed)") {
        SplitCtx sc{4};
        core_gguf::WeightLoad wl;
        REQUIRE(core_gguf::load_weights_split(path.c_str(), gpu_be, cpu_be, test_is_gpu, &sc, "test-split", wl));

        // With 8 tiny tensors (128 bytes each), everything fits in one buffer
        // per partition — no overflow into split_bufs.
        REQUIRE(wl.split_bufs.empty());

        core_gguf::free_weights(wl);
    }

    std::remove(path.c_str());
    ggml_backend_free(gpu_be);
    ggml_backend_free(cpu_be);
}

TEST_CASE("load_weights_split rejects null backends/predicate", "[unit][gguf-split]") {
    const std::string path = "crispasr_test_split_reject.gguf";
    write_multi_tensor_gguf(path, 2, 16);

    ggml_backend_t be = ggml_backend_cpu_init();
    REQUIRE(be);

    core_gguf::WeightLoad wl;
    SplitCtx sc{1};

    SECTION("null gpu_backend") {
        REQUIRE_FALSE(core_gguf::load_weights_split(path.c_str(), nullptr, be, test_is_gpu, &sc, "test", wl));
    }
    SECTION("null cpu_backend") {
        REQUIRE_FALSE(core_gguf::load_weights_split(path.c_str(), be, nullptr, test_is_gpu, &sc, "test", wl));
    }
    SECTION("null predicate") {
        REQUIRE_FALSE(core_gguf::load_weights_split(path.c_str(), be, be, nullptr, &sc, "test", wl));
    }

    std::remove(path.c_str());
    ggml_backend_free(be);
}

TEST_CASE("free_weights clears split_bufs", "[unit][gguf-split]") {
    // Manually construct a WeightLoad with fake split_bufs entries to verify
    // free_weights empties the vector. We allocate real buffers so
    // ggml_backend_buffer_free is exercised (no UAF / double-free).
    ggml_backend_t be = ggml_backend_cpu_init();
    REQUIRE(be);

    core_gguf::WeightLoad wl;
    // Allocate two small buffers and push them into split_bufs.
    wl.split_bufs.push_back(ggml_backend_alloc_buffer(be, 256));
    wl.split_bufs.push_back(ggml_backend_alloc_buffer(be, 256));
    REQUIRE(wl.split_bufs.size() == 2);
    REQUIRE(wl.split_bufs[0] != nullptr);
    REQUIRE(wl.split_bufs[1] != nullptr);

    core_gguf::free_weights(wl);

    REQUIRE(wl.split_bufs.empty());
    REQUIRE(wl.buf == nullptr);
    REQUIRE(wl.buf_cpu == nullptr);

    ggml_backend_free(be);
}
