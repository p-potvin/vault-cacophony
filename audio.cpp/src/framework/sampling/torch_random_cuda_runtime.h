#pragma once

#include "engine/framework/sampling/torch_random.h"

#include <cstddef>
#include <cstdint>

namespace engine::sampling::detail {

void fill_torch_cuda_tensor_iterator_randn_cuda(
    float * output,
    size_t count,
    uint64_t seed,
    uint64_t offset_blocks,
    const TorchCudaSamplingPolicy & policy,
    TorchRandnPrecision precision);

}  // namespace engine::sampling::detail
