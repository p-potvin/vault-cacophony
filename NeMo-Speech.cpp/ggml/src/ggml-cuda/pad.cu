#include "pad.cuh"

#include <stdint.h>

__device__ __forceinline__ int64_t wrap_around(int64_t coord, int64_t size) {
    // + size ensures negatives are handled properly
    return (coord + size) % size;
}

static __global__ void pad_f32(const float * src, size_t s00, size_t s01, size_t s02, size_t s03, float * dst,
                               const int lp0, const int rp0, const int lp1, const int rp1,
                               const int lp2, const int rp2, const int lp3, const int rp3,
                               const int ne0, const int ne1, const int ne2, const int ne3,
                               const bool circular) {
    // Flatten the tensor-slice coordinates into grid.x. The previous 3-D
    // launch put ne2*ne3 in grid.z, whose architectural limit is 65535.
    // Large speech batches legitimately exceed that limit even though the
    // total launch is small enough for grid.x (up to 2^31-1 blocks).
    const uint64_t blocks_per_row = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    const uint64_t linear_block = blockIdx.x;
    const uint64_t row_block = linear_block % blocks_per_row;
    uint64_t slice = linear_block / blocks_per_row;
    const int i1 = slice % ne1;
    slice /= ne1;
    const int i2 = slice % ne2;
    const int i3 = slice / ne2;
    const int i0 = threadIdx.x + row_block * blockDim.x;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t dst_idx = i3 * (ne0 * ne1 * ne2) + i2 * (ne0 * ne1) + i1 * ne0 + i0;

    if (!circular) {
        if ((i0 >= lp0 && i0 < ne0 - rp0) && (i1 >= lp1 && i1 < ne1 - rp1) && (i2 >= lp2 && i2 < ne2 - rp2) &&
            (i3 >= lp3 && i3 < ne3 - rp3)) {
            const int64_t i00  = i0 - lp0;
            const int64_t i01  = i1 - lp1;
            const int64_t i02  = i2 - lp2;
            const int64_t i03  = i3 - lp3;

            const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

            dst[dst_idx] = src[src_idx];
        } else {
            dst[dst_idx] = 0.0f;
        }
    }
    // circular means on a torus, so x and y wrap around
    else {
        const int64_t ne00 = ne0 - lp0 - rp0;
        const int64_t ne01 = ne1 - lp1 - rp1;
        const int64_t ne02 = ne2 - lp2 - rp2;
        const int64_t ne03 = ne3 - lp3 - rp3;

        const int64_t i00 = wrap_around(i0 - lp0, ne00);
        const int64_t i01 = wrap_around(i1 - lp1, ne01);
        const int64_t i02 = wrap_around(i2 - lp2, ne02);
        const int64_t i03 = wrap_around(i3 - lp3, ne03);

        const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    }
}


static void pad_f32_cuda(const float * src, size_t s00, size_t s01, size_t s02, size_t s03, float * dst,
    const int lp0, const int rp0, const int lp1, const int rp1,
    const int lp2, const int rp2, const int lp3, const int rp3,
    const int ne0, const int ne1, const int ne2, const int ne3,
    const bool circular, cudaStream_t stream) {
    const uint64_t blocks_per_row = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    const uint64_t num_blocks =
        blocks_per_row * (uint64_t) ne1 * (uint64_t) ne2 * (uint64_t) ne3;
    GGML_ASSERT(num_blocks <= 0x7fffffffULL);
    pad_f32<<<dim3(num_blocks), CUDA_PAD_BLOCK_SIZE, 0, stream>>>(src, s00, s01, s02, s03, dst,
                                                         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                                                         ne0, ne1, ne2, ne3, circular);
}

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) dst->data;
    cudaStream_t        stream = ctx.stream();

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t lp0      = ((const int32_t *) (dst->op_params))[0];
    const int32_t rp0      = ((const int32_t *) (dst->op_params))[1];
    const int32_t lp1      = ((const int32_t *) (dst->op_params))[2];
    const int32_t rp1      = ((const int32_t *) (dst->op_params))[3];
    const int32_t lp2      = ((const int32_t *) (dst->op_params))[4];
    const int32_t rp2      = ((const int32_t *) (dst->op_params))[5];
    const int32_t lp3      = ((const int32_t *) (dst->op_params))[6];
    const int32_t rp3      = ((const int32_t *) (dst->op_params))[7];
    const int32_t circular = ((const int32_t *) (dst->op_params))[8];

    const size_t s00 = nb00 / ggml_type_size(src0->type);
    const size_t s01 = nb01 / ggml_type_size(src0->type);
    const size_t s02 = nb02 / ggml_type_size(src0->type);
    const size_t s03 = nb03 / ggml_type_size(src0->type);

    pad_f32_cuda(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                 (bool) circular, stream);
}
