#include "conv-transpose-1d.cuh"
#include "convert.cuh"

#include <cstring>

template <typename T>
static __global__ void conv_transpose_1d_kernel(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const T * src0, const float * src1,  float * dst) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }

    const int out_t = global_index % dst_ne0;
    const int out_c = (global_index / dst_ne0) % dst_ne1;
    const int out_b = global_index / (dst_ne0 * dst_ne1);

    float accumulator = 0;

    const int in_end = min(src1_ne0 - 1, out_t / s0);
    const int in_start = max(0, (out_t - src0_ne0 + s0) / s0);

    for (int c = 0; c < src0_ne2; c++) {
        const int kernel_offset = src0_ne0 * (out_c + src0_ne1 * c);
        const int input_offset = src1_ne0 * (c + src1_ne1 * out_b);

        for (int i = in_start; i <= in_end; i++) {
            const int weight_idx = out_t - i*s0;

            const float kernel_weight = ggml_cuda_cast<float>(src0[kernel_offset + weight_idx]);
            const float input_value = src1[input_offset+i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
    GGML_UNUSED_VARS(p0, d0, src0_ne3, src1_ne3, dst_ne3, src1_ne1, dst_ne1, src1_ne2, dst_ne2);
}

template <typename T>
static __global__ void conv_transpose_1d_grouped2_kernel(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const T * src0, const float * src1,  float * dst) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }

    const int out_t = global_index % dst_ne0;
    const int out_c = (global_index / dst_ne0) % dst_ne1;
    const int out_b = global_index / (dst_ne0 * dst_ne1);

    const int in_end = min(src1_ne0 - 1, out_t / s0);
    const int in_start = max(0, (out_t - src0_ne0 + s0) / s0);

    float accumulator = 0;

    const int c0 = out_c * 2;
    const int c1 = c0 + 1;

    for (int i = in_start; i <= in_end; i++) {
        const int weight_idx = out_t - i*s0;

        const int input_offset0 = src1_ne0 * (c0 + src1_ne1 * out_b);
        const int kernel_offset0 = src0_ne0 * (out_c + src0_ne1 * c0);
        accumulator += ggml_cuda_cast<float>(src0[kernel_offset0 + weight_idx]) * src1[input_offset0 + i];

        const int input_offset1 = src1_ne0 * (c1 + src1_ne1 * out_b);
        const int kernel_offset1 = src0_ne0 * (out_c + src0_ne1 * c1);
        accumulator += ggml_cuda_cast<float>(src0[kernel_offset1 + weight_idx]) * src1[input_offset1 + i];
    }

    dst[global_index] = accumulator;
    GGML_UNUSED_VARS(p0, d0, src0_ne2, src0_ne3, src1_ne2, src1_ne3, dst_ne2, dst_ne3);
}

template <typename T>
static void conv_transpose_1d_cuda(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const T * src0, const float * src1,  float * dst,
        const bool use_grouped2,
        cudaStream_t stream) {

    const int num_blocks = (output_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
    if (use_grouped2) {
        conv_transpose_1d_grouped2_kernel<<<num_blocks,CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(
            s0,p0,d0,output_size,
            src0_ne0, src0_ne1,  src0_ne2, src0_ne3,
            src1_ne0, src1_ne1,  src1_ne2, src1_ne3,
            dst_ne0,  dst_ne1,   dst_ne2,  dst_ne3,
            src0,src1, dst);
    } else {
        conv_transpose_1d_kernel<<<num_blocks,CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(
            s0,p0,d0,output_size,
            src0_ne0, src0_ne1,  src0_ne2, src0_ne3,
            src1_ne0, src1_ne1,  src1_ne2, src1_ne3,
            dst_ne0,  dst_ne1,   dst_ne2,  dst_ne3,
            src0,src1, dst);
    }
}

static bool conv_transpose_1d_use_nanocodec_grouped2(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int s0) {
    const bool shape_matches =
        src0->ne[0] == 2*s0 &&
        src0->ne[2] == 2*src0->ne[1] &&
        src1->ne[1] == src0->ne[2] &&
        dst->ne[1] == src0->ne[1] &&
        src1->ne[2] == dst->ne[2] &&
        src1->ne[3] == dst->ne[3];

    const char * name = src0->name;
    const bool is_nanocodec_up_weight =
        name != nullptr &&
        std::strncmp(name, "dec.up.", 7) == 0 &&
        std::strstr(name + 7, ".w") != nullptr;

    return shape_matches && is_nanocodec_up_weight;
}

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;

    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = opts[0];
    const int p0 = 0;//opts[3];
    const int d0 = 1;//opts[4];

    const int64_t output_size = ggml_nelements(dst);
    const bool use_grouped2 = conv_transpose_1d_use_nanocodec_grouped2(src0, src1, dst, s0);

    if (src0->type == GGML_TYPE_F16) {
        conv_transpose_1d_cuda(s0, p0, d0, output_size,
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            (const half *) src0_d, src1_d, dst_d, use_grouped2, stream);
    } else {
        conv_transpose_1d_cuda(s0, p0, d0, output_size,
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            (const float *) src0_d, src1_d, dst_d, use_grouped2, stream);
    }
}
