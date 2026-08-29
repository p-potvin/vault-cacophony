#include "scale.cuh"

#define MAX_GRIDDIM_X 0x7FFFFFFF

static __global__ void scale_f32(const float * x, float * dst, const float scale, const float bias, const int64_t nelements) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = scale * x[i] + bias;
    }
}

static void scale_f32_cuda(const float * x, float * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f32<<<MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, bias, nelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    scale_f32_cuda(src0_d, dst_d, scale, bias, ggml_nelements(src0), stream);
}

static __global__ void scale_add_f32(
        const float * __restrict__ x, const float * __restrict__ residual,
        float * __restrict__ dst, const float scale, const int64_t nelements) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;

    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = residual[i] + scale * x[i];
    }
}

void ggml_cuda_op_scale_add(
        ggml_backend_cuda_context & ctx, const ggml_tensor * scale_node,
        const ggml_tensor * residual, ggml_tensor * dst) {
    const ggml_tensor * x = scale_node->src[0];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(residual));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(x, residual));
    GGML_ASSERT(ggml_are_same_shape(x, dst));

    const float scale = ggml_get_op_params_f32(scale_node, 0);
    const float bias  = ggml_get_op_params_f32(scale_node, 1);
    GGML_ASSERT(bias == 0.0f);

    const int64_t nelements = ggml_nelements(x);
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_add_f32<<<MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) residual->data,
        (float *) dst->data, scale, nelements);
}
