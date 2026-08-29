#include "snake.cuh"
#include "convert.cuh"

#include <cstring>

// Fused Snake activation: y = x + sin^2(a * x) * inv_b
// x: [T, C] (T contiguous), a: [1, C], inv_b: [1, C]
// Supports F32, F16, BF16 data with F32 compute.

template <typename T>
static __global__ void snake_kernel(
        const T     * __restrict__ x,
        const float * __restrict__ a,
        const float * __restrict__ inv_b,
        T           * __restrict__ dst,
        const int    total,
        const uint3  T_len_fastdiv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int c = (int) fastdiv((uint32_t) idx, T_len_fastdiv);

    const float xi = ggml_cuda_cast<float>(x[idx]);
    const float s  = sinf(a[c] * xi);
    dst[idx] = ggml_cuda_cast<T>(xi + s * s * inv_b[c]);
}

// Internal launcher with explicit x/a/inv_b/dst tensors.
// Shared by the public op (reads dst->src) and the fusion path (explicit args).
static void launch_snake(ggml_backend_cuda_context & ctx,
                         const ggml_tensor * x,
                         const ggml_tensor * a,
                         const ggml_tensor * inv_b,
                         ggml_tensor *       dst) {
    const float * a_d     = (const float *)a->data;
    const float * inv_b_d = (const float *)inv_b->data;

    const int   T = (int)x->ne[0];
    const int   C = (int)x->ne[1];
    const int   total = T * C;
    const uint3 T_len_fastdiv = init_fastdiv_values((uint64_t) T);

    const int block_size = 256;
    const int grid_size  = (total + block_size - 1) / block_size;

    cudaStream_t stream = ctx.stream();

    switch (x->type) {
        case GGML_TYPE_F32: {
            snake_kernel<<<grid_size, block_size, 0, stream>>>(
                (const float *)x->data, a_d, inv_b_d, (float *)dst->data, total, T_len_fastdiv);
        } break;
        case GGML_TYPE_F16: {
            snake_kernel<<<grid_size, block_size, 0, stream>>>(
                (const half *)x->data, a_d, inv_b_d, (half *)dst->data, total, T_len_fastdiv);
        } break;
        case GGML_TYPE_BF16: {
            snake_kernel<<<grid_size, block_size, 0, stream>>>(
                (const nv_bfloat16 *)x->data, a_d, inv_b_d, (nv_bfloat16 *)dst->data, total, T_len_fastdiv);
        } break;
        default:
            GGML_ABORT("snake: unsupported type");
    }
}

// Fusion entry: caller supplies x/a/inv_b explicitly from the matched
// mul -> sin -> sqr -> mul -> add pattern. The dst is the trailing add output.
void ggml_cuda_op_snake_fused(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * x,
                              const ggml_tensor * a,
                              const ggml_tensor * inv_b,
                              ggml_tensor *       dst) {
    launch_snake(ctx, x, a, inv_b, dst);
}

template <typename T>
static __global__ void half_snake_kernel(
        const T     * __restrict__ x_snake,
        const T     * __restrict__ x_lrelu,
        const float * __restrict__ a,
        const float * __restrict__ inv_b,
        T           * __restrict__ dst,
        const int    total,
        const int    T_len,
        const int    snake_channels,
        const int    lrelu_channels,
        const int    channels,
        const float  negative_slope) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int t = idx % T_len;
    const int c = (idx / T_len) % channels;
    const int plane = idx / (T_len * channels);

    if (c < snake_channels) {
        const int src_idx = plane * T_len * snake_channels + c * T_len + t;
        const float xi = ggml_cuda_cast<float>(x_snake[src_idx]);
        const float s  = sinf(a[c] * xi);
        dst[idx] = ggml_cuda_cast<T>(xi + s * s * inv_b[c]);
    } else {
        const int lc = c - snake_channels;
        const int src_idx = plane * T_len * lrelu_channels + lc * T_len + t;
        const float xi = ggml_cuda_cast<float>(x_lrelu[src_idx]);
        dst[idx] = ggml_cuda_cast<T>(fmaxf(xi, 0.0f) + fminf(xi, 0.0f) * negative_slope);
    }
}

void ggml_cuda_op_half_snake_fused(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor * x_snake,
                                   const ggml_tensor * x_lrelu,
                                   const ggml_tensor * a,
                                   const ggml_tensor * inv_b,
                                   const ggml_tensor * lrelu,
                                   ggml_tensor *       dst) {
    float negative_slope;
    std::memcpy(&negative_slope, lrelu->op_params, sizeof(float));

    const int T = (int) dst->ne[0];
    const int snake_channels = (int) x_snake->ne[1];
    const int lrelu_channels = (int) x_lrelu->ne[1];
    const int channels = (int) dst->ne[1];
    const int total = (int) ggml_nelements(dst);

    const int block_size = 256;
    const int grid_size  = (total + block_size - 1) / block_size;
    cudaStream_t stream = ctx.stream();

    const float * a_d     = (const float *) a->data;
    const float * inv_b_d = (const float *) inv_b->data;

    switch (dst->type) {
        case GGML_TYPE_F32: {
            half_snake_kernel<<<grid_size, block_size, 0, stream>>>(
                (const float *) x_snake->data, (const float *) x_lrelu->data, a_d, inv_b_d, (float *) dst->data,
                total, T, snake_channels, lrelu_channels, channels, negative_slope);
        } break;
        case GGML_TYPE_F16: {
            half_snake_kernel<<<grid_size, block_size, 0, stream>>>(
                (const half *) x_snake->data, (const half *) x_lrelu->data, a_d, inv_b_d, (half *) dst->data,
                total, T, snake_channels, lrelu_channels, channels, negative_slope);
        } break;
        case GGML_TYPE_BF16: {
            half_snake_kernel<<<grid_size, block_size, 0, stream>>>(
                (const nv_bfloat16 *) x_snake->data, (const nv_bfloat16 *) x_lrelu->data, a_d, inv_b_d,
                (nv_bfloat16 *) dst->data, total, T, snake_channels, lrelu_channels, channels, negative_slope);
        } break;
        default:
            GGML_ABORT("half_snake: unsupported type");
    }
}
