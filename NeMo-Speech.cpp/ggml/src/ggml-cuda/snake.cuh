#include "common.cuh"

// Fusion entry point. Caller supplies x/a/inv_b explicitly.
void ggml_cuda_op_snake_fused(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * x,
                              const ggml_tensor * a,
                              const ggml_tensor * inv_b,
                              ggml_tensor *       dst);

void ggml_cuda_op_half_snake_fused(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor * x_snake,
                                   const ggml_tensor * x_lrelu,
                                   const ggml_tensor * a,
                                   const ggml_tensor * inv_b,
                                   const ggml_tensor * lrelu,
                                   ggml_tensor *       dst);
