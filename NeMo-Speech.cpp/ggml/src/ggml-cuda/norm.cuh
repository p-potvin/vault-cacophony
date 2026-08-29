#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Fused LayerNorm + row-vector mul (gamma) + optional row-vector add (beta).
// add_tensor may be nullptr (norm+mul only).
void ggml_cuda_op_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor,
                             ggml_tensor * add_tensor, ggml_tensor * cast_dst = nullptr);

void ggml_cuda_op_batch_norm_fused(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor *         input,
                                   const ggml_tensor *         mean,
                                   const ggml_tensor *         variance,
                                   const ggml_tensor *         epsilon,
                                   const ggml_tensor *         weight,
                                   const ggml_tensor *         bias,
                                   ggml_tensor *               dst);

void ggml_cuda_op_batch_norm_silu_transpose_fused(ggml_backend_cuda_context & ctx,
                                                  const ggml_tensor *         input,
                                                  const ggml_tensor *         mean,
                                                  const ggml_tensor *         variance,
                                                  const ggml_tensor *         epsilon,
                                                  const ggml_tensor *         weight,
                                                  const ggml_tensor *         bias,
                                                  ggml_tensor *               dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
