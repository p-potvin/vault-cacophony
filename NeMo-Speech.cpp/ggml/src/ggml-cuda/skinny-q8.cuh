// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "common.cuh"

// Skinny-N (9..64 cols) Q8_0 x F32 GEMM specialized for streaming encoders.
// See skinny-q8.cu for the design notes.
bool ggml_cuda_skinny_q8_supported(
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

bool ggml_cuda_skinny_q8_residual_supported(
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mul_mat_skinny_q8(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
    ggml_tensor * dst);

// Fused variant: adds a row-vector bias ([M], F32) in the GEMM epilogue and
// writes the result to `dst` (the bias-add node's buffer).
void ggml_cuda_mul_mat_skinny_q8_bias(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
    const ggml_tensor * bias, ggml_tensor * dst);

void ggml_cuda_mul_mat_skinny_q8_bias_residual(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
    const ggml_tensor * bias, const ggml_tensor * residual, ggml_tensor * dst);
