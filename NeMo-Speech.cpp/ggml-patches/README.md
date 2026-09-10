# ggml-patches

Edge-only changes to the vendored `ggml` submodule, kept as patches so the
submodule stays pinned to clean upstream (`ggml-org/ggml`). Apply them after
checking out / updating the submodule:

```sh
git submodule update --init ggml
scripts/apply-ggml-patches.sh        # applies patches in filename order
```

Patched CUDA and Metal builds expect the patches before CMake configuration.
CPU, Vulkan, and stock-CUDA builds do not. `apply-ggml-patches.sh` uses `git
apply`, skips patches that are already applied, and applies new patches in
filename order. Later patches may build on files changed by earlier patches;
0006 carries the dispatch wiring for the ops/kernels introduced by
0001/0003/0005. Docker builds and `scripts/configure.sh` apply the series
automatically; apply it explicitly before a raw CUDA or Metal CMake
configuration.

Metal requires only `0018-metal-tensor-api-dynamic-k.patch`. The `metal-*`
presets apply the whole series because `apply-ggml-patches.sh` requires all of
them or it fails. Once the submodule moves past upstream ggml `33c9ea5`, drop
0018 as described below and remove `metal-*` from the `case` in
`scripts/configure.sh`.

## Building against patched vs stock ggml

`NEMO_SPEECH_GGML_PATCHED` tells the build whether the vendored ggml contains
this patch series. It defaults to `ON` because the ASR encoder directly uses
the fused relative-position attention op from 0001 and the F16 depthwise-conv
behavior from 0004.

```sh
# Patched ggml (default): apply the patches before configuring CMake.
scripts/apply-ggml-patches.sh
cmake -S . -B build -DGGML_CUDA=ON

# Pristine upstream ggml: do not apply the patches and opt out explicitly.
cmake -S . -B build -DGGML_CUDA=ON -DNEMO_SPEECH_GGML_PATCHED=OFF
```

With `NEMO_SPEECH_GGML_PATCHED=OFF`, the encoder uses stock ggml operations:
unfused relative-position attention and `ggml_conv_1d_dw` im2col lowering.
These paths remain correct across backends but cost latency on CUDA. The
dependent options below default to `ON` only when both `GGML_CUDA` and
`NEMO_SPEECH_GGML_PATCHED` are enabled, and are otherwise forced `OFF`:

| option | effect when enabled |
|---|---|
| `NEMO_SPEECH_FUSED_RELPOS_ATTN` | emit the fused relative-position attention CUDA op |
| `NEMO_SPEECH_DIRECT_DW_CONV` | use the direct CUDA depthwise-convolution kernel |
| `NEMO_SPEECH_FASTCONFORMER_CUDA_FUSIONS` | emit patched sigmoid-GLU and BF16-fusion graph patterns |

The options can be disabled independently for correctness or performance
bisection even in a patched build. Other patches optimize ordinary ggml
operations through their own eligibility checks and may still activate when
`NEMO_SPEECH_GGML_PATCHED=OFF` if the patched sources are present. A genuine
stock comparison therefore requires both a pristine ggml checkout and
`NEMO_SPEECH_GGML_PATCHED=OFF`.

## Patches

- **0001-fused-relpos-attn.patch** - adds relative-position fused attention for
  CUDA, including stride-aware inputs, F16 K/V/P storage, head-merged output,
  and the `NEMO_SPEECH_FUSED_RELPOS_ATTN` encoder path.

- **0002-nvfp4-residual-activations.patch** - keeps the native NVFP4 weight
  path while reducing activation-quantization error. Each activation
  sub-block is quantized once to FP4, its reconstruction residual is quantized
  to a second FP4 block, and both contributions are accumulated by the same
  MMQ tile. The correction is restricted to NVFP4; MXFP4 and non-native paths
  retain their upstream behavior. Backend correctness tests use the standard
  quantized-matmul tolerance rather than the previous relaxed NVFP4 threshold.

- **0003-norm-mul-add-fusion.patch** - fuses affine LayerNorm with row-vector
  scale and optional bias.

- **0004-conv2d-dw-f16-kernel.patch** - supports F16 weights in the direct
  depthwise-convolution CUDA kernel with F32 input and output.

- **0005-skinny-q8-gemm.patch** - adds a Q8_0 x F32 GEMM for skinny streaming
  activations, including planar weights, deterministic K-split reduction, and
  an optional bias epilogue. `GGML_SKINNY_Q8` controls dispatch;
  `GGML_SKINNY_Q8_INPLACE=0` uses separate repack storage when required by a
  multi-stream scheduler.

- **0006-cuda-dispatch-wiring.patch** - wires fused attention, affine
  LayerNorm, skinny-Q8, planar-Q8, and narrow bias/SiLU epilogues into the CUDA
  backend.

- **0007-magpietts-nanocodec.patch** - adds grouped transposed convolution and
  Snake for MagpieTTS and NanoCodec, two-column MMVF epilogues for paired CFG,
  bounded CUDA graph caching, and CUDA architecture handling.

- **0008-cublas-bf16-projections.patch** - flattens contiguous outer activation
  dimensions into shared-weight cuBLAS GEMMs and folds supported BF16 projection
  epilogues into output conversion.

- **0009-fastconformer-cuda-fusions.patch** - adds sigmoid GLU, Macaron
  residual, affine LayerNorm conversion, and BF16 projection fusions for
  FastConformer.

- **0010-cuda-pad-large-batch-grid.patch** - flattens CUDA PAD launches into
  `grid.x` so large batch dimensions do not exceed the `grid.z` limit.

- **0011-cuda-graph-shape-key.patch** - keys cached CUDA graph executables by
  the host graph identity plus a structural signature containing node count and
  endpoint tensor descriptors. This prevents allocator reuse from associating
  a new batch shape/topology with an incompatible executable while avoiding a
  full graph scan on every dispatch.

- **0012-cuda-streaming-cache-copies.patch** - recognizes inner-contiguous F32
  cache-tail views and materializes all batch planes with one pitched
  `cudaMemcpy2DAsync`; adds aligned float4 fast paths for gathering and
  scattering large indexed state-arena rows, including multi-plane K/V arenas.
  The shape/alignment guards keep all other COPY, GET_ROWS, and SET_ROWS cases
  on their existing kernels.

- **0013-cuda-cached-f16-cublas.patch** - optional cached-F16/cuBLAS path for
  skinny Q8 projections on NVIDIA SM80+. It expands immutable Q8 weights once,
  converts only the live activation, and retains FP32 accumulation/output.
  Runtime selection is controlled by `GGML_SKINNY_Q8_CUBLAS_F16` and its
  minimum-N threshold; cuBLAS chooses the implementation for the active GPU.
  Keep it opt-in because the F16 cache consumes additional device memory and
  the performance crossover depends on the GPU and physical batch size.

- **0014-cuda-relpos-extensions.patch** - extends fused relative-position
  attention for the cache-aware and offline FastConformer paths. The
  cache-aware path reads persistent K/V rows directly by state slot and
  circular head, then overwrites only the rows replaced by the current chunk.
  Register-resident NVIDIA SM80+ kernels cover the common R=0, 1, 3, 6, and 13
  streaming shapes for both Nemotron cache geometries, with exact-shape kernels
  retained where they are faster. The offline mask contract accepts both
  per-batch key-padding masks and `[key, query]` L/R masks. Set
  `GGML_CUDA_RELPOS_REGISTER_RESIDENT=0` before process start to
  disable the register-resident specializations without changing the direct
  circular-cache path.

- **0015-cuda-ctc-batch-fusions.patch** - reduces large-batch FastConformer
  overhead by fusing BatchNorm and BatchNorm+transpose+SiLU graph patterns,
  extending SiLU and affine LayerNorm output conversion to F16, and folding
  bias and residual addition into the cached-F16 cuBLASLt projection. The
  eligibility checks preserve the unfused path for unsupported layouts,
  precisions, and GPUs.

- **0016-fix-batched-conv1d-layout.patch** - restores the batch and output-channel
  axes after the flattened Conv1D matrix multiplication. The upstream direct
  reshape interleaves those axes for batches larger than one; batch one keeps
  its original zero-copy path.

- **0019-cuda-graph-dynamic-update.patch** - refreshes CUDA graph node
  parameters when a cached graph is replayed so dynamic pointers and launch
  geometry do not retain values from an earlier execution.

- **0020-bf16-convolution.patch** - adds BF16 im2col and direct depthwise
  convolution support, then fuses bias, BF16 output rounding, and optional
  ReLU epilogues. This preserves the VoiceChat perception stem's native BF16
  behavior without adding standalone conversion kernels.

## Regenerating after editing ggml

Several patches touch the same ggml files, so regenerating a patch from the
fully patched submodule can accidentally fold later changes into it. Edit and
diff at the patch's actual point in the series:
Most base files belong to one patch; 0013 and 0014 are explicit layered
exceptions. Do not regenerate 0001 from a fully patched live tree without first
removing the 0014 delta, or the circular-cache extension will be folded into it.

```sh
# Create a disposable worktree at the pinned upstream commit.
git -C ggml worktree add "$PWD/ggml-patch-work" HEAD
cd ggml-patch-work

# Apply every patch before the one being edited, then stage that baseline.
target=0007-magpietts-nanocodec.patch
for patch in ../ggml-patches/*.patch; do
    [ "$(basename "$patch")" = "$target" ] && break
    git apply "$patch"
done
git add -A

# Apply the target patch, edit it, and capture only its delta.
git apply "../ggml-patches/$target"
# Edit and test the affected files.
git add -N src/ggml-cuda/<new-file>  # only when the patch adds a new file
git diff --binary > "../ggml-patches/$target"

# Return to the repository root, check the patch, and remove the worktree.
cd ..
git diff --check -- "ggml-patches/$target"
git -C ggml worktree remove --force "$PWD/ggml-patch-work"
```

Adjust paths when the disposable worktree is placed elsewhere. If an edited
patch changes context used by later patches, rebase those later patches in the
same way.

Patch 0013 is intentionally layered on top of 0005. To regenerate it without
folding the generic skinny-Q8 implementation into the cached-F16 patch, use a
temporary ggml worktree, apply and stage patches 0001 through 0012 as the
baseline, then copy in only the cached-F16 changes and the SM100 CMake target
correction and run `git diff` against that staged baseline for `CMakeLists.txt`
and `skinny-q8.cu`.

Patch 0014 is intentionally layered on top of 0001 and 0013. Apply and stage
patches 0001 through 0013 in a temporary worktree, copy the edited
`include/ggml.h`, `src/ggml.c`, and `src/ggml-cuda/fused-relpos-attn.cu` into
that worktree, then generate 0014 with `git diff` against the staged baseline.

Generate patches with `git diff` only (GNU `diff`/editors can strip the
leading space on blank context lines, which `git apply` rejects as corrupt).

To verify the complete series, apply every patch in order to a fresh worktree at
the pinned ggml commit, then recursively compare that tree with the live patched
submodule and confirm that all files match.
