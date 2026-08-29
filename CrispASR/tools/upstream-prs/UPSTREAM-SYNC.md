# Inbound sync audit — what we should take FROM upstream ggml

Counterpart to `MASTER-AUDIT.md`, which tracks what we send upstream. This tracks
the other direction, and it exists because of a concrete miss: `core_adaln` failed
on ARM for weeks because our vendored ggml predated **two** upstream SVE fixes.
Nobody was watching the inbound direction at all.

**State (audited 2026-07-29)** — `CrispStrobe/ggml @ crispstrobe-ops`:

    11 commits ahead, 435 behind ggml-org/ggml master
    our base is v0.10.2; upstream master is v0.17.0 — seven minor versions

Of the 11 ahead, 2 are upstream cherry-picks made today (`6aab1bcb`, `f69bdbb3`),
so **9 are genuinely ours**: the CrispStrobe-ops base patch, conv_1d/conv_1d_dw
batch N>1, Metal im2col batch-1 occupancy, the Metal profiler, the Metal teardown
warning, a CUDA CCCL include, gguf empty-key rejection, and webgpu NORM.

## Priority 1 — correctness in paths CrispASR actually exercises

Ranked by how directly they touch what this project runs: conv1d/im2col for audio
codecs and VAD, norm chains in DiT blocks, and CPU tail paths.

| commit | why it matters here |
| --- | --- |
| `b40b6928` metal : fix im2col 1D case (**audio models**) | Upstream says "audio models" in the subject. We are conv1d-heavy on Metal and M1 is the primary dev box. Overlaps our own `23695065` im2col patch — reconcile, do not blind-apply. |
| `98ecb4de` metal : restore im2col implementation for large kernels | Same kernel family; likely wants applying together with the above. |
| `b821278c` ggml : fix A indexing in simd_gemm scalar **tail-column** path | Same *class* as the SVE bug we just fixed — a tail path mishandled. Exactly the shape that hides behind dims that are not multiples of the vector width. |
| `ba5f60df` ggml : fix broken CPU concat for **quantized** types | We concat quantized tensors across several backends. |
| `29e2eef8` ggml-cpu : fix rms_norm_back wrong output under **in-place aliasing** | Norm chains are everywhere in the DiT/TTS paths. |
| `5afc45fe` ggml : fix conv 2d dw | We carry our own conv_1d_dw work (`655c14e4`); adjacent code. |

## Priority 2 — Vulkan

CrispASR ships Vulkan on Windows and has a standing list of Vulkan-only TTS
failures (#304 CosyVoice3, #192 TADA, #215 moss, #171 VibeVoice). Any of these may
be load-bearing:

| commit | note |
| --- | --- |
| `ee317147` vulkan: workaround compiler bug in **conv2d coopmat2** path | Our #171 report (upstream-prs 19) is an RDNA4 **coopmat2** miscompute in flash-attn. Different op, same family — evaluate before filing 19. |
| `c6eff036` vulkan: fix 32-bit integer overflow in CEIL_DIV | |
| `8f3ec498` vulkan: apply bias before softmax in FA, avoid overflow | |
| `f257513c` vulkan: sync on event_wait for transfer-queue async copies (race) | |
| `a55769ce` vulkan: fix step operator for 0 input | |

## Priority 3 — CUDA

| commit | note |
| --- | --- |
| `d71b15e2` CUDA: fix get_rows_back for >65535 rows | We carry a get_rows k-quant patch (upstream-prs 16); same kernel area. |
| `1319de9e` / `6d2504d5` fattn KQ-mask integer overflow | We use flash-attn on CUDA. |
| `00504b04` CUDA: various fixes to `cpy.cu` | upstream-prs 03 patches `cpy.cu`; check for overlap. |
| `5f0b0e89` CUDA: fix ssm_scan_f32 data-races | |
| `b5775f06` CUDA: VMM pool / Turing P2P | |

## Priority 4 — features worth having

`fda9d536` metal col2im_1d, `a1329a7a` metal CONV_2D_DW, `32614331` CPU
RMS_NORM+MUL fusion, `540ffc4f`/`e9eaaef3` concat f16/bf16.

## Two collisions to resolve BEFORE any sync

1. **`col2im_1d` now exists upstream** (`fda9d536`, by Pascal) — and it is NOT the
   same op as ours despite an identical name and signature. Verified against both
   headers during the 2026-07-29 merge:

       upstream  scatter-add, p0 = crop from BOTH sides, T_out = (T_in-1)*s0 + K - 2*p0
       ours #160 gather,      p0 = LEFT offset,          T_out = (T_in-1)*s0 + K -   p0

   `src/core/conv.h:166` (`convt1d_decomp`) depends on OUR gather/left-offset
   semantics — it passes `crop_left` as `p0` and trims `crop_right` with a view.
   Swapping in upstream's op silently changes ConvTranspose1d output LENGTH in every
   TTS decoder that uses it. It compiles, links, and produces wrong audio.

   **CORRECTION**: an earlier draft of this file said "adopt upstream's and delete
   ours". That was wrong — it was written from the commit subject before the two
   headers were compared. Adopting upstream's op REQUIRES rewriting `convt1d_decomp`
   to pass p0=0 and do both crops with views, and that rewrite must be validated
   against TTS audio (f5-tts, cosyvoice3, TADA, vocoders) before CrispASR's pin
   moves. Until then the merge keeps BOTH ops disambiguated, not one silently
   replacing the other.

   `upstream-prs/20-col2im-1d-new-op.md` is still superseded as an OUTBOUND PR —
   upstream has its own op now, so there is nothing to file.

2. **`conv_transpose_1d` arrives twice.** `a056a26f` is *our own* merged PR
   (ggml#1477); we also carry the change locally, so the tightened `i_min`/`i_max`
   loop is already in `ggml-metal.metal:4998`. Not a gap — but a sync will present
   it as a conflict, and the right resolution is upstream's copy.

## Recommendation

Cherry-pick Priority 1 now: those are small, targeted, and in code we run daily.

**Then plan a real sync rather than more cherry-picks.** Seven minor versions of
drift (v0.10.2 → v0.17.0) means every further cherry-pick applies against
increasingly foreign context, and the `MUST RE-APPLY after ggml bump` markers show
this fork already depends on manual re-application. The SVE miss is what that debt
looks like when it comes due: a real bug, shipped, invisible for weeks, found only
because an unrelated CI audit turned the unit tests on.

`MASTER-AUDIT.md` (outbound) is itself stale — it was cross-checked against
upstream on 2026-05-05, 435 commits ago. Re-run it as part of the same effort.
