# Handover — Chatterbox T3: batched classifier-free-guidance (B=2) decode

**Goal.** Cut Chatterbox T3 autoregressive decode time by running the **cond +
uncond CFG passes as one batch-2 forward** instead of two sequential forwards.
This is the single biggest remaining chatterbox perf lever on M1/Metal: T3 AR
decode is the **largest stage** of synthesis (~28 s / ~37–88 % depending on
clip), and the reference impl **gianni-cor/chatterbox.cpp** measured **−42 % T3**
from exactly this change (Q4_0 872→502 ms, F16 1099→602 ms on M3 Ultra; their
`§3.21` "CFG cond+uncond batched into one Metal forward (B=2)").

You are taking a **fresh look** — the design below is a *de-risked starting
point*, not a mandate. Validate the premise yourself and improve on it.

This is a **runtime compute-graph task**. Per `feedback_no_agents_for_runtime_graphs`,
do the graph/alloc math yourself against a **greedy token-parity gate**; do **not**
delegate the graph wiring to a sub-agent. Keep the legacy sequential path; gate
the new one behind an env var; flip the default only after token-parity + a real,
quiet-machine speedup are demonstrated. Keep the gate forever (bisection).

---

## Why this is the lever (and what is NOT)

Per-stage bench (`CHATTERBOX_BENCH=1`), M1 Metal, q8 models:

| stage | time | note |
|---|---|---|
| **T3 AR decode** | **~28 s** | **largest stage**; 2 forwards/token (cond+uncond) |
| S3Gen CFM Euler | ~6–11 s | already trimmed (§207 6 steps; §211 turbo 2-step) |
| HiFT vocoder | ~3–5 s | convT range-fix already in (PR #04, merged upstream) |

Single-token T3 decode is **weight-bandwidth + dispatch bound** (1 token, the full
~30-layer weight set re-read every step). Running cond+uncond as **B=2** reads each
weight **once** for both passes and halves the dispatch count — that is the win.

**Already ruled out this campaign (don't repeat):**
- **CFM single-GPU `ggml_gallocr` graph cache** (§208, `CRISPASR_S3GEN_UNET_GALLOCR`):
  correct (parity 0.999) but a perf DUD — graph-rebuild is ~0.3 % of the
  compute-bound CFM per-step. See HISTORY/LEARNINGS §208.
- **Cached uncond T3 graph** (§212, `CRISPASR_CHATTERBOX_T3_CFG_BUCKET`): the uncond
  pass wasn't bucket-cached (it rebuilt its graph each token); adding a parallel
  bucket cache is bit-identical but **no win** — same §208 lesson (rebuild is
  negligible on compute-bound decode). Kept gated default-OFF.
- **Native Q4_K/Q8_0 CFM** (instead of the §205 dequant-to-F16): NaN at batch=2,
  garbage at batch=1, on **both** sched and gallocr → it is the quantized CFM
  mat-mul path, not the scheduler. Even if fixed it routes through `mul_mv`
  (mat-vec), slower than F16 `mul_mm` for the wide CFM. Helps size, not speed.

The common thread: **graph-cache/alloc tricks don't help compute-bound work.** B=2
is different — it changes the *arithmetic intensity* (weights read once for 2×
batch), which is the actual bottleneck.

---

## Current T3 architecture (read these first — `src/chatterbox.cpp`, current @ §212)

Two T3 variants: Llama-style (`build_graph_t3_kv`, CFG applies here) and GPT-2
(`build_graph_t3_gpt2_kv`, no CFG). **Only the Llama path needs B=2.**

- **`build_graph_t3_kv(c, n_past, n_tokens, use_kv_k, use_kv_v, fixed_kv_len, arena_ctx)`**
  (~1242) — builds one T3 step/prefill graph. Inputs: `inputs_embeds` (D,T),
  `positions` (T), optional `rope_freq_factors`, `causal_mask` (Lk,T). Per layer:
  RMSNorm → `core_attn::kv_self_attn(...)` → residual → RMSNorm → `core_ffn::swiglu`
  → residual. Output `logits` (speech_vocab, 1). Tags every mul_mat + flash_attn
  with `GGML_PREC_F32` at the end (~1419) — **this is load-bearing for T3 sampler
  parity** (Metal mul_mm downconverts to ~F16; PREC_F32 routes Q4_K→`mul_mv_q4_K_q8_K`
  and flash→CPU). Replicate it.
- **`core_attn::kv_self_attn`** (`src/core/attention.h` ~657) — the attention core:
  Q/K/V proj → reshape_3d → RoPE (NEOX, n_rot=hd) → permute → **write new K/V into
  cache** at `[n_past,n_past+T)` (F16 → `ggml_cpy` strided-view path; quant →
  `ggml_set_rows`) → read full `[0,Lk)` → **GQA expand** (`GQA_MANUAL_CONT`:
  reshape_4d/repeat_4d/reshape_3d/cont) → flash_attn_ext (PREC_F32 → CPU) → o_proj.
  KvSelfAttnParams for chatterbox: NEOX rope, GQA_MANUAL_CONT, rope_freq_factors set.
- **KV cache** (~1216): `kv_k`/`kv_v` (cond) and `kv_k_cfg`/`kv_v_cfg` (uncond) are
  separate 4D tensors `(hd, max_ctx, n_kv, n_layers)` — **layer is folded into
  `ne[3]`, so there is NO free tensor dim for a batch axis.** This is the central
  constraint (see design).
- **Decode loop** (~2923): per step — blend `logits = cond + cfg*(cond-uncond)`
  (~2940), sample (`sample_token`), build token embed (`build_speech_token_embed`,
  **same embed for both passes**), then **cond forward** `run_t3_kv(tok_embed,1,n_past)`
  (~2999) and **uncond forward** `run_t3_kv(tok_embed,1,n_past_cfg,kv_k_cfg,kv_v_cfg)`
  (~3008). `n_past` and `n_past_cfg` start equal (both prefill_len) and both `++`
  each step → **they stay in lockstep**.
- **`run_t3_kv`** (~1552) routes T=1 cond to the bucketed cached graph
  (`run_t3_kv_bucket`, ~1495); the uncond path optionally too (§212, gated).
- There is a **second CFG decode loop** at ~4274 (a streaming/diff variant) — check
  whether it needs the same treatment or can stay legacy.

---

## De-risked design (starting point — improve on it)

Key simplification you can rely on: **cond and uncond use the SAME token embedding
and the SAME positions/Lk/mask** (lockstep `n_past`). Only the **KV-cache contents**
differ (cond was prefilled with the conditioned context, uncond with the
unconditioned one). After layer-0 attention the two hidden states diverge, so the
**whole stack** must run B=2 — but the inputs are trivially shared.

The 4D-cache-with-layer-in-`ne[3]` layout means you **cannot** add a 5th batch dim,
and you **must not** restructure `core_attn::kv_self_attn` (shared by ~10 backends).
So:

> **Reuse both existing caches. Batch the heavy GEMMs; split the cheap attention core.**

Per layer, with `hidden_b2` shape `(D, 1, 2)` (batch 0 = cond, batch 1 = uncond):
1. `x = rms_norm(hidden_b2) * attn_norm_w` → (D,1,2).
2. `Q = mul_mat(q_w, x)`, `K = mul_mat(k_w, x)`, `V = mul_mat(v_w, x)` — **batched
   GEMMs, each weight read once** (ggml_mul_mat broadcasts a 2D weight over the
   batch dim). This is the −42 %.
3. reshape Q→(hd,n_q,1,2), K/V→(hd,n_kv,1,2); RoPE (NEOX, positions same for both
   batches — verify ggml_rope_ext handles the `ne[3]=2` batch independently).
4. **Split per batch** (views on `ne[3]`): for b∈{0,1}, permute K_b,V_b → write into
   `cache_b` at `n_past` (cache_0=`kv_k`, cache_1=`kv_k_cfg`), read `[0,Lk)`, GQA
   expand, attend Q_b → `attn_b` (n_q*hd, 1). The attention core is **cheap**
   (single query token, small Lk) so doing it twice is fine.
5. concat `attn_0,attn_1` along batch → (n_q*hd,1,2); `O = mul_mat(o_w, attn_b2)`
   (batched); residual; FFN `swiglu` (batched); residual.
6. Final RMSNorm*output_norm; `logits = mul_mat(speech_head_w, hidden_b2)` → (V,1,2):
   batch 0 = cond logits, batch 1 = uncond. Blend on the host as today.

Then: `build_graph_t3_kv_b2(...)` + `run_t3_kv_b2(...)` (alloc on a dedicated sched,
set the 2 inputs, compute, read both logit rows), and a new decode branch that calls
it once per step instead of the two `run_t3_kv` calls. Bucket/cache it by Lk like the
cond path if profiling shows rebuild matters (it probably won't — see §208/§212).

**Alternative to weigh with fresh eyes:** instead of two existing caches + split
attention, allocate one *new* batched cache `(hd, max_ctx, n_kv*2, n_layers)` (fold
batch into the kv-head dim) or per-layer caches that free `ne[3]` for batch — then a
single batched flash_attn with a block-diagonal mask. Cleaner attention, but a bigger
prefill/cache rewrite and you must keep cond/uncond from attending across each other.
Pick whichever you can get to **bit-parity** fastest.

### Constraints you must respect
1. **PREC_F32 everywhere** (mul_mat + flash_attn). T3's multinomial speech-token
   sampler is uniquely sensitive to ~1e-3 K-projection drift; the existing graph
   tags every node (~1419). The B=2 graph must too, or greedy parity will fail.
2. **RoPE NEOX, n_rot=hd, GQA_MANUAL_CONT, F16 cache cpy** — replicate
   `kv_self_attn` faithfully (bit-parity, not "close").
3. **Don't touch `core_attn::kv_self_attn`** (shared). Inline a chatterbox-specific
   b2 attention, or add an opt-in b2 entry that the other callers never hit.
4. **GPT-2 T3 path has no CFG** — leave it on the legacy path.
5. **Two KV caches already exist and are prefilled** (cond + uncond) — reuse them;
   don't reallocate unless you take the batched-cache alternative.

---

## Validation — MANDATORY (correctness-critical: this drives the sampler)

1. **Greedy token parity is the gate.** Force greedy with `CRISPASR_CHATTERBOX_TEMP=0`
   (deterministic argmax — removes multinomial sensitivity). Fixed seed. The B=2 path
   must emit the **exact same speech-token sequence** as the legacy sequential path.
   Compare the full token list (bump verbosity / use `CHATTERBOX_DEBUG` — note the
   `step=N tok=M` log caps at step<32; print the full `valid` list or dump tokens).
2. **Per-pass logit parity** (debugging): `CRISPASR_CHATTERBOX_DUMP_LOGITS_AT=<n_past>`
   prints the first 8 cond logits; compare B=2 batch-0 vs legacy cond, batch-1 vs
   legacy uncond, within ~1e-4.
3. **End-to-end:** ASR-roundtrip both outputs (`--backend moonshine -m
   moonshine-base-q4_k.gguf`) → identical text. Per `feedback_tts_validation`.
4. **Speedup:** `t3_ar_decode` ms/token from `CHATTERBOX_BENCH=1` must drop
   meaningfully. **Measure on a quiet machine, alternating order** — the M1 has been
   at load 24–50 from parallel sessions and absolute A/B is meaningless under
   contention (the [[project_chatterbox_t3_decode_perf]] noise trap). Report
   before/after ms/tok and total RTF.
5. **q8 + F16 + Q4_K** T3 must all still produce intelligible audio and pass token
   parity. (Q4_K T3 takes the `mul_mv_q4_K_q8_K` PREC_F32 path — make sure the
   batched mul_mat still hits it; check `ggml-metal-ops.cpp` ~2093.)
6. **`crispasr-diff chatterbox`** T3 stages (`t3_text_tokens`, logits) must still pass
   with the env OFF (legacy bit-unchanged) — and ideally ON.

Acceptance: identical greedy tokens (q8+F16+Q4_K) + identical ASR + measurable
quiet-machine ms/tok speedup, legacy path bit-unchanged when the env is off. Only
then propose flipping the default; keep the gate + legacy path forever.

## Failure modes to watch
- **Sampler diverges (tokens differ) but logits look ~right** → missing PREC_F32 on a
  B=2 node, or RoPE applied wrong across the batch dim. Bisect with
  `DUMP_LOGITS_AT` + `DUMP_KPROJ_AT` per batch.
- **Uncond batch attends to cond context (or vice-versa)** → cache-binding/view bug in
  the split step; uncond logits won't match legacy uncond. Check the per-batch cache
  write offsets and the `[0,Lk)` reads.
- **NaN** → an all-zero activation column hitting a quantized mat-vec divide (guarded
  in the q8 quantize kernel, but re-check if you change types), or an uninitialised
  cache tail not masked.
- **No speedup** → you didn't actually batch the GEMMs (each weight still read twice),
  or the machine was contended. Verify `ggml_mul_mat(W, X_b2)` is one op with `ne[2]=2`,
  and measure quiet + alternating.

## References
- HISTORY/LEARNINGS **§207** (CFM 10→6), **§208** (gallocr cache DUD + "graph-cache
  helps only overhead-bound graphs"), **§211** (turbo meanflow 2-step fix), **§212**
  (s3gen thread-env fix + gated cached-uncond T3). LEARNINGS §208.
- `feedback_no_agents_for_runtime_graphs`, `feedback_diff_alignment`,
  `feedback_tts_validation`, `project_chatterbox_t3_decode_perf` (M1-noise A/B
  protocol; threads already wired §176u), `project_chatterbox_perf_levers_211`.
- Reference impl (read-only, **MIT** — may borrow with attribution):
  `/Volumes/backups/code/gianni-chatterbox-cpp` — `src/t3_mtl.cpp`
  `build_step_graph_mtl_b2` / `build_prompt_graph_mtl_b2`, `eval_step_mtl`; gate
  `use_b2 = !meanflow && cfg_rate != 0 && !is_cpu`. Their README §3.21 has the −42 %
  numbers. **Study as existence proof; don't copy their namespace.**
- Models: `/Volumes/backups/ai/crispasr/chatterbox-{t3,s3gen}-q8_0.gguf` (+ `-q4_k`,
  `-f16`). ggml is vendored in-tree (MIT) — Metal kernel changes go in
  `ggml/src/ggml-metal/` with `// CrispASR patch` markers + a paired entry in
  `tools/upstream-prs/`.

## Env vars (add to PLAN §176 + the backend's section)
- `CRISPASR_CHATTERBOX_T3_CFG_B2=1` — opt into the batched-CFG B=2 T3 decode (this
  task). Default off (legacy sequential cond+uncond) until validated.
- (existing, keep working) `CRISPASR_CHATTERBOX_TEMP` (0 = greedy parity gate),
  `CRISPASR_CHATTERBOX_THREADS`, `CRISPASR_CHATTERBOX_SEED`,
  `CRISPASR_CHATTERBOX_T3_CFG_BUCKET` (§212, separate gated path).
