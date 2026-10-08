# P18-P0 — FQ-GS Warp-Quotient Rasterization Oracle — Execution Report

**Verdict: `P18_P0_HARD_DROP`** (terminal — no garden, no P18-P1).

## 1. What was tested

P18-P0 asks whether the per-pixel `G(x)=exp(q(x))` in the frozen C0 raster inner
loop can be accelerated by replacing the per-pixel `exp` with a few *exact*
exponential seeds reconstructed across a warp/subwarp of contiguous pixels via a
multiplicative prefix scan (QSCAN4/8/16), preserving the original 3DGS numerics
and training semantics (no approximate exp, no down-res, no skipped Gaussians,
no tile/order/alpha/loss/optimizer change).

## 2. Setup (all verified before any work)

- **Frozen C0 identity** — all 4 binaries verified (`IDENTITY_VERIFIED`):
  trainer `675b0267…`, c0_ext `d26cccf4…`, core `935ea6c0…`, fused_ssim `dd6a4c79…`; 16K checkpoints room `17c1f590…`, bicycle `0c36f5c7…`.
- **Inner-loop audit** (`P18_BASELINE_INNER_LOOP_AUDIT`) — `MAPPING_OK`: in the forward
  `macro_tile_rasterize_kernel<16>` (TILE=16, `GAUSS_PER_WARP=1`) the warp's 32 lanes cover
  32 contiguous x-pixels of one row for one Gaussian → `q(x)` is a 1-D quadratic in x, so the
  quotient reconstruction applies to contiguous 4/8/16 subwarps. `P18_MAPPING_BLOCKER` was
  **not** triggered. Backward (`higs_blend_bwd_px_kernel`, PX=2) recomputes the weight and has
  the same row structure.
- **Camera set** (`P18_CAMERA_SET.json`) — frozen **before** any timing. N/8 quantiles:
  room (N=311) `[19,58,97,136,174,213,252,291]`; bicycle (N=194) `[12,36,60,84,109,133,157,181]`.
  (P17 used N/12 — different count — so this is a fresh P18 set; P17 provenance recorded.)
- **Hardware** — single clean A100-SXM4-80GB, GPU 3 (`GPU-431ac351-…`), `CUDA_VISIBLE_DEVICES=3`,
  0 MiB / 0 % util throughout (foreign process was on GPU 2 only, a different physical GPU —
  no cross-GPU interference). Contamination: **CLEAN**.

## 3. Numerics (Phase 1) — the reconstruction is CORRECT

Isolated `DIRECT` vs `QSCAN4/8/16` on **synthetic conics** and **real room/bicycle 16K conics**,
both the forward `exp2` domain and the backward `exp` domain, with the fixed γ=2×10⁻⁶ guard:

| | QSCAN4 | QSCAN8 | QSCAN16 |
|---|---|---|---|
| room real max_abs | 1.9e-06 | 1.9e-06 | 2.0e-06 |
| bicycle real max_abs | 2.5e-09 | 4.9e-09 | 1.6e-08 |
| alpha-branch mismatch | 0 | 0 | 0 |
| NaN/Inf | 0 | 0 | 0 |

Absolute G error is ~2×10⁻⁶ (float rounding over ≤16 multiplications) and the **alpha-cull
branch agreement is 100 %**. Relative error inflates only where G is tiny (the culled region),
which the protocol's forward gate (max|ΔRGB|≤2×10⁻⁵) does not constrain. **No speed claim was
made in this phase.** The candidate is numerically sound.

## 4. Performance (Phase 3) — the candidate does NOT win

Real-workload capture (`P18_TRACE_MANIFEST`): ≥8192 warp×Gaussian loop positions per scene,
stratified by live-lane population. The evaluator (`P18_EVAL_MICROBENCH`, CUDA events, median,
CUDA-graph timed, no launch overhead) on the **real trace** (not an all-lanes-active synthetic):

| variant | room (exp2/exp) S_eval | bicycle (exp2/exp) S_eval | median |
|---|---|---|---|
| QSCAN4 | 0.982 / 0.974 | 0.954 / 0.954 | 0.966 |
| QSCAN8 | 0.971 / 0.974 | 0.975 / 0.970 | **0.972** |
| QSCAN16 | 0.974 / 0.973 | 0.964 / 0.967 | 0.969 |

`S_eval = T_DIRECT / T_QSCAN ≈ 0.95–0.98` — **QSCAN is at best a wash, slightly slower** than
DIRECT. Every variant in both primary scenes is far below the 2.5 hard-drop floor.

### Root cause (PTX/SASS static inspection — allowed in place of Nsight)

| kernel | MUFU.EX2 (SFU exp, dynamic) | SHFL | SASS lines |
|---|---|---|---|
| DIRECT (per-pixel) | **1** | 0 | 84 |
| QSCAN4 / 8 / 16 | 4 (leader seeds) | 3 / 4 / 5 | 212 / 228 / 260 |

1. **DIRECT is SFU-exp-bound.** The frozen C0 inner loop is one `MUFU.EX2` (the exp) per pixel,
   with only ~3 FFMAs for the quadratic. The exp *is* the work.
2. **QSCAN's seeds are leader exps on the critical path.** `G0=exp(q(x0))`, `r0=exp(d0)`,
   `k=exp(-dd)` are computed once per subgroup but every lane's `G_j = G0·prefix_j` depends on
   `G0`, so the subgroup still has its exps on the critical path. The quotient does not remove
   the SFU exp from the latency path; it only redistributes it.
3. **The scan adds overhead.** SHFL + FMUL per lane (QSCAN16 = 260 SASS lines vs 84). On an A100
   with a wide MUFU pipeline the exp is already cheap, so the scan overhead cancels the small
   saved FMA work.
4. **C0 Gaussians are tight.** `MAX_EXTEND=3.33` sigma culling → live-lane population is near-zero
   (room mean 0.043, bicycle 0.006, median 0). The inner loop is mostly exp-evals that are culled;
   there is little redundant per-pixel compute for a quotient to eliminate.

## 5. Why the other gates were not evaluated

Phases 4–5 (integrated forward/backward fidelity, DIRECT repeat noise floor) and the integrated
`S_F/S_B/S_FB` timing were **not run**: no variant cleared the evaluator floor (median
`S_eval ≥ 4.0`), and the HARD DROP triggered first. `S_FB` cannot exceed ~1.0 because the
evaluator already shows QSCAN ≈ DIRECT. `P18_FORWARD_FIDELITY.json`, `P18_BACKWARD_FIDELITY.json`,
`P18_INTEGRATED_TIMING.json`, `P18_DIRECT_REPEAT_NOISE.json` are therefore intentionally absent.

## 6. Decision

**`P18_P0_HARD_DROP`** — protocol section 16, clause 1: *all variants in any primary scene have
median `S_eval < 2.5`.* Terminal. Per protocol 16/21: **no** threshold/γ/tile/approximate-exp
change, **no** camera change, **no** quality-gate relaxation, **no** git commit, **no** full
training, **no** automatic entry to P18-P1.

## 7. Interpretation & what a future attempt would need

The FQ-GS quotient reconstruction is **numerically correct** but does **not** reduce the SFU-exp
cost on this hardware — it is a wash (S_eval ~ 0.97). A future attempt would have to (a) eliminate
the SFU exp itself rather than trade it for a scan (e.g. a different exp path, or half-precision
where `exp2` is the sole op), or (b) target a regime with much higher live-lane populations where
the per-pixel FMA work dominates the exp. Both are P18-P1 territory, which protocol 16 forbids
entering after a HARD_DROP. The honest scientific conclusion: **on the frozen C0 / A100 backend,
the warp-quotient exp-reconstruction has no measurable headroom** — the raster inner loop is
already SFU-exp-bound with tight Gaussians, and the quotient cannot beat an exp that is already
cheap.

## 8. Artifact index

| File | Content |
|---|---|
| `P18_SOURCE_IDENTITY.json` | 4 frozen binaries + 2 ckpts verified |
| `P18_CAMERA_SET.json` | frozen 8-camera N/8 set + P17 provenance |
| `P18_BASELINE_INNER_LOOP_AUDIT.json` / `.md` | forward/backward inner-loop audit; MAPPING_OK |
| `P18_TRACE_MANIFEST.json` + `traces/*.npz` | real warp×Gaussian positions, stratified by live-pop |
| `P18_REFERENCE_VALIDATION.json` | DIRECT vs QSCAN4/8/16 numerics (synthetic + real) |
| `P18_EVAL_MICROBENCH.json` | S_eval on the real trace (CUDA events, median) |
| `ptx/p18_eval_sass.txt` | SASS of the evaluator kernels (MUFU/SHFL attribution) |
| `P18_VARIANT_TABLE.json` | per-variant numerics + S_eval + SASS + decision |
| `P18_CONTAMINATION.json` | A100 GPU 3 clean, foreign GPU 2 only |
| `P18_VERDICT.json` | `P18_P0_HARD_DROP` + evidence + rules |
| `P18_REPORT.md` | this report |
| `MANIFEST_P18.json` | full artifact manifest + sha256 |
| `cuda/p18_qscan_ref.cu`, `cuda/p18_eval.cu` | QSCAN kernels (reference + evaluator) |
| `scripts/*` | drivers (identity/cameras, ref-validation, trace/eval) |
