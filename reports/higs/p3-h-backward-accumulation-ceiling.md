# HIGS P3-H — Backward Accumulation Ceiling / Cost Attribution Audit

**Date**: 2026-09-22 · **Env**: `mx` (ssh mx) A100-PCIE-40GB, `higs-13scene-env` torch 2.9.1+cu128, nvcc 12.8 · **Type**: CONTROLLED-REPLAY UPPER-BOUND · **GPU**: idle (tag `idle_gpu`)

**Decision: `HAR_CEILING_WEAK` → P3-HAR should not proceed on the global-atomic-accumulation mechanism.**

---

## 1. Frozen baseline
`C0 V3 = F9 + SCALAR_ADJOINT + H8-MR` (HIGS_BWD_SCALAR_ADJOINT=scalar_adjoint, HIGS_BWD_H8_MR=1, HIGS_PX_RUNTIME=2 → PX=2, block=128).
Fixtures: room / bicycle / garden, cam0, max_long_side=2048. Frozen source worktree `higs_h8_mr_worktree` (git-clean, untouched). Diag source (snapshot) sha `f01b624b…`; frozen production source prefix `e657485f9a187be6`.

## 2—3. Semantic partition & boundary
| # | Component | class | HAR target? |
|---|---|---|---|
| A | pixel-local reverse compositing recurrence (T/alpha) | ORDER_DEPENDENT | NO (preserved in all variants) |
| B | color adjoint accumulation | ASSOCIATIVE_AFTER_RECURRENCE | YES |
| C | opacity adjoint accumulation | ASSOCIATIVE_AFTER_RECURRENCE | YES |
| D | H8 geometry-moment accumulation | ASSOCIATIVE_AFTER_RECURRENCE | YES |
| E | projected/master Gaussian VJP | MANDATORY_GLOBAL_OUTPUT | NO (left real) |
| F | SH VJP | MANDATORY_GLOBAL_OUTPUT | NO (left real) |
| G | densification-facing updates | MANDATORY_GLOBAL_OUTPUT | NO |
| H | other backward overhead | — | NO |

HAR may only target the final associative stage; the reverse alpha/T recurrence was never removed.

## 4. Diagnostic variants (compile-time `P3_SINK_MODE`)
- **V0 FULL** (mode 0) — frozen C0 V3.
- **V1 ACCUM_SINK_ALL** (1) — sinks B+C+D.
- **V2 ACCUM_SINK_GEOM** (2) — sinks D only; B,C real.
- **V3 ACCUM_SINK_APPEAR** (3) — sinks B+C only; D real.
- **V4 WRITE_ONLY/NO_ATOMIC** (4) — full 9 values, plain uncontended unique-slot stores (isolates atomic RMW vs store).

Sink mechanism: per-warp UNCONTENDED slot in a diagnostic scratch buffer; all q/alpha/T/recurrence/local-gradient arithmetic, warp reduction, index mapping, and branch decisions preserved. Only the global atomic scatter destination is replaced.

## 5. Anti-cheating compliance
V1 still executes: q eval, alpha derivative, T recurrence, color/opacity/H8-moment gradient arithmetic, warp `warpSum`, `id_batch` index mapping, and all branch decisions. V1 is **not** a backward early-return. Only the global atomicAdd accumulation / output traffic is removed.

## 6. Source-level accounting
See `accumulation_sites.json`. In `higs_blend_bwd_px_kernel`: B `atomicAdd(v_rgb_ptr+k,…)` CDIM (dest v_colors), C `atomicAdd(v_opacities+g,…)`, D `atomicAdd(v_conic_ptr+{0,1,2})` (v_conics) + `atomicAdd(v_xy_ptr+{0,1})` (v_means2d). All four HAR-eliminable. Projection VJP `gpuAtomicAdd` on v_means/quats/scales and SH VJP `gpuAtomicAdd` on v_coeffs are **not** in V1 scope (reported separately). Background + non-PX blend kernels unguarded.

## 7—8. Protocol & primary metrics (median backward ms, idle GPU, 20w/5r/100s, rotating order)
| scene | V0 | V1 | V2 | V3 | V4 | ALL ceiling %b | GEOM %b | APPEAR %b |
|---|---|---|---|---|---|---|---|---|
| room   | 19.05799 | 19.06804 | 19.07166 | 19.07065 | 19.07492 | **−0.05** | −0.07 | −0.07 |
| bicycle| 19.51897 | 19.41432 | 19.47407 | 19.50773 | 19.42349 | **+0.54** | +0.23 | +0.06 |
| garden | 18.34557 | 18.36332 | 18.35739 | 18.35665 | 18.36654 | **−0.10** | −0.06 | −0.06 |

Full stats (mean/median/std/p10/p90) in `timing_summary.json` (std 0.001–0.01 ms, 5 reps × 100 samples).

## 9. F+B / full-iteration ceilings (median)
| scene | F+B V0 | F+B V1 | ALL ceiling %F+B | full-iteration ceiling |
|---|---|---|---|---|
| room   | 20.50643 | 20.50259 | +0.02 | ~+0.02% |
| bicycle| 21.31694 | 21.21677 | **+0.47** | ~+0.47% |
| garden | 19.40620 | 19.41603 | −0.05 | ~−0.05% |

Forward ≈1.4 ms of the ~20.5 ms F+B. Full-iteration ceiling ≈ F+B ceiling ≤ +0.5%.

## 10. Additivity check
GEOM + APPEAR ≈ ALL is roughly consistent (bicycle 0.23+0.06=0.29 ≈ 0.54), but all terms are at noise level; no additive interpretation is forced because every ceiling is ~0.

## 11. Atomic contention evidence
`COUNTER_UNAVAILABLE` (no CUPTI/hardware-perf permission). Replay: V4 (plain stores) is not faster than V0 (atomic RMW) on any scene → RMW/contention component ≈ 0.

## 12. Gradient-family decomposition (DERIVED)
See `family_decomposition.json`. Geometry/H8 D ≤ +0.23%, appearance B+C ≤ +0.06% (bicycle); ~0 on the other scenes. Neither family carries meaningful removable time.

## 13. H8 remaining aggregation cost
**Negligible as measured on C0 V3.** On C0 V3 with H8-MR enabled, the remaining measured geometry-accumulation ceiling is negligible; this experiment does not independently attribute that result causally to H8.

## 14. R6-A interpretation
**Not retroactively classified as definitively ceiling-small.** DSH-H-R1 shows V0 (the P3-H baseline) is not ABI-identical to the original production C0 V3 binary, so R6-A's original baseline equivalence is not proven. The observed sub-1% ceiling is **plausible** for the accumulation mechanism but is not an independent, proven attribution for the historical R6-A failure.

## 14b. DSH-H-R1 production-identity addendum (`P3H_PRODUCTION_IDENTITY_FAIL`)
- Production composed .so `7ca1c6bf…` located (`higs_c0_cache_composed`).
- `higs_blend_bwd_px_kernel<3,2,true,true>`: PROD REG=64, V0 REG=56; both SHARED/local=0 (dynamic smem set via cudaFuncSetAttribute); **SASS NOT identical** (864 vs 696 static instrs; opcode histogram differs).
- Root cause: the diagnostic build adds a trailing `v_p3h_scratch` kernel param (unused at P3_SINK_MODE=0) → different ABI/mangling, register allocation, scheduling. Accumulation body at P3=0 is compile-time identical.
- Gate: kernel/resource audit shows a material difference → **P3H_PRODUCTION_IDENTITY_FAIL**.
- Consequences: P3-H is **retained as diagnostic only**; P3-0's 3–6% `ATOMIC_FREE_ORACLE` is **NOT superseded** (remains authoritative); HAR remains DROP; no new candidate.

## 15. HAR gate
**`HAR_CEILING_WEAK`** — ALL_ACCUM_CEILING ≤ +0.54% backward (<6%) on ALL THREE scenes. Per spec, P3-HAR should not proceed on this mechanism unless P3-0 identifies a *different* removable mechanism.

## 16. P3-0 cross-check (structure; no P3-0 numbers used)
If P3-0 predicts removable accumulation ms ≫ DSH-H ceiling (~0 ms) → it double-counts. ∵ ceiling ≈ $T(V0)-T(V1)$ ≈ 0 ms, any material P3-0 estimate above ~0 is overreach.

## 17. Training-semantics safety
Frozen worktree untouched (git clean, no P3_SINK_MODE). All 5 diagnostic `.so` hashed (`binary_hashes.json`); built only in `higs_p3h_cache`, never deployed. V0 byte-hash differs from production `.so` (build-flag provenance only; mode-0 semantics identical; controlled V0–V4 delta valid).

## 18. Deliverables
Report + `artifacts/higs-p3-h/` (all 12 JSONs + raw_timing.csv) + `scripts/p3/p3_h_accum_ceiling.py` (build/hash/emit-static/smoke/timing) + `scripts/p3/p3h_run_timing*.sh`. P3-0 artifacts untouched.

## 19. Timing evidence grade: HIGH
Idle GPU, tag `idle_gpu`, low noise (std ≤0.01 ms), 5 reps × 100 samples, rotating order, same process/GPU/fixture. Confound: sink variants carry ~0.01 ms fixed overhead, so the already-~0 ceilings are if anything stated optimistically.

## 20. Next action
**Do not pursue HAR on global atomic accumulation.** Re-run this controlled-replay tag only if P3-0 surfaces a *different* removable mechanism (e.g. the projection VJP → master-Gaussian reduction, or the SH VJP). If P3-0's predicted removable accumulation ms is non-trivial, forward the measured ~0 ms ceiling as a falsification of its accumulation-time model.