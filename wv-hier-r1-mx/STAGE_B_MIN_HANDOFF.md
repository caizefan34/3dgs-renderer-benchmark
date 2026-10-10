# Stage B-min — Execution Handoff (MX A100)

**Protocol:** `WV_HIER_R1_MX_STAGE_B_REVISED_20261008` · **Date:** 2026-10-08 · **Round:** 6
**Status of this host:** MX unreachable → GPU oracle **NOT_EXECUTED here**. This runbook is for an idle MX A100.
**This is a SUPERVISION ORACLE, not a performance benchmark.** All timing from full-render + masked-loss is `SUPERVISION_ORACLE`; never claim sparse-CUDA performance.

---

## 0. Hard constraints (do NOT violate)
- **No CUDA modification / no recompile.** The CUDA core `_wv0_core.py` / `gsplat_cuda_WV0.so` is used UNCHANGED.
- **No training.** 0 optimizer steps. This is a gradient-quality oracle only.
- **No scene expansion.** Bicycle `__iter15000` only.
- **Do NOT preempt TEMPORAL_COVERAGE-R1.** Snapshot `nvidia-smi` before/after; abort if no clearly idle A100.

## 1. Pre-flight (on MX)
```bash
nvidia-smi                         # confirm an idle A100-40GB (not GPUs 0-3 if reserved for TEMPORAL_COVERAGE-R1)
python -c "import torch,gsplat; print(torch.__version__, gsplat.__version__)"   # expect torch 2.4.1+cu124, gsplat 1.5.3
# verify input hashes match the audit:
sha256sum bicycle__iter15000.pth gsplat_cuda_WV0.so _wv0_core.py
```

## 2. Files to place together
Copy into one MX working dir:
- `_wv0_grad_stage_b.py` — the turnkey oracle patch (CRN + HWCS + N1/N2 + paired metrics + raw-CSV writer). Imports `_wv0_core.render_full` UNCHANGED.
- `_wv0_core.py` + `gsplat_cuda_WV0.so` — the audited CUDA core (unchanged).
- `stage_b_min_runner.py` — CPU self-test + dispatch guard (already validated on the dev host).

Set hash env vars (used to stamp the raw CSV):
```bash
export WV0_CHECKPOINT_SHA256=<sha of bicycle__iter15000.pth>
export WV0_SOURCE_SHA256=<sha of _wv0_core.py>
export WV0_SO_SHA256=<sha of gsplat_cuda_WV0.so>
```

## 3. Sanity (CPU, no GPU) — already PASS on dev host, re-run on MX
```bash
python stage_b_min_runner.py --selftest      # masks/shapes/inclusion≈p/variance<0.4%/N1N2 logic
python _wv0_grad_stage_b.py --selftest       # mask shapes + inclusion for 5 methods
```
Expected: 5 methods (G16, G4_N1, G4_N2, HWCS-1, HWCS-2); inclusion 0.2497–0.2506; variance-identity <0.4%; N1/N2 identical pixel mask, distinct work (W_N1=7.517 vs W_N2=4.510 at ρ=6.01).

## 4. Execute (GPU)
The oracle exposes `run_camera(gparams, cam, sh_degree, seed)` per (camera, seed). Cameras = IDs {0, 8}; 32 predetermined paired seeds; methods = G16, G4_N1, G4_N2, HWCS-1, HWCS-2; marginal p=0.25. One common `(U_parent, U_child)` per (camera, seed) shared across all methods (CRN). Write one raw row per (camera, seed, method) to `stage_b_raw_results.csv` (schema in STAGE_B_REPORT §7). Stamp `timing_class=SUPERVISION_ORACLE`.

## 5. What is ALREADY validated (do NOT re-derive)
- **Sampling design** (OBSERVED mask bits, 96 seeds × 3 grids): inclusion ≈0.25; spatially Bernoulli-independent (max adjacent-cov 0.00032); per-seed std = binomial √(p(1−p)/n). → CRN/iid design confirmed on real masks.
- **Variance identity** `V_HWCS=(1−α)V_G4+α·V_G16` (MC, <0.6% err, all regimes). Ordering is scene-dependent (s2−s1 sign flips) — confirm on REAL gradients, don't assume.
- **Cost model** (DERIVED ρ proxy): GAM-invariant; HWCS beats N1 at κ=1 (18/18) but **loses to N2 at κ=1 (0/18)**; beats N2 only if real N2 overhead o_N2 > 1.14–1.29× (unmeasured).

## 6. Decision rules (per camera, paired)
- Compute `d_1(s)=NMSE_HWCS1(s)−[(2/3)NMSE_G4(s)+(1/3)NMSE_G16(s)]` (α=1/3) and `d_2` (α=1/9) per seed.
- Per-camera paired bootstrap 95% CI (resample paired seed vectors). Pool across cameras only if labeled+stratified.
- Quality gate: `NMSE_G16 − NMSE_HWCS1` with paired CI + effect size. **A single-seed deviation is NOT a formula failure.**
- Insufficient power → `ORACLE_INCONCLUSIVE` (never auto-DROP).

## 7. Deliverables to produce on MX
1. `stage_b_raw_results.csv` (per-sample rows, schema §7).
2. `stage_b_variance_identity.csv` (per-seed d_1,d_2 + paired CI per camera).
3. `stage_b_n1_n2_counts.csv` (real projected counts; label COUNTS/PROXY vs NOT_MEASURED emitted_intersections).
4. `nvidia-smi` before/after snapshot.

## 8. Verdict vocabulary (ONLY these)
`ORACLE_PASS` · `ORACLE_INCONCLUSIVE` · `MODEL_OR_IMPLEMENTATION_MISMATCH` · `STRUCTURAL_POTENTIAL` · `TIMING_NOT_MEASURED`.
**`STRUCTURAL_POTENTIAL ≠ CODEX_GO`.** A CUDA prototype needs INDEPENDENT design + performance thresholds: measured GPU kernel/F+B time vs BOTH N1 and N2 on matched p, gradient-quality vs G16, E2E time-to-quality target. **No count-based κ threshold (e.g. κ≤0.5) is a valid CUDA GO gate.**

## 9. Fallback if MX stays unreachable
If no idle MX A100 is available, the Stage B verdict is `TIMING_NOT_MEASURED` and the structural verdict remains `STRUCTURAL_POTENTIAL_WEAK_N2_CONDITIONAL` (N1-conditional, κ-conditional, proxy-only). Do NOT fabricate GPU results. Do NOT reopen the prior WV0 `BLOCKED_RAW_EVIDENCE` verdict; no old GO is restored.
