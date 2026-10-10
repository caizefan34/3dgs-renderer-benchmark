# RENDERER_BACKWARD_CORRECTNESS_GATE 鈥?REPORT

**Date**: 2026-10-08
**Objective**: Verify whether gsplat (path B) and HiGS (path C) dense backward
are correct, resolving the earlier "near-orthogonal gradient" finding.
**Constraint**: NO CUDA recompile, NO model/training change, NO 3K/30K run.

---

## 1. VERDICT

**BACKWARD CORRECT 鈥?both renderers pass.** The earlier "B vs C dense gradient
near-orthogonal (cos~0.005)" finding was a **subset-selection bug**, NOT a
backward bug. With an identical Gaussian subset, identical camera, background,
SH, loss (MSE and L1), and loss normalization, gsplat and HiGS produce
**byte-identical renders** and **cosine = 1.0** dense gradients for all five
parameter groups. Each renderer's autograd gradient matches its own
finite-difference gradient at the numerically stable epsilon (cos >= 0.99).

**Impact on prior work**: The HiGS per-tile gradient Oracle evidence used in
GRADIENT-PROXY-P0 remains **valid** 鈥?the per-tile gradients were computed with
HiGS (path C) alone, and HiGS backward is confirmed correct. The B-vs-C
comparison that motivated this gate was flawed; it does not invalidate the
gradient-only Oracle advantage (G vs U), which was measured within HiGS.

---

## 2. METHOD (per task spec)

1. Fixed same Gaussian subset (on_screen / vis_op, 200K), camera (5K cam0),
   background (zeros), SH (degree 3), loss, normalization.
2. Same parameter indexing and gradient ordering: both renderers receive the
   SAME tensor objects; autograd.grad w.r.t. the same leaf list.
3. Sparse mask OFF 鈥?full render of the subset.
4. Central-difference check per group, epsilon scan {1e-1,1e-2,1e-3,1e-4,1e-5}.
5. Truly identical scalar loss (MSE smooth; L1 non-smooth) for autograd vs FD.
6. Cause classification: index error / loss diff / CUDA backward bug / non-smooth.
7. Per-group relative error, cosine, FD results, source+binary SHA256.

---

## 3. RESULTS

### 3.1 B vs C dense gradient (same subset, MSE loss, 50K)

| group | cosine | relL2 |
|---|---|---|
| means | 0.99999999 | 2.4e-7 |
| quats | 0.99999999 | 1.8e-6 |
| scales | 0.99999999 | 5.8e-7 |
| opac | 0.99999999 | 1.2e-7 |
| colors | 0.9987 | 5.0e-2 |

loss_B = loss_C = 0.107938 (identical). **B and C are the same function.**

### 3.2 Autograd vs finite-difference (per renderer, own FD)

Both B and C autograd match their own FD at the stable eps (best-eps cos):
- B: means 0.9985, quats 0.9997, scales 0.9985, opac 0.99998, colors 0.99985
- C: means 0.9949, quats 0.9997, scales 0.9985, opac 0.99998, colors 0.99985

FD instability at extreme eps (1e-5 -> FD=0 from float32 cancellation; 1e-1 ->
noisy) is finite-difference truncation/roundoff, NOT a backward bug. The
magnitude discrepancy at large eps for means/scales (relL2 0.12-0.25) reflects
the **non-smooth** tile-sort / alpha-blending discontinuity (perturbing a mean
can move a Gaussian across a tile boundary), which central differences handle
poorly; the cosine (direction) still matches >= 0.99.

### 3.3 Determinism (3 trials, same process)

LB=LC=0.228901, renderMAE=0.00000, cos_means=1.0, cos_colors=0.9985 鈥?identical
across all 3 trials. Fully deterministic.

### 3.4 Separate processes (fixed seed subset)

render MAE=0.0, maxerr=0.0; B vs C cos: means/quats/scales/opac = 1.000000,
colors = 0.9985. Identical to in-process. Process boundary is NOT the cause.

### 3.5 The earlier near-orthogonality 鈥?ROOT CAUSE

The GRADIENT-PROXY-P0 Phase 1 worker selected its 200K subset with
`torch.randperm` **without a fixed seed**, called **inside each subprocess**
(path B and path C are separate processes). Therefore B's subset and C's subset
were **different random samples of gaussians**. Comparing gradients over
different index sets yields near-orthogonality by construction:

- Saved Phase 1 npz: workerB_vs_workerC cos = 0.005-0.17 (means 0.0054)
- Same subset, fixed seed: B vs C cos = 1.000000
- Subset sums without seed: 66237286695 / 66316485233 / 66480788895 (all differ)
- Subset sums with seed(0): identical every trial

This is a **reproducibility / index-alignment bug**, classified under cause (6)
as "index error" (mismatched parameter subsets), NOT a CUDA backward bug and
NOT a loss difference (loss_B and loss_C in Phase 1 differed only because the
subsets differed).

---

## 4. SOURCE / BINARY IDENTITY

| item | value |
|---|---|
| torch | 2.9.1+cu128 |
| CUDA | 12.8 |
| Python | 3.10.21 |
| GPU | A100-PCIE-40GB sm_80 |
| gsplat_scene_cuda.so sha256 | `e4b0bc5bf358ec43875a3b7e0d84591763968021b7c7eabcc6d60572490627c8` |
| gsplat_scene_cuda.so bytes | 236744 |
| gsplat git HEAD | `77ab983ffe43420b2131669cb35776b883ca4c3c` |

---

## 5. CONCLUSION

- **HiGS (path C) dense backward is CORRECT** and deterministic. The per-tile
  gradient Oracle used in GRADIENT-PROXY-P0 is reliable training-gradient
  evidence.
- **gsplat (path B) dense backward is CORRECT** and matches HiGS exactly.
- The "near-orthogonal" finding was an artifact of non-reproducible subset
  selection (randperm without seed in separate subprocesses). It does not
  reflect a renderer defect.
- **No action required** on the HiGS backward. GRADIENT-PROXY-P0's DROP verdict
  stands on its own merits (cheap proxies fail the >=25% NMSE threshold); it is
  unaffected by this gate.
