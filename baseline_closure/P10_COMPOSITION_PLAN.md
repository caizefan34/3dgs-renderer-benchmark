# P10_COMPOSITION_PLAN — H8-MR on current gsplat main (port + correctness only)

**Status: PREPARED, NOT STARTED.** This is the EPIC-05 P6 gate restated as the P10 first-choice
composition: *"H8 generality validation in pristine gsplat / isolated drop-in path; no C0
production changes; required output: gradient equivalence + 3-scene microbench"*
(`epic05_publication_closure_extract/README.md`, `DSH_EPIC05_EXECUTION.md` Job 8,
`epic05_manifest.json` `"P6_H8_GENERALITY"`).

## Hard rules (non-negotiable)

1. **Do not change H8 mathematics.** The six moments and the reconstruction formulas must be
   byte-for-byte the same algebra as the frozen production patch
   `patches/higs-h8-mr.patch` (SHA-256 `8fb23acca9841ae4d24035de9b2d0ba7725742186b7db149c44cc656415fe312`,
   spec: `reports/higs/h8-mr-production.md`). The port adapts *where* the math runs, never *what* it computes.
2. **Do not touch C0 / A0 / B1A** frozen binaries, worktrees, or build inputs.
3. **No long training.** No 30K convergence run. Deliverables are correctness + 3-scene
   microbench (+ optionally a 200-iteration trainability smoke, allowed because the
   transformation does not change training semantics).
4. **GPU work is gated on P7 completion** (same launch gate as P8/P9). All P10 GPU steps are
   short, but they still wait for P7 and clean-GPU verification.
5. **No new optimization branches.** If the port shows an *additional* optimization idea, it is
   recorded as a hypothesis and stopped there.

## Composition target

| component | pin |
|---|---|
| host library | gsplat main `512d366b67073d77ca099ede742683c165dfc23b` (v1.6.0; local clone `public_baselines/gsplat`) |
| ported math | H8-MR (moment-space / opacity-absorbed moment adjoint) per `patches/higs-h8-mr.patch` |
| port scope | the dense 3DGS backward path only: `gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh` (blend backward), the serial-batch backward kernel's scatter site, and the EWA projection backward (`ProjectionEWA3DGS*.cu`). Sparse/CSR, 2DGS, lidar, from-world paths are OUT OF SCOPE. |

## Why the mapping is natural (verified against v1.6.0 source)

The v1.6.0 per-(gaussian, pixel) backward (`rasterize_to_pixels_3dgs_blend_bwd`,
`RasterizeToPixels3DGSDevice.cuh:104-173`) computes per intersection, with
`v_sigma = -opac*vis*v_alpha`, `delta = (dx, dy)`, conic `(A, B, C)`:

```
v_conic = (0.5*v_sigma*dx*dx, v_sigma*dx*dy, 0.5*v_sigma*dy*dy)   // 3 atomics after warp reduce
v_xy    = (v_sigma*(A*dx + B*dy), v_sigma*(B*dx + C*dy))          // 2 atomics
v_opac  = vis*v_alpha                                            // 1 atomic
```

H8-MR replaces the per-intersection geometry reconstruction with per-Gaussian moments
(summed over that Gaussian's pixel intersections):

```
Sx = Σ v_sigma*dx      Sy = Σ v_sigma*dy      R0 = Σ vis*v_alpha
Sxx = Σ v_sigma*dx*dx  Sxy = Σ v_sigma*dx*dy  Syy = Σ v_sigma*dy*dy
```

and reconstructs ONCE per Gaussian in the projection VJP (forward conics available there —
the EWA projection backward recomputes cov2d/conic from means/scales/quats):

```
v_xy  = (A*Sx + B*Sy, B*Sx + C*Sy)
v_conic = (0.5*Sxx, Sxy, 0.5*Syy)
v_opac = R0
```

This is the identical algebra to `h8-0-moment-adjoint-gate.md` / `h8-0r` / `h8-mr-production.md`
(vx = A·Sx + B·Sy, vy = B·Sx + C·Sy, vA = 0.5·Sxx, vB = Sxy, vC = 0.5·Syy, v_opacity = R0).

Memory: the 5 moment floats (Sx, Sy, Sxx, Sxy, Syy) replace the 5 scattered geometry floats
(v_conic 3 + v_xy 2) — **no new global buffers**; `v_opac`/`R0` keeps its existing scalar slot.
Per-intersection arithmetic drops the 4-FMUL/2-FADD conic·delta reconstruction in favor of
5 moment increments, with the 6-FMUL/2-FADD reconstruction amortized by intersection
multiplicity (~28–42× in the HiGS measurements).

## Porting steps (analysis first, then code, then gates)

1. **Mapping analysis (no GPU).** Enumerate exact insertion points in v1.6.0:
   the blend-backward signature (`*_local` outputs → moment outputs), the warp-reduce +
   atomic-scatter site in `RasterizeToPixels3DGSSerialBatchBwd.cu` (and any other dense-path
   launchers that consume `rasterize_to_pixels_3dgs_blend_bwd`), the `dL_dconic`/`dL_dmean2d`
   buffer hand-off into `ProjectionEWA3DGSFused.cu`, and the env-dispatch pattern
   (`GSPLAT_H8_MR=1`, default OFF, so one binary serves both arms — the HiGS env-switch
   convention). Conditions that must be preserved verbatim: the `opac*vis <= MAX_ALPHA`
   guard (clamped-alpha intersections contribute no geometry gradient), saturation skips,
   background-term handling, and `compute_abs`.
2. **absgrad note.** `v_xy_abs` (absgrad) CANNOT be reconstructed from signed moments
   (|A·Sx+B·Sy| ≠ Σ|…|). The abs path keeps its per-intersection accumulation unchanged —
   exactly as in the HiGS production patch, where `HIGS_BWD_ABSGRAD` accumulates per
   intersection while the signed path goes through moments. Only the signed v_conic/v_xy/v_opac
   scatters are replaced.
3. **Implement + build.** Env-gated (`GSPLAT_H8_MR=1|0`), built JIT in the frozen
   higs-13scene-env (torch 2.9.1+cu128, sm_80). Both arms from ONE binary. Record build
   identity (source SHA + extension SHA-256) per the G4 convention.
4. **Correctness gate (short GPU).** ≥15 randomized configs (H8-MR production pattern):
   random means/scales/quats/opacities/SH, random upstream gradient `v_render`; compare ALL
   gradients (means2d, conic→scales/quats via projection VJP, opacities, colors/SH) between
   `GSPLAT_H8_MR=0` and `=1`:
   - forward: bit-identical (forward untouched — assert, don't assume);
   - backward: rel-L2 ≤ 1e-3 per tensor (the frozen H8-MR production gate), sign-consistency
     of nonzero entries, no NaN/Inf.
   FAIL ⇒ the port is wrong or the mapping is unsound; fix or stop. NEVER loosen the gate.
5. **3-scene microbench (short GPU).** room / bicycle / garden, seed-42 camera sequence,
   forward+backward timing at the microbench convention (2048-max-side, per
   `h8-mr-production.md`): report BWD gain, F+B geomean, and the resource profile
   (registers/spills, `resource_usage` convention). Optional 200-iter trainability smoke.
6. **Deliverables.**
   - `patches/h8-mr-gsplat-main-512d366b.patch` (NEW file; the frozen HiGS patch stays untouched as the math reference)
   - `artifacts/p10-h8-mr-gsplat/`: `correctness.json`, `timing.json`, `resource_usage/`, `provenance.json`
   - `reports/p10/p10-h8-mr-gsplat-main.md` in the §30 report format, registered in the
     authoritative registry with the frozen status vocabulary
   - verdict string pre-registered: `P10_H8_MR_GSPLAT_GENERALITY_{PASS|FAIL}` + a wall-gain
     number; NO claim beyond generality (this is not a new headline, not a C0 change, and
     per EPIC-05 rules it must not be used to re-attribute the C0/B1A gap).

## Risks / open questions (resolve in step 1, before coding)

- v1.6.0 restructuring (`gsplat/strategy/`, `torch.library` backend, new launcher split
  Serial/Parallel/Sparse) means the scatter site may exist in more than one dense launcher —
  enumerate all consumers of `rasterize_to_pixels_3dgs_blend_bwd` and port only those on the
  dense classic path; anything else stays env-off.
- The EWA projection backward must have the forward conic available at reconstruction time;
  if it recomputes conic in a different precision than the blend backward used, the
  reconstruction uses the projection kernel's own conic (same source values) — verify
  equivalence in the correctness gate rather than assuming.
- `packed=True` paths and `RasterizeToPixelsFromWorld3DGS*` are out of scope and remain env-off.
- If the correctness gate cannot be met with unmodified H8 algebra, the outcome is
  `P10_H8_MR_GSPLAT_GENERALITY_FAIL` with the divergence documented — that is a valid,
  reportable result. Redesigning H8 to pass is prohibited.
