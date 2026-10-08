# 3DGS Moonshot Parallel P0 Portfolio — Execution Summary

**Date:** 2026-09-30  **Execution host:** EPIC-05 A100 (4x NVIDIA A100-SXM4-80GB)  **Operator:** frozen higs_c0 renderer (gsplat 1.4.0, torch 2.9.1+cu128)

Three independent cheap-falsification tracks were run in parallel from the frozen packet. Each
produces a PROMOTE / DROP / BLOCKED verdict + JSON deliverables + report. Stop discipline honored:
no threshold tuning, no full training before P0, no novelty claim without a literature audit.

## Headline: all three tracks DROP

| Track | Verdict | Decisive gate | One-line reason |
|---|---|---|---|
| A — PSC-GS (Projection-Space Consensus) | **DROP** | S_upper_8 < 1.30 (1.035); P_hold_pos_B8 <= 0.55 (0.513) | Projected-state updates give ~4% amortization and do not hold a consistent shared 3D state. |
| B — ReplayGS (radiance-operator replay) | **DROP** | B2 rho <= 0.30 (observed 31-76) | Replay is EXACT (cosine=1.0) but 30-250x MORE expensive than the native fused SH kernel. |
| C — Profiled-SH / VarPro-GS | **DROP** | gate2 ratio >= 1.5x in >=3/4 cells (observed 0/4) | L-BFGS does not beat SH-Adam on held-out loss; appearance inner problem overfits 2 train cams. |

## Track A — PSC-GS: DROP (was BLOCKED)
- Prior state: BLOCKED, `EXECUTION_HOST_MISMATCH` (reprobe ran on local WSL RTX 5070; frozen /root artifacts absent).
- Resolution: re-probed on EPIC-05 (the intended A100 host) as root — ALL frozen identities match
  byte-for-byte (checkpoint b31e5a6b, C0 ext d26cccf4, core gsplat_cuda.so 935ea6c0, trainer 675b0267).
- The frozen probe had never executed on the correct host; 4 minimal bug fixes made it runnable
  (contiguous projection inputs; NaN-safe conic pinv; conic->Sigma pinv; 3 einsum dim fixes). The
  gates/analyzer were NOT altered.
- Result: S_upper_8=1.035 (<1.30) and P_hold_pos_B8=0.513 (<=0.55) => DROP. H1 supported (3D consistency required).
- Details: trackA/TRACK_A_SUMMARY.md, trackA/PSC_P0_REPORT.md.

## Track B — ReplayGS: DROP
- B1 fidelity: EXACT in all 4 cells (pooled SH-grad cosine=1.000000, rel_l2 ~1e-5). The frozen-geometry
  radiance-operator algebra is validated.
- B2 cost: rho = T_replay/T_full = 31.19-76.22 across cells (gate <=0.30). Profiling proves the work is
  compute-bound on 229M-290M per-hit SH shading; the PyTorch replay (~1.0-1.3 s) cannot beat the native
  fused kernel (15-46 ms). B5 cache (9-12 GB) fits the A100. Failure is fundamental/fair.
- B3 schedule: mixed — room_25000 replay schedule is ~2.7-2.8x more effective per step than full steps
  (U_4=2.85, U_8=2.67, PSNR-safe); room_20000 U_8=1.56 but PSNR regresses; bicycle U undefined (F1<=0 overfit).
  Moot because B2 fails.
- Details: trackB/REPLAYGS_REPORT.md.

## Track C — Profiled-SH / VarPro-GS: DROP
- Reuses Track B's EXACT operator (does not assume its performance result). Operator reproduces native
  SH gradients (cosine=1.0).
- C0 inner oracle (L-BFGS vs SH-Adam, 4 cycles, 2 inner cams, held-out loss): ratio L-BFGS/Adam =
  1.091 / 0.164 / nan / nan. 0/4 cells reach the 1.5x gate. bicycle Adam itself degrades held-out loss
  (overfit to 2 inner cams).
- Gate 2 fails decisively => DROP. Gate 3 (outer oracle) not run (cannot rescue a PROMOTE).
- Details: trackC/PROFILEDSH_REPORT.md.

## Cross-track observations
- All three tracks share the SAME frozen room@25K / bicycle checkpoints and the EXACT higs_c0 renderer.
- Track B and Track C both confirm the SH appearance subproblem is mathematically exact/replayable
  (cosine=1.0), but neither yields a usable speedup/optimizer win under the native fused kernel.
- The recurring blocker is the native fused CUDA SH-shading kernel (15-46 ms), which any PyTorch
  replay/second-order inner loop cannot beat for per-hit SH evaluation at 10^8-hit scale.

## Novelty / crowding (post-hoc, all DROP => no claims made)
No track PROMOTEd, so no novelty claim is made. For the record, the candidate narrow boundaries
(PSC projected-state sync; ReplayGS frozen-geometry radiance operator; Profiled-SH inner appearance
problem) all overlap crowded prior art, notably "Instant Colorization of Gaussian Splats (2026)"
(visibility-weighted LS on fixed splats) and the generic second-order/VarPro/pruning directions that
NOVELTY_CROWDING_GUARD.md flags as NOT defensible.

## Stop discipline compliance
- No threshold tuned after observing results.
- No full training run; P0 falsification probes only.
- No novelty claim without the literature audit (audit noted above; all DROP).
- Frozen gates (Track A analyzer; Track B/C pre-registered gates) applied unchanged.

## Reproducibility
- Track B/C code + logs: moonshot_p0/trackB/, moonshot_p0/trackC/ (local + EPIC-05 /root/epic05_moonshot/).
- Track A frozen packet + bug-fixed probe: moonshot_p0/trackA/ (local + EPIC-05 /root/3dgs-renderer-benchmark/PSC_P0_RUN/).
- Environment: EPIC-05 A100, /root/miniforge3/envs/gsplat/bin/python, CUDA 12.8.
