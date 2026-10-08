# SecantGS S0 — Temporal Update-Subspace ORACLE — Report

**Date:** 2026-09-30  |  **Campaign:** SECANTGS-S0  |  **Prereg:** `_packet_sg0/S0_PREREG.md` (FROZEN)
**Execution site:** Remote Azure A100-80GB (host `dsw-933622-64cc845c7d-lqswc`, EPIC-05). **No local GPU used** (per goal constraint).

## Verdict

**`S0_DROP`** (frozen analyzer `s0_analyze.py`). No single fixed m∈{2,4} passes all four cells.
This closes the **temporal-span secant mechanism** at the oracle upper bound: even with the *true* next update revealed to fit coefficients, representing it in the span of the previous 2–4 executed Adam updates does **not** recover native next-step loss progress.

## Question

For a topology-frozen 3DGS trajectory, does the actual next Adam update lie in a useful low-dimensional temporal subspace formed by the previous 2–4 executed Adam updates? S0 is an ORACLE span test: the true next update is used only to fit oracle coefficients (an upper bound on any deployable secant method).

## Frozen identity (fail-closed gate: PASS)

| item | value | prereg | match |
|------|-------|--------|-------|
| C0 extension `.so` | `d26cccf46c79…511` | `d26cccf…` | OK |
| C0 core gsplat `.so` | `935ea6c0dd04…37e` | `935ea6c0…` | OK |
| loss | 0.8*L1+0.2*(1-SSIM), fused-ssim pad=same, 1080p | same | OK |
| optimizer | native Adam, per-group lr/eps/betas from frozen ckpt | same | OK |

Checkpoints `room@{10000,15000}`, `bicycle@{10000,15000}` from `/root/epic05_n1/runs/n1snap_c0f_s42_*/` (hashes in `S0_SOURCE_IDENTITY.json`).

## Design

- **Cells:** room@10K/15K, bicycle@10K/15K. N = 710,652 / 841,781 / 2,960,907 / 3,654,566.
- **Cameras (corrected true-holdout, CVT0 §3-4 authority):** TRUE HOLDOUT = {i: i%8==0}; TRAIN POOL = rest; disjoint. 8 train + 8 holdout, midpoint-stratified per pool.
  - room train `[20,59,98,137,175,214,253,292]`, holdout `[16,56,96,136,168,208,248,288]`
  - bicycle train `[12,36,60,84,109,133,157,181]`, holdout `[8,32,56,80,112,136,160,184]`
- **Rollouts:** 4 per cell (offsets 0/2/4/6 cyclic over the 8 train cameras), K=16 native Adam steps, identical deep-copied ckpt+opt, topology frozen. Per step: source loss before/after, raw executed delta_t, peer-train + true-holdout forward-only metrics.
- **Inner product (optimizer-unit, no whitening):** `<u,v>_W = Σ_g <u_g/lr_g, v_g/lr_g>`, lr_g frozen per group at start ckpt.
- **Oracle:** m∈{2,4}; basis = previous-m deltas, target = delta_t; ridge `λ=1e-8·trace(G)/m`; probe valid iff trace(G)>0 and all projected metrics finite.
- **Probe counts:** m=4 → 48/cell (t=5..16 × 4 rollouts), m=2 → 56/cell (t=3..16). All valid (invalid_fraction 0.0).

## Results (medians / p10 across valid probes)

| cell | m | E_span | P_src | peer_win | thold ΔPSNR med/p10 |
|------|---|--------|-------|----------|---------------------|
| room@10000 | 2 | 0.820 / 0.694 | **-0.026 / -0.186** | 0.000 | -0.010 / -0.032 |
| room@10000 | 4 | 0.838 / 0.712 | **-0.025 / -0.195** | 0.000 | -0.010 / -0.032 |
| room@15000 | 2 | 0.848 / 0.723 | **-0.036 / -0.120** | 0.036 | -0.005 / -0.025 |
| room@15000 | 4 | 0.855 / 0.737 | **-0.032 / -0.105** | 0.000 | -0.004 / -0.023 |
| bicycle@10000 | 2 | 0.857 / 0.719 | **+0.067 / -0.093** | 0.000 | -0.002 / -0.016 |
| bicycle@10000 | 4 | 0.870 / 0.762 | **+0.093 / -0.081** | 0.000 | -0.002 / -0.011 |
| bicycle@15000 | 2 | 0.873 / 0.757 | **+0.071 / -0.063** | 0.000 | +0.003 / -0.001 |
| bicycle@15000 | 4 | 0.876 / 0.765 | **+0.121 / -0.046** | 0.000 | +0.003 / -0.000 |

**Gates (S0_PROMOTE):** median P_src ≥ 0.70, p10 P_src ≥ 0.40, peer_win ≥ 0.35. **All cells fail P_src and peer_win** despite E_span ≫ 0.65. → DROP.

## Interpretation (the decisive dissociation)

- **Parameter-space span energy is high** (E_span 0.82–0.88, uniform across xyz/scaling/rotation/opacity/shs): the next update's *W-norm* is largely contained in the span of the previous few updates.
- **But it does not convert to loss progress** (P_src ≈ 0 or negative; peer_win ≈ 0). The oracle-projected state leaves source/peer/holdout loss essentially unchanged (e.g. room@10000 off0 t=3: L_prev=0.04645, L_proj=0.04645, L_native=0.04333).
- **Coefficients** are dominated by the *immediately preceding* delta (≈0.77–0.89), near-zero on older ones — consecutive Adam updates are variance-normalized to similar parameter-space magnitude, so the span looks energetic, but the loss-relevant content is camera-specific and does not carry forward.
- **Held-out PSNR** is near-neutral (median −0.01…+0.003 dB) — the projected state is not *worse* on holdout, it simply buys nothing.

This is exactly the failure mode S0 was built to expose (prereg §0: "The decisive test is not parameter cosine alone"): high subspace energy ≠ loss-progress capture.

## Timing (descriptive, not a speedup)

Per-cell native step ≈ 30–94 ms; forward-only validation-like pass ≈ 4–24 ms (room faster than bicycle at 1080p). Amdahl ceiling inputs recorded in `S0_TIMING_CEILING_*.json`; T(q)=T_full+(q-1)·T_fwd. No realized speedup claimed.

## Forbidden-rescue compliance (prereg §12)

No m>4, no per-scene m, no L-BFGS reinterpretation, no learned predictor, no sketch tuning before oracle pass, no camera-sequence changes, no full training. DROP closes this mechanism. Analyzer owns the verdict; S0 PROMOTE would have permitted — but did not trigger — S1.

## Artifacts (`sg0_run_s0/`)

- `out/`: `S0_SUMMARY.json`, `S0_VERDICT.json`, `S0_SOURCE_IDENTITY.json`, `S0_ENVIRONMENT.json`, `S0_CAMERA_SEMANTICS.json`, `S0_ROLLOUTS.json`, `S0_MANIFEST.json`
- `cells/`: per-cell `S0_CELL_*`, `S0_ORACLE_PROJECTIONS_*`, `S0_UPDATE_GRAMS_*`, `S0_LOSS_PROBES_*`, `S0_TRUE_HOLDOUT_*`, `S0_TIMING_CEILING_*`
- `src/`: `s0_lib.py`, `s0_cell.py`, `s0_finalize.py`, `s0_analyze.py` (frozen packet analyzer), launch scripts
- `logs/`: per-cell run logs; `S0_PLAN.md`: execution plan
