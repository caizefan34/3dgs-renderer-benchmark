# C51-CD0-MX P0 Report — Compute-Densification Co-Design Scientific Gate

## Verdict
`P0_NO_NOVELTY` — the proposed co-design mechanism is not novel and its primary lever is already falsified.
No P1 GPU work authorized.

## 1. OBSERVED (fresh read-only audit on THIS host, no GPU compute)
- Host: `Giannis` — Windows-11-10.0.26200-SP0, Python 3.13.13.
- GPU present but UNUSED for compute: NVIDIA GeForce RTX 5070 Laptop, 8 GB, driver 592.01, CUDA 13.1.
  `nvidia-smi` shows only desktop processes (6% util, 2.7 GB desktop). NOT a clean A100; make no clean-GPU claim.
- Repo: `C:\Users\36570\3dgs-renderer-benchmark` at git HEAD `387a7b31f35fa0b9ab1d160fa12181e1a60c77c2`, branch `r6-exact-backward-screening`.
  Tracked changes: only submodule gitlinks dirty (`public_baselines/gsplat`, `clockgs/third_party/dgr_clock`);
  NO tracked C51 source/report files modified this stage.
- Package integrity: `PACKAGE_SHA256_MANIFEST.json` verified — `evidence/legacy_evidence.json` SHA256 matches manifest exactly (`6e327317…`).
- Strong baselines FastGS/Faster-GS/nerficg: checkouts ABSENT on this host (they live under remote `/mnt/storage_pool/liaoyuanjun/…`). Not re-runnable here; comparison is archival/literature only.
- Paired-runs fail-closed analyzer confirmed: header-only CSV → `FAIL_CLOSED NO_RAW_RUNS` (exit 2). No fabricated pairs.

## 2. ARCHIVAL (prior C51 Stage4A/4B/5, A100 cohort — NOT re-run, NOT merged with this host)
- Stage4A (Room, A100-PCIE-40GB, CUDA 11.8, gsplat 1.5.3, config-buggy 5K): K50-B1 microbench 10.2% at K50; 5K E2E +6.7%; 30K E2E +7.8%; B2 quality-fails (-0.49 dB); B1/B3 quality-preserve. Single scene, separate cohort.
- Stage4B (Room/Bicycle/Garden, A100-80GB, clean 30K): K50-B1 E2E speedup +8.2% / +24.4% / +28.9%, quality preserved/improved. Intrinsic sparse backward (Condition C, densification preserved) = +0.5% / −6.6% / −1.7% — never clears the 5% gate.
- Stage5 (Room 30K, A100-80GB): A=57.25 ms, B=52.90, C=56.96, D=57.48. 94% of K50-B1's +8.2% comes from reduced Gaussian population (densification response), only +0.5% from intrinsic sparse backward. C≈D semantically equivalent. D (full bwd + Python mask) is SLOWER than baseline.

## 3. RECOMPUTED (from frozen archival `legacy_evidence.json`, CPU-only)
Reproduced with the package's own `scripts/recompute_legacy.py` — status `ARCHIVAL_RECOMPUTED_NOT_NEW_MEASUREMENT`:
- Stage4B 30K iter speedup: Room 1.0823x (ΔPSNR +1.030 dB), Bicycle 1.2443x (+0.498), Garden 1.2886x (+0.037).
- Stage5 Room: B/A 1.0822x; C/A 1.0051x (+0.51%); D/A 0.9960x (−0.40%).
- These match the frozen report tables exactly. `total_wall_speedup`, `LPIPS`, `time_to_quality`, `multiseed_CI` all `NOT_MEASURED`.

## 4. Competing hypotheses (Stage 0)
- **H1 — causal benefit from sparse backward at matched population: FALSIFIED.** Condition C (densification preserved) is +0.5%/−6.6%/−1.7% across scenes; the pure CUDA skip saves ~0.5–0.9 ms but mask overhead (~0.7–2.0 ms) consumes it. Stage5 Table 2/6, Stage4B §10.
- **H2 — explicit compute-aware birth budget beats independent decisions: NOVELTY_THREAT/CLOSED_OVERLAP.** Decomposes into (a) skip-backward = H1 (falsified), (b) density/capacity control = FastGS (CVPR2026) multiview-consistent densify/prune, (c) coupled topology/capacity scheduling = repo CLOSED VTS0_DROP + ReBudgetGS (N1_SMALL_BUT_ACTIONABLE) + LAW-OPT-FRAC (optimizer 5.94–14% of wall, Amdahl-bound <1.05x). Per §0C: no non-trivial independent mechanism → `P0_NO_NOVELTY`.

## 5. Prior-art / novelty delta
See `NOVELTY_DELTA_MATRIX.md`. Summary: every component maps to published prior art (FastGS, Faster-GS)
or a CLOSED law (VTS0, ReBudgetGS, VOS, IGF, LAW-OPT-FRAC, LAW-D0). "Joint", "budget", "coupled",
"multi-view" are relabelings, not new mechanisms. The one certified forward lever (TFC footprint
tightening, PAPER0 MAIN_CANDIDATE) is a different axis (forward pipeline), not the backward/optimizer
coupling this track proposes.

## 6. Key answers to the Stage-0E closing questions
1. Old C51 speedup is a **density/population trajectory** effect, not a new algorithm (94% of room +8.2%).
2. K50-B1 has **no new headroom** vs FastGS/Faster-GS — the speedup is fewer Gaussians (covered by FastGS density control); intrinsic backward skip is falsified (+0.5%).
3. There is **no novel & testable** compute-densification coupling mechanism that is non-isomorphic to prior art/CLOSED laws.
4. The minimal falsification that WOULD test a coupling lever is the archival Stage5 A/B/C/D design re-run on the CURRENT strong baseline (Faster-GS fused, A100/EPIC-05) — see `P1_PREREG.md`. Not launched.
5. Under NO-GO: **STOP the C51-CD0 co-design mainline**; first-priority authorized GPU work stays Temporal Coverage / TFC.

## 7. Decision & authorization boundary
- Verdict `P0_NO_NOVELTY`. No GPU training, no CUDA recompile, no 30K, no original-code edits authorized this stage.
- Do not resurrect C51-CD0 without new independent evidence overturning a CLOSED law or a certified new backward/coupling cost model on the current strong baseline.
- This stage consumed no GPU. Temporal Coverage (TFC) remains first-priority authorized GPU work.
