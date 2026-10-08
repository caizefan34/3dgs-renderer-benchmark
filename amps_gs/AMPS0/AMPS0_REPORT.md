# AMPS0 Report — Attribute-aware Mixed-Precision Optimizer States for Gaussian Splatting

**Generated:** 2026-10-05
**Lane:** AMPS0 (independent; does not modify CGT/DPS0/IGF/RetireGS/ClockGS/Sparse-tail K50-B1 verdicts)
**Working name:** AMPS-GS = Attribute-aware Mixed-Precision Optimizer States for Gaussian Splatting
**Strong baseline:** Faster-GS fused (`FasterGSFused` @ `ae665af`, NeRFICG @ `e458ac6`), A100/EPIC-05
**Local host:** RTX 5070 Laptop 8GB (numerics only; ALL local timing non-authoritative)
**Terminal verdict:** `DROP_INSUFFICIENT_OPTIMIZER_HEADROOM` (P0 headroom gate; P1 numerics independently corroborate `DROP_STATE_PRECISION_NUMERICS`)

---

## 1. OBSERVED (directly measured / read from artifacts)

- **Optimizer fraction of iteration time (authoritative A100):**
  - Full-Adam (gsplat continuation, room@15k post-densification, N=933590): **5.94%** (`M0_BASELINE_PROFILE`).
  - SH-Adam wall fraction (LAVOS0, all 3 scenes): **14%** (`LAVOS0_HEADROOM`).
  - Both are far below the 0.20 headroom floor (objective section 5).
- **Exact analytical traffic model (host-independent):** 59 scalars/Gaussian; fused Adam = 28 bytes/scalar (FP32 m/v), 20 bytes/scalar (BF16 m/v). BF16 on SH-rest (45 scalars) cuts optimizer HBM traffic **21.8%**; BF16 on ALL (C4) cuts **28.6%** (`AMPS0_TRAFFIC_MODEL.json`).
- **E2E speedup ceiling (memory-bound-optimizer assumption):** at measured p, C1–C3 ≈ 1.031–1.034x, C4 ≈ 1.042x — **all below the 1.05x traffic gate** (`AMPS0_TRAFFIC_MODEL.json`).
- **Prior speed bound for the strictly stronger visible-skip optimizer variant:** best-case **1.006x** (`M0_SPEED_BOUND`).
- **P1 shadow numerics (real gradient stream, pure-PyTorch oracle, params frozen at FP32):**
  - BF16 preserves update **direction**: cosine median 0.99998, sign-disagreement 0% (all 3 scenes).
  - BF16 fails update **magnitude** tolerance: relative-L2 median ≈ 1.94–1.98e-2, p95 ≈ 3.8e-2 — **exceeds the 5e-3 / 2e-3 eligibility gates** (`AMPS0_SHADOW_NUMERICS*.json`, `AMPS0_CROSS_SCENE_NUMERICS.json`).
  - Consistent across room (497 steps), garden (508), bicycle (509).

## 2. EVIDENCE (traceable artifacts)

| Claim | Artifact | Source |
|---|---|---|
| f_opt = 0.059 (full-Adam) | `AMPS0_HEADROOM_PROFILE.json` | `dea_gs_m0/out/M0_BASELINE_PROFILE.json` (A100) |
| f_opt = 0.14 (SH-Adam, 3 scenes) | `AMPS0_HEADROOM_PROFILE.json` | `a100_out/synthetic/LAVOS0_HEADROOM.json` (A100) |
| Traffic model + E2E ceiling | `AMPS0_TRAFFIC_MODEL.json` | analytical (this lane) |
| Visible-skip speed bound 1.006x | `AMPS0_HEADROOM_PROFILE.json` | `dea_gs_m0/out/M0_SPEED_BOUND.json` |
| P1 numerics (room/garden/bicycle) | `AMPS0_SHADOW_NUMERICS*.json` | pure-PyTorch oracle (this lane) |
| Cross-scene consistency | `AMPS0_CROSS_SCENE_NUMERICS.json` | aggregated |
| Strong baseline freeze | `AMPS0_BASELINE_AUDIT.json` | `fgs_strong_baseline_final/` |
| Prior art | `AMPS0_PRIOR_ART_AUDIT.md` | internal knowledge + local repo |

## 3. INFERENCE (reasoned from evidence)

- The optimizer is a **minor** fraction of iteration time on the strong fused baseline (6–14%), not the "substantial" cost the AMPS0 hypothesis presumes.
- Because the optimizer is small AND a streaming kernel, shrinking its bytes yields negligible E2E speedup; the ceiling (≤1.04x) is below the gate even before conversion overhead.
- BF16 moment storage introduces ~2% per-step magnitude error that **compounds** over hundreds of Adam updates (bias-corrected m/√v amplifies v-rounding), violating the frozen numerical tolerance even though direction is preserved.
- AMPS0 is **strictly weaker** than the already-analyzed visible-skip variant (1.006x best case), so it cannot exceed that ceiling.

## 4. HYPOTHESIS (untested, not claimed)

- A memory-capacity-bound regime (e.g., bicycle/garden at ~4.7M Gaussians near A100 limits) might make optimizer-state bytes a binding constraint — but that is a **capacity** claim, not a speed claim, and is out of AMPS0 scope.
- A fused-backend that keeps m/v in registers (never round-trips HBM) would nullify any state-precision benefit — plausible but NOT measured (no local Faster-GS fused build).

## 5. NOT_MEASURED (explicitly out of scope or blocked)

- **Authoritative A100 timing** of any AMPS0 candidate (no local A100; Faster-GS fused CUDA backend not buildable on RTX 5070 — no MSVC, wrong toolchain). All local work marked `performance_evidence_valid=false`.
- **Faster-GS fused per-component breakdown** (forward/backward/optimizer/loss split on the fused backend) — only total wall-time exists in `fgs_strong_baseline_final/`.
- **P2 optimizer microbench, candidate freeze, lifecycle tests, P3A/P3B training** — NOT_REACHED_FAIL_CLOSED (see stub artifacts).
- **FP8/INT8 states, stochastic rounding** — excluded by objective section 8/27.
- **Trajectory divergence** — P1 isolates state-storage error with params frozen; no trajectory claim is made or needed.

## 6. CONCLUSION

**AMPS0 is DROP.** It fails two independent, fail-closed gates:

1. **P0 headroom gate (primary):** measured optimizer fraction 5.94%–14% is below the 0.20 floor on all scenes; even the most aggressive candidate (C4, 28.6% traffic cut) yields ≤1.042x E2E at measured p, below the 1.05x gate. The central premise — that modern 3DGS training cost has shifted "substantially" into Adam — is **not supported** on the strong baseline.
2. **P1 numerical eligibility gate (secondary):** BF16 moment storage produces ~2% relative update-magnitude error (exceeding the 5e-3 gate) while preserving direction, consistently across all 3 scenes. The mechanism is numerically non-viable within the frozen tolerance **independent of headroom**.

**Novelty status:** `NOVELTY_NOT_ESTABLISHED`. The prior-art audit classifies the primary mechanism (geometry-FP32 + appearance-BF16 Adam states fused into 3DGS) as `NO_DIRECT_GS_MATCH_FOUND`, but with `PARTIAL_OVERLAP` from generic 8-bit/block-wise optimizers and GS parameter-compression families. A fresh literature audit is mandatory before any paper claim (section 26); arxiv export API was network-blocked in this environment, so the audit is from internal knowledge + local repo and is honestly marked as such.

**What would reopen the lane:** (a) a measured strong-baseline (Faster-GS fused) optimizer fraction ≥0.30 on ≥2 scenes; AND/OR (b) evidence the fused backend materializes optimizer state in HBM per-iteration with measurable bandwidth that BF16 would halve; AND/OR (c) a precision format that meets the 5e-3 numerical gate (BF16 does not). Absent these, AMPS0 should not be pursued further and must not be rescued by changing the frozen policy.

---

## Reproducibility

- Pure-PyTorch oracle gradient stream: `amps_gs/AMPS0/p1_shadow.py` (scene-parameterized, cached streams in `amps_gs/AMPS0/raw/grad_stream_*.pt`).
- Traffic model: `amps_gs/AMPS0/traffic_model.py`. Headroom: `amps_gs/AMPS0/headroom_profile.py`. Baseline: `amps_gs/AMPS0/baseline_audit.py`. Cross-scene: `amps_gs/AMPS0/cross_scene.py`. Memory: `amps_gs/AMPS0/memory.py`.
- All numbers trace to `raw/` evidence and the cited A100 artifacts. All local timing is non-authoritative.
