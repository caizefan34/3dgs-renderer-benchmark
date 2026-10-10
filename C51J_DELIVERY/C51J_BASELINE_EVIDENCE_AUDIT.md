# C51J_BASELINE_EVIDENCE_AUDIT.md

**Track:** CVPR27-C51-JOINT-MX — Baseline (1.164×) Evidence Audit
**Generated:** 2026-10-10 (UTC)
**Author role:** Execution Agent (MX/A100, 3DGS training, profiling, acceptance, archival)
**Platform of this audit:** Windows-11 host `Giannis` (RTX 5070 Laptop 8GB) + remote MX `bms-39468022-001` (8× A100-PCIE-40GB)

---

## 0. One-line verdict

The frozen **1.164× wall-clock speedup** for `gsplat_30k_fused_prune10_rclip05` is an **A100 confirmatory-matrix result** supported only by its **aggregate summary + verdict JSON** on this host. The **132-job per-scene / per-seed raw results and its dedicated frozen protocol file are NOT locatable** (neither locally nor on the MX storage pool at audit time). Therefore this 1.164× is recorded as **historical aggregate support**, NOT as independently recomputed 132-job evidence. No 132-job recomputation is claimed.

---

## 1. Platform reality (P0, verified this session)

| Check | Result | Evidence |
|-------|--------|----------|
| This host | Windows-11 `Giannis`, RTX 5070 Laptop 8GB, driver 592.01 | `p0_c51_cd0_20261008/env_snapshot.txt`, `SOURCE_IDENTITY.json` |
| MX reachability | **Reachable via SSH host `mx`** (36.140.146.31:26372, user `liaoyuanjun`) | live SSH probe this session: `hostname → bms-39468022-001`, `nvidia-smi → A100-PCIE-40GB` |
| MX GPUs | **8× A100-PCIE-40GB** (40 GB each) | live `nvidia-smi` |
| MX exclusive GPU | **NOT AVAILABLE** — all 8 A100s occupied by unrelated python jobs (compute apps present on every index) | live `nvidia-smi --query-compute-apps` |
| MX storage `/mnt/storage_pool` | **100% full** (~420 MB free), **94% of inodes used** (821,605 free) | live `df -B1` / `df -i` |
| MX home `/` | 94% used (~28 GB free) | live `df` |
| MX default Python/torch | `torch import failed` in default shell; NVCC release **11.5**; project uses frozen env `/root/miniforge3/envs/gsplat/bin/python` (torch ABI mismatch documented for default python) | live probe + `h0_run/H0_PROGRESS.md` |
| C51J output dir | Created: `/mnt/storage_pool/liaoyuanjun/CVPR27_C51_JOINT_MX/{audit,raw_logs}` | live `mkdir` + `ls -ld` |

**P0 consequence:** Storage-full + no exclusive GPU ⇒ the audit-first branch mandated by the objective is active. No new large-file artifacts and no 30K training can run now. Deliverables are written locally and only lightweight audit copies are staged to MX.

---

## 2. The 1.164× claim — what is actually provable

### 2.1 Verified-locally (authoritative aggregate)

File: `paper/higs/tables/confirmatory-accel15-summary.json`
- Matrix `confirmatory_accel15_11s3`, **132 jobs planned / 132 complete**, 11 scenes × 3 seeds × 4 methods, 33 paired cells.
- Candidate `gsplat_30k_fused_prune10_rclip05` vs `gsplat` (paired, 33 cells):
  - **wall_speedup_ratio mean = 1.16422**, `ci95_lower = 1.03437`, `required_min = 1.1111`, `accel_passed = true`
  - PSNR Δmean **+0.0577 dB** (CI lo −0.0222, hi +0.1505) → `ni_passed=true` (non-inferior)
  - SSIM Δmean **−0.000071** (CI lo −0.00109, hi +0.00101) → `ni_passed=true`
  - LPIPS Δmean **+0.001412** (CI lo **+0.000464**, hi +0.00249) → `ni_passed=true`
  - wall_time Δmean **−71.8 s**; TTQ Δmean **−49.5 s** (CI hi −18.98 → faster, `ni_passed=true`)
  - peak GPU mem Δmean **−23.3 MiB** (n.s.); energy Δmean **−28,663 J**; final Gaussian count Δmean **−1,362,967**
- Verdict file: `artifacts/training-ablation/results/confirmatory-accel15-verdict.json` → `all_gates_pass=true` for the 6 pre-registered gates.

### 2.2 Gate semantics — important nuance (verified against raw JSON)

The non-inferiority gate for LPIPS is defined as `lpips_ci_hi = 0.00249 < +0.005` (degradation ceiling). Because **LPIPS is lower-is-better**, a positive Δmean = +0.0014 means the aggregate LPIPS got marginally *worse*, but stayed inside the pre-registered +0.005 acceptance bound. The gate passes on the **bound**, not on evidence of improvement.

This matches the caveat in `experiments/training_solver_design/FULL_RESOLUTION_PARETO_DIRECTION.md`:
> "预注册的质量不劣门通过，但没有证明三个质量指标共同提高，LPIPS 的聚合均值反而变差。"

**Correction to that design doc's shorthand:** it wrote "LPIPS CI 下界 +0.000464" as the passing figure; the *passing* gate is the **upper** bound +0.00249 (the degradation ceiling). The +0.000464 CI lower bound is the floor of the degradation interval. Both are recorded here so the two documents cannot drift apart.

### 2.3 NOT locatable (so NOT claimed)

Searched local workspace + MX `/mnt/storage_pool/liaoyuanjun` (read-only, `find`):
- **132-job per-run raw JSONs** (per-scene × per-seed PSNR/SSIM/LPIPS/wall/TTQ/VRAM): **absent**. Only the aggregate summary + verdict exist locally; nothing on MX.
- **Dedicated frozen protocol file** for `confirmatory_accel15_11s3`: the closest local artifacts are `benchmark/higs-paper-protocol.json` and `src/higs_paper_protocol.py`, but `FULL_RESOLUTION_PARETO_DIRECTION.md` explicitly states the historical data path is no longer visible and data identity is unverified. **The per-run protocol file itself is not found.**
- **`gsplat_30k_fused_prune10_rclip05` implementation source**: not found by that name on MX or locally. Confirmed by MX-wide grep: `confirmatory_accel15`/`accel15`/`fused_prune` per-run dirs and JSONs are **absent**. The name is an accel-matrix ablation arm label, not a standalone trainer file.

### 2.3.1 What IS locatable on MX (strengthened provenance, verified this round)

Searched MX `/mnt/storage_pool/liaoyuanjun` read-only:

| Item | Status | Location / commit |
|------|--------|-------------------|
| **Rich paper protocol** | FOUND | `tco0_work/benchmark/higs-paper-protocol.json` (has `matrices`, per-method `trainer_cfg`, gsplat commit `77ab983ffe43420b2131669cb35776b883ca4c3c`) |
| Strong-baseline trainers | FOUND | `strong_baselines/fastgs` @ **`d4d33b6f45ed97709f4dd63673d97f6f6584063c`**; `strong_baselines/speedy-splat` @ **`b9dd42d1be29f595282c390871872788ee07718d`**; `strong_baselines/nerficg/scripts/train.py` — each with `train.py` + `full_eval.py` |
| fastgs prune logic | FOUND | `strong_baselines/fastgs/train.py`: `final_prune_fastgs(min_opacity=0.1, pruning_score=...)` at densification interval; opacity-reset interval — this is the naive opacity-pruning control reference |
| **Naive fixed-budget control (Speedy-Splat)** | FOUND | `strong_baselines/speedy-splat/train.py`: `prune(prune_ratio)` scheduled between `prune_from_iter`/`prune_until_iter` via `densify_prune_ratio` / `after_densify_prune_ratio` — a clean simple-fixed-budget pruning control for the P3 `opacity_pruning_control` arm |
| Multiple gsplat eval checkouts | FOUND | `gsplat-clean-mx`, `higs_c0_final30k_worktree`, `higs_p2_1a_worktree`, etc. (`gsplat/cuda/_torch_impl_eval3d.py`) |
| EPIC-05 | NOT PRESENT | no `epic05`/`EPIC05` dir on MX at audit time |

#### 2.3.2 Naive-budget control characterization (verified round 5, commit-pinned)

For the P3 `opacity_pruning_control` arm the objective requires a "simple fixed-budget / opacity-pruning control". Two commit-pinned candidates are locatable and usable:

| Control | Commit | Mechanism | Role |
|---------|--------|-----------|------|
| **Speedy-Splat** | `b9dd42d1be29f595282c390871872788ee07718d` | `prune(prune_ratio)` on a schedule (`prune_from_iter`→`prune_until_iter`, every `prune_interval`), ratios `densify_prune_ratio` / `after_densify_prune_ratio` | **Simple fixed-budget pruning control** (the naive bar the new method must beat) |
| **FastGS** | `d4d33b6f45ed97709f4dd63673d97f6f6584063c` (2026-09-16) | multiview-consistent opacity pruning: `densify_and_prune_fastgs` every `densification_interval` (size_threshold=20 after `opacity_reset_interval`), plus `final_prune_fastgs(min_opacity=0.1, pruning_score=…)` every 3k iters after 15k | **Opacity-pruning control** (error/multiview-guided, a stronger naive control) |

Both `train.py` files are present with their eval counterparts. These give Codex two concrete, SHA-pinned naive controls to run as P4 comparators (native vs simple-budget vs candidate), rather than an invented control. Note: running them still requires the storage+GPU blocker to clear.
| Mip-NeRF360 data | PRESENT | `benchmark_data/mipnerf360/{room,bicycle,garden}` + 8 more scenes |

**Key structural finding:** the paper-protocol `methods` block contains `original_3dgs, gsplat, higs_full, higs_proposed, speedy_splat, turbo_gs` — it does **NOT** contain `gsplat_30k_fused_prune10_rclip05`. This independently confirms the 1.164× candidate is an **accel-matrix ablation arm** (a gsplat config combination), not a named paper method — consistent with "gsplat optimization combination, not HiGS native." The nearest *recoverable* strong reference on MX is **FastGS** (opacity-prune training control), which directly informs the P3 `opacity_pruning_control` arm.

Per the objective's rule: "若找不到，不得宣称已独立重算 132-job 的全部证据；将它标为历史汇总支持。" → **This is exactly the classification used here.**

---

## 3. Independent recomputation performed this session

Recomputation is **scope-limited to what the surviving evidence supports**:

| Recomputation | Status | Basis |
|---------------|--------|-------|
| 132-job paired bootstrap (33 cells, 10k iters, seed 12345) | **NOT recomputed** — raw per-run data absent | aggregate JSON only |
| Non-inferiority gate arithmetic (PSNR/SSIM/LPIPS) | **Verified** against stored gate values | `confirmatory-accel15-verdict.json` |
| Speed-ratio CI lower bound vs required_min 1.1111 | **Verified**: 1.03437 < 1.1111, but `accel_passed` uses the mean (1.1642) vs 1.1111; CI-lower gate is separate | summary JSON |
| Which scenes drive the LPIPS regression (prune vs rclip) | **NOT possible** — per-scene LPIPS absent | flagged as open |
| TTQ improvement | **Supported by aggregate** (−49.5 s, CI hi −18.98) | summary JSON |

**Explicit statement (per objective):** the 1.164× result is a **gsplat optimization-combination** (`fused render + opacity pruning every 10th densification + radius clip 0.05`), **NOT a native HiGS method**. It is recorded as historical aggregate support.

---

## 4. Independent recomputation of the C51 Stage-5 causal evidence (P2 cross-link)

The Stage 5 four-condition experiment (A/B/C/D) IS fully reconstructable from `reports/phase-c51-stage5.md` with complete per-scene tables. Its true cost table is delivered separately as `C51J_WORKLOAD_COST.csv` and audited in `C51J_C51_CAUSAL_AUDIT.md`. Headline cross-check against the 1.164× baseline:

- The C51 K50-B1 joint mechanism (sparse backward + densification response) produced +8.2% / +24.4% / +28.9% E2E speedup on room/bicycle/garden (A100). **94% of that speedup is densification-induced** (fewer Gaussians), not the backward kernel. The intrinsic sparse-backward (condition C, population preserved) is **+0.5% / −6.6% / −1.7%** — it FAILS the 5% gate everywhere.
- This is a **different experiment protocol** from the 132-job matrix (3-scene single-seed C51 ablation vs 11-scene 3-seed confirmatory). They must **not** be merged. See causal audit for the boundary.

---

## 5. Deliverable provenance

| Item | Location | Integrity |
|------|----------|-----------|
| 132-job aggregate | `paper/higs/tables/confirmatory-accel15-summary.json` (local, read-only) | read this session |
| 132-job verdict | `artifacts/training-ablation/results/confirmatory-accel15-verdict.json` (local) | read this session |
| gate-nuance correction | §2.2 of this file | derived from raw JSON |
| MX live probe | `bms-39468022-001` SSH, 2026-10-10T15:43Z | live |
| C51J output dir | `/mnt/storage_pool/liaoyuanjun/CVPR27_C51_JOINT_MX/` | created this session |
| 132-job per-run raw | **NOT FOUND** (local + MX) | §2.3 |
| paper protocol | `tco0_work/benchmark/higs-paper-protocol.json` (MX) | §2.3.1 |
| fastgs trainer | `strong_baselines/fastgs` @ `d4d33b6f…` (MX) | §2.3.1 |
| speedy-splat trainer | `strong_baselines/speedy-splat` @ `b9dd42d1…` (MX) | §2.3.1 |
| Mip-NeRF360 data | `benchmark_data/mipnerf360/{room,bicycle,garden}` (MX) | §2.3.1 |
| EPIC-05 | **NOT PRESENT** on MX | §2.3.1 |

No unique experiment data was deleted or moved. This audit performed **read-only** operations on MX (plus two `mkdir` and lightweight `/tmp` script uploads that do not consume meaningful storage). The 1.164× candidate is confirmed to be an accel-matrix ablation arm (absent from the paper-protocol `methods` block), not a named method.

---

## 6. Standing conclusions for downstream (Codex / P3–P5)

1. Treat 1.164× as **historical aggregate support on A100**, not as freshly recomputed proof.
2. Do **not** cite the C51 Stage-5 K50-B1 speedup as independent confirmation of the 132-job result; different protocol, different seed/scene count.
3. A **new executable frozen protocol** for any fresh confirmation must be authored from scratch (the historical per-run protocol file is missing). This is P3 work.
4. P4 (Codex candidate verification) and P5 gating **cannot run now**: no exclusive A100 and storage 100% full. They are BLOCKED on resources, not on science.
