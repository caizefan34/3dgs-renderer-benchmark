# PUBLICATION PHASE — First Checkpoint (§27)

Date: 2026-09-24 (post FINAL-30K closure)
Authoritative FINAL-30K result preserved unchanged: 13/13 scenes faster, 26/26 valid runs, geomean speedup 1.0685×, mean ΔPSNR +0.070 dB, FINAL_PASS (`reports/higs/final-30k-c0-v3-vs-b1a.md`).

---

## 1. B0 Graphdeco exact repo/commit/config

- **Repo**: `graphdeco-inria/gaussian-splatting` (official upstream).
- **Commit**: **`54c035f`** — frozen. This is the exact commit the benchmark's `baseline/reference_v1/config.py` documents the optimizer values against ("official Graphdeco OptimizationParams at commit 54c035f"), making it the identity anchor for the matched-optimizer claim.
- **Build**: cloning (recursive) to `/mnt/storage_pool/liaoyuanjun/graphdeco-b0-pub`; building the original `diff-gaussian-rasterization` and `simple-knn` CUDA packages under higs-13scene-env (torch 2.9.1+cu128). Build in progress (background).
- **Config / method classification**: **`ORIGINAL_METHOD_PROTOCOL`** for densification only. The original method has no absgrad statistic; its densification is the published signed-2D-gradient L2-norm criterion (threshold 0.0002). Everything else (optimizer, lrs and expon schedule, loss 0.8·L1 + 0.2·(1−SSIM), densify window 500–15000/100, opacity reset 3000, SH/1000, pruning 0.005/20, percent_dense 0.01, COLMAP init) is IDENTICAL to reference_v1 by construction — reference_v1 was documented against this exact commit.
- **Wrapper policy (per §2)**: matched training wrapper = the unified trainer's infrastructure (GTDataset/colmap_reader data loading, seed-42 camera sequence, eval grid, timing instrumentation) with the Graphdeco METHOD verbatim (its densification statistic/threshold transcribed from `reference/graphdeco/gaussian_model.py`, which is the upstream file). The method is NOT modified; the difference from the matched protocol is exactly one item (densification statistic: signed-grad L2-norm ≥ 2e-4 vs absgrad ≥ 8e-4) and will be stated in every table that includes B0.
- **Loss note**: the unified trainer uses a separable SSIM (same formula, window 11, sigma 1.5, separable conv implementation); the original uses a joint 2D window. Mathematically identical SSIM, FP-rounding-level difference — infrastructure-level, disclosed.

## 2. B1 clean gsplat exact repo/commit/config

- **Repo**: official gsplat (nerfstudio-project/gsplat), as present locally at the accutile tree's origin history.
- **Commit**: **`937e29912570c372bed6747a5c9bf85fed877bae` = tag `v1.5.3`** (2025-07-04). Same base commit as the B1A accutile tree, so **B1 → B1A isolates exactly one delta: the AccuTile patch** (9 files: `IntersectTile.cu` core + plumb in `_backend/_wrapper/Intersect.cpp/h/Common.h/Ops.h/rendering.py/setup.py`).
- **Build**: `/mnt/storage_pool/liaoyuanjun/gsplat-b1-clean-v153` = byte-copy of the accutile tree with the 9 patched files git-reverted to HEAD → git-verified pristine v1.5.3 (accutile grep = 0 hits), glm submodule present. Building as extension `gsplat_cuda_b1clean` with the identical b1a_build.py flag set (build in progress, background).
- **Config**: `rasterization(absgrad=True, tile_size=16, packed=False, radius_clip=0.0, eps2d=0.1, render_mode="RGB")` — the B1A call minus `accutile`. `absgrad` is **native upstream gsplat 1.5.3** (verified in the pristine signature), so B1 runs the FULL matched absgrad densification protocol (MATCHED_PROTOCOL, not ORIGINAL_METHOD_PROTOCOL).
- **eps2d choice**: 0.1 (same as B1A) so the B1↔B1A delta is accutile-only. The eps2d axis is separately quantified by P4 (which adds B1A@0.3 and C0@0.1). Upstream default is 0.3; this choice is documented and disclosed in all tables.

## 3. Unified trainer support for B0

**Supported with a new arm (`--arm b0`)**, not yet wired. The trainer's render-call abstraction gets a third branch calling the original `diff_gaussian_rasterization` package (build in progress). The densification block gets a B0 branch implementing the original signed-grad statistic/threshold, transcribed from the upstream `gaussian_model.py` (already in-repo at `reference/graphdeco/gaussian_model.py` as the fidelity source). Optimizer/loss/schedule/eval/seed all come from the shared recipe (matched to graphdeco @54c035f by design). B0 gate (§2 checklist: dataset loading, COLMAP init, camera convention, resolution, seed, loss, optimizer, densification schedule, pruning, SH progression, evaluation, checkpoint schedule) runs as functional verification + 3-scene gate runs (room/bicycle/garden) before the 13-scene phase.

## 4. Unified trainer support for B1

**Supported with a new arm (`--arm b1`)**, not yet wired. Mechanically identical to the existing `b1a` bootstrap (sys.path insert of the clean tree + sys.modules injection of the `gsplat_cuda_b1clean` extension), with the render call minus `accutile=True`. Same GTDataset, same seed-42 camera sequence, same loss, same optimizer, same absgrad densification semantics, same resolution, same eval grid, same seed — MATCHED_PROTOCOL by construction.

## 5. A0/A1/A2/A3 exact binary definitions

All four ablation variants run the **SAME frozen binary pair** as FINAL-30K C0: extension `experimental_gaussian_render_inference_scene_cuda.so` (sha256 `9baf8655…`) + core `gsplat_cuda.so` (sha256 `361b216b…`), built from the C0 worktree (HEAD `77ab983f` + the C0 working-tree patch). The three composition modules are **runtime-switchable** in this binary (verified in source):

- F9 (gatherless projected producer, forward): `HIGS_DISABLE_F9=1` disables it; the baseline (gather-based) forward branch is retained in the binary and is authoritative for every unsupported mode (`gaussian_inference.py:1757`).
- SCALAR_ADJOINT (blend-backward adjoint form): `HIGS_BWD_SCALAR_ADJOINT=scalar_adjoint` enables; unset = original tensor-valued adjoint (`HigsNativeBackward.cu:1362`, template param `SCALAR_VAL`).
- H8-MR (opacity-absorbed moment-space geometry adjoint): `HIGS_BWD_H8_MR=1` enables; requires scalar_adjoint (enforced by TORCH_CHECK, consistent with cumulative order).
- `HIGS_PX_RUNTIME` is a blend-backward **launch-shape** variant (pixels-per-thread lanes 0/1/2/4/8; identical math, defaults to 2). It is part of the frozen C0 substrate, NOT one of the three modules. All ablation arms hold PX=2 constant. (The functional gate will verify PX-invariance of gradients to substantiate "launch shape, not algorithm".)

| arm | forward producer | blend backward adjoint | H8-MR | env (beyond the frozen absgrad) | 13-scene runs |
|---|---|---|---|---|---|
| **A0** (internal parent) | gather-based (baseline branch) | tensor adjoint | off | `HIGS_DISABLE_F9=1` | **NEW ×13** |
| **A1** = A0+F9 | F9 gatherless | tensor adjoint | off | — | **NEW ×13** |
| **A2** = A1+SCALAR | F9 gatherless | scalar adjoint | off | `HIGS_BWD_SCALAR_ADJOINT=scalar_adjoint` | **NEW ×13** |
| **A3** = A2+H8 = C0_V3_FINAL30K | F9 gatherless | scalar adjoint | on | `HIGS_BWD_H8_MR=1` | **REUSE FINAL-30K** |

All arms additionally set `HIGS_BWD_ABSGRAD=1` (the pure densification-statistic auxiliary, verified by gates C/D to change no gradient and no output) and `HIGS_PX_RUNTIME=2`. A0 is the exact internal parent (Trainable HiGS B2 lineage, core `361b216b`) with the three algorithmic modules disabled inside the same frozen binary — no structurally different substitute.

## 6. Reusable existing runs (§6 audit)

| dataset | reusable? | reason |
|---|---|---|
| FINAL-30K C0 × 13 | **YES** | exact A3 binary + frozen protocol + clean publication-grade |
| FINAL-30K B1A × 13 | **YES** (Table 1 B1A row; P4 0.1-corner) | exact binary + frozen protocol + clean |
| C0-T1/T2 nested timing (3 scenes, 2K/5K mechanism tests) | **NO** | not 30K, not publication protocol (§6: do not promote mechanism tests) |
| Prior B1A 13-scene cohort (older env) | **NO** for matched tables | different env (pre torch 2.9.1+cu128 rebuild); superseded by FINAL-30K B1A |
| Faster-GS / Speedy-Splat / FastGS prior runs (C4.2-era, native/c42 variants) | **NO** for publication | system-level results from a different research era; unknown GPU cleanliness; publication needs fresh official-config runs with provenance |
| Any A0/A1/A2 30K runs | **NONE EXIST** | HiGS side never ran 13-scene 30K before FINAL-30K |

## 7. Missing ablation run count

- Stage A (§7): A0/A1/A2 × {room, bicycle, garden} = **9 new runs** (A3 reused).
- Stage B (§8): A0/A1/A2 × remaining 10 scenes = **30 new runs**.
- Total P2: **39 new runs**.

## 8. eps2d sensitivity implementation plan (P4)

**Fully runtime — no rebuilds.** `eps2d` is a keyword argument of both render entry points: `gsplat.rasterization(..., eps2d=…)` (B1A/B1) and `rasterize_gaussian_higs_dynamic(..., eps2d=…, default 0.3)` (C0/A0/A1/A2). The publication trainer gets a `--eps2d` override that flows to the active arm's render call.

2×2 on {room, bicycle, garden} @ 30K, nothing else changed:

| | B1A (accutile) | C0 (A3) |
|---|---|---|
| eps2d 0.1 | FINAL-30K reuse | **NEW ×3** |
| eps2d 0.3 | **NEW ×3** | FINAL-30K reuse |

**6 new runs.** Pre-registered conclusion rule (written before any run): compare C0@0.3 vs C0@0.1 and B1A@0.1 vs B1A@0.3 per scene; classify `EPS2D_NEGLIGIBLE` if geomean wall-time delta ≤ 1% AND |ΔPSNR| ≤ 0.10 dB on ≥2/3 scenes; `EPS2D_MATERIAL_CONFOUND` if the eps2d effect on C0 wall time exceeds half the C0-vs-B1A speedup effect or shifts ΔPSNR by more than the FINAL-30K mean; `EPS2D_SMALL_BUT_PRESENT` otherwise.

## 9. Faster-GS reproduction status (P5)

- Code present: `/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg` (Faster-GS runs in the NeRFICG framework) + `fastergs_configs/` + working trainer wrapper (`fastergs_train.py`) and scheduler from the C4.2 era.
- Prior results: 8 jobs completed (native + c42 variants) — usable as SANITY REFERENCE ONLY; not publication evidence (protocol era + cleanliness unproven).
- Publication plan per §12–§14: record the frozen nerficg/NeRFICG commit; reproduce official supported training configuration; verify datasets/scenes; run room/bicycle/garden first; classify components (renderer-only vs optimizer/densification alterations) BEFORE any 13-scene expansion; keep in the SYSTEM_LEVEL table unless an official renderer-only configuration exists.
- Priority: after P1/P2/P4 underway (§23).

## 10. Current clean GPU pool

**0/8 clean at checkpoint time** (all eight A100s occupied by foreign ~22–23 GB jobs; utilization 3–67%). Historical behavior (FINAL-30K phase): foreign jobs flap; clean windows open unpredictably. The dynamic scheduler (scan all 8, two-scan ≥120 s stability + <500 MiB + zero processes, never touch foreign jobs) handles this; launches happen when windows open.

## 11. Proposed GPU allocation

Single dynamic pool over all 8 A100s, one queue with §23 priority, up to N-clean-GPUs concurrent (never two publication jobs per GPU):

1. B1 gate runs (room/bicycle/garden) → then B1 remaining 10 scenes
2. B0 gate runs (after build + arm wiring) → then B0 remaining 10 scenes
3. A0/A1/A2 Stage A (9 runs) → Stage B (30 runs)
4. P4 eps2d (6 runs)
5. P6 multi-seed (16 runs)
6. Faster-GS 3-scene gate
7. P3 standalone isolation (9 runs, lowest priority)

## 12. Estimated run matrix (not wall-clock ETA)

| phase | variants | scenes | seeds | runs |
|---|---|---|---|---|
| P1 B0 | 1 | 13 | 42 | 13 new (+3-scene gate incl.) |
| P1 B1 | 1 | 13 | 42 | 13 new (+3-scene gate incl.) |
| P2 Stage A | A0,A1,A2 | 3 | 42 | 9 |
| P2 Stage B | A0,A1,A2 | 10 | 42 | 30 |
| P4 eps2d | C0@0.1, B1A@0.3 | 3 | 42 | 6 |
| P6 multi-seed | B1A,C0 | 4 | 43,44 | 16 |
| P5 Faster-GS | 1 system | 3 → 13 | official | 3 (then 10) |
| P3 standalone | F9, SA, H8 alone | 3 | 42 | 9 |
| **Total before P5/P3** | | | | **87** |
| **Total incl. P5(1 system)+P3** | | | | **109** |

Reused (no new runs): FINAL-30K C0 (A3) ×13, B1A ×13.

## 13. Protocol blockers

**None.** Wait conditions, not blockers: (a) zero clean GPUs at this instant — the scheduler waits for windows; (b) B1/B0 builds in flight (~1 h); (c) trainer arms a0/a1/a2/b1/b0 to be wired (code work, in progress this round). No protocol incompatibility identified: B1 is MATCHED_PROTOCOL (native absgrad), B0 differs by exactly one documented method item (ORIGINAL_METHOD_PROTOCOL densification statistic).

## 14. Exact next runs to launch

1. **Now (background, no clean GPU needed)**: B1 pristine build; B0 graphdeco clone + rasterizer build; publication trainer arms.
2. **Functional gate (small footprint, may share a GPU, FUNCTIONAL_ONLY, no timing claims)**: A0/A1/A2 smoke — verify A0 forward frame bit-identical to A3 (F9 exactness), gradients within FP envelope, PX-invariance check, absgrad statistic present in all arms; B1 smoke (render + absgrad + trainer loop).
3. **First timing launches (clean-GPU scheduler, in priority order)**:
   - `b1/room`, `b1/bicycle`, `b1/garden` (30K, seed 42, publication grade) → B1_REPRODUCTION_PASS check → remaining 10 B1 scenes
   - `a0/room`, `a1/room`, `a2/room`, then bicycle/garden for each (Stage A)
   - `b0/room`, `b0/bicycle`, `b0/garden` once the B0 arm lands
4. **Then**: Stage B expansion, P4 corners, P6 seeds, Faster-GS gate.
