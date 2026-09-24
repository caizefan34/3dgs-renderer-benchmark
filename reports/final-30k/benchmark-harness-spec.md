# FINAL 30K Benchmark — Frozen Harness Specification

**Status:** `FINAL_30K_HARNESS_FROZEN` — the harness is frozen and ready. The benchmark is **NOT run**.
**Date:** 2026-09-22 (initial) / **R1 refresh:** metadata only — P2_FINAL_V2 composition updated (P2-1C removed, R2-gated), TTQ censoring semantics frozen, garden quality policy frozen. **No protocol redesign; the benchmark protocol is unchanged.**
**Goal:** When the final stack is selected, the complete 30K benchmark can start immediately with no protocol redesign and no ambiguity over which historical numbers are authoritative.

---

## 1. Scope and non-goals

**In scope (this task):**
- Freeze the 13-scene real-dataset manifest.
- Freeze the training protocol (every experiment-critical value explicit).
- Define the immutable candidate identities (B1A_ACCUTILE / C0_V3 / P2_FINAL_V2) and the identity/protocol gate.
- Define runtime + quality metric collection, TTQ, failure handling, resume policy, GPU contamination policy, aggregation rules, and output schema.
- Provide harness/infrastructure scripts (provenance, validation, aggregation, TTQ, scaffold).

**Out of scope (explicitly NOT done):**
- ❌ Run the FINAL 30K benchmark.
- ❌ Modify renderer CUDA kernels.
- ❌ Reopen C0.
- ❌ Introduce new optimization candidates.

---

## 2. Final 13-scene manifest (FROZEN)

The SAME 13 scenes used by the B1A AccuTile 13-scene validation and the C42 13-scene benchmark. Do NOT substitute scenes.

| # | Scene | Dataset | Data path (rel. to dataset root) |
|---|-------|---------|----------------------------------|
| 1 | bicycle | Mip-NeRF 360 | `mipnerf360/bicycle` |
| 2 | bonsai | Mip-NeRF 360 | `mipnerf360/bonsai` |
| 3 | counter | Mip-NeRF 360 | `mipnerf360/counter` |
| 4 | flowers | Mip-NeRF 360 | `mipnerf360/flowers` |
| 5 | garden | Mip-NeRF 360 | `mipnerf360/garden` *(data-quality outlier: manually-created COLMAP; see garden policy below)* |
| 6 | kitchen | Mip-NeRF 360 | `mipnerf360/kitchen` |
| 7 | room | Mip-NeRF 360 | `mipnerf360/room` |
| 8 | stump | Mip-NeRF 360 | `mipnerf360/stump` |
| 9 | treehill | Mip-NeRF 360 | `mipnerf360/treehill` |
| 10 | train | Tanks & Temples | `tanksandtemples/train` |
| 11 | truck | Tanks & Temples | `tanksandtemples/truck` |
| 12 | drjohnson | Deep Blending | `deepblending/drjohnson` |
| 13 | playroom | Deep Blending | `deepblending/playroom` |

Distribution: Mip-NeRF 360 ×9, Tanks & Temples ×2, Deep Blending ×2.

**Resolution rule:** 1080p-class (max_side=1920, per-scene aspect preserved; `GTDataset(resolution='1080p')`) — the same resolution the B1A 13-scene run used. Progressive downscale dirs (`images_2/4/8`) exist per scene but the primary run is 1080p-class.

**Split rule:** Official dataset split. Train = COLMAP/SfM training views; test/eval = official test cameras. During-training eval uses every 10th camera (`range(0, n_cameras, max(1, n_cameras//10))`); final eval (iter 30000) uses ALL cameras.

**Camera count:** per-scene, dataset-derived (`n_cameras = len(dataset)`), recorded in `provenance.json` per run — NOT a protocol choice, so NOT fixed in the manifest.

**Environment-dependence:** dataset root and all paths are exposed via `harness/final30k/config_schema.json`, NOT hardcoded to a user-specific temporary path.

**Garden quality policy (frozen BEFORE any FINAL-30K results):**
- Garden **remains included** in the primary 13-scene quality aggregate (all 13 scenes, no exclusions).
- A **pre-labeled sensitivity analysis** excluding garden (reason: known dataset/registration concerns from manually-created COLMAP) MAY additionally be reported — labeled as such.
- **Do NOT exclude garden only from unfavorable quality conclusions.** Either garden is in the primary aggregate (it is), or the exclusion is a pre-labeled sensitivity analysis applied regardless of the direction of the result.
- Machine-readable: `artifacts/final-30k/manifest.json` → `garden_quality_policy`.

Source of truth: `b1a_13scene_analyze.py` lines 11-14, `c42_13scene_analysis.py` lines 32-36, `reports/accutile30k/13scene-quality.md`. Full machine-readable manifest: `artifacts/final-30k/manifest.json`.

---

## 3. Frozen training protocol (FROZEN)

Every experiment-critical value is explicit (no hidden defaults). The harness **ABORTS** if two candidates resolve to different experiment-critical settings. Source of truth: `baseline/reference_v1/config.py` (`ReferenceV1Config`) + `b1a_13scene_train.py`.

| Field | Frozen value |
|---|---|
| **Training steps** | 30000 |
| **Resolution** | 1080p-class (max_side=1920) |
| **Loss** | `(1-0.2)*L1 + 0.2*(1-SepSSIM)`, `lambda_dssim=0.2`; SepSSIM window=11 sigma=1.5 |
| **Optimizer** | Per-parameter Adam, full-parameter (same recipe as B1A 13-scene; the FINAL 30K candidate set is compared under full-Adam, not selective/visible Adam) |
| **LR — position** | init 0.00016 → final 0.0000016, delay_mult 0.01, max_steps 30000 |
| **LR — feature** | 0.0025 |
| **LR — opacity** | 0.025 |
| **LR — scaling** | 0.005 |
| **LR — rotation** | 0.001 |
| **Densification** | from 500 to 15000, interval 100, `absgrad` threshold 0.0008 (gsplat adaptation, NOT original-3DGS signed 0.0002), percent_dense 0.01 |
| **Pruning** | min_opacity 0.005, max_screen_size 20 (size threshold active only after first opacity reset) |
| **Opacity reset** | every 3000 iters (during densify window) |
| **SH schedule** | sh_degree 3, oneupSHdegree every 1000 iters |
| **Initialization** | COLMAP SfM sparse (`points3D.bin` per scene); `create_from_pcd(spatial_lr_scale=scene_extent)`; scene_extent = max pairwise camera-center distance (50 sampled) |
| **Seed** | 42 (primary publication; torch/numpy/python all 42) |
| **Eval schedule** | iters 500,1000,2000,5000,10000,15000,20000,25000,30000 |
| **Checkpoint schedule** | 5000,10000,15000,20000,25000,30000 (only iter_30000.pt retained on disk; all checkpoint n_gaussians recorded for TTQ) |

Full machine-readable protocol: `artifacts/final-30k/protocol.json`.

---

## 4. Seed policy

- **Primary publication seed = 42** — the existing authoritative 13-scene protocol already uses seed=42 (`b1a_13scene_train.py` `config.seed=42`, `provenance.seed=42`), so no other seed is established.
- Seeds recorded: `torch seed`, `numpy seed`, `python random seed`, `CUDA seed` (derived from torch seed). The C0-T2 fixture's seed 4200 is a **microbenchmark upstream-gradient seed**, distinct from the training seed.
- **No multi-seed requirement** for the primary benchmark (skill §23: exploratory screening allows 1 seed). The harness is structured so additional seeds can be added later by appending to the `seeds` list in `config_schema.json` without changing naming/log schema.

---

## 5. Environment provenance (recorded per run)

`harness/final30k/capture_provenance.py` captures, per run:

```
host, GPU index, GPU UUID, GPU model, driver version,
torch version, CUDA runtime,
nvcc version = the ACTUAL compiler used for built binaries (config nvcc_path), NOT PATH nvcc,
git commit, git dirty state (+ dirty file count),
renderer binary SHA256, patch SHA256s,
environment variables affecting renderer behavior (HIGS_DISABLE_F9, HIGS_BWD_H8_MR, CUDA_VISIBLE_DEVICES, TORCH_EXTENSIONS_DIR, TORCH_CUDA_ARCH_LIST, OMP_NUM_THREADS, ...),
GPU contamination snapshot (memory.used, utilization.gpu, compute PIDs).
```

**Target environment:** host `bms-39468022-001`, GPU `A100-PCIE-40GB / SM80`, torch `2.9.1+cu128`, CUDA `12.8`. nvcc recorded from `higs-13scene-env/bin/nvcc` (12.8.93), NOT PATH nvcc (host nvcc 11.5 deliberately not used).

---

## 6. Candidate identity (immutable)

`artifacts/final-30k/candidate_registry.json` defines three immutable candidates. The harness **ABORTS** if the expected binary identity does not match.

| Candidate ID | Role | Binary SHA256 | Key patches |
|---|---|---|---|
| `B1A_ACCUTILE` | Modern training baseline (gsplat 1.5.3 + True AccuTile) | pinned at build time (recorded per run) | true-accutile tree |
| `C0_V3` | Frozen Phase-1 stack (F9 + SCALAR_ADJOINT + H8-MR, V3) | `7ca1c6bf...` (asserted in-harness) | F9 `1faa05c1...`, H8-MR `8fb23ac8...` |
| `P2_FINAL_V2` | Future Phase-2 stack (IF P2-1A R2 passes) | TBD (P2-1A prototype `.so` `1799facc...` is **INVALID** — deterministic FP16/FP32 entry-contract failure; final binary derives from P2_1A_R2 if R2 passes) | P2-1A `ab860d77...` (R2 repair patch TBD) |

For each candidate the harness records: source commit, binary SHA256, patch SHA256s, runtime feature toggles, build environment. **P2_FINAL_V2 is accepted-later** — the harness protocol is invariant to which candidates are present, so it can be added without protocol change when P2-1A R2 passes. **P2_FINAL_V2 composition: C0 V3 + P2-1A native hierarchical forward (if R2 runtime passes) + optional E2. P2-1C is REMOVED from the planned final stack (DROP).**

**Quality language (B1A precedent):** B1A_ACCUTILE is `QUALITY_NEUTRAL_WITHIN_VARIATION` — **NOT quality-improving overall**. Aggregate ΔPSNR +0.110 / ΔSSIM -0.0003 / ΔLPIPS -0.0009 are within FP/stochastic variation. Garden ΔPSNR -0.714 is a genuine per-scene delta. Do NOT describe B1A as quality-improving overall.

---

## 7. Required runtime metrics (per scene)

`timing.json` (machine-readable):

```
total training wall time (s)
iterations/sec
mean iteration time (ms)
median iteration time (ms)
forward time (ms)
backward time (ms)
nested F+B time (ms)          # the training-loop shape: forward immediately followed by backward
optimizer time (ms)
loss time (ms)
peak VRAM (GB)
final N_GS
```

**Measurement definition is identical across candidates** (same nested-event / same-boundary protocol; the C0-T2 nested-event invariants `fb ≥ forward`, `fb ≥ backward`, `gap ≥ -0.5 ms` are the reference). No incompatible measurement definitions between candidates.

---

## 8. Quality metrics (per scene, at eval checkpoints)

`quality.json` (30000-step, all cameras) + per-checkpoint quality in `training_curve.csv`:

```
PSNR (dB)        20*log10(1/sqrt(MSE)); higher better
SSIM             SepSSIM window=11 sigma=1.5; higher better
LPIPS            lpips.LPIPS(net='vgg'), inputs mapped to [-1,1]; LOWER better
L1               auxiliary
```

**Exactly the same evaluation implementation for every candidate** — one shared evaluator (`evaluate_model` logic), its code SHA256 recorded in `provenance.json`. **No candidate-specific metric paths.** Eval resolution = training resolution (full-res, not downscaled). Test camera set identical across candidates per scene. Recorded: metric code version (evaluator SHA256), evaluation resolution, test camera set.

---

## 9. Time-to-quality (TTQ)

`artifacts/final-30k/ttq_spec.json` + `harness/final30k/derive_ttq.py`:

- **Thresholds are derived from the FROZEN baseline reference (B1A_ACCUTILE) final-eval quality — NOT chosen after seeing final results.** This prevents "changing quality targets after seeing results."
- For each scene/metric, threshold = the B1A reference value at iter 30000. TTQ = cumulative wall time at the first eval checkpoint where the candidate meets the threshold (PSNR/SSIM `>=`; LPIPS `<=`).
- **Censoring semantics (frozen):** for a scene that NEVER reaches the frozen threshold within 30000 steps → `ttq = null`, `status = TTQ_NOT_REACHED`, `censor_time = total_wall_time_30000`. **`censor_time` is a right-censoring bound, NOT an achieved TTQ.** Do NOT treat censor_time as an achieved TTQ in any comparison or geomean.
- Preserved at minimum: **time-to-PSNR** (always meaningful); **time-to-SSIM** and **time-to-LPIPS** where the curve crosses.
- `TTQ_speedup = TTQ_baseline / TTQ_candidate`. **The primary TTQ geomean includes ONLY comparable reached/reached pairs** (both baseline and candidate reached the threshold) and **explicitly reports reach counts** (`n_reached_pairs`, `n_not_reached`, `n_scenes`). Censored scenes are excluded from the geomean, never substituted with censor_time.
- **Do NOT claim TTQ until threshold definitions are frozen** — they are frozen here, derived from the B1A reference.

---

## 10. Training curve artifacts (machine-readable)

`training_curve.csv` (no graph-only results):

```
step, wall_time, iter_time, loss, N_GS, VRAM_gb, PSNR(if evaluated), SSIM(if evaluated), LPIPS(if evaluated)
```

---

## 11. Failure handling

A failed scene **MUST NOT silently disappear** from aggregate statistics. `final_status.json` status vocabulary:

```
SUCCESS | OOM | CUDA_ERROR | NUMERIC_FAILURE | QUALITY_FAILURE | INTERRUPTED
```

Aggregate reports explicitly show `missing_or_failed_scenes` (scene_id + status) and `n_success`. Aggregates compute over SUCCESS scenes only, with `n_success` stated. A run with any failed scene is marked `NOT_PUBLICATION_GRADE_FOR_AGGREGATE`. No scene is dropped from the denominator without being recorded (skill §32).

---

## 12. Resume policy

**Preferred publication policy: fresh run from step 0.** No resume for the primary benchmark.

If an infrastructure failure forces a resume, the run MUST record:
```
resume step
checkpoint hash
reason
wall-time accounting policy
```
Wall time from incompatible fragments is **NOT silently combined** — a resumed run is either (a) re-run fresh, or (b) reported with an explicit `resumed=true` flag and the wall-time accounting policy documented. A resumed run is marked non-publication-grade for the aggregate unless the wall-time accounting policy is proven equivalent to a fresh run.

---

## 13. GPU contamination policy

Before the final run, `capture_provenance.py` records:
```
memory.used (per GPU)
utilization.gpu (per GPU)
compute PIDs (process_name + used_memory)
```

- The final benchmark timing should preferably use a **low-interference GPU** (idle, no compute PIDs, low utilization).
- If a run experiences **competing workload interference**, it is **marked** (`gpu_snapshot.json` shows non-zero compute PIDs / high utilization) and classified **NOT publication-grade**.
- Contaminated timing is **NOT silently classified as publication-grade**.

---

## 14. Output structure

```
artifacts/final-30k/
    manifest.json                  # frozen 13-scene manifest
    protocol.json                  # frozen training protocol
    candidate_registry.json        # immutable candidate identities
    aggregation_spec.json          # frozen aggregation rules
    quality_metric_spec.json       # frozen quality metric protocol
    ttq_spec.json                  # frozen TTQ thresholds
    README.md
    <candidate>/                   # B1A_ACCUTILE | C0_V3 | P2_FINAL_V2
        <scene>/                   # 13 scenes
            provenance.json        # env + binary/patch identity + GPU snapshot
            config.json            # resolved protocol config
            training_curve.csv     # machine-readable curve
            timing.json            # runtime metrics
            quality.json           # final PSNR/SSIM/LPIPS
            memory.json            # peak VRAM
            checkpoints.json       # checkpoint schedule + N_GS (for TTQ)
            final_status.json      # SUCCESS/OOM/.../INTERRUPTED
            gpu_snapshot.json      # pre-run GPU contamination
            MANIFEST.sha256
reports/final-30k/
    benchmark-harness-spec.md      # this file
    benchmark-summary.md           # populated AFTER the benchmark runs
```

The per-run template files are scaffolded by `harness/final30k/init_run_dirs.py` (351 templates = 3 × 13 × 9 already created; no numbers populated). **The benchmark is NOT run.**

---

## 15. Aggregation rules (FROZEN)

`artifacts/final-30k/aggregation_spec.json` + `harness/final30k/aggregate_results.py`:

- **Speedup** `= T_baseline / T_candidate` (ratio; >1 = faster). **Time reduction** `= 1 - T_candidate / T_baseline` (fraction). **Do NOT mix them** — report each with its unit.
- Speedup/reduction aggregates: **geomean** (primary) + arithmetic mean (secondary). A slower scene is INCLUDED in the geomean (not dropped).
- **Quality deltas:** `ΔPSNR = candidate - baseline` (higher better), `ΔSSIM = candidate - baseline` (higher better), `ΔLPIPS = candidate - baseline` (**lower better**; negative = improvement). **Do NOT transform LPIPS sign.**
- Per-scene reported BEFORE any aggregate; aggregates over SUCCESS scenes with `n_success` stated.
- Baseline for speedup = B1A_ACCUTILE.

---

## 16. Quality language rules

Reporting templates distinguish:

```
faster
quality-neutral within observed variation
quality improved
quality degraded
```

**Do NOT describe** `ΔPSNR > 0 but ΔSSIM/LPIPS mixed` **as globally improved quality** without qualification. The B1A precedent (aggregate ΔPSNR +0.110 with mixed ΔSSIM/ΔLPIPS, garden -0.714) demonstrates why: the correct classification is *quality-neutral within observed variation*, not *quality improved*. Full rules in `artifacts/research-registry/prohibited_claims.json` → `language_discipline_prohibited`.

---

## 17. Validation checklist (before the benchmark may start)

`harness/final30k/validate_candidates.py` enforces:

1. ✅ Every authoritative numeric claim has a source-artifact reference (registry).
2. ✅ All superseded values are marked SUPERSEDED_FOR_PUBLICATION (supersession_map.json).
3. ✅ All candidate states are mutually consistent (candidate_status.json vocabulary only).
4. ✅ No invalid C0 timing remains publication-authoritative (C0-T2 balanced values are the ONLY C0 publication timing).
5. ✅ P2-0 measured/derived/hypothetical distinctions are explicit (authoritative_results.json / P2 registry).
6. ✅ Binary SHA256 identity matches expected (abort on mismatch).
7. ✅ Patch SHA256s present and matching (abort on mismatch).
8. ✅ All candidates resolve to IDENTICAL experiment-critical settings (abort on mismatch).
9. ✅ Manifest = 13 scenes, matches B1A 13-scene + C42 13-scene lists.
10. ✅ TTQ thresholds frozen (derived from B1A reference, not post-hoc).

**Result: `FINAL_30K_HARNESS_FROZEN` = PASS** (see final response item 19).
