# S1 — Experimental Protocol, Datasets, and Hardware Cohorts

Fact-only extraction for the 3DGS Renderer Benchmark summer research package.
Project period 2026-07-11 .. 2026-09-18. Every numeric value is copied from the
cited source file; no value is inferred or reconciled across sources except
where an explicit disagreement is flagged in §6.

---

## 1. REFERENCE V1 definition

The Reference V1 baseline is the **only** allowed reference for all future paper
experiments; historical results under `CURRENT_PROJECT_SEMANTICS` are declared
not directly comparable
(`reports/reference-v1-baseline-lock.md` lines 23–25, 299).

Canonical config: `configs/reference_v1/room_30k.yaml`. Lock report:
`reports/reference-v1-baseline-lock.md`. Correction record:
`reports/phase-r0.1-evidence-correction.md`.

- **Semantic label**: `REFERENCE_BASELINE_V1`
  (`reports/reference-v1-baseline-lock.md` line 10;
  `configs/reference_v1/room_30k.yaml` line 59). Renamed to
  `REFERENCE_V1_ABSGRAD` in the R0.1 correction because densification uses
  `absgrad=True` with `grow_grad2d=0.0008`, not literal original-3DGS
  signed-gradient 0.0002
  (`reports/phase-r0.1-evidence-correction.md` lines 34–38).
- **Scene / iterations**: `mipnerf360/room`, 30,000 iterations
  (`configs/reference_v1/room_30k.yaml` lines 9–11).
- **Renderer**: gsplat 1.5.3 (absgrad mode)
  (`reports/reference-v1-baseline-lock.md` line 13).
- **Pinned official source**: `graphdeco-inria/gaussian-splatting @ 54c035f7834b564019656c3e3fcc3646292f727d`
  (`reports/reference-v1-baseline-lock.md` line 15;
  `benchmark/higs-paper-protocol.json` original_3dgs commit).

### 1.1 SfM initialization
- Init from sparse SfM point cloud read via `colmap_reader.py` from COLMAP
  `points3D.bin`; adaptation A7 "SfM init from COLMAP points3D.bin"
  (`reports/reference-v1-baseline-lock.md` lines 171, 250).
- Initial Gaussian count on Room: 112,627
  (`reports/reference-v1-baseline-lock.md` line 33, table row iter 500;
  provenance line 235).
- Scene extent: 9.9537
  (`reports/reference-v1-baseline-lock.md` line 235).

### 1.2 Loss function
- L1 + λ·D-SSIM, with **λ = 0.2** (`lambda_dssim: 0.2`,
  `configs/reference_v1/room_30k.yaml` line 42).
- (Equivalently (1−λ)·L1 + λ·D-SSIM as in the official Graphdeco formulation; the
  config only stores `lambda_dssim`.)

### 1.3 Optimizer (Adam)
Single persistent Adam created once in `training_setup`, never reconstructed;
topology changes use `cat_tensors_to_optimizer` / `_prune_optimizer`
(`reports/reference-v1-baseline-lock.md` lines 108–111). Learning rates
(`configs/reference_v1/room_30k.yaml` lines 32–39):

| Param | LR |
|---|---|
| position_lr_init | 0.00016 |
| position_lr_final | 0.0000016 |
| position_lr_delay_mult | 0.01 |
| position_lr_max_steps | 30000 |
| feature_lr | 0.0025 |
| opacity_lr | 0.025 |
| scaling_lr | 0.005 |
| rotation_lr | 0.001 |

Position LR uses exponential scheduling with delay multiplier over 30,000 steps
(official Graphdeco defaults; `configs/reference_v1/room_30k.yaml` lines 14, 31).

### 1.4 Densification & pruning
From `configs/reference_v1/room_30k.yaml` lines 18–29:

| Param | Value |
|---|---|
| densify_grad_threshold (τ) | 0.0008 (gsplat absgrad; official signed = 0.0002; 4× higher as recommended by gsplat docs) |
| percent_dense | 0.01 |
| densify_from_iter | 500 |
| densify_until_iter | 15000 |
| densification_interval | 100 |
| min_opacity (prune threshold) | 0.005 |
| max_screen_size | 20 |
| opacity_reset_interval | 3000 |

- Selection criterion: `max(scale) <= /> percent_dense * scene_extent`
  (official) vs per-event median in the old project
  (`reports/phase-c0-canonical-parity-audit.md` Table A, deviation #6).
- Gradient source for densification: view-space `means2d.absgrad`, pixel-scaled
  `grad[:,0]*=width/2`, `grad[:,1]*=height/2`, visibility-filtered
  (`radii>0`) (`reports/reference-v1-baseline-lock.md` lines 122–126,
  adaptation A2/A3 lines 164–167).
- Opacity reset: global `min(opacity, 0.01)`; opacity Adam moments zeroed
  (`reports/reference-v1-baseline-lock.md` deviation #10 line 87, line 110).
- Pruning is opacity + screen-size + world-size in Reference V1 (deviation #9,
  line 86); in the actual Room 30K run, screen-size and world-size pruning
  never triggered (0 events), so all 639,803 prunes were opacity-based
  (lines 58–62).

### 1.5 Camera sampling & resolution
- Render: `tile_size: 16`, `packed: false`, `absgrad: true`, SH degree 3
  (`configs/reference_v1/room_30k.yaml` lines 45, 54–56).
- Canonical benchmark (Tier A): 100 cameras ordered case-insensitively by
  image name, evenly spaced including both endpoints; center-crop FOV to 16:9,
  render at 1920×1080, GT area-resized at metric time
  (`benchmark/protocol.json` camera_trajectory; `docs/protocol.md` lines 11–13).
- Original photos remain byte-identical (`docs/datasets.md` lines 41–42).

### 1.6 Evaluation split
- Official train/test splits must be used; for original-3DGS training pass
  `--eval` (`docs/official_dataset_training.md` lines 13–14;
  `data/datasets/official_training_datasets.json` policy.required_eval_mode).
- HiGS paper protocol: `eval_split: "official"`
  (`benchmark/higs-paper-protocol.json` training.eval_split).
- Confirmatory protocol: 3 held-out eval cameras, evaluated every 300 steps;
  final report uses step-30000 eval (`paper/confirmatory-protocol.md` §3–§4).
- Official validation matrix cameras: bicycle 281, garden 250, room 217
  (`configs/epic05/official_validation_matrix.json` official_cohort).

### 1.7 Seed(s)
- Reference V1 Room 30K: **seed = 42** (provenance, frozen
  `camera_sequence.npy`) (`reports/reference-v1-baseline-lock.md` line 236;
  `reports/phase-a100/experiment_protocol.md` JSON schema seed 42).
- HiGS paper protocol seeds: **0, 1, 2**
  (`benchmark/higs-paper-protocol.json` training.seeds).
- Confirmatory (pre-registered) seeds: **3, 4, 5** (never used for config
  selection) (`paper/confirmatory-protocol.md` §3).
- Official validation matrix baseline seed: 0; top-level seed: 42
  (`configs/epic05/official_validation_matrix.json` baseline.seed vs seed).

### 1.8 Iteration count & eval checkpoints
- 30,000 iterations (`configs/reference_v1/room_30k.yaml` line 11).
- `eval_iterations: [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]`
  (`configs/reference_v1/room_30k.yaml` line 48).
- `instrument_iterations: [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]`
  (C49/C50/C53) (line 51).
- R0.1 continuation checkpoints: 2000, 5000, 10000, 14000 (a 14K run, not 30K)
  (`reports/phase-r0.1-evidence-correction.md` §2.1).

---

## 2. Datasets — the 13 scenes

Scene set defined in `c42_13scene_analysis.py` (SCENES dict, lines 31–36) and
`data/datasets/official_training_datasets.json` (official_sources).

| # | Dataset family | Scene id | Native res | Cameras | Gaussians | License |
|---|---|---|---|---|---|---|
| 1 | Mip-NeRF 360 | bicycle | 4946×3286 | 281 | 6,131,954 | CC BY-NC 4.0 |
| 2 | Mip-NeRF 360 | flowers | — | — | — | CC BY-NC 4.0 |
| 3 | Mip-NeRF 360 | garden | 4946×3286 | 250 | 5,834,784 | CC BY-NC 4.0 |
| 4 | Mip-NeRF 360 | stump | — | — | — | CC BY-NC 4.0 |
| 5 | Mip-NeRF 360 | treehill | — | — | — | CC BY-NC 4.0 |
| 6 | Mip-NeRF 360 | room | 1944×1296 | 217 | 1,593,376 | CC BY-NC 4.0 |
| 7 | Mip-NeRF 360 | counter | — | — | — | CC BY-NC 4.0 |
| 8 | Mip-NeRF 360 | kitchen | — | — | — | CC BY-NC 4.0 |
| 9 | Mip-NeRF 360 | bonsai | — | — | — | CC BY-NC 4.0 |
| 10 | Tanks and Temples | truck | — | — | — | (Graphdeco tandt_db) |
| 11 | Tanks and Temples | train | — | — | — | (Graphdeco tandt_db) |
| 12 | Deep Blending | drjohnson | — | — | — | — |
| 13 | Deep Blending | playroom | — | — | — | — |

Sources for metadata:
- 13-scene list: `c42_13scene_analysis.py` lines 31–36;
  `data/datasets/official_training_datasets.json` official_sources.scene_ids
  (Mip-NeRF 360 lists bicycle, bonsai, counter, garden, kitchen, room, stump —
  flowers/treehill are added in the 13-scene benchmark, see §6).
- Gaussian/camera/resolution for bicycle, garden, room:
  `configs/epic05/official_validation_matrix.json` official_cohort.
- License: `configs/epic05/official_validation_matrix.json` per-scene `license`.
- Tanks&Temples + Deep Blending share Graphdeco `tandt_db.zip` bundle
  (682,628,995 bytes, SHA-256
  `816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1`)
  (`docs/datasets.md` line 19; `reports/dataset_report.md` line 12).
- Official sources URLs: Mip-NeRF 360 jonbarron.info/mipnerf360; Tanks and
  Temples tanksandtemples.org/download; Deep Blending via the Graphdeco
  tandt_db.zip URL (`data/datasets/official_training_datasets.json`).
- Tier A canonical 1080p cases (5): Garden, Truck, Train, Bicycle, Bonsai
  (`docs/benchmark_suite.md`; `docs/datasets.md` suite v3.1.0).
- Checkpoint hashes for the 5 canonical cases:
  `reports/dataset_report.md` table (garden
  `16701d5e...`, truck `65ecf405...`, train `70e055d4...`, bicycle
  `64d357cb...`, bonsai `a16af6d8...`).

---

## 3. Hardware cohorts

Every run records GPU model/UUID/VRAM, CPU, RAM, OS, driver, CUDA, Python,
PyTorch, renderer commit, build command, benchmark commit, power limit, clock
policy; "Any difference creates a separate cohort unless the suite explicitly
declares it irrelevant" (`docs/protocol.md` lines 43–46; `docs/hardware.md`
lines 6–15).

Cross-cohort rule: "Results from different GPUs never share a primary ranking"
(`docs/hardware.md` line 4). "All new results are explicitly tagged … and are
never mixed with synthetic stress benchmarks" (`README.md` lines 173–175).
"Direct cross-hardware speed comparison between old and new A100 is NOT VALID"
(`reports/phase-a100/environment_fingerprint.md` lines 100–105).
"Absolute wall times are never compared across hardware pools"
(`paper/confirmatory-protocol.md` §9, line 132).

### Cohort A — EPIC-05 (primary Tier A host)
Source: `reports/machine_report.md`; `reports/reproducibility.md`.
- Host: EPIC-05
- GPU: 8 × NVIDIA A100-SXM4-80GB, compute capability 8.0
- Selected GPU: physical GPU 2, UUID `GPU-12b6b703-727b-0c6e-1433-4e6161c54938`
- VRAM: 81,920 MiB (nvidia-smi) / 81,152.8 MiB (PyTorch)
- Power limit: 400 W
- CPU: 2 × Intel Xeon Platinum 8369B @ 2.90 GHz, 64 physical / 128 logical
- RAM: ~1.97 TiB (2,163,293,777,920 bytes)
- OS: Ubuntu 22.04.5 LTS, kernel `5.10.134-16.3.al8.x86_64`
- Driver: 580.105.08
- CUDA toolkit (build): 12.9 (`nvcc 12.9.86`); PyTorch CUDA runtime: 12.8
- Python: 3.10.20
- PyTorch: 2.9.1+cu128
- Benchmark commit: `dc9bb4e9231ae2fdf90fa9c40bcd6e0dbd7d104f`; protocol SHA-256
  `892e18890501c408dc6746af69f17e16973604f0c02a3caddf1954d3bf1fede2`
- 4 conda envs: original3dgs, gsplat, speedy, tcgs

### Cohort B — mx / bms-39468022-001 (A100 PCIe 40GB; Reference V1 host)
Source: `reports/phase-a100/environment_fingerprint.md`;
`reports/phase-a100/current_environment_fingerprint.md`;
`reports/phase-a100/experiment_protocol.md`.
- Host: mx (bms-39468022-001)
- GPU: 8 × NVIDIA A100-PCIE-40GB, compute capability 8.0
- VRAM: 40,960 MiB / 40,441 MiB per GPU
- Power limit: 250 W
- OS: Ubuntu 22.04.4 LTS, kernel `5.15.0-181-generic`
- Driver: 595.71.05; CUDA driver 13.2
- CUDA runtime (conda): 11.8; NVCC: 12.4.131
- Python: 3.10.19 / 3.10.12
- PyTorch: 2.7.1+cu118 (current); 2.4.1+cu124 (anysplat env, 2026-09-04)
- gsplat: 1.5.3 (pip wheel, JIT)
- Reference V1 provenance env: gsplat 1.5.3, torch 2.7.1+cu118, CUDA 11.8, A100-PCIE-40GB
  (`reports/reference-v1-baseline-lock.md` line 233)

### Cohort C — RTX 5070 Laptop (historical + official validation)
Source: `reports/archive/windows-rtx5070-2026-07/generated/machine_environment.json`;
`configs/epic05/official_validation_matrix.json`; `paper/confirmatory-protocol.md` §7, §9.
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU, compute capability 12.0
- VRAM: 8,151 MiB (8.15 GB)
- OS: Windows-10-10.0.26200-SP0
- Driver: 592.01
- CUDA: 13.0; PyTorch CUDA: 13.0
- PyTorch: 2.12.1+cu130
- Python: 3.10.20
- Commit at archive: `b46e8f27fbc3beea89a12f25c35ce8b296f24cd9`
- Note: could not fit frozen HiGS 30K workload (8.15 GB VRAM); 720p leg moved
  to EPIC-05 A100 (`paper/confirmatory-protocol.md` §9).

### Cohort D — RTX 4090 (reference host template, not measured)
Source: `docs/hardware.md` lines 19–32.
- Template `hardware_profile_id: rtx4090-linux-cuda12`
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, 450 W, clock_policy locked
- RAM: 65,536 MB; OS: Ubuntu 24.04
- (This is a template profile; no measured results are published on it.)

---

## 4. Round / phase numbering convention

Three parallel numbering schemes are used in the project.

### 4.1 EPIC-05 optimization phases
- EPIC-05 = the 8×A100-SXM4-80GB host, also the name of the rendering
  optimization study (`configs/epic05/optimization_matrix.json`;
  `docs/epic05/optimization-protocol.md`).
- Optimization modules M0–M6 (tile, packed, SH, radius_clip, eps2d, block size)
  and interaction experiments I1–I10
  (`configs/epic05/optimization_matrix.json` ablation_modules,
  interaction_experiments).
- Phase 3 = official real-scene validation using Mip-NeRF 360 pretrained
  checkpoints (`docs/epic05/optimization-protocol.md` §"Phase 3";
  `README.md` "Official Dataset Validation (EPIC-05 Phase 3)").

### 4.2 C-numbers (mechanism discovery phases)
Report headers: `Phase C0 — Canonical 3DGS Semantic Parity Audit`;
`Phase C17`, `C42`–`C53`, `C51 Stage 4A/4B/5`, `C52 Stage 0`
(`reports/phase-c0-canonical-parity-audit.md`; `reports/phase-c44...`,
`phase-c49...`, `phase-c50...`, `phase-c51...`, `phase-c52...`,
`phase-c53...`). C0 is the audit that identified 11 semantic deviations later
corrected in Reference V1. C49 = gradient concentration; C50 = gradient
temporal predictability; C51 = sparse backward; C52 = predictive density
control; C53 = computational utility / workload persistence.

### 4.3 R-numbers (validation / certificate decision gates)
Report headers: `Phase R0.1 — Evidence Protocol Correction + C51 Go/No-Go`;
`R0.2`, `R0.3`, `R1`, `R2`, `R2.1`, `R3` (`reports/phase-r0.1...`,
`phase-r1...`, `phase-r2...`, `phase-r3...`). R3 decision criteria are
CR1–CR4 (`reports/r3_1/decision_protocol.md`): CR1 correctness (zero
violations), CR2 JOINT weighted-work removal (≥30% keep / 10–30% modify /
<10% drop), CR3 signal coverage (≥3 windows), CR4 SPD certifiability
(<10% disabled). Final decision C_KEEP / C_MODIFY / C_DROP.

### 4.4 Round N (sequential HiGS implementation rounds)
- Rounds 1–65 in `reports/higs-trainability-implementation.md` (Round 13 …
  Round 30) and `reports/higs-training-speedup-research-2026-08-03.md`
  (Rounds 31–65). Round 30 = EPIC-05 environment recovery; Round 65 =
  progressive-resolution × decay quality-max unit. The confirmatory protocol
  freezes the "round-65 final operating points"
  (`paper/confirmatory-protocol.md` §2).
- Mechanism milestones M1–M6 (M2 done Round 31; M4 partial → Round 41d;
  M5 Round 42; M6 Rounds 51–58) (`reports/higs-training-speedup-research...`).

---

## 5. Key cross-references

Canonical definitions live at:
- Protocol (hashable): `benchmark/protocol.json` (matrix-primary-1080p-v2);
  explained in `docs/protocol.md` ("the authoritative, hashable protocol is
  benchmark/protocol.json").
- Suite: `benchmark/suite.json` (3dgs-renderer-matrix v3.1.0);
  `docs/benchmark_suite.md`.
- Reference V1 config: `configs/reference_v1/room_30k.yaml`; lock report
  `reports/reference-v1-baseline-lock.md`; corrected baseline
  `reports/phase-r0.1-evidence-correction.md` (tag
  `baseline/reference-v1-absgrad`, commit `32ab80e`).
- HiGS paper protocol: `benchmark/higs-paper-protocol.json` (iterations 30000,
  seeds [0,1,2], eval_split official); submission design `docs/higs-paper-plan.md`.
- Confirmatory protocol: `paper/confirmatory-protocol.md` (seeds 3/4/5,
  30k steps, gates).
- Hardware cohort rules: `docs/hardware.md`; cohort inventory
  `reports/machine_report.md`, `reports/reproducibility.md`.
- Dataset policy: `data/datasets/official_training_datasets.json`;
  `docs/official_dataset_training.md`; canonical asset hashes
  `reports/dataset_report.md`; suite `docs/datasets.md`.

Token budget / iteration counts defined at:
- Training iteration horizon: 30,000 — `benchmark/higs-paper-protocol.json`
  training.iterations; `configs/reference_v1/room_30k.yaml` iterations;
  `paper/confirmatory-protocol.md` §2 (`--steps` 30000);
  `benchmark/training.json` iterations.
- Eval cadence: `configs/reference_v1/room_30k.yaml` eval_iterations and
  instrument_iterations; confirmatory `--eval-every 300`
  (`paper/confirmatory-protocol.md`).
- (No file named "token budget" exists; the project expresses compute budget
  as iteration counts and measured frames/repeats —
  `benchmark/protocol.json` timing: 30 warmup, 100 measured, 5 repeats.)

---

## 6. Ambiguities and disagreements between docs

1. **λ naming.** `configs/reference_v1/room_30k.yaml` stores `lambda_dssim: 0.2`
   with no explicit (1−λ)·L1 form; the Graphdeco convention is
   (1−λ)·L1 + λ·D-SSIM. The two are equivalent given λ=0.2, but the config does
   not state the L1 coefficient.
2. **Densification threshold 0.0008 vs 0.0002.** Reference V1 uses 0.0008
   (gsplat absgrad) while official Graphdeco uses 0.0002 (signed gradient).
   This is a deliberate gsplat adaptation (A3), but the baseline is named
   "Reference V1" while R0.1 renames it `REFERENCE_V1_ABSGRAD` — the two names
   refer to the same frozen baseline
   (`reports/phase-r0.1-evidence-correction.md` lines 34–38).
3. **Seed conflict in `official_validation_matrix.json`.** The `baseline.seed`
   is 0 while the top-level `seed` is 42 (lines 19 vs 92). Which applies to the
   validation runs is ambiguous from this file alone.
4. **C50 persistence misattribution.** The baseline-lock report carries a
   correction banner: `persistence=0.861` was misattributed to C50 (gradient
   temporal persistence) in 3 places; it actually belongs to C53-Validation2
   (workload/tile persistence). C50 gradient persistence is near-zero
   (`reports/reference-v1-baseline-lock.md` lines 3–8, 206, 272).
5. **C49 gradient signal.** The R0 report's C49 measured G_dens
   (densification gradient `means2d.absgrad`), not G_opt (optimization
   per-parameter `.grad`); R0.1 measures both separately
   (`reports/phase-r0.1-evidence-correction.md` lines 16–17).
6. **13-scene set vs official_sources list.** `data/datasets/official_training_datasets.json`
   lists Mip-NeRF 360 scene_ids as [bicycle, bonsai, counter, garden, kitchen,
   room, stump] (7 scenes, omitting flowers and treehill). The 13-scene
   benchmark adds flowers and treehill (`c42_13scene_analysis.py`). The two
   sources disagree on whether flowers/treehill are official-training scenes.
7. **Hardware cohort naming.** The same physical A100-PCIE-40GB host is called
   "mx" in `reports/phase-a100/experiment_protocol.md` and
   "bms-39468022-001" in the fingerprint reports; the project's "cohort ID"
   for it is not stated as a single canonical string. EPIC-05 is both a host
   name and a study name.
8. **RTX 4090 template.** `docs/hardware.md` gives an RTX 4090 reference host
   template (`rtx4090-linux-cuda12`), but no measured cohort is published on
   it; it is a template, not a realized cohort.
