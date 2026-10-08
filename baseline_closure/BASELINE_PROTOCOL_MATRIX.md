# BASELINE_PROTOCOL_MATRIX

Public-baseline freeze matrix for the P8 / P9 / P10 campaign stages.
Machine-readable source of truth: [`BASELINE_FREEZE.json`](BASELINE_FREEZE.json) (same content; this
document is the human-readable view). Freeze date: **2026-09-25**. Upstream HEADs verified via
`git ls-remote` on that date; local shallow clones under `public_baselines/`.

**Launch gate:** no performance run is started by this freeze. P7 residual-attribution closure
(another DSH) must complete first; then the clean-GPU snapshot convention applies (two scans
≥120 s apart, zero processes, <500 MiB) and the cohort policy (never mix A100-PCIE / A100-SXM4
cohorts; never mix torch/CUDA envs within one comparison).

**Label taxonomy** (strict; a row never carries a label it does not earn):

- **MATCHED_RENDERER** — all P8 controls literally satisfied; eps2d recorded and matched to the
  maximum technically possible.
- **MATCHED_RENDERER_WITH_DISCLOSURE** — unified matched trainer, but ≥1 disclosed irreducible
  deviation. Renderer-level comparisons allowed ONLY with the disclosure attached.
- **SYSTEM_LEVEL_CONTEXT** — the method's own training system preserved end-to-end. No
  renderer-level winner claims from these rows.

---

## 1. Identity and pinning

| field | Original Graphdeco 3DGS | current gsplat main | Taming 3DGS rasterizer-only | Faster-GS | FastGS | Speedy-Splat |
|---|---|---|---|---|---|---|
| arm id | `b0` | `gsplat_main` | `taming_rasterizer` | `faster_gs` | `fastgs` | `speedy_splat` |
| repository | graphdeco-inria/gaussian-splatting | nerfstudio-project/gsplat | nullptr81/3dgs-accel | nericg-project/faster-gaussian-splatting | fastgs/FastGS | j-alex-hanson/speedy-splat |
| commit SHA | `54c035f7834b564019656c3e3fcc3646292f727d` | `512d366b67073d77ca099ede742683c165dfc23b` | `1207926e75a601c5d5134ae60c988340ea5c23d0` | `117bdbacfd7df6c1f06d9d959b1f5ab02d4d3847` | `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f` | `34c45c6d9b8bd6110231864f2f358b6d3abbf73d` |
| branch | `main` (upstream unchanged) | `main` (v1.6.0) | `rasterizer` | `main` | `main` | `speedy-splat` (repo default) |
| submodules | diff-gaussian-rasterization `9c5c2028` (dr_aa), simple-knn `86710c2d`, fused-ssim `1272e21a` | none | rasterizer **in-tree**; fused-ssim gitlink `84422e0d` (SSH URL in .gitmodules → https on clone) | FasterGSCudaBackend **in-tree** (vendored) | rasterizer **in-tree** (`diff_gaussian_rasterization_fastgs`) | speedy-splat-rasterizer `b38abb6d`, simple-knn `44f76429`, SIBR_viewers `14199886` |
| prior pin / drift | P1-validated, unchanged | new arm (B1 lineage at v1.5.3 is the precedent) | never run in this repo | P5 pin `3cb0b75` → upstream moved; **re-gate required** | P5 cites `d4d33b6f` (c42-branch commit); native main pin `44e02a5` = this freeze; **re-gate required** | P5 pin `b9dd42d1` **no longer resolvable upstream (404)**; submodule gitlinks unchanged; **re-gate required** |
| local clone | `.c0-official-reference` (verified HEAD) | `public_baselines/gsplat` | `public_baselines/taming-3dgs-rasterizer` | `public_baselines/faster-gs` | `public_baselines/fastgs` | `public_baselines/speedy-splat` |

## 2. Environments

| field | b0 | gsplat_main | taming_rasterizer | faster_gs | fastgs | speedy_splat |
|---|---|---|---|---|---|---|
| native env (upstream) | py 3.7.13, torch 1.12.1, cu11.6 | torch ≥2.7 (cu126/cu128-class wheels); JIT or prebuilt; sm_80 covered | py 3.7.13, torch 1.12.1, cu11.6 | CUDA 12.8 recommended; NeRFICG conda env; validated: py 3.11, torch 2.7.1+cu118 (env `nerficg`) | py 3.7.13, torch 1.12.1, cu11.6 (tested nvcc 11.8); validated: py 3.10, torch 2.1.0+cu118 (env `strong_native`) | py 3.7.13, torch 1.12.1, cu11.6; validated: py 3.10, torch 2.1.0+cu118 (env `strong_native`) |
| P8 matched build | rebuilt vs higs-13scene-env (torch 2.9.1+cu128, nvcc 12.8.93) via `scripts/publication/b0_graphdeco_build.sh` | build under higs-13scene-env; record build identity | rebuild vs cu128 (unverified — build-system-only fixes allowed, math changes forbidden) | n/a (P9 only) | n/a (P9 only) | n/a (P9 only) |

## 3. Training semantics (upstream defaults unless a P8/P9 override is stated)

| field | b0 | gsplat_main | taming_rasterizer | faster_gs | fastgs | speedy_splat |
|---|---|---|---|---|---|---|
| training recipe | Inria: densify 500–15000, interval 100, signed-grad ≥2e-4, opacity reset 3000, prune opa<0.005, size thresh 20 | DefaultStrategy: refine 500–15000 every 100, grow_grad2d 2e-4 (absgrad off by default), reset_every 3000 → 0.01, prune_opa 0.005 | stock Inria on this branch (densification-scheme changes NOT released here) | NeRFICG: densify 600–14900, interval 100, thresh 2e-4, opacity reset 3000 (+extra 500), PPISP off, filter3d off, morton 5k–15k, MCMC off | densify 500–15000, **interval 500**, clone 2e-4, split abs-grad per-scene (0.0006–0.002), importance>5 gate, 50% prune-budget sampling, final prune (opa<0.1 ∨ score>0.9) every 3000 in (15k,30k), opacity clamp ≤0.8 | stock Inria densification + **score-based pruning**: from 6000 every 3000 prune 80% (during densify window), 30% after 15k, until 30k |
| densification / pruning statistic | signed L2-norm of xy viewspace grad, view-averaged | running mean of L2 norm of NDC-scaled 2D position grads (absgrad tensor if enabled) | same as b0 | L2 xy-grad norm accumulated inside CUDA backward, normalized by visibility count | dual: signed grad (clone) + abs grad (split), plus multi-view pixel-error importance counts (10 sampled views, loss_thresh 0.1) | Inria stat + extra backward pass per train camera accumulating per-Gaussian sensitivity scores |
| image resolution behavior | `-r -1` auto-1600px; full_eval: images_4 (outdoor) / images_2 (indoor) | `data_factor=4` (images_4) both train+eval | identical to b0 | IMAGE_SCALE_FACTOR 0.25 (MipNeRF360) / 1.0 (T&T/DB) | `-i images` + auto-1600px | train.sh: `-i images` + auto-1600px (README: higher-res than full_eval protocol) |
| loss definition | 0.8·L1 + 0.2·(1−SSIM), 11×11 σ1.5 | torch.lerp(L1, 1−SSIM, 0.2); fused_ssim if importable | identical to b0 (+ fused SSIM kernels in the same extension) | 0.8·L1 + 0.2·fused_dssim (deterministic) | 0.8·L1 + 0.2·(1−fused_ssim) (backward non-deterministic) | 0.8·L1 + 0.2·(1−SSIM), 11×11 |
| eps2d / antialiasing | hardcoded cov-diag **+0.3**; eigenvalue floor 0.1; antialiasing flag off | **eps2d=0.3** library default (eigenvalue form); trainer doesn't pass it; antialiased off; **P8 override: 0.1** to match B1A | hardcoded cov-diag **+0.3** (same as b0) | dilation 0.3 (0.1 for proper antialiasing); tile 16×12 | cov-diag **+0.3**; tile assignment = compact box `mult` 0.5/0.7 (<1 ⇒ **approximate** render) | cov-diag **+0.3**; SnugBox/AccuTile (exact) |
| optimizer | torch Adam lr=0 eps=1e-15, Inria groups; xyz expon 1.6e-4→1.6e-6 | Adam fused, lr·√BS, eps 1e-15/√BS, means ExponentialLR 0.01^(1/30k); means 1.6e-4, scales 5e-3, opacities 5e-2, quats 1e-3, sh0 2.5e-3, shN 1.25e-4 | same as b0; **sparse_adam NOT equivalent** (no decay for invisible, no bias correction) — excluded from matched arm | FusedAdam (custom CUDA); Inria-equivalent lrs; opacity 2.5e-2 | two torch Adams; step schedule 1/16→1/32→1/64; f_dc/f_rest split lrs; opacity 2.5e-2 | stock Inria Adam |
| default iterations | 30 000 | 30 000 (`max_steps`) | 30 000 | 30 000 | 30 000 | 30 000 |
| seed | safe_state seed 0; P8 uses `patches/original-3dgs-seed.patch` → 42 | set_random_seed(42+rank); train loader **shuffled** — frozen camera sequence imposed by matched trainer | seed 0 (upstream) | RANDOM_SEED 0 | seed 0 | seed 0 |
| evaluation protocol | render.py+metrics.py: PSNR, SSIM 11×11, **LPIPS-VGG** | eval @ [7000, 30000]; torchmetrics PSNR/SSIM; LPIPS alex(normalize=True) or vgg | stock render.py+metrics.py (LPIPS-VGG) | NeRFICG built-in; TEST_STEP 8; PSNR/SSIM/LPIPS torchmetrics; N parsed from train.log | train_base.sh → render.py (`--mult`) + metrics.py (LPIPS-VGG) | train.sh → metrics (LPIPS-VGG) + compute_scene_metrics.py (FPS from kernel_times, 20 repeats) |
| final Gaussian count (prior evidence) | P1 13-scene frozen run trees (`aggregates/p1_b0_vs_b1a.json`); N/B1A spans 0.72×–1.54× | TBD (P8 produces) | none — TBD | c42 cohort native total 24.10 M; range 767 K–3.29 M | c42 cohort native total 4.04 M; P5 range 132 K–716 K | c42 cohort native total 3.20 M; P5 range 81 K–512 K (bicycle/kitchen walls missing in P5 ledger) |
| classification | **MATCHED_RENDERER_WITH_DISCLOSURE** | **MATCHED_RENDERER** (at eps2d=0.1) | **MATCHED_RENDERER_WITH_DISCLOSURE** (conditional on gradient-equivalence gate) | **SYSTEM_LEVEL_CONTEXT** | **SYSTEM_LEVEL_CONTEXT** | **SYSTEM_LEVEL_CONTEXT** |

## 4. Disclosed deviations (travel with every citation of the row)

| arm | deviation | why irreducible |
|---|---|---|
| b0 | densification statistic = original signed-grad 2e-4 (not absgrad 8e-4) | forcing absgrad on B0 would not be B0 (wrapper policy, `reports/publication/matched-baselines.md`) |
| b0, taming_rasterizer | 2D filter = hardcoded cov-diag +0.3, not parameterizable eps2d | graphdeco-family rasterizer API has no eps2d knob |
| taming_rasterizer | quaternion normalization removed (fwd) + dnormvdv commented out (bwd); separate dc/sh API; sampled-state buffers | upstream design; exactness claimed by the authors, verified by the P8 pre-gate, never assumed |
| gsplat_main | none (v1.6.0 restructure vs v1.5.3 handled by a new trainer arm + 200-iter smoke) | — |
| faster_gs | densification statistic computed inside the custom backend; window [600, 14900] | the backend is the method |
| fastgs | pruning-budget densification; compact-box `mult`<1 truncates contributions (approximate render); fused-SSIM backward non-determinism | the method's design point |
| speedy_splat | pruning integrated in the training loop; custom rasterizer fork (exact math, extra outputs) | the pruning IS the method |

## 5. eps2d cross-family note

Two filter parameterizations coexist and cannot be equated by argument: gsplat-family
(eigenvalue-form eps2d: B1A 0.1, C0/A0 0.3, gsplat_main default 0.3) and graphdeco-family
(hardcoded cov-diag +0.3: b0, taming, fastgs, speedy). P4 (`reports/publication/eps2d-sensitivity.md`,
verdict MATERIAL_CONFOUND on the quality axis) governs the 0.1-vs-0.3 axis inside the gsplat
family. Every P8 table states eps2d per arm; the frozen B1A(0.1)/C0(0.3) asymmetry keeps its P4
disclosure and is never described as eps2d-matched.

## 6. What this freeze does NOT do

- No run, gate, or timing is launched (P7 gate; see `BASELINE_FREEZE.json.launch_gate`).
- C0 / A0 / B1A frozen binaries are untouched; internal arms are re-frozen by pointer to
  `reports/higs/authoritative-experiment-registry.md`.
- H8-MR is not redesigned; P10 (`P10_COMPOSITION_PLAN.md`) is port/adaptor + correctness prep only.
- The P5/c42 garden data caveat (manually-created COLMAP) is not retro-fixed: old evidence keeps
  its caveat; P9 adapters point at the canonical 13-scene layout going forward.
