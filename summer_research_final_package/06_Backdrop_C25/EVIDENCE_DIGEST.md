# S5 — Backward-Pass Profiling & Optimization (Candidate C25 and successors)

**Scope.** Backward-pass profiling and optimization in this 3DGS-renderer-benchmark repo (project period 2026-07..2026-09). Focus: candidate C25 (sparse/selective backward), the "tiny fraction of Gaussians dominates backward cost" finding, the ~1.7 ms fixed floor, gradient sparsity (C26/C50), and the sparse-backward implementation chain (C51 / C51-R / C51-Stage4A / C51-Stage4B). Every claim cites a file path.

> Note on terminology: "T5′" is the original C25/C26 candidate name (sparse-tail backward reorganization via per-pixel termination compaction); "C25" is the screening phase. The predictive Gaussian-level sparse backward (C49→C50→C51→C51-R→Stage4A/B) is a *different* mechanism that grew out of the same backward-bottleneck motivation. Both are covered.

---

## 1. Iteration budget / time decomposition

The iteration decomposition is **strongly loss-protocol-dependent**. Three measurement regimes give very different backward shares. All measured on A100 PCIe 40 GB (SM80, 108 SMs), gsplat 1.5.3, CUDA 11.8, tile_size=16, Mip-NeRF360 scenes.

### 1a. C25/C26 — L1-style training iteration, steady state (A100)

CUDA-event segmentation, steady-state (reps 1–4, first-iter warmup excluded), cameras 0 & 80, 5 reps each:

| Segment | Mean (ms) | Share of T_iter |
|---|---|---|
| Forward | 2.68 | 28.9 % |
| **Backward** | **5.66** | **61.0 %** |
| Loss | 0.08 | 0.9 % |
| Optimizer update | 0.84 | 9.1 % |
| Other (sync/overhead) | ~0.00 | ~0.0 % |
| **T_iter** | **9.3** | 100 % |

Source: `reports/phase-c25/c25_training_candidate_screening.md` (candidate D) and `results/a100/c17-c33/phase-c25/c25_training_candidate_screening.json`.

C26's J3 candidate corroborates with a slightly different split: Forward 2.71 ms (30.7 %), Backward 6.10 ms (69.2 %), Optimizer 0.88 ms (10.0 %, of which SH update = 0.70 ms = 79.6 % of optimizer). Source: `reports/phase-c26/c26_candidate_tournament.md`.

### 1b. Pipeline-only breakdown (excludes loss/optimizer), `backward_candidate_analysis.md`

Averaged across 6 cameras/scene (CUDA-event timing, A100):

| Scene | N_Gs | Isect | Sort | Offset | Proj+Raster Fwd | Backward | Total |
|---|---|---|---|---|---|---|---|
| room | 1.59 M | 0.40 (2.8 %) | 0.62 (4.4 %) | 0.03 (0.2 %) | 2.37 (16.9 %) | 10.62 (75.7 %) | 14.03 |
| bicycle | 6.13 M | 0.63 (2.1 %) | 0.91 (3.1 %) | 0.05 (0.2 %) | 4.74 (16.2 %) | 22.91 (78.4 %) | 29.23 |
| garden | 1.84 M | 0.28 (2.5 %) | 0.44 (4.0 %) | 0.02 (0.2 %) | 1.93 (17.6 %) | 8.28 (75.6 %) | 10.95 |

Source: `reports/backward_candidate_analysis.md`. Backward dominates 75–78 % of the renderer pipeline across all scenes; sort (C17 target) is only 3–4 %.

### 1c. C43 — torch.profiler breakdown **with SSIM loss** (A100, room, 1.53 M Gs)

This is the critical reconciliation: when the SSIM loss is included, the picture inverts.

| Stage | Time (ms) | % of T_iter |
|---|---|---|
| SSIM loss | 75.85 | **77.2 %** ← true bottleneck |
| Backward (total) | 17.87 | 18.2 % |
| ├ rasterize_bwd_kernel | 6.587 | 6.7 % (63.5 % of backward) |
| ├ memcpy DtoD (grad buf init) | 1.445 | 1.5 % (13.9 % of backward) |
| ├ vectorized_elementwise | 0.974 | 1.0 % |
| ├ spherical_harmonics_bwd | 0.276 | 0.3 % |
| ├ projection_ewa_3dgs_fused_bwd | 0.249 | 0.3 % |
| └ other elementwise | ~0.84 | 0.9 % |
| Render (fwd rasterize) | 4.57 | 4.6 % |
| **T_iter** | **98.33** | 100 % |

Source: `reports/phase_c43_backward_optimization_report.md` (Track C). The backward *rasterizer kernel* is only 6.7 % of a full SSIM iteration; the entire backward is 18.2 %. "Even completely eliminating [the rasterizer kernel] would yield only +6.7 % e2e." C42 SSIM-downscale (scale 0.75 → +32.9 % e2e; scale 0.5 → +59 %) is the only optimization clearing the 5 % e2e threshold in this regime.

### 1d. C51 Stage4A/4B — L1-only / canonical training (A100, room)

Stage4A (L1-only, 5K): backward kernel ≈ 50 % of T_iter (13 ms / 17 ms); speedup transfer ratio (E2E/kernel) ≈ 63–66 %. Source: `reports/phase-c51-stage4a.md`.
Stage4B (canonical L1+D-SSIM, room 30K): Fwd+Bwd 49.49 ms, Optimizer 7.31 ms, Densify 0.11 ms, Mask 0.68 ms, **Total 57.25 ms**. Source: `reports/phase-c51-stage4b.md`.

**Reconciliation.** The backward-as-dominant-bottleneck narrative (61–69 %) holds under L1-style / render-pipeline-only timing; under the canonical SSIM-bearing iteration the backward rasterizer kernel is ~6.7 % and SSIM is the true bottleneck. The realizable ceiling of any backward-only optimization therefore depends heavily on the loss protocol and on whether C42-style SSIM downscaling is already applied.

---

## 2. Backward-pass structure

Backward = 4 CUDA kernels launched in sequence by autograd (`loss.backward()`). Source: `reports/epic05/phase8c_backward_path_trace.md`, `reports/epic05/phase15_backward_dataflow_audit.md`.

| Kernel | Grid / Block | tile-dependent? | Cost share |
|---|---|---|---|
| `rasterize_to_pixels_3dgs_bwd_kernel` | I × tile_h × tile_w / tile_size² | YES (grid & block) | ~63–97 % of backward |
| `spherical_harmonics_bwd` | ceil(nnz/256) / 256 | No | <1–2.7 % |
| `projection_ewa_3dgs_fused_bwd` | ceil(N/256) / 256 | No | <1–2.4 % |
| `quat_scale_to_covar_preci_bwd` | ceil(N/256) / 256 | No | <1 % |

For 1080p, tile16: grid = 1 × 68 × 120 = 8160 blocks, 256 threads/block. The backward kernel binary is **identical** for tile16/tile32 (cuobjdump-verified: REG=40, SHARED=1024, STACK=0); only launch dims differ. Source: `phase8c_backward_path_trace.md`.

**Traversal.** Back-to-front (reverse of forward). Each block = one tile; each thread = one pixel. Per batch: cooperative load of Gaussians into shared memory → compute alpha → update transmittance T (`T *= 1/(1−alpha)`) and buffer (`buffer[k] += rgb[k]·alpha·T`) → compute gradients v_rgb/v_conic/v_xy/v_opacity → warpSum reduce → warp leader atomicAdd. Source: `results/a100/phase-c51/source_audit.json`, `reports/phase_c51_sparse_backward.md` §3.

**atomicAdd usage.** ~10 atomic adds per valid Gaussian-pixel pair: `v_rgb[CDIM=3]`, `v_conic[3]`, `v_xy[2]`, `v_opacity[1]`. warpSum reduces across 32 threads first, then one thread/warp writes (32× contention reduction). A large Gaussian visible in many tiles still receives concurrent atomicAdds from many pixel-warps. Source: `phase8c_backward_path_trace.md` §1.4, `phase15_backward_dataflow_audit.md` §2.3.

**Shared memory.** tile16, CDIM=3: id_batch 256×4 + xy_opacity 256×12 + conic 256×12 + rgbs 256×12 = **10 240 bytes** (10 KB), within A100's 48 KB default. Backward shared mem > forward (forward lacks rgbs_batch). tile32 → ~20 KB. Source: `phase15_backward_dataflow_audit.md` §2.2, `backward_candidate_analysis.md` §4.

**Existing skip mechanisms** (pre-C51): `last_ids[pix_id]` (skip Gaussians beyond last forward contributor — occluded), `ALPHA_THRESHOLD` (1/255, skip near-transparent), `warp.any(valid)` (warp-level early exit). Source: `source_audit.json`.

**Data dependencies.** Backward re-reads all forward-saved tensors (`means2d`, `conics`, `colors`, `opacities`, `tile_offsets`, `flatten_ids`, `render_alphas`, `last_ids`); no intersection rebuild. Projection backward **recomputes** geometry (covar/mean_c/covar2d) from quats/scales — memory-optimal (storing 3D matrices would cost ~27 floats/Gs vs 8 currently). Source: `reports/phase-a100/backward_dependency_audit.md`, `phase15_backward_dataflow_audit.md` §3.

**Memcpy DtoD (1.45 ms, 13.9 % of backward)** is gradient-buffer zero-init — a candidate for fusion with the kernel prologue. Source: `phase_c43_backward_optimization_report.md` Track C.

---

## 3. C25 finding: 1 % of Gaussians → ~29 % of backward cost; the ~1.7 ms floor

### Measurement

Controlled Gaussian subsampling at fixed fractions (camera 5, 5 reps/config, A100), same camera, same sorted order, same tile geometry — only Gaussian count varied. Source: `reports/phase-c25/c25_training_candidate_screening.md` (T5′).

| Gaussians | Fraction | Fwd (ms) | Bwd (ms) | Bwd vs full |
|---|---|---|---|---|
| 1 593 376 | 100 % | 1.91 | 5.86 | 100 % |
| 796 688 | 50 % | 1.30 | 3.84 | 65.4 % |
| 398 344 | 25 % | 0.99 | 2.81 | 47.9 % |
| 159 337 | 10 % | 0.86 | 2.12 | 36.1 % |
| 79 668 | 5 % | 0.88 | 1.82 | 30.9 % |
| **15 933** | **1 %** | 0.74 | **1.69** | **28.8 %** |

**The claim:** 1 % of Gaussians still incurs ~29 % (28.8 %) of full backward cost. Reducing Gaussians 99× (1.6 M → 16 K) reduces backward time only 3.5×. Source: c25 report + `c25_training_candidate_screening.json` (verdict KEEP_CANDIDATE, `end_to_end_training_gain_pct_bound: 14`).

### What the ~1.7 ms fixed floor represents

The 1.69 ms residual at 1 % is a **large fixed-cost floor independent of active Gaussian count**, dominated by (c25 report): (1) kernel launch latency / grid scheduling; (2) fixed tile-traversal costs (tile iteration, sorted-position overhead regardless of per-pixel activity); (3) synchronization / global-memory writes for gradient accumulation that must complete. It is the cost of the backward rasterizer kernel's structural overhead, *not* proportional to active lanes.

### Is the floor scheduler overhead or wasted pixel work? (C26 falsification)

C26 replicated across 3 cameras/GPUs (cam5/GPU0, cam0/GPU6, cam1/GPU7), refining 1 % → ~20 % (19.5 %, 20.3 %, 21.4 %). A depth-tail sweep (Experiment B) divided sorted traversal into 6 depth intervals and measured terminated-pixel fraction:

| Sorted frac | cam5 terminated | cam0 | cam1 |
|---|---|---|---|
| 0–50 % | 46.8 % | 36.5 % | 37.4 % |
| 75–90 % | **97.6 %** | 96.9 % | 94.6 % |
| 90–95 % | 99.4 % | 98.4 % | 97.7 % |
| 99–100 % | 100.0 % | 100.0 % | 100.0 % |

**Beyond 75 % sorted depth, >95 % of per-pixel work is on already-terminated pixels.** Source: `reports/phase-c26/c26_candidate_tournament.md`. The falsification test concluded **Outcome 1**: the 1.7 ms floor is *sparse-tail cost* (wasted iteration over terminated pixel positions where gradient contribution is mathematically zero), **not** grid/scheduler overhead. The wasted work is concentrated at the *end* of each pixel's sorted range, where warp divergence is extreme (>95 % lanes inactive). This is "NOT standard sparsity" — not randomly distributed.

**Scenes:** Mip-NeRF360 room (1.59 M Gaussians, 1080p, 311 cameras); camera 5 for subsampling; cameras 0 & 80 for decomposition. C26 added cameras 0/1 on GPUs 6/7.

**Opportunity bound:** ~14 % T_iter (C25) / 14–20 % (C26). Mechanism: per-pixel active mask from forward `last_ids` + warp-level `__ballot_sync` compaction to skip terminated positions. Correctness risk LOW (skips only zero-gradient operations → gradients bit-identical on active positions). Implementation cost HIGH (~300 lines CUDA). Prior-art separation: pixel-level termination compaction not addressed by FastGS/Faster-GS/SkipGS/TileGS (all Gaussian- or view-level). T5′ was the sole **STRONG KEEP** survivor of 15 candidates (C24–C26). Sources: c25, c26 reports.

---

## 4. C26 sparsity analysis (and the C49/C50 gradient-sparsity bridge)

C26's sparsity finding is the depth-tail terminated-pixel sparsity above. A *separate* gradient-sparsity thread (C49→C50) established the basis for the predictive sparse backward:

- **C49** (referenced in C50): gradient distribution highly concentrated — top 50 % → ~97 % of gradient; top 32 % → ~90 %; top 10 % → 60 %. Oracle gradient filtering preserves quality (top 10 % → only −0.06 dB). Source: `reports/phase_c50_gradient_predictability.md` §1.
- **C50** temporal stability (room, 5000 iters, 4409 measurements): Pearson correlation of consecutive gradient norms ≈ 0.978–0.980; Spearman (ranking) ≈ 0.988–0.989; stable across early/middle/late phases. **Recall@32 % = 0.977**, Coverage@32 % = 0.912, Coverage@50 % = 0.973 using the *previous-iteration* gradient norm as predictor (available pre-backward). EMA smoothing **hurts** (beta=0.9 → 0.940; beta=0.99 → 0.888); opacity is a poor predictor (recall@32 % = 0.559). Decision: **KEEP → proceed to C51**. Source: `phase_c50_gradient_predictability.md`.
- **C51 multi-scene predictor validation:** room (recall 0.977) and bicycle (0.974) pass; **garden fails** (Coverage@32 % = 0.834 < 0.85, Coverage@50 % = 0.937 < 0.95) — outdoor scene has less concentrated gradient. Implication: K must be scene-adaptive; default K=50 % safer. Source: `reports/phase_c51_sparse_backward.md` §5.

---

## 5. Sparse-backward implementations (C51 / C51-R / Stage4A / Stage4B)

### Design space (C51 Stage 1 source audit)

Four CUDA designs audited against C49's validation contract. Source: `results/a100/phase-c51/source_audit.json`, `reports/phase_c51_sparse_backward.md` §3.

| Design | Mechanism | T/buffer correct? | Matches C49? | Est. speedup |
|---|---|---|---|---|
| A | Skip atomicAdd only | Yes | Yes | <5 % |
| **B** | Skip gradient compute, keep T/buffer | Yes | Yes | 10–25 % |
| C | Skip loading + alpha + T/buffer | **No** | **No** | 20–35 % (unvalidated) |
| D | Batch compaction (rebuild flatten_ids) | No | No | 30–50 % (complex) |

Critical finding: C49 validated Design A/B (zero output gradients *after* correct T/buffer), **not** C/D. Design B chosen as primary (mask test after alpha computation, before gradient arithmetic; T/buffer always update → important Gaussians' gradients stay exact).

### C51 (simulation, Python masking) — MODIFY

V1 (mask before densification) **broken**: clone count collapsed 83–86 % (58 814 → 8 308), PSNR −2.39 to −3.04 dB. Root cause: masked gradients fed `accumulate_positional_gradient()` → densification saw artificially low gradients. **V2 fix** (accumulate before mask): K50 −0.55 dB, K32 −0.73 dB, K20 −0.95 dB; gradient cosines >0.99 (xyz 0.993–0.995, others >0.998). **PSNR gate (<0.2 dB) FAILED** for all. Key finding: **gradient masking must be decoupled from densification**. Decision: MODIFY. Source: `reports/phase_c51_sparse_backward.md`.

### C51-R (error-controlled, simulation) — KEEP → CUDA

8 configs on 8 A100s. Track A (sparsity sensitivity): K80 −0.18 dB, K90 −0.14 dB (both pass 0.2 dB gate; cosines 1.0000). Track B (periodic refresh every 50–500 iters): **no improvement** — degradation is steady-state bias, not accumulating error (gap appears by iter 1000 and plateaus). Track C (post-densification K50 at 30K): **−0.01 dB** (essentially zero) vs C45 baseline. Recommended CUDA candidate: **K=80 %**. Decision: KEEP → proceed to CUDA Stage 4. Source: `reports/phase_c51r_sparse_backward.md`.

### C51 Stage4A (real CUDA implementation) — K50-B1 passes all gates

Modified 6 gsplat files (all backed up `.orig_stage4a`): `RasterizeToPixels3DGSBwd.cu` (mask into `mask_batch[tr]`; 3 gradient branches), `Rasterization.h/.cpp`, `Ops.h`, `_wrapper.py`, `rendering.py`. Three sub-designs: **B1** (skip all gradient compute for masked; sparse densification), **B2** (compute only v_means2d for masked → densification sees full xyz grad), **B3** (delayed-densification ablation, same kernel as B1). Mask = `uint8_t importance_mask[N]`, top-K by EMA of previous `||xyz.grad||` (decay=0.9, eps=1e-6).

Microbenchmark (1.59 M Gs, single camera): kernel speedup linear in skip fraction — K50-B1 = **10.2 %**, K80-B1 = 4.6 %. Gradient correctness exact: selected cosine = 1.000000 (xyz/scales/rotations/shs), 0.999990 (opacity); skipped max_abs = 0.0 (B1/B3); forward bit-exact (max diff = 0.0). B2 correctly leaves xyz grad non-zero for masked (max_abs 0.002).

5K training (7 configs): **Only K50-B1 passes all four gates** — Gate A (correctness) ✅, Gate B (−0.03 dB, −0.0027 SSIM) ✅, Gate C (**+6.7 % E2E**) ✅, Gate D (54 % clone / 68 % split) ✅. **B2 fails quality** (−0.49 dB): partial-update inconsistency (position updates without appearance updates). Non-obvious: "densification-safe" B2 is *worse* for quality than "freeze-everything" B1. 30K (K50-B1 vs baseline): +7.8 % speedup, **+0.59 dB** (better), clone 24.4 % / split 59.1 %. Source: `reports/phase-c51-stage4a.md`.

### C51 Stage4B (canonical validation, 3 scenes, 30K) — KEEP with caveats

Canonical config (0.8·L1 + 0.2·D-SSIM, Adam, C45 "moderate" pruning; `prune_and_reset` bug fixed). CUDA kernel frozen from Stage4A.

| Scene | Method | PSNR | SSIM | Time (ms) | E2E speedup | Final GS |
|---|---|---|---|---|---|---|
| Room | Baseline | 29.21 | 0.8845 | 57.25 | — | 2 155 755 |
| Room | K50-B1 | 30.24 | 0.8903 | 52.90 | **+8.2 %** | 1 505 069 |
| Bicycle | K50-B1 | 21.74 | 0.6424 | 96.82 | **+24.4 %** | 7 673 479 |
| Garden | K50-B1 | 23.10 | 0.6861 | 92.70 | **+28.9 %** | 7 331 782 |

All 3/3 scenes pass quality (no degradation; ΔSSIM actually *positive*) and speedup (>5 %) gates. Predictor: Recall@50 = 95.7 %, mask Jaccard 91.8 %, mask churn 4.3 %.

**Caveats (important):**
1. **Quality improvement is likely a densification side effect**, not sparse backward. K50-B1 has 30–36 % fewer Gaussians (clone ratio 14.9 % room → 73.1 % garden). 5K OracleDens ablation (preserve densification) eliminates the quality difference → hypothesis: fewer Gaussians → less overfitting.
2. **Speedup on outdoor scenes dominated by optimizer savings** from fewer Gaussians, not pure kernel skip. Room component breakdown: Fwd+Bwd −3.00 ms (69 % of saved), Optimizer −2.02 ms (47 %), Mask +0.68 ms (−15.6 %). On bicycle/garden optimizer savings (14.3–14.5 ms) *exceed* kernel savings (13.1–15.3 ms). Pure sparse-backward speedup ≈ 5 % room, 11–13 % outdoor.
3. **Post-densification mode shows NO speedup** (−0.1 %): after densification stops, gradient landscape becomes uniform; mask construction overhead (0.389 ms) exceeds kernel savings. Sparse backward is effective *only during* densification (iter 500–15000).
4. **B1's densification decoupling is inseparable** from the sparse-backward effect — it is a combined sparse-backward + densification-modification technique, not a standalone optimization.

Decision: **KEEP** (with caveats). Source: `reports/phase-c51-stage4b.md`.

---

## 6. Other backward optimization candidates tested

| Candidate | Phase | Verdict | Key evidence |
|---|---|---|---|
| B — Gradient workload reordering | C25 | DROP | 8 cameras, near-uniform access (p50 6–10 tiles/Gs, p99/p50 <60×); no heavy tail. Source: c25 report |
| C — F/B asymmetric tile policy | C25 | DROP | 6 cameras × 6 tiles × 5 reps; 0 % aggregate vs tile=16; true `tile_f≠tile_b` not testable via config. Source: c25 report |
| E — View-adaptive backward gating | C25 | DROP | 23 cameras; T_iter spread 1.13×, backward spread 1.15×; bwd-loss corr r=0.0003. Source: c25 report |
| I1 — Training-phase-aware renderer policy | C26 | KEEP | 7 phases; 3.4× T_iter range; cheap observable (nz tiles). 5–10 %. Source: c26 report |
| J3 — Async optimizer/renderer overlap | C26 | MAYBE | Optimizer 0.88 ms (10 %); 80 % is SH update; double-buffering 5–8 %, complexity high. Source: c26 report |
| F1 — Fwd→bwd state reuse | C26 | DROP | R=0.27 repeated work, all SH recomputation (~3 % of backward). Source: c26 report |
| G1 — Intersection representation redesign | C26 | DROP | 9.05 MB/iter, not a bottleneck. Source: c26 report |
| G2 — Sort→raster co-design | C26 | DROP | Sort 0.71 ms (8 %); correctness risk high. Source: c26 report |
| Track A — tile_size=8 (batch=64) | C43 | DROP | +26.2 % bwd but +64 % fwd regression → +3.7 % e2e (<5 %); tile32 infeasible on SM80 (register limit). Source: `phase_c43_backward_optimization_report.md` |
| Track B — Alpha/vis caching | C43 | DROP | exp() ≈ 16.7 % of bwd kernel → only 0.9–1.8 % e2e (SSIM dominates). Source: C43 report |
| Track D — Warp specialization | C43 | DROP | Kernel structure doesn't support it; bottleneck is exp() on SFU (shared across warps). Source: C43 report |
| Alpha caching | `backward_candidate_analysis.md` | NEED EVIDENCE | Potentially +15–23 %, memory trade-off (400 MB cache). (Superseded by C43's measured <2 %.) Source: `reports/backward_candidate_analysis.md` |
| Batch size increase | `backward_candidate_analysis.md` | ITERATE | +7–12 % est., low risk. (C43 measured +3.7 % → DROP.) Source: same |
| accel10/accel21 — HiGS SkipGS-style backward gating | epic05 (commits) | honest negative | No config passes all gates; segmented timing showed loss_compute ≈ 40–45 % of step cost; skipbwd_27k 1.21× speed but PSNR −0.25 fails. Sources: commits 96c495c, 0e047cb, 032fcea, 7eed9c0, e4deaff |

C43's bottom line: with SSIM loss, only C42 (SSIM downscale) clears the 5 % e2e threshold (+17 to +59 %); all backward-only optimizations fall below 3 % e2e standalone. Backward optimization becomes worthwhile only *combined* with C42 (at scale 0.5, backward = 35.8 % of total, alpha caching → ~5 % e2e).

---

## 7. Waiting for cache? Lost work? Redo invariants?

- **Phase 8B → 8C thermal-throttling correction (major redo).** Phase 8B claimed tile32-vs-tile16 backward ratios of **137×–248×** and "H7: backward sensitivity 10–30× larger than forward." Phase 8C re-measured on a cold GPU and found the real ratio is **2.6×–6.3×**, fully explained by the 4× fewer tile-Gaussian intersections (tile geometry). Root cause of the original extreme claim: GPU thermal throttling from running tile16 fwd+bwd for all 6 checkpoints sequentially without cooldown (bimodal ~875 ms ↔ ~2528 ms pattern). H7 **FALSIFIED**. Phase 8C/8D are authoritative. Source: `reports/epic05/phase8b_real_snapshot_fwdbwd.md` (§12 correction notice), `reports/epic05/phase8c_backward_workload_analysis.md`, `reports/epic05/phase8c_hypothesis_matrix.md`.
- **C51 V1 densification bug (implementation error, not fundamental).** Mask applied before `accumulate_positional_gradient()` → clone collapse (58 814 → 8 308), −2.39 dB. Fixed in V2 (accumulate before mask) → +1.4–2.1 dB recovery. Source: `reports/phase_c51_sparse_backward.md` §4.2–4.3, §6.
- **C51 Stage4B `prune_and_reset` shape bug.** Crashed when opacity 1D (from `load_ply`); `reset_val` created as 2D. Fixed with dimensionality check. Caused 5K quality collapse (opacity reset at iter 3000) → 5K ablation quality unreliable; only 30K numbers definitive. Source: `reports/phase-c51-stage4b.md` §1.2, §12.
- **accel21 skip-bwd pending-backward crash.** Commit 032fcea: added `cancel_pending_backward` to HiGS renderer to handle SkipGS per-view gating crash; full 66-job matrix rerun launched for uniform provenance. Source: commit 032fcea.
- **Nsight Compute BLOCKED** on both platforms: WDDM (RTX 5070 laptop) and ERR_NVGPUCTRPERM (A100 without root). nvprof unsupported on SM80. Memory/atomic contention quantification (H7-E) remains BLOCKED. Source: `phase8c_hypothesis_matrix.md`, `phase_c43_backward_optimization_report.md` Track C.

---

## 8. Unresolved questions

1. **Memory/atomic contention impact** (H7-E): hardware-counter measurement BLOCKED on both GPUs. Source-code analysis shows `gpuAtomicAdd` in the critical path but actual contention unquantified. Source: `phase8c_hypothesis_matrix.md`.
2. **Is the +dB quality improvement in Stage4B a true densification side effect?** Needs 30K OracleDens ablation (5K version supports the hypothesis but quality was collapsed). Source: `phase-c51-stage4b.md` §13.
3. **Garden-scene gradient coverage** (Coverage@32 % = 0.834) fails the predictability threshold — scene-adaptive K needed before production. Source: `phase_c51_sparse_backward.md` §5.
4. **Post-densification sparse backward is ineffective** — mechanism breaks down when gradient landscape is uniform. Confines the technique to the densification window (iter 500–15000). Source: `phase-c51-stage4b.md` §4.4, §7.
5. **Decomposition discrepancy:** backward = 61 % (L1, C25) vs 6.7 % rasterizer-kernel (SSIM, C43). The realizable ceiling of backward-only optimization depends on loss protocol and whether C42 SSIM-downscale is applied. Not yet resolved into a single canonical number. Source: §1 above.
6. **Why does forward scale superlinearly (4.66×) while backward scales near-linearly (1.07×)** with intersection count (iter5000→30000)? Asymmetry unexplained; may indicate different computational complexity w.r.t. tile density. Source: `phase8c_backward_workload_analysis.md` §5.
7. **Per-pixel termination compaction (T5′, the original C25 pixel-level mechanism) was never CUDA-implemented.** The implemented C51 chain is the *Gaussian-level* predictive sparse backward (different mechanism). The pixel-level T5′ prototype (active mask + `__ballot_sync` compaction) remains specified but unbuilt. Source: `reports/phase-c26/c26_candidate_tournament.md` "Recommended next steps."

---

## Commit hashes found (git log --all, grep c25|backward|sparse|latent)

No direct C25/C49/C50/C51 phase commits matched the grep; those reports appear committed under broader/batched feature commits. Backward/sparse-relevant commits:

- `0bae4b3` feat: Phase 8C backward path decomposition + workload analysis
- `0e047cb` feat(higs-ablation): accel21 per-view SkipGS backward-gating (honest negative)
- `032fcea` fix(higs-ablation): accel21 skip-bwd pending-backward crash
- `7eed9c0` feat(higs-ablation): pre-register accel21 SkipGS backward-gating
- `8ff4643` feat(higs-ablation): accel17 SH-progressive-schedule (SH3 eval+backward dominant cost)
- `96c495c` feat(higs-ablation): accel10 SkipGS-style backward-gating complete (honest negative)
- `e4deaff` feat(higs-ablation): pre-register accel10 backward-gating
- `64d4ef8` feat(higs-ablation): accel6 calibration-gated sparse-window (honest negative)
- `c16c252` feat(higs-ablation): pre-register Phase-8 scheduling-levers (accel8)
- `2f9b769` fix(higs-ablation): repair accel4 phase-split sparse trainer (stale-ssim double backward)
- `02daeb0` feat(higs-ablation): accel4 phase-split sparse exploration (honest negative)
- `30d6826` bench(higs): round-57 sparse-pixel rasterization (M6 close)
- `98f6dc9` bench(higs): round-39 backward tile compaction (blend grid over active tiles)
- `29e1988` docs(higs): round-28 macro-tile backward feasibility (lever closed)
- `9e6eb32` perf(higs): pixels-per-thread blend VJP + backward decomposition
- `574a998` perf(higs): fixed-grid SH VJP removes per-backward D2H sync
- `4df0b14` perf(higs): scatter master gradients in native backward kernels
- `f2b93de` feat(higs): native backward for depth render modes
- `73bebec` feat(higs): native CUDA backward, culling semantics, tests
- `6bfa646` feat: add _HigsAutogradFunction for native autograd backward (Stage B)
