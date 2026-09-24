# S3 — Tile Geometry / Tile Size, Occupancy, Block Geometry, Kernel Behavior

*Evidence digest for the summer research project on 3DGS rendering performance.*
*Working dir: `C:\Users\36570\3dgs-renderer-benchmark`.* All paths are repo-relative.

---

## 1. Research Questions Addressed (RQs)

The tile-geometry work across EPIC-05 and the `phase-cNN` candidate series addressed these RQs (consolidated from `reports/epic05/hardware-aware-tile-study-2026-08-19.md` §13, `reports/epic05/tile-mechanism-training-study-2026-08-19.md` §1, `reports/phase-c19/c19-1_rasterizer_mechanism.md`, `reports/epic05/phase13b_expanded_tile_sweep.md` §7):

- **RQ-T1** Does a universal optimal `tile_size` exist? (No — hardware- and scene-dependent.)
- **RQ-T2** Can tile-size selection be modeled as a hardware/workload-aware optimization? (Partially, n=2 GPUs.)
- **RQ-T3** Why does tile32 dominate on A100 (1.42–3.93×) but tile16 on RTX 5070 Laptop (tile32 = 0.67–0.83×)? What is the causal chain `tile_size → block geometry → occupancy/stalls → kernel runtime`?
- **RQ-T4** Is the per-block resource footprint (registers/shared memory) the cause, or the launch geometry (threads/block)?
- **RQ-T5** Does `tile_size` affect correctness/quality (forward pixels, gradients, training PSNR)?
- **RQ-T6** Does the snapshot (inference) optimal tile predict the full-training optimal tile?
- **RQ-T7** Is there an interior optimum between 16 and 32, and is the space >32 usable?

---

## 2. Candidate History (every candidate touching tile geometry)

### C1 — Sort-mechanism verification
*`reports/phase-c17-c2/c1_sort_mechanism_verification.md`.* Confirmed gsplat sorts by `isect_id = image_id | tile_id | depth` (64-bit key, ~46–47 bits used). Established that the sort key embeds tile geometry; tile size changes the tile-bit width and total tile count. Foundational for C17-0.

### C17-0 — Tile-segmented depth-only sort
*`reports/a100_validation/c17_0_tile_segmented_sort.md`; `patches/c17_0_additions.cu`.*
- **Hypothesis:** replacing the global CUB 46-bit radix sort with per-tile 32-bit depth sort (histogram + scan + counting-sort scatter + `DeviceSegmentedRadixSort`) reduces passes 6→4 and is faster.
- **Experiment:** 6 cameras × 3 scenes on A100-PCIE-40GB.
- **Result:** sort **2.5× slower** (room 1.14→2.87 ms; E2E −44.7%). `isect_ids` match 100%; `flatten_ids` differ only at same-depth ties (non-blocking). Counting-sort `atomicAdd` scatter (~1.3 ms) is the bottleneck.
- **Decision: DROP.** Counting sort is unsuitable; CUB global sort is already optimal. Points to **C17-1** (fused in-tile sort in shared memory).

### C17-1 — Tile-local intersection prototype
*`third_party_patches/c17_1/c17_1_tile_local.cu.inc`; commit `b562562` "feat: add C17 tile-local intersection prototype"; `reports/phase-a100/c17_2_data_integrity_cross_tile_audit.md`.*
- **Hypothesis:** fuse the per-tile sort into the Pass-2 CTA (block-wide radix sort in shared memory, 32-bit depth only), eliminating the global sort and the atomic scatter.
- **Experiment/Result:** prototype compiled; correctness of intersection encoding audited. Evolves into the C17-2 cross-tile differential membership redesign.
- **Decision:** CONTINUE → C17-2 redesign (see below).

### C17-2 — Data integrity + cross-tile overlap audit
*`reports/phase-a100/c17_2_data_integrity_cross_tile_audit.md`; `results/a100/phase-a100/c17_2_data_integrity_cross_tile_audit.json`.*
- **Hypothesis:** within-tile membership has exploitable structure (delta locality, run-length) for compression; cross-tile overlap enables differential encoding.
- **Result:** Within-tile delta/run-length **falsified** (median |Δ|=26–745, max run 2–3; effectively random in Gaussian-ID space). Cross-tile overlap **substantial** (horizontal Jaccard P50 0.43–0.62) and **order-consistent (100%)**. Reconciled the apparent n_isect discrepancy (174M vs 1.6M) to *different Gaussian sources* — trained checkpoint (large radii, ~158 tiles/G) vs SfM PLY (~9 tiles/G) — not a bug.
- **Decision:** CONTINUE WITH REDESIGN → "cross-tile differential membership."

### C19-0 — A100 rasterizer profiling gate
*`reports/phase-c19/c19-0_a100_rasterizer_profiling.md`.*
- Established the **canonical** A100 baseline (official Mip-NeRF 360 room ckpt, `packed=False`, 1.59M Gs): forward 3.08 ms; `rasterize_to_pixels_3dgs_fwd_kernel` = 55.5% of forward CUDA; CUB sort 9.9%; pass1 10.6%. Tile workload uniform (all 8160 tiles active, max/mean 1.5×). **PAT (Parallel-Adaptive Tile Subdivision) NO-GO** — no imbalance to correct.

### C19-1 — Rasterizer mechanism characterization
*`reports/phase-c19/c19-1_rasterizer_mechanism.md`; `results/a100/c17-c33/phase-c19/c19-1_tile_scaling.json`.*
- 7-point sweep (8/12/16/20/24/28/32) on A100, room. **Per-intersection cost rises 8.7× (t8→t32: 0.320→2.791 ns/int) while ints/tile rises only 1.6×** — structurally non-linear. **Abrupt 3.1× jump at t16→t20** (199→217 ints/tile). Diagnosed (indirectly) as register-pressure-primary; proposed H1 (register-aware tile subdivision), H2 (per-tile budget gating), H3 (WMMA). Diagnosed the old "CUB sort dominates 65%" baseline as an artifact of a non-canonical checkpoint (1.1M Gs, `packed=True`, 158 ints/G).

### C19-2 — Rasterizer isolation & resource-pressure gate ★ DECISIVE
*`reports/phase-c19/c19-2_rasterizer_isolation.md`; `results/a100/c17-c33/phase-c19/c19-2_block_geometry_control.json`.*
- **Controlled replay** with block fixed at 16×16 (256 threads, 7168 B shmem), varying only ints/tile 100→320: ns/int **decreases** (0.40→0.29); only a mild 1.6× jump at 205 ints. The canonical 3.1× cliff does **not** reproduce.
- **Decision: H1 NO-GO.** Register pressure is **not** the primary mechanism. Primary = **occupancy collapse from increasing block dimensions** (threads/block 64→1024; blocks/SM drops; ns/int correlates with threads/block R²≈0.93). Proposes **C19-3** occupancy-aware fixed-block (16×16) rasterization of larger tiles — projected 5–15× faster at t32.

### C25 — Asymmetric tile policy (per-camera adaptive)
*`results/a100/c17-c33/phase-c25/c_asymmetric_tile.json`; `scripts/phase-c25/c_asymmetric_tile.py`.*
- **Hypothesis:** per-camera tile selection beats a fixed tile.
- **Experiment:** 6 cameras, room, A100, tiles 8/12/16/20/24/28. Aggregate best fixed = tile16 (6.63 ms); per-camera best varies (cam0→tile28, cam3→tile16). Speedup over best-fixed = **0.0%** (verdict MAYBE). Intersection distribution is extremely heavy-tailed per camera (P99 0.48M–1.04M vs P50 1–134K).

### C27 — Parallel mechanism validation (tile-related: I1, T5', H)
*`reports/phase-c27/c27_parallel_mechanism_validation.md`; `results/a100/c17-c33/phase-c27/c27_parallel_mechanism_validation.json`.*
- **I1 (training-phase-aware renderer policy) KEEP:** tile24 beats tile16 by 12–15% in mid-training (3K–20K iters) on 2 cameras/2 GPUs; predictor = non-zero tile count. **T5' (depth-tail backward compaction) STRONG KEEP** (12–22%): 24.7% of backward sorted positions are in the 75–100% depth tail where 98.1% of work is on terminated pixels. **H (Gaussian lifecycle) KEEP backup:** top 10% visible Gs = 61.8% of intersection cost; cost–size correlation 0.68.

### C35 — Cross-renderer gate (tile-format coupling)
*`reports/rtx5070/c35_cross_renderer_gate.md`.*
- **C35-1 (adaptive hybrid blend) DROP:** no sparse/dense bimodality (render time CV=0.05, fixed-overhead dominated). **C35-2 (fwd/bwd decoupling) MAYBE:** projection+intersection shared across backends but per-pixel blend state incompatible (CUDA saves `last_ids`, PyTorch saves full graph). **C35-3 (logical/physical ID separation) KEEP:** solves ID-shift invalidation after prune.

### C36 — Cross-renderer architecture gate
*`reports/rtx5070/c36_architecture_gate.md`.*
- **R1 (canonical execution IR) MAYBE:** projection state (means2d/conics/depths/radii) shared by gsplat+HiGS (2/3); Inria fused kernel exposes nothing. **R2 (phase-specific handoff) KEEP:** gsplat↔Inria checkpoint handoff feasible (identical params, camera adapter, identical L1+D-SSIM loss); SH clamp differs (not bit-exact but training-compatible). **R3 (component substitution) DROP:** intersection+rasterization tightly coupled to tile format (gsplat 16×16 fine tiles, HiGS 128×128 macro-tiles = 8×8 fine tiles, Inria fully fused); no cross-backend component substitutable.

### C37 — Cross-renderer common mechanism
*`reports/rtx5070/c37_common_mechanism.md`.*
- **C37-A (alpha/contribution work amplification) DROP:** ~395× amplification inherent to tile architecture; best savings <0.3 ms (<0.3% T_iter). ~54% of per-tile pixel checks wasted (elliptical footprint vs square tile). **C37-B (intersection/sort amplification) DROP:** sort = 0.25 ms (0.2% T_iter). **C37-C (training-state reuse) KEEP:** >99.9% of projected state unchanged per step (Δparams 0.018%/step → <0.05 px error); copy 0.01 ms vs recompute 1.7 ms (170×).

### C42 / C43 — Adaptive tile screening
*`results/a100/phase-c42/c43_adaptive_tile_screening.json`; `scripts/phase-c42/c43_adaptive_tile_screening.py`.*
- Adaptive tile16↔32 by `n_visible` threshold (50000) on A100, room (1.07M Gs). All 13/31 cameras chose tile32 (n_visible 52K–483K all > threshold). tile32 mean 3.43 ms vs tile16 36.39 ms (**90.6% speedup** — note tile16 contaminated by two 185–234 ms outliers; clean cameras ~4–5 ms). ΔPSNR = 0, ΔSSIM = 0 (identical pixels). Verdict KEEP, but the threshold is degenerate here.

### Phase 4 / Phase 5 (EPIC-05) — hardware-aware + mechanism
*`reports/epic05/hardware-aware-tile-study-2026-08-19.md`; `reports/epic05/tile-mechanism-training-study-2026-08-19.md`.* See §4–§6. Phase 5's `cuobjdump` finding (same binary, REG=40, SHARED=1024 B) **revised the Phase 4 occupancy model** and falsified shared-memory/register-spill explanations.

### Phase 7C — Tile-size source trace
*`reports/epic05/phase7c_tile_source_trace.md`.* Confirmed `tile_size` touches exactly **4 launch parameters** (grid.x, grid.y, block.x, block.y) and **2 derived values** (tile_width, tile_height), affecting 3 CUDA kernel invocations (intersect_tile, intersect_offset, rasterize_to_pixels). Not used in projection/SH/optimizer/densification.

### Phase 13A / 13B / 13C — Tile selection oracle, expanded sweep, training validation
*`reports/epic05/phase13a_tile_selection_oracle.md`; `reports/epic05/phase13b_expanded_tile_sweep.md`; `reports/epic05/phase13c_tile_training_validation.md`; `results/epic05/phase13b/expanded_tile_results.json`.* See §3, §6. tile20 = strongest universal *snapshot* candidate; **snapshot ≠ training** (room: snapshot tile20, training tile32).

### Phase 14A — tile20 training anomaly
*`reports/epic05/phase14a_tile20_anomaly.md`; `results/epic05/phase14a_tile20_anomaly.json`.* tile20 is the forward winner for room (10.247 ms) but the **worst 30K training config** (561.8 min, 0.27× vs tile16). Root cause = **asynchronous topology spillover** (42/294 densification events overrun; 157.8 s outlier at iter 8050). tile20's non-power-of-2 400-thread block is hypothesized to cause suboptimal densification kernels. Falsified "tile20 renderer pathology."

---

## 3. Tile-Size Sweep Results

### 3.1 A100-PCIE-40GB, room, official Mip-NeRF 360 ckpt, 1080p, `packed=False`
Source: `results/a100/c17-c33/phase-c19/c19-1_tile_scaling.json` (C19-1) and `c19-2_block_geometry_control.json` (canonical block-geometry control, 15 runs/point).

| Tile | Grid | Tiles | Threads/blk | Shmem B | Total ints | Mean ints/tile | Rasterize ms (C19-2) | ns/int |
|:---:|:----:|:----:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8 | 240×135 | 32400 | 64 | 1792 | 5,465,619 | 168.7 | 0.977 | 0.179 |
| 12 | 160×90 | 14400 | 144 | 4032 | 2,640,904 | 183.4 | 1.137 | 0.431 |
| 16 | 120×68 | 8160 | 256 | 7168 | 1,626,135 | 199.3 | 1.172 | 0.721 |
| 20 | 96×54 | 5184 | 400 | 11200 | 1,125,232 | 217.1 | 1.249 | 1.110 |
| 24 | 80×45 | 3600 | 576 | 16128 | 849,047 | 235.8 | 2.409 | 2.838 |
| 28 | 69×39 | 2691 | 784 | 21952 | 684,937 | 254.5 | 2.976 | 4.344 |
| 32 | 60×34 | 2040 | 1024 | 28672 | 564,323 | 276.6 | 2.862 | 5.072 |

Full forward (C19-0, kernel-decomposed): t8 7.43 / t16 2.55 / t32 3.20 ms; rasterize-only t16 1.05, t32 1.59. Intersection duplication falls 30.9×→3.2× (t8→t32) exactly as geometry predicts; **no tile imbalance** (max/mean < 1.6×, all tiles active). Quality: PSNR/SSIM/LPIPS identical across all tile sizes (pixel-equivalence `max_abs_diff=0.0`, see §3.3).

### 3.2 RTX 5070 Laptop, 1080p, real cameras, SfM init (Phase 13B)
Source: `results/epic05/phase13b/expanded_tile_results.json`; `reports/epic05/phase13b_expanded_tile_sweep.md` §2. Median ms.

| Tile | room fwd | room fb | bicycle fwd | bicycle fb | garden fwd | garden fb |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4 | 37.42 | 68.07 | 50.61 | 150.66 | 50.81 | 140.02 |
| 8 | 13.67 | 37.93 | 68.98 | 122.84 | 34.18 | 112.83 |
| 12 | 11.81 | 36.35 | 33.65 | 133.25 | 26.63 | 98.35 |
| 16 | 11.37 | 34.83 | 35.86 | 119.09 | 27.68 | 96.97 |
| 20 | **10.25** | 38.98 | 31.81 | **96.96** | 28.88 | **92.74** |
| 24 | 10.41 | 39.39 | **31.40** | 98.69 | 29.41 | 110.42 |
| 28 | 11.34 | 43.14 | 36.94 | 112.69 | 34.36 | 129.84 |
| 32 | 11.78 | 46.15 | 34.11 | 140.23 | 36.99 | 128.43 |

Forward optima: room tile20, bicycle tile24, garden tile12. Fwd+bwd optima: room tile16, bicycle tile20, garden tile20. **tile20 is the only tile in the top-3 for all 3 scenes for both metrics.** All 8 sizes {4,8,12,16,20,24,28,32} are supported by gsplat v1.5.3; >32 unsupported (1024-thread block limit). Empty-tile ratio = 0.0% everywhere.

### 3.3 RTX 5070, Phase 4 (bicycle/garden/room, official ckpts)
Source: `reports/epic05/hardware-aware-tile-study-2026-08-19.md` §2. stable_mean_ms: bicycle t8 21.82 / t16 20.72 / t32 27.71; garden 18.41 / 16.70 / 23.94; room 10.39 / 7.09 / 8.65. 4K bicycle: t16 35.10 / t32 31.55 (tile32 wins at 4K, but t16 had 45 cold-start outliers; clean repeats t16 24.7 vs t32 31.5). VRAM: room training peak 2324 MB for both t16 and t32.

### 3.4 A100-PCIE-40GB 30K training (new A100)
Source: `reports/phase-a100/final_conclusions.md` §2.2; `reports/phase-a100/m1_gap_analysis.md`.

| Scene | t16 PSNR | t20 PSNR | t24 PSNR | t32 | Wall (t16/t20/t24) min |
|---|---|---|---|---|---|
| room | 29.39 | 29.38 | 29.44 | ❌ BLOCKED | 49.4 / 49.6 / 49.8 |
| bicycle | 19.32 | 19.22 | 19.19 | ❌ BLOCKED | 54.4 / 54.7 / 54.6 |
| garden | 21.04 | 21.12 | 21.11 | ❌ BLOCKED | 50.1 / 49.6 / 50.2 |

ΔPSNR < 0.1 dB across 16/20/24. **tile32 backward blocked** ("CUDA too many resources") on gsplat 1.5.3+pt24cu124.

### 3.5 RTX 5070 room 30K training
Source: `reports/epic05/phase13c_tile_training_validation.md` §6; `reports/epic05/phase14a_tile20_anomaly.md` §2.

| Tile | Wall (min) | Iter/s | Best PSNR | Final Gs | Iters >10s |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 16 | 150.3 | 3.33 | 29.27 | 1,193,480 | 0 |
| 20 | 561.8 | 0.89 | 29.54 | 1,158,369 | 57 (1.9%) |
| 32 | 94.8 | 5.27 | 29.39 | 1,146,273 | 0 |

**Training winner = tile32** (1.58× vs tile16), **decoupled** from snapshot winner (tile20). bicycle/garden 30K blocked locally (8 GB VRAM).

---

## 4. Hardware Dependence: A100 vs RTX 5070 Laptop

### 4.1 Hardware matrix
Source: `reports/epic05/hardware-aware-tile-study-2026-08-19.md` §6; `reports/epic05/tile-mechanism-training-study-2026-08-19.md` §2.

| Property | A100 (SXM4-80GB, Phase 4) | A100-PCIE-40GB (C19) | RTX 5070 Laptop |
|---|:---:|:---:|:---:|
| Compute cap | 8.0 | 8.0 | 12.0 |
| SMs | 108 | 108 | 36 |
| Shared mem/SM | 164 KB | 164 KB | 100 KB |
| Max threads/SM | 2048 | 2048 | 1536 |
| Registers/SM | 65536 | 65536 | 65536 |
| L2 | 40 MB | 40 MB | 32 MB |
| Mem bandwidth | 2039 GB/s | 1555 GB/s | ~144 GB/s est. |
| PyTorch/gsplat | — | 2.7.1+cu118 / 1.5.3 | 2.13.0+cu130 / 1.5.3 |

### 4.2 Opposite optima
- **A100 (synthetic, Phase 4):** tile32 optimal, 1.42–3.93× over tile16 (`hardware-aware-tile-study` §1). 50k 44.15/15.89/11.20; 400k 263.27/43.65/11.11 ms (t8/t16/t32).
- **RTX 5070 (official scenes, Phase 4):** tile16 optimal; tile32 = 0.67–0.83× (slower) (`hardware-aware-tile-study` §2).
- **A100-PCIE (canonical ckpt, C19-0):** tile16 optimal for full forward (2.55 ms); tile32 1.26× slower. **The A100 "tile32 wins" result is not reproduced on the canonical workload** — it came from synthetic/saturated workloads and a non-canonical checkpoint.

### 4.3 Occupancy / block geometry (the key interaction)
Source: `reports/epic05/tile-mechanism-training-study-2026-08-19.md` §6; `reports/phase-c19/c19-2_rasterizer_isolation.md` §3.

Because the kernel binary is identical (REG=40, SHARED=1024 B), the only per-tile difference is **threads/block** (t16=256, t32=1024) and grid dims.

| GPU | tile16 blocks/SM | tile16 occ | tile32 blocks/SM | tile32 occ | Limiting factor |
|---|:---:|:---:|:---:|:---:|---|
| RTX 5070 (1536 thr/SM) | 6 | 100% | 1 | 66.7% | **thread count** (1024 > 1536/2) |
| A100 (2048 thr/SM) | 8 | 100% | 2 | 100% | threads allow 2 blocks |

On RTX 5070, tile32's 1 block/SM (66.7% occ) reduces warp-level parallelism → `rasterize_to_pixels` **1.91× slower** with tile32 (room, 7,544→14,419 µs), while intersect/offset are 0.26–0.83× *faster* (fewer tiles) (`tile-mechanism-training-study` §5.1). On A100, both tile sizes reach 100% occupancy (40 regs), so tile32's advantage = reduced tile/bin overhead + bandwidth making extra per-block work cheap.

### 4.4 Same-binary / compiled-binary evidence
Source: `reports/epic05/hardware-aware-tile-study-2026-08-19.md` §5; `reports/epic05/tile-mechanism-training-study-2026-08-19.md` §4; `reports/epic05/phase7c_tile_source_trace.md` §5.
- RTX 5070: all three tile sizes use the **same compiled CUDA extension** (`backend_hash b08bf0e8…`, git `0cf93be`), only `tile_size` differs at runtime.
- `cuobjdump --dump-resource-usage`: `rasterize_to_pixels_3dgs_fwd_kernel<float,SH=3>` **REG=40, SHARED=1024 B, STACK=0**; bwd **REG=48, SHARED=1024 B**. No separate tile32 kernel, no spilling.
- A100-PCIE: gsplat 1.5.3 JIT-compiled, sm_80; **tile32 backward BLOCKED** ("CUDA too many resources") — a build/resource-limit artifact, not a mechanism. (Note: this contradicts the older SXM4 A100 where tile32 ran; likely PyTorch/CUDA-version dependent — `pt24cu124` vs older.)

### 4.5 Cross-tile overlap / block geometry (A100)
- `c17_2_data_integrity_cross_tile_audit.md` §5: horizontal Jaccard P50 0.62/0.51/0.43 (room/bicycle/garden); vertical < horizontal; **100% order consistency** of shared Gaussians across neighbor tiles (tautological from depth-sorted membership).
- `c19-2` §3.3: occupancy cliff when blocks become too large to fit multiple per SM (t16 ~2 blocks/SM → t24 ≤3 → t32 1–2).

---

## 5. Mechanism Conclusions

1. **tile_size is a runtime launch parameter only.** The same compiled kernel binary serves all tile sizes; `tile_size` changes exactly `block=(tile_size,tile_size)`, `grid=(tile_width,tile_height,I)`, and the tile-quantization in `intersect_tile`. Forward output is bit-identical (`max_abs_diff=0.0`); gradients are identical within FP precision (§7) (`phase7c_tile_source_trace.md`, `gradient-correctness-tile-size-2026-09-01.md`).

2. **The per-intersection cost cliff (t16→t20, 3.1× on A100) is caused by occupancy collapse from increasing block dimensions, not register pressure or shared-memory pressure.** The C19-2 controlled replay (fixed block 16×16, ints 100→320) shows ns/int *decreasing*; the cliff does not reproduce. ns/int correlates with threads/block (R²≈0.93). Register pressure is a minor contributor (mild 1.6× at 205 ints) (`c19-2_rasterizer_isolation.md` §2, §5).

3. **The effect is NOT universal — it is hardware- AND scene-dependent.** Hardware: RTX 5070 (1536 thr/SM) caps tile32 at 1 block/SM (66.7% occ) → rasterize 1.91× slower; A100 (2048 thr/SM) allows 2 blocks/SM (100% occ) → tile32 wins on saturated/synthetic workloads but tile16 wins on the canonical ckpt. Scene: on RTX 5070, room/bicycle favor tile16, garden (5.8M Gs) favors tile32 by median; denser scenes benefit from reduced tile/bin overhead (`tile-mechanism-training-study` §5.3). The hardware-aware heuristic `select_tile_size` (≥164 KB shmem/SM → 32, else 16) matches the optimal tile for the 2-GPU cohort but the 164 KB threshold is not universal (n=2).

4. **Snapshot optimum ≠ training optimum.** Room RTX 5070: snapshot forward tile20 (10.25 ms), 30K training tile32 (94.8 min, 1.58×). The backward pass (~70% of step time) dilutes the forward rasterize penalty; tile20's forward win is negated by training-system topology spillover (Phase 14A).

---

## 6. Falsified Hypotheses

| Hypothesis | Evidence that falsified it | Source |
|---|---|---|
| **register-pressure-primary** (spill at ~200 ints/tile causes the t16→t20 cliff) | Controlled replay at fixed block=16×16 over 100→320 ints/tile: ns/int *decreases* (0.40→0.29), cliff does not reproduce; only mild 1.6× at 205 ints. STACK=0 from cuobjdump. | `c19-2_rasterizer_isolation.md` §2, §5; `tile-mechanism-training-study` §4 |
| **work-per-tile-primary** (more ints/tile → more cost) | Replay shows sub-linear (better) scaling with ints/tile at fixed geometry; canonical cliff co-varies with block geometry, not ints/tile. | `c19-2_rasterizer_isolation.md` §3.2 |
| **sort-count-primary** (CUB sort dominates → reduce intersections) | On canonical A100 baseline sort = 9.9% (C19-0) / 0.25 ms·0.2% T_iter (C37-B); rasterize dominates (55.5%). "Sort dominates 65%" was an artifact of a non-canonical checkpoint. | `c19-0_a100_rasterizer_profiling.md` §3; `c37_common_mechanism.md` C37-B; `c19-1` §1 |
| **shared-memory-pressure** (tile32's 14 KB shmem/block causes slowdown) | Actual shmem = 1024 B/block (cuobjdump), ≤1% of SM shmem. Falsified. | `tile-mechanism-training-study` §4, §12 |
| **tile32 compiles with more registers** | Same binary, REG=40 for both tile sizes. Falsified. | `tile-mechanism-training-study` §4.3, §12 |
| **tile32 universally optimal** | RTX 5070 tile32 = 0.67–0.83× (slower); A100-PCIE canonical tile32 1.26× slower than tile16. | `hardware-aware-tile-study` §11; `c19-0` §6 |
| **164 KB shmem universal threshold** | n=2 GPUs; A100-PCIE (164 KB) canonical prefers tile16. | `hardware-aware-tile-study` §11 |
| **training matches inference tile preference** | Room RTX 5070: inference tile16, training tile32. | `tile-mechanism-training-study` §9 |
| **tile20 renderer pathology** (tile20's 561 min training = renderer bug) | Root cause = async topology spillover (42/294 densification events overrun; 157.8 s outlier at iter 8050, distributed across fwd/bwd/opt). | `phase14a_tile20_anomaly.md` §4 |
| **PAT — parallel-adaptive tile subdivision** (tile imbalance is the problem) | All tiles active, max/mean < 1.6×; no imbalance to correct. | `c19-0` §11; `c19-1` §2.1 |
| **C17-0 segmented sort faster than global CUB** | 2.5× slower (counting-sort atomicAdd scatter). | `c17_0_tile_segmented_sort.md` |
| **within-tile membership compressible** (delta/run-length) | Median |Δ|=26–745, max run 2–3; effectively random. | `c17_2_data_integrity_cross_tile_audit.md` §8 |
| **synthetic camera predicts real-camera tile winner** | 4/15 disagreements; synthetic sees <0.3% of outdoor Gaussians. | `phase13a_tile_selection_oracle.md` §7 |

---

## 7. Canonical vs Fixed-Geometry Replay

Two distinct "replay" notions appear, both distinguishing **correct behavior** from **launch-geometry artifacts**:

### 7.1 Fixed-geometry intersection replay (C19-2) — the decisive control
- **Protocol:** block geometry **fixed** at 16×16×1 (256 threads, 7168 B shmem); intersections/tile varied 100→320 by truncating/replicating `flatten_ids` per tile; same `rasterize_to_pixels_3dgs_fwd_kernel`. 15 runs/point. (`c19-2_rasterizer_isolation.md` §2; `results/a100/c17-c33/phase-c19/c19-2_block_geometry_control.json`.)
- **What it isolates:** the intersection-count variable, holding block geometry constant — vs the canonical sweep where both co-vary.
- **Result:** ns/int decreases 0.40→0.29 (sub-linear); no 3.1× cliff. ⇒ the canonical cliff is a **block-geometry (occupancy) artifact**, not an intersection-count (register) effect. This is the cleanest "correct behavior vs replay artifact" separation in the dataset.

### 7.2 Canonical correctness (gradient + pixel replay)
- **Gradient replay:** `torch.autograd.gradcheck` (eps=1e-4, atol=1e-3, rtol=1e-3, nondet_tol=1e-5) on 10-Gaussian subset, 64×64 crop; finite-difference central (ε=1e-4, 1e-5, n=100) on means/scales/rotations/opacity/shs. tile16 vs tile32: grad-norm relative diff < 5e-7; FD max_abs=0.0 (1.03e-7 for shs); gradcheck PASS all 5 param groups both tile sizes; packed and dense modes match. (`reports/epic05/gradient-correctness-tile-size-2026-09-01.md`; `results/epic05/gradient/gradcheck_*.json`.)
- **Pixel replay:** `pixel_equivalence_vs_tile16.max_abs_diff = 0.0` for all tile sizes {4…32} × 3 scenes (Phase 13B JSON). Forward bit-exact; backward grad-identical within FP ⇒ tile_size is a **correct, differentiable** optimization of the launch geometry, not a numerical change.
- **Sort tie-breaking (C17-0):** `isect_ids` match 100% across global vs segmented sort; `flatten_ids` differ only at identical-depth ties (non-blocking, <0.01 dB). Confirms depth ordering is preserved; tie differences are a sort-stability artifact, not a correctness bug.

---

## 8. Open Questions / Contradictions

1. **A100 tile32 contradiction.** Old SXM4 A100 (Phase 4): tile32 1.42–3.93× faster (synthetic). New PCIE-40GB (C19-0): tile32 1.26× *slower* than tile16 (canonical ckpt); and tile32 **backward blocked** ("too many resources"). Likely PyTorch/CUDA-version (`pt24cu124` vs older) + workload (synthetic saturated vs canonical sparse). Not fully reconciled; needs same-binary re-profile on both. (`hardware-aware-tile-study` §1 vs `c19-0` §6, `final_conclusions.md` §4.)

2. **Snapshot vs training decoupling.** Room RTX 5070: snapshot tile20, training tile32. Only room has 30K data; bicycle/garden 30K blocked on 8 GB. Whether the decoupling generalizes is open. (`phase13c_tile_training_validation.md` §6.)

3. **tile20 training anomaly mechanism.** Phase 14A attributes the 561.8 min to async topology spillover from tile20's non-power-of-2 400-thread block, but "Direct topology kernel timing isolation requires modifying the training loop" — not done. (`phase14a_tile20_anomaly.md` §6 BLOCKED.)

4. **NCU blocked throughout.** `ERR_NVGPUCTRPERM` on RTX 5070 (WDDM); ncu broken on A100 (section path, `HOME=/tmp`, `CUDA_VISIBLE_DEVICES`). No direct occupancy/stall/L2-hit counters; all occupancy numbers are computed, not measured. (`tile-mechanism-training-study` §14; `c19-2` §4.)

5. **C19-3 (occupancy-aware fixed-block rasterization) not implemented.** Projected 5–15× faster at t32 but unverified. (`c19-2` §7.)

6. **Adaptive threshold degeneracy.** C43's n_visible>50000 threshold chose tile32 for all cameras (52K–483K); the 90.6% speedup is partly a tile16-outlier artifact (2 cameras 185–234 ms). Threshold needs recalibration and cross-scene validation. (`c43_adaptive_tile_screening.json`.)

7. **n=2 GPUs, single consumer GPU (Blackwell-specific?).** The hardware-aware heuristic rests on 2 GPUs; 164 KB threshold not universal. (`hardware-aware-tile-study` §12.)

8. **C25 asymmetric tile: 0% speedup over best-fixed** despite per-camera optima varying — the heavy-tailed per-camera intersection distribution (P99 up to 1.04M) may need a different predictor than tile-count. (`c_asymmetric_tile.json`.)

---

*Digest compiled from ~20 reports and ~10 JSON/CSV result files across `reports/epic05/`, `reports/phase-c19/`, `reports/phase-a100/`, `reports/rtx5070/`, `reports/phase-c27/`, `reports/a100_validation/`, and `results/a100/c17-c33/`, `results/epic05/phase13b/`, `results/a100/phase-c42/`. See §2 for per-candidate sources.*
