# R4: 13-Scene CUDA Final Validation — Progress Report

## Status: Phase 6 (Checkpoint Benchmark) — In Progress

## Completed Phases

### Phase 1-2: Room CUDA Correctness Sanity (R4-0) ✅
**Result: PASS**

- MODE0 vs MODE1 (same code, two runs): All gradients match within numerical noise
- max_abs < 2e-6, L2 < 1e-9 for all gradient families
- Camera 2 showed all-zero gradients (degenerate camera — no Gaussian overlap)
- **Conclusion**: Baseline reproducibility confirmed. GPU atomicAdd non-determinism is the only source of inter-run variation.

### Phase 3: CUDA Extension Compilation ✅
**Result: BUILD SUCCESSFUL**

- Custom CUDA kernel `rasterize_to_pixels_bwd_with_skip` compiled with:
  - CUDA 12.8 from `/mnt/storage_pool/liaoyuanjun/higs-13scene-env` (complete installation with cicc)
  - gsplat 1.5.3 headers from local `candidate_c_source_audit/canonical_source/`
  - nv/target headers from conda env `~/miniforge3/envs/anysplat/include/`
  - sm_80 gencode for A100
- Extension: `experiments/r4/build/ext_skip.so` (215.9 KB)
- Exposes: `rasterize_to_pixels_bwd_with_skip(means2d, conics, colors, opacities, backgrounds, masks, width, height, tile_size, tile_offsets, flatten_ids, render_alphas, last_ids, v_render_colors, v_render_alphas, skip_mask, absgrad)`

### Phase 4: MODE2 CUDA Skip Test ✅
**Result: PASS**

Room 30K checkpoint (952K Gaussians), 2 cameras, 5% budget:

| Comparison | Camera 0 | Camera 1 |
|---|---|---|
| MODE0 vs MODE1 (skip off) | ALL PASS (max_abs < 1e-6) | ALL PASS (max_abs < 1e-6) |
| MODE0 vs MODE2 (skip on) | Expected diff (35% skip) | ALL PASS (budget conservative) |

- MODE0 vs MODE1: Custom CUDA kernel with all-False skip mask matches baseline → kernel implementation is correct
- MODE0 vs MODE2: Camera 0 shows expected gradient differences (L2 ~1e-4) from 35% skip; Camera 1 shows near-zero differences (L2 ~1e-10) → certificate bounds are conservative
- Skip fraction: 35.20% (cam0), 34.25% (cam1) at 5% budget

### Phase 5: CUDA Insertion Audit ✅
**Document**: `reports/r4/r4-cuda-insertion-audit.md`

Documents kernel launch config, per-thread workload, certificate insertion point, R3.1 weighted-work → CUDA mapping, branches introduced, atomicAdd avoided, correctness preservation.

### Phase 6: Multi-Scene Checkpoint Benchmark (In Progress)

**Room** (952K Gaussians, 45.6M intersections per camera):

| Budget | Skip Fraction | v_xyz L2 | v_opacity L2 | v_shs L2 |
|---|---|---|---|---|
| 1% | 16.37% | 1.37e-05 | 6.45e-04 | 2.13e-06 |
| 5% | 35.20% | 1.37e-05 | 6.45e-04 | 2.13e-06 |

Camera 1 (more typical):
| Budget | Skip Fraction | v_xyz L2 | v_opacity L2 | v_shs L2 |
|---|---|---|---|---|
| 1% | 15.85% | 6.61e-10 | 1.97e-10 | 3.19e-12 |
| 5% | 34.25% | 5.29e-11 | 7.25e-11 | 4.61e-12 |

**Bicycle** (3.9M Gaussians, 705M intersections per camera):

| Budget | Skip Fraction | v_xyz L2 | v_opacity L2 | v_shs L2 |
|---|---|---|---|---|
| 1% | 13.54% | 1.68e-08 | 3.26e-08 | 3.96e-09 |
| 5% | 30.52% | 1.68e-08 | 3.26e-08 | 3.96e-09 |

**Key finding**: Bicycle (largest scene, 705M intersections) shows the smallest gradient errors — L2 ~1e-8 at 30% skip. The certificate is extremely conservative for large scenes because the budget is relative to the total bound, which grows with scene size.

## Files Created

| File | Description | Status |
|---|---|---|
| `experiments/r4/rasterize_to_pixels_bwd_with_skip.cu` | Custom CUDA backward kernel with skip_mask | ✅ Compiled |
| `experiments/r4/ext_skip.cpp` | pybind11 binding | ✅ Compiled |
| `experiments/r4/build_cuda_extension.py` | JIT build script | ✅ Working |
| `experiments/r4/compute_skip_mask.py` | Python skip mask computation (vectorized) | ✅ Working |
| `experiments/r4/r4_sanity.py` | R4-0 correctness sanity check | ✅ PASS |
| `experiments/r4/r4_mode2_test.py` | MODE2 CUDA skip test | ✅ PASS |
| `experiments/r4/r4_checkpoint_benchmark.py` | Multi-scene checkpoint benchmark | 🔄 Running |
| `experiments/r4/r4_train_scene.py` | 13-scene training script | 📝 Written |
| `experiments/r4/run_r4_13scene_v2.sh` | 13-scene parallel launcher | 📝 Written |
| `reports/r4/r4-cuda-insertion-audit.md` | CUDA insertion audit | ✅ Complete |

## Next Steps

1. **Complete checkpoint benchmark** (garden scene + bicycle camera 1)
2. **Launch 13-scene 30K training** (baseline + Candidate C, 8 GPUs, 2 batches)
3. **Aggregate results** and produce final verdict
