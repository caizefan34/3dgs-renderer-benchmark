# H3-FWD-1A-R Independent Audit Report

## Summary

**Audit Decision: AUDIT_NEEDS_REPAIR**

The Codex H3-FWD-1A-R Macro-F4 forward rasterization results were independently audited across six dimensions: structural, ordering, timing, memory, interpretation, and last-ID semantics. The representation passes ordering, timing methodology, memory accounting, and interpretation checks. However, the structural check **fails** for the bicycle scene due to a macro tile boundary classification error (1 missing + 1 extra pair, 0.000141624% defect rate). Room and garden pass all checks with zero defects.

## Evidence Sources

All evidence was fetched from `mx` (8×A100-PCIE-40GB, 36.140.146.31:26372):

| File | Description |
|------|-------------|
| `/tmp/h3_fwd_1a_r/room.json` | Room structural + ordering results |
| `/tmp/h3_fwd_1a_r/bicycle.json` | Bicycle structural + ordering results (final) |
| `/tmp/h3_fwd_1a_r/bicycle.log` | Bicycle results from earlier run (non-determinism comparison) |
| `/tmp/h3_fwd_1a_r/garden.json` | Garden structural + ordering results |
| `/tmp/h3_fwd_1a_r/timing.json` | CUDA event timing (room/cam0) |
| `/tmp/h3_fwd_1a_r/memory.json` | Memory accounting (room/cam0) |
| `/tmp/h3_fwd_1a_r_validate.py` | Validation script (structural + ordering) |
| `/tmp/h3_fwd_1a_r_timing.py` | Timing script |
| `/tmp/h3_fwd_1a_r_memory.py` | Memory script |
| `/tmp/h3_fwd_0_oracle.py` | CPU FP32 oracle (ground truth) |
| `/tmp/h3_fwd_1a_r/*_order.csv` | Per-tile ordering differences |
| `/tmp/h3_fwd_1a_r/*_pair_diff.csv` | Per-tile structural differences |

Binary: `/tmp/higs_h3_fwd_1a_build/.../experimental_gaussian_render_inference_scene_cuda.so`
SHA256: `78ace12b294d274209b4711b95da578ade4ac2ed213cbe25c372cb5e86d7c890`
Patch: `patches/higs-h3-fwd-1a-macro-f4.patch` (9403 bytes)

## 1. Structural Check — **FAIL**

### Room — PASS
| Metric | Value |
|--------|-------|
| N_B2_tile_pairs | 953,144 |
| N_macro_entries | 124,150 |
| N_reconstructed_fine_pairs | 953,144 |
| CPU-CUDA macro entries equal | true |
| CPU-CUDA fine pairs equal | true |
| Missing pairs | 0 |
| Extra pairs | 0 |
| Duplicate pairs | 0 |
| Wrong Gaussian IDs | 0 |

### Bicycle — **FAIL**
| Metric | Value |
|--------|-------|
| N_B2_tile_pairs | 1,412,189 |
| N_macro_entries | 311,675 |
| N_reconstructed_fine_pairs | 1,412,189 |
| CPU-CUDA macro entries equal | **false** (CPU: 311,674, CUDA: 311,675) |
| CPU-CUDA fine pairs equal | **false** (CPU: 1,412,188, CUDA: 1,412,189) |
| Missing pairs | **1** |
| Extra pairs | **1** |
| Duplicate pairs | 0 |
| Wrong Gaussian IDs | **2** |
| Defect rate | 0.000141624% |

**Defect details** (from `bicycle_pair_diff.csv`):
- Tile 863: B2 count=168, macro count=167, **missing=1**
- Tile 7960: B2 count=131, macro count=132, **extra=1**

**Root cause hypothesis**: Floating-point precision difference between the CUDA float32 ellipse-tile intersection test and the CPU double-precision oracle at macro tile boundaries. A Gaussian near the boundary between macro tiles 863 and 7960 is classified into one macro tile by CUDA and a different one by the oracle. The defect is **consistent across runs** (present in both `bicycle.json` and `bicycle.log`; `macro_offsets` SHA256 identical across runs).

### Garden — PASS
| Metric | Value |
|--------|-------|
| N_B2_tile_pairs | 533,928 |
| N_macro_entries | 66,724 |
| N_reconstructed_fine_pairs | 533,928 |
| CPU-CUDA macro entries equal | true |
| CPU-CUDA fine pairs equal | true |
| Missing pairs | 0 |
| Extra pairs | 0 |
| Duplicate pairs | 0 |
| Wrong Gaussian IDs | 0 |

## 2. Ordering Check — **PASS**

All ordering differences across all three scenes are **equal-depth ties** (inversions == equal_depth_ties for every affected tile). No real depth-ordering errors were found.

| Scene | Exact order match % | Inversions | Equal-depth ties | Inversions > ties |
|-------|--------------------:|-----------:|-----------------:|------------------:|
| Room | 99.35% | 72 | 72 | 0 |
| Bicycle | 99.27% | 82 | 104 | 0 |
| Garden | 99.97% | 3 | 3 | 0 |

**Non-determinism on bicycle**: `bicycle.json` and `bicycle.log` show different `sorted_ids` SHA256 hashes (0b6c209d vs 489d85fb) and different tie counts (104 vs 101), but identical `macro_offsets` hashes (1655529f). This indicates non-deterministic GPU segmented sort within equal-depth groups — **semantically harmless** since equal-depth Gaussians can be in any order without affecting rendering correctness.

## 3. Timing Check — **PASS** (with coverage note)

### Protocol — PASS
| Parameter | Value |
|-----------|-------|
| Clock | CUDA Events |
| Warmup | 20 |
| Measurements per rep | 100 |
| Repetitions | 5 |
| Total samples | 500 |
| Interleaved | true |
| CPU reconstruction in measured region | No |

### Results (room/cam0)
| Metric | B2 F4 | Macro F4 | Speedup |
|--------|------:|---------:|--------:|
| Median (ms) | 0.557 | 0.366 | **34.35%** |
| Mean (ms) | 0.622 | 0.405 | — |
| P10 (ms) | 0.541 | 0.352 | — |
| P90 (ms) | 0.919 | 0.600 | — |

### Stage breakdown (Macro-F4)
| Stage | Median (ms) |
|-------|------------:|
| Count | 0.115 |
| Scan prefix | 0.024 |
| Fill | 0.145 |
| Segmented sort | 0.043 |
| Batch metadata | 0.027 |

**Coverage note**: Timing was only run for room/cam0. Bicycle and garden were not timed. The timing methodology is valid but coverage is incomplete.

**timing.log issue**: `/tmp/h3_fwd_1a_r/timing.log` contains a traceback from an earlier failed run (`torch.ops.gsplat.isect_offset_encode` not found). The `timing.json` was generated by a later successful run using the fixed import (`from gsplat.cuda._wrapper import isect_offset_encode`). No impact on results.

## 4. Memory Check — **PASS** (with coverage note)

### Validation masks — PASS
Validation masks (496,600 bytes) are explicitly tracked **separately** from persistent state (499,424 bytes). They are validation-only and **NOT counted** as production persistent state.

### Memory comparison (room/cam0)
| Metric | B2 F4 | Macro F4 | Reduction |
|--------|------:|---------:|----------:|
| Persistent (MB) | 10.95 | 0.48 | **95.65%** (22.99×) |
| Peak allocator delta (MB) | 22.59 | 1.92 | **91.51%** |

**B2 F4 persistent breakdown**: isect_metadata 7.63 MB + flatten_ids 3.81 MB + tile_offsets 0.04 MB = 11.48 MB

**Macro F4 persistent breakdown**: macro_offsets 1.41 KB + macro_sorted_ids 496.6 KB + batch_offsets 1.41 KB = 499.42 KB

**Coverage note**: Memory was only measured for room. Bicycle and garden memory not available.

## 5. Interpretation Check — **PASS**

### Representation compression ≠ fine-tile work compression ≠ pixel-Gaussian compute compression

**Representation compression** = N_B2_tile_pairs / N_macro_entries:
- Room: 7.68× (953,144 / 124,150)
- Bicycle: 4.53× (1,412,189 / 311,675)
- Garden: 8.00× (533,928 / 66,724)

This is the **macro entry compression ratio** — the reduction in the number of entries needed to represent the same tile-Gaussian coverage. It is NOT:
- **Fine-tile work compression** (which is 1.0 — all fine pairs are reconstructed from masks)
- **Pixel-Gaussian compute compression** (which depends on the rendering kernel, not the representation)

### Mask popcount
The `mask_popcount` mean equals `representation_compression` because each macro entry on average covers (N_fine_pairs / N_macro_entries) fine tiles. Max=32 means some Gaussians cover all 32 fine tiles (8×4 macro tile) within their macro tile.

## 6. Last-ID Check — **PASS**

The `last_ids` in the HiGS backward kernel (`higs_blend_bwd_px_kernel`) stores the **tile-local sorted-list position** (the index within the per-tile sorted Gaussian list), NOT:
- The Gaussian's global ID
- The global flatten index

The Macro-F4 representation does NOT change last-ID semantics. It only changes how the sorted list is stored (grouped by macro tile with 32-bit masks vs. flat per-tile lists). When the forward rasterizer expands a macro entry to fine tiles via masks, each fine tile receives a contiguous range of Gaussian IDs from the macro entry's sorted list. The tile-local position in the backward kernel corresponds to the position within this per-tile expanded list.

## Audit Decision: AUDIT_NEEDS_REPAIR

### Rationale

The H3-FWD-1A-R Macro-F4 representation passes ordering, timing, memory, and interpretation checks. However, the structural check **fails** for the bicycle scene due to a macro tile boundary classification error (1 missing + 1 extra pair, 0.000141624% defect rate). This is a minor but real defect that should be repaired before promotion to macro raster.

### Conditions for promotion to AUDIT_PASS_PROMOTE_TO_MACRO_RASTER

1. **Fix bicycle macro tile boundary classification**: achieve 0 missing, 0 extra, 0 wrong Gaussian IDs on all three scenes
2. **Extend timing** to bicycle and garden scenes (currently only room timed)
3. **Extend memory measurement** to bicycle and garden scenes (currently only room measured)

### What was NOT done

- No new Macro-F4 implementation was written
- No algorithm redesign was attempted
- No optimization phase was opened

## Artifacts

All audit artifacts are in `artifacts/higs-h3-fwd-1a-audit/`:
- `evidence_check.json` — structural + ordering audit
- `timing_check.json` — timing audit
- `memory_check.json` — memory audit
- `interpretation_check.json` — interpretation + last-ID audit
- `analysis.json` — overall audit analysis and decision
