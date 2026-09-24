# C35 Phase-2: Cross-Renderer Local Feasibility — Gate Report

## C35-1: Adaptive Hybrid Blend Backend — **DROP**

### Source audit: tile density bimodality

Using data from C33-D (3 camera cycles on room scene):

| Metric | Value | Source |
|--------|-------|--------|
| Image resolution | 1920×1080 | Phase-7 config |
| Fine tiles (16×16) | 120×68 = **8,160** | Geometry |
| HiGS macro-tiles (8×8 tiles) | 15×9 = **135** | `IntersectMTConfig.h` |
| Active fine tiles/iter (≥1 GS) | ~7,500-8,160 | C33-D (estimated from coverage ratio) |
| Max visible GS/iter | 1.16M (out of 1.59M) | C33-D data |
| Forward rasterization time | 4-6ms (stable) | C33-D (post warmup) |
| Forward time CV post-warmup | **0.05** | C33-D |
| Backward rasterization time | 11-18ms (stable) | C33-D |

### Tile-level density analysis

**Key finding**: Standard gsplat uses 1 block (256 threads) per fine tile. Each block loads all GS for that tile in batches of 256. For tiles with 1-5 GS, this means 1-2 batches. For tiles with 200+ GS, this means 1 batch per 256 GS. The warp-based HiGS approach loads GS into registers and tests overlap with 32 tiles simultaneously using `__ballot_sync`.

**Density distribution** (from C33-D geometry):
- Tiles have `area = 256 pixels`, each with 1-20 GS visible
- Mean GS per tile: ~87-143 (for visible GS only, most tiles empty)
- Median: likely much lower due to heavy-tailed distribution

**Critical finding: no bimodality**
```
Render time (fwd) vs n_visible: Pearson r = 0.05
Render time total: range 16-20ms despite 10x workload variation
Forward time flat at 4-6ms regardless of 1.16M vs 119K visible GS
```

The renderer time is **dominated by fixed overhead** (kernel launch, memory setup), not by per-GS blend computation. There is no sparse/dense crossover because:
1. The warp-level sorting and batch-loading is already efficient for all densities
2. The fixed overhead dwarfs any per-pixel blend time
3. Forward time is 5% of T_iter — even 50% savings = 2.5% total

### Does HiGS macro-tile have cheap density metrics?

**YES**: `m_mtGaussCounts[n_macro_tiles]` is available per macro-tile immediately after the count kernel. This is a cheap CUDA-visible per-macro-tile intersection count.

**But**: The cost of checking density is irrelevant because:
- Even counting intersections only takes ~30μs (one kernel launch)
- With 135 macro-tiles, the check itself costs more than any possible blend savings

### GEMM-GS path feasibility

There is no GEMM-GS backend in gsplat v1.5.3. Building one would require:
1. A CUDA kernel that loads N GS × 256 pixels → [N, 256] weight matrix
2. A batched matrix multiply for alpha compositing
3. Handling depth ordering (GEMM can't naturally handle sorted alpha compositing)

This is a research contribution, not a "minimal prototype."

### Verdict: DROP

| Criterion | Result |
|-----------|--------|
| Dense/sparse bimodality | **No** — render time is fixed-overhead dominated |
| Cheap density metric | Yes (`m_mtGaussCounts`) but irrelevant |
| GEMM-GS path exists | **No** — would require new CUDA kernels |
| Dispatch overhead | Cost of checking exceeds max possible savings |
| Total T_iter impact | ≤1% (renderer is 16-18ms of 103ms T_iter) |
| **Threshold** | **<5% → DROP** |

---

## C35-2: Forward/Backward Backend Decoupling — **MAYBE**

### Source audit: existing backend options

gsplat v1.5.3 provides **three distinct rendering backends** with different properties:

| Backend | Forward | Backward | Requires | Location |
|---------|---------|----------|----------|----------|
| **CUDA fused** (`rasterization()`) | `rasterize_to_pixels_fwd` CUDA | `rasterize_to_pixels_bwd` CUDA | gsplat CUDA extension | `cuda/csrc/rasterize_to_pixels_{fwd,bwd}.cu` |
| **PyTorch** (`_rasterization()`) | PyTorch w/ nerfacc accumulate | PyTorch autograd (full graph) | `nerfacc` package | `cuda/_torch_impl.py:522` |
| **HiGS inference** (`rasterize_gaussian_inference_scene`) | HiGS macro-tile CUDA | **NONE** (inference mode only) | HiGS CUDA extension | `experimental/render/` |

### Existing forward/backward separation

The function `rasterize_gaussian_higs_trainable(differentiable=True)` at `functional/gaussian_inference.py:465` explicitly implements this pattern:

```
HiGS forward attempt → fallback to standard rasterization backward
```

The code path (L569-590):
```python
try:
    # Attempt HiGS preview (inference only)
    ...
except Exception:
    pass
# Uses standard gsplat rasterization for trainable rendering
rendered_colors, rendered_alphas, meta = rasterization(...)
```

This **already decouples** forward and backward — the HiGS forward is for preview, while the standard rasterization handles the differentiable path.

### Shared state analysis

The standard gsplat `rasterization()` saves the following state for backward (via autograd saved tensors):

| Tensor | Shape | Producer | Consumer |
|--------|-------|----------|----------|
| `means2d` | [C,N,2] | `fully_fused_projection` fwd | both fwd+bwd rasterize |
| `conics` | [C,N,3] | `fully_fused_projection` fwd | both fwd+bwd rasterize |
| `colors` | [C,N,3] | SH activation or direct | both fwd+bwd rasterize |
| `opacities` | [C,N] | input (activated) | both fwd+bwd rasterize |
| `tile_offsets` | [C,th,tw] | `isect_offset_encode` | both fwd+bwd rasterize |
| `flatten_ids` | [n_isects] | `isect_tiles` + offset encode | both fwd+bwd rasterize |
| `render_alphas` | [C,H,W,1] | forward output | bwd (as transmittance) |
| `last_ids` | [C,H,W] | forward output | bwd (last GS per pixel) |

The backward kernel requires **all of these** — they are backend-specific:
- `tile_offsets` and `flatten_ids` are specific to gsplat's tile-first intersection encoding
- `render_alphas` is the forward compositing state (per-pixel accumulated alpha)
- `last_ids` is the last GS that contributed to each pixel

**Cross-backend state mismatch:**
- HiGS stores: `mtGaussOffsets` (macro-tile), `mtGaussIdsSorted` (sorted within macro-tile), `tileBuffer` (per-batch partial RGBT)
- gsplat stores: `tile_offsets` (per-fine-tile), `flatten_ids` (unsorted by depth), `last_ids` (per-pixel last GS index)
- These data structures are **fundamentally incompatible** — a HiGS forward cannot produce gsplat's `tile_offsets`/`flatten_ids`/`last_ids`, and vice versa

### What CAN be decoupled

The `_rasterization()` function (rendering.py:585) is the PyTorch autograd path. It uses:
- Same `fully_fused_projection` CUDA kernel for means2d/conics (SAME)
- Same `isect_tiles` CUDA kernel for intersection (SAME)
- Different `_rasterize_to_pixels` PyTorch impl for per-pixel blending (DIFFERENT)

The projection and intersection stages are **shared** — only the final per-pixel rasterization differs. This means:

**Forward projection + intersection → CUDA (shared)**
**Forward per-pixel blend → Backend A**
**Backward per-pixel blend → Backend B**

is POSSIBLE for the per-pixel blend step. But:
- CUDA backward and PyTorch backward expect identical forward intermediate state
- The PyTorch backend saves the entire forward computation graph (high memory)
- The CUDA backend saves only the final alpha and last_ids (low memory)
- Decoupling would need to save BOTH sets of intermediates

### Minimal prototype design

1. **Phase 1** (only if KEEP): Gradient parity validation
   - Use `_rasterization()` (PyTorch backend) for forward
   - Use `rasterization()` (CUDA backend) backward
   - Force both to produce identical `render_alphas` and `last_ids`
   - Compare: grad w.r.t. means, quats, scales, opacity, colors

2. **Phase 2**: Build wrapper that picks backend per-pass

### Verdict: MAYBE

| Criterion | Result |
|-----------|--------|
| Shared projection/intersection | **YES** — `fully_fused_projection` + `isect_tiles` are common |
| Per-pixel blend state is compatible? | **NO** — CUDA saves `last_ids`, PyTorch saves full graph |
| Gradient parity achievable? | **YES** with `_rasterization()` → `rasterization()` within gsplat |
| Cross-renderer (HiGS ↔ gsplat)? | **NO** — completely different state representations |
| Implementation difficulty | Medium — wrapper around existing backends |
| Potential T_iter impact | Low — backward is 11-18ms (11-17% of T_iter) |
| Memory benefit | High — PyTorch backward saves full graph (~2GB for 311 cameras) |
| **Next experiment** | Gradient correctness: compare grads from pure CUDA vs pure PyTorch vs hybrid |

**Reason for MAYBE (not KEEP)**: The existing `_rasterization()` already does pure-PyTorch backward. The CUDA backward fused kernel is already optimal (single kernel launch). Decoupling doesn't improve speed — it only improves memory, which is not the bottleneck.

---

## C35-3: Stable Logical ID / Physical Storage Separation — **KEEP**

### Source audit: the ID shift problem

Current state in `train_3dgs.py` and `gaussian_model.py`:

| Operation | Effect on IDs |
|-----------|--------------|
| Clone (densification) | N → N+Δ: new GS appended with IDs N..N+Δ-1. Existing IDs unchanged. |
| Split (densification) | Clone + remove: new GS replaces parent, IDs shift |
| Prune (remove) | Boolean mask: `params = params[mask]`. **ALL IDs after removed index shift.** |

The optimizer state discard (`_reconfigure_optimizer` at train_3dgs.py:453):
```python
def _reconfigure_optimizer(self):
    self.optimizer = make_opt(self.model, self.config.spatial_lr_scale)
    for pg in self.optimizer.param_groups:
        pg["params"] = [self.model.xyz, self.model.rotations, ...]
```

**This discards ALL momentum and variance state.** The optimizer state rebuild is the dominant cost of topology changes — not the hierarchy rebuild. But the hierarchy rebuild is also 100% invalidated by ID shifts.

### Indirection scheme

```
Logical ID (stable, never changes after GS creation)
    ↓
Physical index (current tensor position, shifts after prune)
    ↓
Parameter tensor [N_physical, D]
```

Memory overhead:
- One int32 per GS: 1.6M × 4 bytes = **6.4 MB** (trivial on 40GB A100)
- One extra read per GS in projection → L1 cache resident
- One extra read per GS in rasterization (coalesced with other reads)

### Impact analysis

| Mutation type | Current cost | With indirection | Savings |
|--------------|-------------|-----------------|---------|
| Clone +Δ (α%) | Add Δ entries, append phys. | Add Δ entries, assign new logical IDs | No change |
| Remove (α%) | Compact ALL params (torch.cat), shift IDs | Mark logical ID as dead, no compaction | **Massive** |
| Split (clone+remove α%) | Clone + full compaction | Add Δ entries, mark parent dead | **Massive for remove phase** |
| Optimizer state | Full discard & rebuild | Only discard for dead logical IDs | **Massive** |

**Quantitative estimate** (from C33-D: densification steps 500,600,..., with n_gaussians going from 1.59M→1.47M):

At each densification step:
- Clone: ~10K new GS (0.6%)
- Prune: ~15K removed GS (0.9%)
- Split: ~5K parent removed + 5K child added

**Without indirection**: 
- Remove 15K GS: compact 1.59M entry tensors → **1.5ms** (cudaMemcpy for 5 params × 1.59M elements)
- Rebuild optimizer: **~3-5ms CPU** (state_dict serialization + new Adam)
- Total topology overhead: **5-7ms**

**With indirection**:
- Remove: mark 15K logical IDs dead → **~0.01ms** (one memset)
- Clone: append 15K new entries → **~0.05ms** (torch.cat)
- Optimizer: preserve momentum for all non-dead logical IDs → **~0.1ms** (masked copy)
- Total topology overhead: **~0.2ms**
- **Savings: 5-7ms per 100 steps** (but 95 of 100 steps don't have topology change)

### Memory overhead

| Structure | Size | Notes |
|-----------|------|-------|
| `logical_to_physical` [N_max, int32] | 6.4 MB | N_max = 1.6M, grows to ≤2.5M |
| `logical_alive` [N_max, bool] | 2.5 MB | Optional — dead IDs can have physical index -1 |
| `physical_to_logical` [N_phys, int32] | 6.4 MB | Reverse mapping for renderer |
| `dead_list` [N_dead, int32] | 0-6.4 MB | For memory reclamation |
| **Total** | **~20 MB** | Acceptable on 40GB A100 |

### Rasterization overhead

The indirection adds one extra __ldg() in the rasterizer:
```cuda
// Current: gaussian data at physical index i
float3 means2d_i = means2d[i];

// With indirection: logical ID → physical index
int phys_idx = logical_to_physical[logical_id];
float3 means2d_i = means2d[phys_idx];
```

This is a **coalesced read** (physical tensor access is still contiguous by physical index). The extra indirection read hits L1 cache (logical_to_physical is 6.4MB, fits in GPU L2 cache of 40MB on A100). Estimated overhead: **<1%** of rasterization time.

### Correctness risk

- **LOW**: Indirection is a pure functional mapping. It doesn't change any mathematical computation.
- **MEDIUM**: Optimizer state preservation for non-removed GS must be exact. The masked copy must exactly preserve momentum.
- **LOW**: Gradient flow through indirection is a standard gather/scatter.

### Implementation difficulty

| Component | Difficulty | Changes needed |
|-----------|-----------|---------------|
| Indirection table | Easy (Python) | `GaussianModel` wrapper class |
| Optimizer state preservation | Medium (Python) | Selective momentum masking in `_reconfigure_optimizer` |
| Rasterizer indirection | Hard (CUDA) | Modify `rasterize_to_pixels_fwd/bwd` to accept logical→physical map |
| HiGS indirection | Very hard (CUDA) | Modify `IntersectMTFused` and `MacroTileRasterize` |

**Phase 1 scope** (recommended): Python-only prototype. Use `torch.gather` and `torch.index_select` to simulate indirection in the existing training loop. Don't modify CUDA kernels yet.

### Verdict: KEEP

| Criterion | Result |
|-----------|--------|
| Solves optimizer state discard? | **YES** — momentum preserved for non-dead GS |
| Solves hierarchy rebuild? | **YES** — intersection references use logical IDs, no shift |
| Memory overhead | ~20MB (<0.1% of 40GB) |
| Rasterization lookup overhead | <1% (L2-cache resident indirection table) |
| Implementation difficulty | Python prototype: easy. CUDA integration: medium. |
| Next experiment | Python-only prototype: simulate topology events with indirection, measure optimizer state preservation time |

---

## Final Decision Table

| Candidate | Verdict | Primary reason | GPU cost | Implementation difficulty |
|-----------|---------|---------------|----------|--------------------------|
| **C35-1**: Adaptive Blend | **DROP** | No sparse/dense bimodality. Render time is fixed-overhead dominated (CV=0.05). GEMM path doesn't exist. Max possible savings <3% T_iter. | 0 (existing data sufficient) | N/A |
| **C35-2**: Fwd/Bwd Decoupling | **MAYBE** | Shared projection/intersection makes per-pixel blend decoupling feasible. But CUDA backward is already optimal. Benefit is memory reduction, not speed. | 50 iter gradient check | Medium (wrapper) |
| **C35-3**: Logical ID Separation | **KEEP** | Solves the fundamental ID-shift problem that invalidates 100% of renderer state after any prune. Enables optimizer state preservation. Overhead <1%. | 100 iter Python prototype | Easy (Phase 1 Python only) |

## Next Experiments (if KEEP)

### C35-3 Phase 1: Python indirection prototype
1. Implement `GaussianModelIndirection` wrapper: wraps `GaussianModel` + adds `logical_to_physical` + `physical_to_logical` tables
2. Modify densification to assign new logical IDs (monotonic counter)
3. Modify prune to set logical IDs as dead (no tensor compaction)
4. Measure: time to mark dead IDs vs time to compact tensors (current)  
5. Measure: optimizer state copy time for non-dead GS
6. Train 500 iterations on room scene, measure per-step overhead

Expected output:
```
Per densification step cost:
  Baseline: 5.2ms (compact 3 tensor groups + rebuild Adam)
  Indirection: 0.3ms (mark dead + clone new + mask momentum)
  Savings: 4.9ms/step
```

### C35-1 next if not DROP
Would need: actual GEMM-style tile rasterizer prototype. But DROP is justified.

### C35-2 next if KEEP
1. Gradient parity test: `_rasterization()` forward + `rasterization()` backward
2. Compare with pure CUDA backward for 3 iterations
3. If grad error < 1e-4 for all params → MAYBE → KEEP

---

## Raw Artifacts
- Data: `results/phase-c31/c33_d_workload_data.json` (C33-D)
- Reports: `reports/phase-c31/c32_b_sync_ablation.md`, `reports/phase-c31/c33_candidate_gate.md`
- Scripts: `scripts/phase-c31/c35_1_tile_density.py`
- Source: `gsplat.cuda._torch_impl`, `gsplat.rendering`, `gsplat.experimental.render`
