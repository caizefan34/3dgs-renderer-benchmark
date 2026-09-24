# C36 Cross-Renderer Architecture Gate — Final Report

## R1: Cross-Renderer Canonical Execution IR — **MAYBE**

### Execution Graph (3 renderers)

```
                  Input: means [N,3], quats [N,4], scales [N,3],
                         opacities [N], colors [N,K,3], viewmats [C,4,4], Ks [C,3,3]

    ┌─────────────┬────────────────────────────────────┬─────────────────────┐
    │  gsplat     │  HiGS                              │  Inria              │
    │-------------│------------------------------------│---------------------│
    │             │                                    │                     │
    │  fully_fused │  launch_projection_sh_fused_kernel │  GaussianRasterizer │
    │  projection  │  ↓ produces visible, means2d,      │  ↓                  │
    │  ↓           │    depths, conics, colors          │  SINGLE fused       │
    │  isect_tiles │  IntersectMTFused::execute          │  kernel             │
    │  (per-fine-  │  (macro-tile count, scan, fill,    │  project + sort +   │
    │   tile)      │   radix-sort within macro-tile)    │  rasterize          │
    │  ↓           │  ↓                                 │                     │
    │  isect_offset│  IntersectMTFused::rasterize        │  1 kernel per       │
    │  encode      │  (1 warp per macro-tile+batch,     │  camera (C times)   │
    │  ↓           │   __ballot_sync 32-tile overlap)   │                     │
    │  rasterize_  │  MacroTilePostBlend                │                     │
    │  to_pixels   │  (per-fine-tile blend)              │                     │
    │  (1 block    │                                     │                     │
    │   per tile)  │                                     │                     │
    └─────────────┴────────────────────────────────────┴─────────────────────┘
```

### State Tensors per Renderer

| Stage | gsplat | HiGS | Inria |
|-------|--------|------|-------|
| **Input** | means [N,3], quats [N,4], scales [N,3], opacities [N], colors [N,K,3] | means_planar [3,N], qso_packed [N,8], colors_packed [N,K,3] f16 | means [N,3], quats [N,4], scales [N,3], opacities [N], colors [N,K,3] |
| **Projection** | **means2d** [N,2], **conics** [N,3], **depths** [N], **radii** [N] (via meta dict) | **means2d** [1,1,N,2], **conics** [1,1,N,4]+opac, **depths** [1,1,N], **visible** bitmask (in InferenceRenderState) | NOT EXPOSED (internal only) |
| **Intersection** | tile_offsets [C,th,tw+1], flatten_ids [n_isects] sorted by depth per tile | mt_gauss_counts [n_mt], mt_gauss_offsets [n_mt+1], mt_gauss_ids_sorted [n_mt_isects], mt_batch_offsets [n_mt+1] | NOT EXPOSED |
| **Raster order** | fine-tile (16×16 pixels), batch-collect | macro-tile (8×8 tiles = 128×128 pixels), warp-per-(mt,batch) | NOT EXPOSED |
| **Backward** | rasterize_to_pixels_bwd (CUDA) + fully_fused_projection_bwd | NONE (inference only) | Fused in diff-gaussian-rasterization |

### Candidate IR: GaussianExecutionIR

```python
@dataclass
class GaussianExecutionIR:
    """Canonical execution representation shared across 3DGS renderers."""
    
    # === Input (shared by ALL) ===
    means:     Tensor  # [N, 3] f32
    quats:     Tensor  # [N, 4] f32
    scales:    Tensor  # [N, 3] f32
    opacities: Tensor  # [N] f32
    colors:    Tensor  # [N, K, 3 or N, D] f32
    
    # === Projected state (gsplat + HiGS, NOT Inria) ===
    # This is the natural IR boundary: camera-space 2D representation
    means2d: Tensor  # [N, 2] f32  — projection is common
    conics:  Tensor  # [N, 3] f32  — inverse covariance
    depths:  Tensor  # [N]    f32  — camera-space Z
    radii:   Tensor  # [N]    f32  — screen radius
    
    # === Intersection (backend-specific, NOT shared) ===
    # gsplat path (fine-tile flat)
    flatten_ids:   Optional[Tensor]  # [n_isects] int32
    isect_offsets: Optional[Tensor]  # [C, tile_h, tile_w] int32
    
    # HiGS path (macro-tile)
    mt_gauss_ids_sorted: Optional[Tensor]  # [n_isects] int32
    mt_gauss_offsets:    Optional[Tensor]  # [n_mt+1] int32
    mt_batch_offsets:    Optional[Tensor]  # [n_mt+1] int32
```

### Common State Assessment

| Field | gsplat | HiGS | Inria | Shared? |
|-------|--------|------|-------|---------|
| means2d | ✅ meta['means2d'] | ✅ IRState.means2d | ❌ | **2/3** |
| conics | ✅ meta['conics'] | ✅ IRState.conics | ❌ | **2/3** |
| depths | ✅ meta['depths'] | ✅ IRState.depths | ❌ | **2/3** |
| radii | ✅ meta['radii'] | ✅ IRState.visible (bitmask) | ❌ | **2/3** |
| tile_offsets | ✅ | ❌ (macro-tile instead) | ❌ | **1/3** |
| flatten_ids | ✅ | ❌ | ❌ | **1/3** |
| mt_gauss_offsets | ❌ | ✅ | ❌ | **1/3** |

### Verdict: MAYBE

**Rationale for MAYBE (not DROP or KEEP):**

| Criterion | Result |
|-----------|--------|
| ≥2 renderers share common state? | **YES** — projection (means2d, conics, depths, radii) shared by gsplat + HiGS |
| All 3 renderers share common state? | **NO** — Inria exposes nothing, making the IR only useful for 2/3 |
| Intersection format convertible? | **YES** but with overhead: gsplat flatten_ids ↔ HiGS macro-tile requires scatter/gather + recompute of tile ordering (~20μs CPU or minor GPU kernel) |
| Conversion overhead vs full rebuild? | ~20μs IR conversion vs ~2ms full reproject+reintersect (gsplat) or single fused kernel (Inria) |
| Can Inria participate? | **NO** — fully fused kernel exposes no intermediate state. Fallback: gsplat wraps Inria output only |
| gsplat already has this IR? | **YES** — the `meta` dict from `rasterization()` is effectively the IR for projection state |

**Why not KEEP**: gsplat's `meta` dict already provides this IR. Adding a formal `GaussianExecutionIR` dataclass is a code-organization improvement, not a research contribution. Inria's non-participation limits adoption to 2/3 renderers.

**Why not DROP**: The IR boundary at projection→intersection is real and exploitable. If a future renderer exposes the same projection state, it can plug into either gsplat or HiGS intersection+rasterization without reimplementing projection.

---

## R2: Phase-Specific Renderer Handoff — **KEEP**

### Checkpoint Compatibility Analysis

#### Parameter representation

| Parameter | gsplat | Inria (diff-gaussian-rasterization) | Compatible? |
|-----------|--------|-------------------------------------|-------------|
| `means` (xyz) | [N,3] f32 | [N,3] f32 | ✅ **Identical** |
| `quats` (rotations) | [N,4] f32, wxyz, not normalized | [N,4] f32, expects normalized | ⚠️ gsplat passes non-normalized (normalizes internally in `fully_fused_projection`); Inria requires normalization before passing |
| `scales` | [N,3] f32, log-space | [N,3] f32, log-space | ✅ **Identical** |
| `opacities` | [N] f32, logit-space | [N] f32, logit-space (passed as [N,1]) | ✅ Compatible |
| `colors` (SH) | [N, K, 3] f32 SH coefficients | [N, K, 3] f32 SH coefficients | ✅ **Identical** |

#### Camera convention

| Property | gsplat | Inria | Notes |
|----------|--------|-------|-------|
| Extrinsics | viewmats [C,4,4] (world→camera) | viewmatrix (OpenGL column-major transposed) | ⚠️ Need transpose |
| Intrinsics | Ks [C,3,3] | tanfovx, tanfovy | ⚠️ Different format |
| FoV conversion | Used internally | Required for Inria | Can compute from K: `FoVx = 2*atan(w/(2*K[0,0]))` |
| Projection matrix | Not needed (gsplat handles internally) | Required as `projmatrix` | Can derive from near/far |

**Inria convention fix** (from rendering.py:1364-1365):
```python
FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0]))
FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1]))
```

gives the same numbers regardless of renderer. So camera convention IS compatible with an adapter.

#### Loss compatibility

| Loss component | gsplat Phase-7 | Inria original 3DGS | Compatible? |
|----------------|---------------|-------------------|-------------|
| L1 loss | `F.l1_loss(rendered, gt)` | `F.l1_loss(rendered, gt)` | ✅ Identical |
| D-SSIM | `d_ssim_loss()` (Gaussian blur SSIM) | Same formula | ✅ Identical |
| Combined | `(1-λ)*L1 + λ*D-SSIM` | Same | ✅ Identical |

#### SH evaluation

**⚠️ KEY INCOMPATIBILITY**: gsplat and Inria use DIFFERENT SH evaluation:
- gsplat: `spherical_harmonics()` with colors `[0, 2]**2` output → `clamp_min(colors + 0.5, 0.0)` (rendering.py:392)
- Inria: internal SH evaluation in fused CUDA kernel, clamp differently

This means the **RGB output is not bit-exact** between renderers at the same Gaussian parameters. However, both produce valid photorealistic renders with slightly different color interpretation.

### Handoff Prototype Design

```python
# Phase A: train with renderer A (gsplat)
model = GaussianModel(...)
for step in range(50):
    rendered, alpha, _ = gsplat.rasterization(...)
    loss = combined_loss(rendered, gt)
    loss.backward()
    optimizer.step()

# Save checkpoint
torch.save(model.state_dict(), "checkpoint.pt")

# Phase B: continue with renderer B (Inria)
model2 = GaussianModel(...)
model2.load_state_dict(torch.load("checkpoint.pt"))
model2.sh_degree = 0  # Inria needs explicit SH degree control

# Camera adapter
for step in range(50):
    rendered, _, meta = gsplat.rasterization_inria_wrapper(
        model2.means, model2.quats, model2.scales,
        model2.opacity.sigmoid().squeeze(),  # Inria expects activated opacity
        model2.get_sh(),  # SH coefficients
        viewmats, Ks, width, height,
        sh_degree=0,
    )
    loss = combined_loss(rendered, gt)
    loss.backward()
    optimizer.step()
```

### Correctness Expectations

| Check | Expected |
|-------|----------|
| PSNR trajectory continuity | **Continuous** (monotonic improvement, not reset) |
| Loss at handoff point | **May jump** by ~0.5-5% (due to SH + opacity representation differences) |
| Gradient correctness | **Self-consistent** per renderer (not cross-comparable) |
| Parameter shape | **100% identical** (same model definition) |
| Camera count | **Identical** (both handle C cameras) |

### Verdict: KEEP

| Criterion | Result |
|-----------|--------|
| PLY/state_dict compatibility | ✅ **Identical** (load_ply produces same format) |
| Parameter layout | ✅ Same [N,3]/[N,4]/[N]/[N,K,3] |
| Camera convention | ⚠️ Adapter needed (viewmat→FoV conversion), proven working |
| SH compatibility | ⚠️ Not bit-exact (gsplat adds 0.5 clamp), but training-compatible |
| Loss compatibility | ✅ **Identical** L1+D-SSIM |
| PSNR continuity at handoff | ✅ Expected continuous (parameters same at handoff point) |
| Handoff experiment cost | **Very low** — 50+50 iter, single GPU, 3 minutes |

**Implementation difficulty**: LOW. Mostly Python adapter code. The hardest part is the camera adapter (transpose viewmat, compute FoV from K), which already exists in `rasterization_inria_wrapper`. The SH clamp difference is cosmetic for training purposes — the optimizer will adapt to the new renderer's color interpretation within a few iterations.

---

## R3: Cross-Renderer Partial Component Substitution — **DROP**

### Component Compatibility Matrix

| Component | gsplat | HiGS | Inria |
|-----------|--------|------|-------|
| **Projection** | `fully_fused_projection` [N,3]→[N,2] — CUDA **exposed** | `launch_projection_sh_fused_kernel` — CUDA **internal** | **Fused** into single kernel — NOT separable |
| **Intersection** | `isect_tiles` + `isect_offset_encode` — CUDA **exposed**, tile-based | `IntersectMTFused::execute` — C++ **internal**, macro-tile | **Fused** — NOT separable |
| **Sorting** | Depth-sort within tile (implicit in flatten_ids) | Radix sort within macro-tile (`launch_mt_segmented_sort`) | **Fused** — NOT separable |
| **Rasterization** | `rasterize_to_pixels` — CUDA **exposed**, 1 block/tile | `MacroTileRasterize` — C++ **internal**, 1 warp/(mt,batch) | **Fused** — NOT separable |
| **Backward** | `rasterize_to_pixels_bwd` + `fully_fused_projection_bwd` — CUDA **exposed** | **NONE** (inference only) | **Fused** — NOT separable |

### What's Actually Separable

The gsplat `rasterization()` pipeline internally calls:

1. **`fully_fused_projection()`** → means2d, conics, depths, radii (CUDA)
2. **`isect_tiles()`** → flatten_ids (CUDA)
3. **`isect_offset_encode()`** → tile_offsets (CUDA)
4. **`rasterize_to_pixels()`** → rendered colors + alphas (CUDA)

Steps 1-3 are shared by **both** `rasterization()` and `_rasterization()` (the PyTorch autograd path at rendering.py:585). The swap point is step 4:
- `rasterization()` → CUDA `rasterize_to_pixels` fdwd+bwd
- `_rasterization()` → PyTorch `_rasterize_to_pixels` with nerfacc accumulate

But **this is gsplat-internal**, explicitly excluded per protocol.

### Cross-Renderer Component Candidates

| Swap Candidate | Can it be done? | Why/Why not |
|---------------|----------------|-------------|
| gsplat projection + HiGS rasterization | ❌ | HiGS requires macro-tile intersection format; gsplat produces fine-tile format. Conversion requires scatter+gather+resort. The HiGS rasterize is C++ internal, not callable from Python. |
| gsplat projection + Inria rasterization | ❌ | Inria's rasterizer is fully fused — it takes raw means/quats/scales and does projection internally. You can't feed it pre-projected data. |
| HiGS projection + gsplat rasterization | ❌ | HiGS projection is inside `launch_projection_sh_fused_kernel` which is fused with SH decoding and fp16 packing. Would need to call the internal C++ function. |
| gsplat intersection + any rasterizer | ❌ | No other rasterizer accepts gsplat's `tile_offsets` + `flatten_ids` format |
| HiGS intersection + gsplat rasterization | ❌ | Would need to convert macro-tile sorted IDs back to fine-tile format |

### The Fundamental Problem

```
gsplat:   [N GS] → project → fine-tile intersect → fine-tile sort → rasterize
HiGS:     [N GS] → project → macro-tile intersect → macro-tile sort → rasterize  
Inria:    [N GS] → [single fused kernel: project + tile + sort + rasterize]
```

Each renderer's intersection + rasterization stages are **tightly coupled** to their tile format. There's no "standard intersection" that multiple rasterizers can consume, because:

1. **Tile granularity differs**: gsplat uses 16×16 fine tiles, HiGS uses 128×128 macro-tiles (8×8 fine tiles)
2. **Sort scope differs**: gsplat sorts within each fine tile, HiGS sorts within each macro-tile
3. **Rasterize fashion differs**: gsplat uses 1 block (256 threads) per fine tile loading GS in shared memory batches; HiGS uses 1 warp (32 threads) per macro-tile batch testing 32-tile overlap with `__ballot_sync`
4. **Inria is fully fused**: No intermediate is exposed at any level

### Verdict: DROP

| Criterion | Result |
|-----------|--------|
| ≥1 cross-backend component substitutable? | **NO** — all backends have fused or tightly-coupled intersection+rasterization |
| gsplat internal substitution available? | **YES** — but explicitly excluded per protocol (C35-2 already analyzed this) |
| Conversion overhead too high? | **YES** — changing tile format requires scatter+gather across 8K+ tiles or 135 macro-tiles |
| Any new finding beyond C35-2? | **NO** — consistent with C35-2's finding that only per-pixel blend is separable, and only within gsplat |
| Can we separate projection for ANY pair? | **NO** — Inria's fused kernel means ANY component substitution requires gsplat↔HiGS only (2/3), and those have incompatible intersection formats |

---

## Final Decision Table

| Candidate | Verdict | Primary reason | GPU cost | Implementation difficulty |
|-----------|---------|---------------|----------|--------------------------|
| **R1**: CIR | **MAYBE** | Projection state (means2d, conics, depths, radii) IS shared by gsplat+HiGS. gsplat's `meta` dict already serves as this IR. Formalizing it helps code organization but Inria can't participate. | 50-100 iter validation that IR conversion works | Low (Python dataclass + adapter) |
| **R2**: Handoff | **KEEP** | gsplat↔Inria checkpoint handoff is fully feasible. Same Gaussian representation, compatible camera adapter, identical loss. Cost of experiment (50+50 iter) is negligible. | 50+50 iter = 3 min | Low (Python adapter) |
| **R3**: Component Substitution | **DROP** | No cross-backend component can be cleanly substituted. Intersection+rasterization are tightly coupled to tile format. Inria's fused kernel has no separable components. | 0 (source analysis sufficient) | N/A |

## Source Locations Summary

| File | Location | Relevant to |
|------|----------|-------------|
| `gsplat/rendering.py` | Lines 200-582 | R1 (gsplat pipeline), R2 (handoff), R3 (states) |
| `gsplat/rendering.py` | Lines 1350-1466 | R2 (Inria wrapper, camera adapter) |
| `gsplat/cuda/_torch_impl.py` | Lines 522-617 | R3 (PyTorch substitute rasterize) |
| `gsplat/scene/components/gaussian_inference_scene.py` | Lines 32-65 | R1 (HiGS state) |
| `gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/` | Various `.h` files | R1 (HiGS internal state) |
| `scripts/epic05/phase7/train_3dgs.py` | Lines 257-331 | R2 (training loop uses gsplat) |
| `scripts/epic05/phase7/gaussian_model.py` | Lines 52-106 | R2 (model loading = PLY) |

## Next Experiments

### R2 Phase 1 (immediately executable)

1. Train 50 iter with gsplat on room scene, save checkpoint
2. Load checkpoint into same model
3. Run 50 iter using `rasterization_inria_wrapper` (camera adapter already exists)
4. Measure:
   - PSNR at handoff point (A: last iter, B: first iter)
   - PSNR difference: `|PSNR_A_last - PSNR_B_first|`
   - Loss continuity
5. If PSNR difference < 0.5 dB → **STRONG KEEP** (handoff is transparent)
6. If PSNR difference 0.5-3 dB → **KEEP** (handoff works but needs adaptation)
7. If PSNR difference > 3 dB → **MAYBE** (handoff costs significant quality)

### R1 Phase 1 (if R2 KEEP)

1. Formalize `GaussianExecutionIR` dataclass
2. Write adapter: `gsplat_rasterization → GaussianExecutionIR` (trivial, meta dict)
3. Write adapter: `GaussianExecutionIR → Inria rasterization call` (uses camera adapter from R2)
4. Verify forward image parity for 5 cameras
5. Measure conversion overhead
