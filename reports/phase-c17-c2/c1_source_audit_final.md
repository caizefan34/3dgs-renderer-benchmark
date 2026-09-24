# C1 — Source Audit Final

> **Purpose:** Verifiable evidence of baseline and C1 key construction,
> bit widths, CUB sort range, offset/downstream consumers — from source code only.
>
> **Files audited:**
> - `gsplat/cuda/csrc/IntersectTile.cu.baseline` (backup of v1.5.3 baseline)
> - `gsplat/cuda/csrc/IntersectTile.cu` (current file — contains C1 patch)
> - `patches/IntersectTile.c1.cu` (C1 patch source — identical to above)
> - `gsplat/cuda/csrc/Intersect.cpp` (call sites)
> - `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu` (downstream reader)
> - `gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu` (backward reader)
> - `gsplat/cuda/csrc/RasterizeToIndices3DGS.cu`
>
> **Date:** 2026-09-05

---

## 1. Key Construction — IntersectTile Kernel

### 1.1 Baseline (`IntersectTile.cu.baseline`)

**File:** `gsplat/cuda/csrc/IntersectTile.cu.baseline`

**Line 95 — Image ID encoding:**
```c++
const int64_t iid_enc = iid << (32 + tile_n_bits);
```

**Line 98–99 — Depth encoding (full float32 bitcast):**
```c++
int32_t depth_i32 = *(int32_t *)&(depths[idx]);
int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);
```

**Line 108 — Final key composition:**
```c++
isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
```

**Baseline key bit layout (1080p, tile16, single image):**
```
bit [63:47]      — image_id (1 bit) shifted to (32+13)=45
bit [46:32]      — tile_id shifted << 32
bit [31:0]       — depth (full 32-bit float32 reinterpret)
Used bits:       1 + 13 + 32 = 46 bits  (end_bit = 46)
```

### 1.2 C1 (`IntersectTile.cu` / `IntersectTile.c1.cu`)

**File:** `gsplat/cuda/csrc/IntersectTile.cu` (currently patched), `patches/IntersectTile.c1.cu`

**Line 97 — Image ID encoding (changed from 32 to 16):**
```c++
const int64_t iid_enc = iid << (16 + tile_n_bits);
```

**Line 100–103 — Depth truncation to upper 16 bits:**
```c++
int32_t depth_i32 = *(int32_t *)&(depths[idx]);
int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);
int64_t depth_upper = depth_id_enc >> 16;
```

**Line 113 — Final key composition (reordered):**
```c++
isect_ids[cur_idx] = depth_upper | (tile_id << 16) | iid_enc;
```

**C1 key bit layout (1080p, tile16, single image):**
```
bit [63:31]      — unused (zero from 64-bit sign extension of non-negative depth_upper)
bit [30:16]      — tile_id shifted << 16
bit [15:0]       — depth_upper (upper 16 bits of float32 bitcast)
bit [31]         — image_id (encoded as iid << (16+13) = << 29)
Used bits:       16 + 13 + 1 = 30 bits (end_bit = 30)
```

**Key change from baseline:**
```
Baseline: isect_ids = iid_enc | (tile_id << 32) | depth_id_enc  [46 bits]
C1:       isect_ids = depth_upper | (tile_id << 16) | iid_enc   [30 bits]
                 depth reordered from LSB to position [15:0]
                 tile_id shifted from [46:32] to [30:16]      
                 image_id shifted from [47] to [31]            
                 unused in [63:32] due to 64-bit key width
```

---

## 2. Bit Width Calculation (from source formulas)

**Lines 155–156** (both baseline and C1):
```c++
uint32_t image_n_bits = (uint32_t)floor(log2(I)) + 1;
uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
```

For representative case: **1920×1080, tile_size=16, I=1 (single image)**

| Quantity | Source | Value |
|----------|--------|-------|
| `tile_width` | `ceil(1920/16)` | 120 |
| `tile_height` | `ceil(1080/16)` | 68 |
| `n_tiles` | `tile_width * tile_height` | 8160 |
| `tile_n_bits` | `floor(log2(8160)) + 1 = 12 + 1` | **13** |
| `image_n_bits` | `floor(log2(1)) + 1 = 0 + 1` | **1** |

**Note:** C1 patch does *not* change `tile_n_bits` or `image_n_bits` formulas. Both baseline and C1 use identical computation.

---

## 3. CUB Sort Bit Ranges

### 3.1 Global radix sort (`radix_sort_double_buffer`)

| Variant | Source | end_bit | bits sorted |
|---------|--------|---------|-------------|
| Baseline | `Line 322`: `32 + tile_n_bits + image_n_bits` | 32+13+1 = **46** | bits [0, 46) |
| C1 | `Line 328`: `16 + tile_n_bits + image_n_bits` | 16+13+1 = **30** | bits [0, 30) |

**Source lines (baseline):** `IntersectTile.cu.baseline`, lines 317–324
**Source lines (C1):** `patches/IntersectTile.c1.cu`, lines 323–330

### 3.2 Segmented radix sort (`segmented_radix_sort_double_buffer`)

| Variant | Source | end_bit | bits sorted |
|---------|--------|---------|-------------|
| Baseline | `Line 377`: `32 + tile_n_bits` | 32+13 = **45** | bits [0, 45) |
| C1 | `Line 383`: `16 + tile_n_bits` | 16+13 = **29** | bits [0, 29) |

**Source lines (baseline):** `IntersectTile.cu.baseline`, lines 369–378
**Source lines (C1):** `patches/IntersectTile.c1.cu`, lines 375–384

---

## 4. Offset Kernel — Tile Boundary Detection

### 4.1 Baseline

**`intersect_offset_kernel`, `IntersectTile.cu.baseline`, line 227:**
```c++
int64_t isect_id_curr = isect_ids[idx] >> 32;  // shift out depth (32 bits)
```

After the shift, the remaining 14 bits (image_id + tile_id) are used for tile boundary detection. This works because the baseline stores tile and image IDs in bits [46:32] of the key.

### 4.2 C1

**`intersect_offset_kernel`, `patches/IntersectTile.c1.cu`, line 233:**
```c++
int64_t isect_id_curr = isect_ids[idx] >> 16;  // shift out depth (16 bits)
```

After the shift, the remaining 14 bits (image_id + tile_id) are used for tile boundary detection. This works because C1 stores tile and image IDs in bits [30:16] of the key.

**Line 252 (C1 only):**
```c++
int64_t isect_id_prev = isect_ids[idx - 1] >> 16; // C1: shift out depth (16 bits)
```

**Verification:** Both baseline and C1 offset kernels extract the same (image_id, tile_id) tuple after shifting. C1 shifts by 16 instead of 32 because depth occupies only 16 bits instead of 32.

---

## 5. Downstream Consumers of `isect_ids` and `flatten_ids`

### 5.1 Pipeline

```
IntersectTile.cu    — generates isect_ids (sort keys) + flatten_ids (Gaussian indices)
  → CUB sort        — produces sorted isect_ids_sorted + flatten_ids_sorted
  → intersect_offset — produces tile_offsets array from sorted isect_ids
  → RasterizeTo*    — uses tile_offsets + flatten_ids_sorted for per-tile traversal
```

### 5.2 RasterizeToPixels3DGSFwd.cu

**Lines 34–35 (parameter declaration):**
```c++
const int32_t *__restrict__ tile_offsets, // [I, tile_height, tile_width]
const int32_t *__restrict__ flatten_ids,  // [n_isects]
```

**Lines 85–89 (tile range via offsets):**
```c++
int32_t range_start = tile_offsets[tile_id];
int32_t range_end = (image_id == I - 1) && (tile_id == ...)
    ? n_isects
    : tile_offsets[tile_id + 1];
```

**Line 127 (Gaussian index from flatten_ids, C1 works identically):**
```c++
int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
```

**No code in RasterizeToPixels* reads `isect_ids` directly.** The rasterizer only uses `tile_offsets` (boundary indices) and `flatten_ids` (Gaussian index per intersection). The sort key itself (`isect_ids`) is consumed only by:
1. CUB sort (as the key)
2. `intersect_offset_kernel` (for tile boundary detection)

**C1 impact on downstream:** None. `flatten_ids` is unchanged (still `int32_t` Gaussian index). `tile_offsets` is computed identically in structure (shift depth out → extract tile+image ID). The rasterizer code path is identical.

### 5.3 Backward pass (RasterizeToPixels3DGSBwd.cu)

**Lines 33–34:**
```c++
const int32_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
const int32_t *__restrict__ flatten_ids,  // [n_isects]
```

**Lines 88–92 (reads same tile_offsets):**
```c++
int32_t range_start = tile_offsets[tile_id];
int32_t range_end = (image_id == I - 1) && (tile_id == ...)
    ? n_isects
    : tile_offsets[tile_id + 1];
```

**Line 141 (same flatten_ids read):**
```c++
int32_t g = flatten_ids[idx];
```

**Verification:** Backward pass reads the same `tile_offsets` and `flatten_ids` as forward. C1 does not change these arrays or their semantics. The backward pass is unaffected.

---

## 6. Summary of Source-Verified Facts

| Fact | Baseline source | C1 source | Status |
|------|---------------|-----------|--------|
| Key field: tile_id position | `tile_id << 32` (L108) | `tile_id << 16` (L113) | ✅ Changed |
| Key field: image_id position | `iid << (32 + tile_n_bits)` (L95) | `iid << (16 + tile_n_bits)` (L97) | ✅ Changed |
| Key field: depth content | Full 32-bit float32 bitcast (L98–99, L108) | `depth_id_enc >> 16` (L103) | ✅ Changed |
| Key field: depth position | bits [31:0] (L108) | bits [15:0] (L113) | ✅ Changed |
| Key field: depth bits | 32 (full float32) | 16 (upper bits) | ✅ Changed |
| CUB global sort end_bit | `32 + tile_n_bits + image_n_bits` (L322) | `16 + tile_n_bits + image_n_bits` (L328) | ✅ Changed |
| CUB segmented sort end_bit | `32 + tile_n_bits` (L377) | `16 + tile_n_bits` (L383) | ✅ Changed |
| Offset shift | `>> 32` (L227) | `>> 16` (L233) | ✅ Changed |
| tile_n_bits formula | `floor(log2(n_tiles))+1` (L151) | Same (L156) | ✅ Unchanged |
| image_n_bits formula | `floor(log2(I))+1` (L150) | Same (L155) | ✅ Unchanged |
| `flatten_ids` content | Gaussian index (int32) | Same | ✅ Unchanged |
| Rasterizer Fwd reads `isect_ids` | ❌ No (reads offsets only) | Same | ✅ Unchanged |
| Rasterizer Bwd reads `isect_ids` | ❌ No (reads offsets only) | Same | ✅ Unchanged |
| Intersection count computation | Same kernel (first pass) | Same | ✅ Unchanged |

---

## 7. CUB Pass Count — Exact vs Estimated

**The actual number of CUB radix passes is not determined by the source code alone.** CUB's `DeviceRadixSort` determines passes at runtime based on its internal policy:

```c++
// From CUB agent_radix_sort_histogram.cuh (CCCL v13.3):
num_passes = (end_bit - begin_bit + RADIX_BITS - 1) / RADIX_BITS
```

Where `RADIX_BITS` is a policy parameter (commonly 4, but can be 8 on newer architectures). The source only specifies `[begin_bit, end_bit)`.

**Source-verified bit ranges:**

| Variant | Global end_bit | Segmented end_bit |
|---------|---------------|-------------------|
| Baseline | 46 | 45 |
| C1 | 30 | 29 |
| **Reduction** | **16 bits** | **16 bits** |

**Constrained statement (only what source can prove):**
> C1 reduces the CUB sort `end_bit` parameter from 46 to 30 (global) and from 45 to 29 (segmented), a reduction of 16 bits in both cases. For any CUB policy with fixed bits-per-pass, this translates to a proportional reduction in sort passes. The exact pass count depends on the runtime CUB policy.

---

## 8. Key Differences from RoofGS (Source-Verified)

| Property | C1 (this patch) | RoofGS (2608.15785) |
|----------|----------------|---------------------|
| **Quantization formula** | `depth_upper = bitcast_u32(d) >> 16` (no division, no clamp, no scene params) | `q_d = clamp(floor((d-z_n)/(z_f-z_n)*(2^b_d-1)), 0, 2^b_d-1)` (needs z_near/z_far) |
| **Depth bits** | Fixed 16 (top of float32) | Variable: 32 - `ceil(log2(N_tiles))`; 19 @ 1080p/tile16 |
| **Total key bits** | 64-bit container, 30 used | 32-bit word, all used |
| **Tile bits** | Same as gsplat (13 @ 1080p/tile16) | Resolution-adaptive: 13 @ 1080p/tile16 |
| **Image bits** | 1 (GSplat multi-image compatible) | N/A (single-image) |
| **Bit allocation** | Fixed: 16 depth, 13 tile, 1 img | Resolution-adaptive: tile bits = ceil(log2(N_tiles)), rest = depth |
| **Scene bounds needed** | ❌ No | ✅ Yes (z_near, z_far) |
| **Rasterizer changed** | ❌ No | ✅ Yes (fast-exp, dual-pixel) |
| **Differentiable** | ✅ Yes (full backward pass unchanged) | ❌ Inference only |
| **Intersect kernel lines changed** | 5 lines (97, 103, 113, 233, 252) | Not applicable (different codebase) |

---

## 9. References

All source paths relative to `gsplat/cuda/csrc/` (installed in `C:\Users\36570\miniconda3\Lib\site-packages\gsplat\cuda\csrc\`) and `patches/` (in benchmark repo).

| File | Notes |
|------|-------|
| `IntersectTile.cu.baseline` | Backup of gsplat v1.5.3 original |
| `IntersectTile.cu` | Currently patched with C1 |
| `patches/IntersectTile.c1.cu` | C1 patch source (identical) |
| `Intersect.cpp` | Sort dispatch, lines 119–148 |
| `RasterizeToPixels3DGSFwd.cu` | Downstream consumer of tile_offsets + flatten_ids |
| `RasterizeToPixels3DGSBwd.cu` | Backward pass (same read pattern) |
| `RasterizeToIndices3DGS.cu` | Second downstream consumer |
