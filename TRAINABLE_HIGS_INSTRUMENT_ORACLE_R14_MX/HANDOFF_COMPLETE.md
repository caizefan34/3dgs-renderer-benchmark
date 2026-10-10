# HiGS Source Handoff - Complete

## STATUS: SOURCE_HANDOFF_READY

**Date:** 2026-10-08  
**Platform:** MX (NVIDIA A100-PCIE-40GB)

---

## Extracted Files

All files saved to: `/mnt/storage_pool/liaoyuanjun/HIGS_SOURCE_HANDOFF/`

### Core Source Files
1. **IntersectTile.cu** - Main CUDA source with counting kernel
2. **HIGS_PYTHON_WRAPPER.py** - Python entry point (`rasterize_gaussian_higs_frozen`)
3. **HIGS_BINDINGS.cpp** - C++ pybind11 bindings
4. **R8_WORKING_ENTRYPOINT.py** - Verified working test script

### Documentation
5. **COUNTING_CALL_GRAPH.md** - Complete call chain analysis
6. **BUILD_CONFIGURATION.txt** - Build commands and requirements
7. **RUNTIME_BINARY_IDENTITY.json** - Binary SHA256 and verification

---

## Key Findings for Research Lead

### 1. Counting Kernel Location
**File:** `IntersectTile.cu`  
**Function:** `intersect_tile()`  
**Lines:** ~700-850

**Key sections:**
- Count kernel: Lines ~700-780
- Prefix sum (CUB): Lines ~720-740
- Allocation: Lines 833-834
- Fill kernel: Lines ~850+

### 2. n_isects Calculation
**Location:** Lines ~720-740

```cpp
int64_t n_isects = 0;
for(const auto device_id: c10::irange(device_count))
{
    n_isects += device_totals[device_id];
}
```

**Prefix sum:** Uses `cub::DeviceScan::InclusiveSum`  
**Accumulator dtype:** `int64_t`

### 3. Allocation
**Location:** Line 833

```cpp
isect_ids = at::empty({n_isects}, opt.dtype(at::kLong));
flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
```

**Sizes:**
- `isect_ids`: `n_isects × 8 bytes`
- `flatten_ids`: `n_isects × 4 bytes`
- **Total:** `n_isects × 12 bytes`

### 4. Per-Gaussian Count Location
**In count kernel:** Each Gaussian's tile count computed and accumulated into `device_totals`

**To capture:** Need to modify count kernel to write individual counts

### 5. Projection Data
**Current state:** Computed internally but NOT exposed

**Available:** Means2d, radii, depths computed before counting  
**To capture:** Add diagnostic output buffer in CUDA kernel

### 6. Data Types
| Variable | Dtype | Max Safe Value |
|----------|-------|----------------|
| Per-Gaussian count | `int32_t` | 2,140 |
| `n_isects` | `int64_t` | 9.2×10^18 |
| `isect_ids` | `int64` | 9.2×10^18 |
| `flatten_ids` | `int32` | 2.1×10^9 |

### 7. Integer Safety
**At 4.4M Gaussians:** `n_isects` could reach ~14.9B (from R10 OOM)
- Exceeds int32 range (2.1B)
- Fits in int64 range (9.2×10^18)
- **Current implementation uses int64: SAFE**

---

## Modification Strategy for Research Lead

### Option 1: Add Debug Output to Count Kernel

**File:** `IntersectTile.cu`  
**Function:** Count kernel

**Changes:**
1. Add parameters:
   ```cpp
   int32_t* debug_counts,      // [n_gaussians * n_cameras]
   float2* debug_means2d,      // [n_gaussians * n_cameras]
   float* debug_radii,         // [n_gaussians * n_cameras * 2]
   int debug_sample_size
   ```

2. Write per-Gaussian values before accumulation

3. Update function signature in C++ binding

4. Return debug tensors to Python

### Option 2: Python-Side Capture
**Not possible** - per-Gaussian projected data not exposed

---

## Build Instructions

```bash
cd /home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/
source /home/liaoyuanjun/miniforge3/bin/activate /mnt/storage_pool/liaoyuanjun/higs-13scene-env
python setup.py build_ext --inplace
```

**Expected output:** `gsplat/experimental/render/kernels/csrc.so`

---

## Verification

**Binary SHA256:** `b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3`

**Verified at:** 580K Gaussians, B1 camera (R8 campaign)

**Test script:** `R8_WORKING_ENTRYPOINT.py`

---

## Next Steps

1. Review `IntersectTile.cu` counting kernel logic
2. Implement diagnostic output buffer
3. Update C++ bindings
4. Recompile CUDA extension
5. Capture per-Gaussian projected data
6. Implement Python AABB oracle
7. Compare GPU counts vs oracle counts

---

## Handoff Complete

All source code and documentation extracted and ready for instrumentation patch design.
