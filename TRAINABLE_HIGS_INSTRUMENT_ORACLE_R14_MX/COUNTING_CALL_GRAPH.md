# HiGS Source Handoff - Counting Call Graph

## Overview
This document traces the HiGS Gaussian-tile intersection counting call chain from Python entry point to CUDA kernel.

## Call Chain

### 1. Python Entry Point
**File:** `HIGS_PYTHON_WRAPPER.py`  
**Function:** `rasterize_gaussian_higs_frozen()`  
**Line:** ~200-400

```python
result = rasterize_gaussian_higs_frozen(
    means, quats, scales, opacities, colors,
    viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
    width=960, height=540, sh_degree=3,
    enable_culling=True, use_higs_culling=True
)
```

**Returns:**
- `frame`: Rendered RGB image
- `alpha`: Alpha channel
- `densification_info`: Statistics dict with `n_isects`
- `metadata`: Additional runtime info

### 2. C++ Extension Binding
**File:** `HIGS_BINDINGS.cpp`  
**Function:** `rasterize_gaussian_higs_frozen` (pybind11 wrapper)

**Key operations:**
- Validates input tensor shapes
- Calls `render_kernels::rasterize_gaussian_higs_frozen()`
- Returns result dict to Python

### 3. CUDA Kernel Entry
**File:** `IntersectTile.cu`  
**Function:** `intersect_tile()`  
**Line:** ~700-850

**Key sections:**

#### 3.1 Count Kernel (First Pass)
**Line:** ~700-780

```cpp
// Count intersections per Gaussian-camera pair
int64_t n_isects = 0;
for(const auto device_id: c10::irange(device_count))
{
    // Synchronize and get count from device
    device_intersection_offsets[device_id] = n_isects;
    device_intersection_counts[device_id] = device_totals[device_id];
    n_isects += device_totals[device_id];
}
```

**Count kernel:** Counts tile coverage for each Gaussian using AABB overlap  
**Output:** `device_totals[device_id]` = sum of counts on device

#### 3.2 Prefix Sum (CUB)
**Line:** ~720-740

```cpp
CUB_WRAPPER_ASYNC(stream, cub::DeviceScan::InclusiveSum,
    device_totals.data_ptr(),
    device_intersection_offsets.data_ptr(),
    device_count);
```

**Purpose:** Compute prefix sum of counts to get offsets  
**Accumulator dtype:** `int64_t` (confirmed from source)

#### 3.3 Allocation
**Line:** 833-834

```cpp
isect_ids = at::empty({n_isects}, opt.dtype(at::kLong));
flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
```

**Allocation sizes:**
- `isect_ids`: `n_isects × 8 bytes` (int64)
- `flatten_ids`: `n_isects × 4 bytes` (int32)
- **Total:** `n_isects × 12 bytes`

**OOM location:** Line 833 (`isect_ids` allocation)

#### 3.4 Fill Kernel (Second Pass)
**Line:** ~850+

**Purpose:** Write actual intersection records into `isect_ids` and `flatten_ids`

## Key Data Types

| Variable | Dtype | Shape | Purpose |
|----------|-------|-------|---------|
| `n_isects` | `int64_t` | scalar | Total intersection count |
| `isect_ids` | `int64` | `[n_isects]` | Intersection IDs |
| `flatten_ids` | `int32` | `[n_isects]` | Flattened Gaussian-camera indices |
| `device_totals` | `int64_t*` | `[device_count]` | Per-device count sums |
| `device_intersection_offsets` | `int64_t*` | `[device_count]` | Prefix sum offsets |

## Per-Gaussian Tile Count Location

**In count kernel:** Each Gaussian's tile count is computed and accumulated into `device_totals`

**Location:** Within the count kernel (called before prefix sum)

**To capture per-Gaussian counts:** Need to modify count kernel to write individual counts to output buffer

## Projection Data Availability

**Current state:** Projected means2d and radii are computed internally but NOT exposed

**Available in GPU memory:**
- Projection happens before counting
- Means2d, radii, depths computed per Gaussian-camera pair
- Used by count kernel but not saved

**To capture:** Need to add diagnostic output buffer in CUDA kernel

## Integer Safety

| Checkpoint | Dtype | Max Value | Status |
|------------|-------|-----------|--------|
| Per-Gaussian count | `int32_t` | 2^31-1 | SAFE (max ~2040 tiles) |
| `n_isects` | `int64_t` | 2^63-1 | SAFE |
| `isect_ids` | `int64` | 2^63-1 | SAFE |
| `flatten_ids` | `int32` | 2^31-1 | SAFE (if N < 2B) |

**At 4.4M Gaussians:** `n_isects` could reach ~14.9B (from R10 OOM), which exceeds int32 range but fits in int64.

## Modification Points for Instrumentation

### Option 1: Add Debug Output to Count Kernel
**File:** `IntersectTile.cu`  
**Location:** Count kernel function

**Changes needed:**
1. Add output buffer parameters: `int32_t* debug_counts`, `float2* debug_means2d`, `float* debug_radii`
2. Write per-Gaussian values before accumulation
3. Update function signature in C++ binding
4. Return debug tensors to Python

### Option 2: Python-Side Capture (Limited)
**Not possible** without CUDA modification because per-Gaussian projected data is not exposed.

## Build Information

**Build system:** PyTorch C++ extension (setup.py)  
**CUDA version:** 12.8  
**PyTorch version:** 2.9.1+cu128

**Build command:**
```bash
python setup.py build_ext --inplace
```

**Output binary:** `gsplat/experimental/render/kernels/csrc.so`

## Next Steps for Research Lead

1. **Add diagnostic output** to count kernel in `IntersectTile.cu`
2. **Update C++ binding** to accept and return debug tensors
3. **Recompile** CUDA extension
4. **Capture** per-Gaussian projected data for 1K sample
5. **Implement** Python AABB oracle
6. **Compare** GPU counts vs oracle counts

## Files in Handoff

- `IntersectTile.cu` - Main CUDA source with counting logic
- `HIGS_PYTHON_WRAPPER.py` - Python entry point
- `HIGS_BINDINGS.cpp` - C++ pybind11 bindings
- `R8_WORKING_ENTRYPOINT.py` - Working test script
- `BUILD_CONFIGURATION.txt` - Build details
- `RUNTIME_BINARY_IDENTITY.json` - Binary SHA256
- `COUNTING_CALL_GRAPH.md` - This document
