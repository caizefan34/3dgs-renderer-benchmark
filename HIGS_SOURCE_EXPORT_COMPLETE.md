# HiGS Source Export - Complete

## Export Successful

**Archive:** `HIGS_SOURCE_HANDOFF_ACTUAL.tar.gz`  
**Location:** `C:\Users\36570\3dgs-renderer-benchmark\`  
**Size:** 407,979 bytes (398 KB)  
**Created:** 2026-10-08 15:10:04

---

## Archive Contents

### Summary
- **Total files:** 92 source files
- **Missing required:** None
- **Source root:** `/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx`

### Binary Identities
1. **Main CUDA extension:**
   - Path: `gsplat/csrc.so`
   - SHA256: `40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50`
   - Size: 20,753,960 bytes

2. **HiGS render kernels:**
   - Path: `gsplat/experimental/render/kernels/csrc.so`
   - SHA256: `b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3`
   - Size: 1,415,536 bytes

---

## Key Source Files

### Core Counting Logic
- **IntersectTile.cu** (42,919 bytes)
  - SHA256: `b2468d28a329e7e5d7693f76eae1ba93c927865b50c1ce106c591e42843c8c0f`
  - Contains counting kernel and n_isects calculation

### Python Interface
- **HIGS_PYTHON_WRAPPER.py** (120,068 bytes)
  - SHA256: `7003ebf5b158b9eaf00d834873a9c76186bf54800789841aefba92d564a3c63a`
  - Entry point: `rasterize_gaussian_higs_frozen()`

### C++ Bindings
- **HIGS_BINDINGS.cpp** (65,187 bytes)
  - SHA256: `73a8be29a80d8717149867f18be37217e1a31f1ec145e4f7ef89d3c43c0fa973`
  - Pybind11 bindings

### Working Test Script
- **R8_WORKING_ENTRYPOINT.py** (3,172 bytes)
  - SHA256: `9f2bc47de41c2842ba92c73747f8b6a75884fd3d16a6d75ec944671aaf13f5e6`
  - Verified at 580K Gaussians

---

## Extracted Location

**Full path:** `C:\Users\36570\3dgs-renderer-benchmark\HIGS_SOURCE_HANDOFF_ACTUAL\`

**Directory structure:**
```
HIGS_SOURCE_HANDOFF_ACTUAL/
├── SOURCE_MANIFEST.json
├── SHA256SUMS.txt
├── source/
│   ├── gsplat/
│   │   ├── cuda/
│   │   │   ├── csrc/
│   │   │   │   ├── IntersectTile.cu
│   │   │   │   ├── Rasterization.cpp
│   │   │   │   └── ... (more CUDA files)
│   │   │   └── _torch_impl.py
│   │   └── experimental/
│   │       └── render/
│   └── ... (more Python files)
└── previous_handoff/
    ├── IntersectTile.cu
    ├── HIGS_PYTHON_WRAPPER.py
    ├── HIGS_BINDINGS.cpp
    └── R8_WORKING_ENTRYPOINT.py
```

---

## Key Findings from Source

### 1. Counting Kernel Location
**File:** `source/gsplat/cuda/csrc/IntersectTile.cu`  
**Function:** `intersect_tile()`  
**Lines:** ~700-850

### 2. n_isects Calculation
```cpp
int64_t n_isects = 0;
for(const auto device_id: c10::irange(device_count))
{
    n_isects += device_totals[device_id];
}
```

### 3. Allocation
```cpp
isect_ids = at::empty({n_isects}, opt.dtype(at::kLong));
flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
```

### 4. Prefix Sum
Uses CUB library: `cub::DeviceScan::InclusiveSum`

---

## Archive SHA256

**Full archive:** `990d6d9ac2bd40cc596a230ddf006b59114a422e95560c669c096c2c071c629e`

---

## Next Steps

1. Extract archive: `tar -xzf HIGS_SOURCE_HANDOFF_ACTUAL.tar.gz`
2. Review `IntersectTile.cu` counting kernel
3. Implement diagnostic output
4. Recompile CUDA extension
5. Capture per-Gaussian projected data
6. Implement Python AABB oracle
7. Compare GPU counts vs oracle counts

---

## Status: SOURCE_HANDOFF_READY

All source code and documentation extracted and ready for instrumentation patch design.
