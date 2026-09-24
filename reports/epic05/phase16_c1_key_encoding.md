# Phase 16 C1 — Key Encoding Verification Report

## Source: Actual Implementation (gsplat v1.5.3)

### Source File: `IntersectTile.cu`

**Key encoding at line 108:**
```cuda
isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
```

Where:
- `iid_enc = iid << (32 + tile_n_bits)` (line 95)
- `depth_id_enc = static_cast<uint32_t>(depth_i32)` (line 99), where `depth_i32 = *(int32_t *)&(depths[idx])` (line 98) — **bit-level reinterpret of float32**

**Actual bit layout:**
```
Bit 63 ... (32+tile_n_bits) | (32+tile_n_bits-1) ... 32 | 31 ... 0
     image_id (Xc bits)     |       tile_id (Xt bits)   | depth (32 bits)
```

### Source File: `Intersect.cpp`

**Bit width calculation** (lines 50-51):
```cpp
uint32_t image_n_bits = (uint32_t)floor(log2(I)) + 1;
uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
```

**Sort range** (IntersectTile.cu line 322):
```cpp
cub::DeviceRadixSort::SortPairs(
    d_keys, d_values, n_isects,
    0,                              // begin_bit
    32 + tile_n_bits + image_n_bits, // end_bit
    ...
);
```

---

## Verified Answers to All 10 Questions

### 1. Key actual type
**`int64_t`** (64-bit signed integer). Declared in both the CUDA kernel (line 46) and the CUB DoubleBuffer (line 310-311).

### 2. Actual bit layout
```
Bits [63 : 32+tile_n_bits]   — image_id (image_n_bits wide)
Bits [32+tile_n_bits-1 : 32] — tile_id (tile_n_bits wide)  
Bits [31 : 0]                — depth (full float32 bitcast to int32, zero-extended)
```

### 3. `image_n_bits`
**`image_n_bits = floor(log2(I)) + 1`**, where `I` = number of camera views.
- Single image (forward/render): `I=1` → `image_n_bits = floor(log2(1)) + 1 = 1`
- Training with batch: `I` = batch size

### 4. `tile_n_bits`
**`tile_n_bits = floor(log2(n_tiles)) + 1`**, where `n_tiles = tile_width × tile_height`.
- For 1080p, tile_size=16: `tile_width=ceil(1920/16)=120`, `tile_height=ceil(1080/16)=68`
- `n_tiles = 120 × 68 = 8160`
- `tile_n_bits = floor(log2(8160)) + 1 = floor(13.0) + 1 = 14`

| Resolution | tile_size | tile_width | tile_height | n_tiles | tile_n_bits |
|-----------|-----------|-----------|-------------|---------|-------------|
| 1080p     | 16        | 120       | 68          | 8160    | 14          |
| 1080p     | 20        | 96        | 54          | 5184    | 13          |
| 1080p     | 32        | 60        | 34          | 2040    | 12          |
| 4K        | 16        | 240       | 135         | 32400   | 15          |

### 5. Depth used bit count
**Full 32 bits.** The entire float32 is bitcast and stored as the lower 32 bits of the key. No truncation, no quantization.

### 6. `begin_bit` / `end_bit`
- `begin_bit = 0`
- `end_bit = 32 + tile_n_bits + image_n_bits`
- This **includes all depth bits** (bits 0-31) plus tile_id bits plus image_id bits.

### 7. CUB actual sort bit range
Full key range from bit 0 to the end of the image_id field. For single-image 1080p tile16:
- `end_bit = 32 + 14 + 1 = 47` → CUB sorts **47 bits**

### 8. Depth encoding: IEEE float bitcast?
**YES.** Line 98:
```cuda
int32_t depth_i32 = *(int32_t *)&(depths[idx]);
```
This does a **bit-level reinterpretation** of `float32` → `int32`, preserving IEEE 754 total ordering for positive floats (all depths are positive). The `static_cast<uint32_t>` then zero-extends to 64 bits.

**Key property:** For positive float32 `a, b`: `a < b` iff `bitcast_to_int32(a) < bitcast_to_int32(b)` as unsigned integers.

### 9. Depth actual numerical range
In real 3DGS scenes (Mip-NeRF 360):
- Room: ~0.2m to ~6.0m (indoor scene)
- Bicycle: ~0.5m to ~50m+ (outdoor, distant background)
- Garden: ~0.3m to ~30m+ (outdoor, varied depth)

Float32 representation (upper 16 bits shown as hex):
| Depth  | Float32 hex | Upper 16 bits | Int32 value |
|--------|------------|---------------|-------------|
| 0.01   | 0x3C23D70A | 0x3C23        | 1008983298  |
| 0.1    | 0x3DCCCCCD | 0x3DCC        | 1036831949  |
| 1.0    | 0x3F800000 | 0x3F80        | 1065353216  |
| 5.0    | 0x40A00000 | 0x40A0        | 1084227584  |
| 10.0   | 0x41200000 | 0x4120        | 1092616192  |
| 50.0   | 0x42480000 | 0x4248        | 1111490560  |
| 100.0  | 0x42C80000 | 0x42C8        | 1120403456  |

### 10. Existing quantization/normalization?
**NONE.** The depth is stored as raw float32. No normalization, no clamping, no quantization exists in the intersection pipeline.

---

## Sort Pass Calculation

**CUB DeviceRadixSort** uses 4-bit radix passes (the default radix bits = 4).

For single-image 1080p, tile_size=16, `end_bit = 47`:
- `passes = ceil(47 / 4) = 12 passes` ✓

Each pass reads and writes 12 bytes per element (8B key + 4B value double-buffered):
- Per pass memory traffic: `12 × n_isects` bytes
- Total sort traffic: `12 × n_isects × 12 = 144 × n_isects` bytes
- For n_isects = 175M (tile16): `144 × 175M = 25.2 GB` of memory traffic per sort call

---

## Key Observation for C1

The depth field occupies the **lower 32 bits** of the key. The sort range (`end_bit`) includes **all 32 bits of depth**.

**Critical finding:** Since `depth_id_enc` occupies bits [0:32] of the key, telling CUB to sort only to `end_bit = 16 + tile_n_bits + image_n_bits` would sort the upper 16 bits of depth (which correspond to the exponent + top 7 mantissa bits of the float32), leaving the lower 16 bits unsorted.

This is **not equivalent** to reducing depth precision — the full depth value remains in the key for the offset kernel to read. Only the sorting granularity changes.

### Concurrent modification needed

When the key encoding is changed, the **offset kernel** (IntersectTile.cu line 209-257) must also be updated because:
- Line 227: `int64_t isect_id_curr = isect_ids[idx] >> 32;` — reads the tile_id by shifting out the depth
- Line 228: `int64_t iid_curr = isect_id_curr >> (tile_n_bits);` — extracts image_id from the upper bits

The offset kernel currently assumes the depth is in bits [0:32]. If we change the encoding, this assumption breaks.
