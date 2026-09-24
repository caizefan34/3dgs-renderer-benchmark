"""
Phase 16 C1 — Toy CUB Radix Sort Benchmark

Compares baseline sort key (47-bit) vs compressed key (31-bit) for CUB sorting.

Key insight: 
  Baseline: 64-bit key with depth(32) | tile_id(14) | image_id(1) → 47 bits sorted
  C1:       64-bit key with depth_upper(16) | tile_id(14) | image_id(1) → 31 bits sorted
             (but key REORGANIZED: depth_upper in bits [0:15], tile_id in bits [16:...])
  
  This reduces CUB radix passes from 12 to 8 (33% reduction).
  
  Memory traffic: each pass reads/writes 12B per element.
  Baseline: 12×12×N = 144N bytes
  C1:        8×12×N =  96N bytes  (33% reduction)

NVIDIA RTX 5070 Laptop GPU, CUDA 12.x
"""

import json
import time
import os
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "results" / "epic05" / "phase16"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Scenario: single image 1080p, tile_size=16
# tile_width=120, tile_height=68, n_tiles=8160, tile_n_bits=14
# image_n_bits=1 (single image)
BASELINE_END_BIT = 32 + 14 + 1  # 47
C1_END_BIT = 16 + 14 + 1        # 31
TILE_N_BITS = 14
IMAGE_N_BITS = 1


def generate_realistic_keys(n: int, seed=42) -> torch.Tensor:
    """Generate realistic isect_ids with proper tile_id + depth encoding."""
    torch.manual_seed(seed)
    
    # Simulate: 200K Gaussians producing ~6x intersections at tile16
    nnz = max(n // 6, 100000)
    
    # Generate realistic depth values (log-uniform, typical range)
    log_depths = torch.empty(nnz).uniform_(np.log(0.01), np.log(100.0))
    depths = log_depths.exp().float()
    
    # Simulate tile assignment (each Gaussian covers 1-12 tiles)
    tile_counts = torch.randint(1, 12, (nnz,))
    # Adjust to match requested n
    tile_counts = (tile_counts.float() * n / tile_counts.sum().float()).round().int()
    tile_counts = torch.clamp(tile_counts, 1, 20)
    # Truncate/pad to exact n
    if tile_counts.sum() > n:
        tile_counts[tile_counts.argmax()] -= (tile_counts.sum() - n)
    elif tile_counts.sum() < n:
        tile_counts[0] += (n - tile_counts.sum().item())
    
    # For each Gaussian, generate tile_ids and encode keys
    n_tiles = 120 * 68  # 8160 for 1080p tile16
    isect_ids = []
    for g in range(len(tile_counts)):
        depth = depths[g]
        depth_u32 = depth.view(torch.int32).item() & 0xFFFFFFFF
        
        n_t = tile_counts[g].item()
        # Random tiles that this Gaussian covers
        tile_ids = torch.randint(0, n_tiles, (n_t,))
        
        for tid in tile_ids.tolist():
            # Baseline encoding
            iid_enc = 0 << (32 + TILE_N_BITS)  # single image
            baseline_key = iid_enc | (int(tid) << 32) | depth_u32
            isect_ids.append(baseline_key)
    
    # Trim to exact n
    isect_ids = torch.tensor(isect_ids[:n], dtype=torch.int64)
    return isect_ids


def generate_random_keys(n: int) -> torch.Tensor:
    """Generate random keys for baseline timing."""
    # Simulate baseline isect_ids: tile_id at [32:46], depth at [0:31]
    rng = torch.Generator().manual_seed(42)
    
    # Random tile_ids (uniform across 8160 tiles)
    tile_ids = torch.randint(0, 8160, (n,), generator=rng)
    # Random depth values (log-uniform across [0.01, 100.0])
    log_depths = torch.empty(n).uniform_(np.log(0.01), np.log(100.0), generator=rng)
    depths = log_depths.exp()
    depth_u32s = depths.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    
    # Baseline keys
    keys = (tile_ids.int().to(torch.int64) << 32) | depth_u32s
    return keys


def measure_sort_time(keys: torch.Tensor, end_bit: int, n_warmup=3, n_measure=20) -> dict:
    """Time CUB DeviceRadixSort::SortPairs."""
    n = len(keys)
    
    # Create double buffers
    keys_sorted = torch.empty_like(keys)
    values = torch.arange(n, dtype=torch.int32, device='cuda')
    values_sorted = torch.empty_like(values)
    
    # Get temp storage size
    from torch.utils.cpp_extension import load
    # Use torch's CUB wrapper directly
    d_keys = keys.contiguous().cuda()
    d_values = values.cuda()
    d_keys_out = keys_sorted.cuda()
    d_values_out = values_sorted.cuda()
    
    # Use torch.sort as baseline reference (doesn't use CUB, but gives correct answer)
    # For CUB timing, we use cub.DeviceRadixSort via gsplat's existing mechanism
    # Actually, let's use the CUB wrappers from gsplat's code
    import torch.utils._pytree as pytree
    
    # Get CUB sort from gsplat (already loaded)
    from gsplat.cuda._backend import _C
    
    # Actually, gsplat doesn't expose a raw CUB benchmark.
    # Let's use the cub module from torch's C extension build.
    # Alternative: use the gsplat renderer's radix sort path directly.
    
    # Simplest approach: modify gsplat source and measure the renderer timing
    # But for microbenchmark, let's implement a CUB wrapper:
    
    # Get temp storage
    from torch.cuda import Stream
    
    keys_dev = keys.contiguous().cuda()
    values_dev = torch.arange(n, dtype=torch.int32, device='cuda')
    keys_dev_out = torch.empty_like(keys_dev)
    values_dev_out = torch.empty_like(values_dev)
    
    try:
        from cub import DeviceRadixSort
        cub_available = True
    except ImportError:
        cub_available = False
    
    if not cub_available:
        # Fallback: use torch.sort as proxy (not ideal but gives us baseline)
        # Actually, let's try to access CUB through the already-compiled gsplat extension
        # by calling sort directly on our own data
        # We can rebuild with the right end_bit
        
        # For now, measure torch.sort timing as a rough proxy for relative comparison
        print(f"  CUB not directly accessible — using torch.sort as proxy (n={n:,}, end_bit={end_bit})")
        
        # Note: torch.sort uses a different algorithm than CUB radix sort,
        # so absolute timings are not comparable. Relative scaling with n is meaningful.
        
        # Warmup
        for _ in range(n_warmup):
            _ = torch.sort(keys_dev)
        
        # Measure
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(n_measure):
            start.record()
            sorted_keys, sorted_values = torch.sort(keys_dev, stable=True)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        times = np.array(times)
        
        # Memory: torch.sort doesn't use temp storage the same way
        # Estimate: radix sort temp storage ≈ key_bytes + val_bytes + overhead
        temp_bytes_estimate = n * 12 * 2  # double buffer for 8B key + 4B value
        # CUB uses ~2x the data size as temp for double buffer sorting
        
        return {
            "n": n,
            "end_bit": end_bit,
            "sort_bits": end_bit,
            "label": "torch.sort (proxy)",
            "times_ms": {
                "mean": float(times.mean()),
                "median": float(np.median(times)),
                "std": float(times.std()),
                "min": float(times.min()),
                "max": float(times.max()),
                "all": times.tolist(),
            },
            "temp_storage_bytes_estimated": temp_bytes_estimate,
            "theoretical_passes": (end_bit + 3) // 4,
            "note": "Using torch.sort as proxy for CUB radix sort. Absolute values not comparable to CUB."
        }
    
    # CUB path
    keys_dev = keys.contiguous().cuda()
    values_dev = torch.arange(n, dtype=torch.int32, device='cuda')
    keys_dev_out = torch.empty_like(keys_dev)
    values_dev_out = torch.empty_like(values_dev)
    
    d_keys_obj = cub.DoubleBuffer(keys_dev.data_ptr(), keys_dev_out.data_ptr())
    d_values_obj = cub.DoubleBuffer(values_dev.data_ptr(), values_dev_out.data_ptr())
    
    # Get temp storage size
    temp_bytes = cub.DeviceRadixSort.SortPairs(d_keys_obj, d_values_obj, n, 0, end_bit, cub.null_stream())
    temp_storage = torch.empty(temp_bytes, dtype=torch.uint8, device='cuda')
    
    # Warmup
    for _ in range(n_warmup):
        cub.DeviceRadixSort.SortPairs(
            d_keys_obj, d_values_obj, n, 0, end_bit,
            torch.cuda.current_stream().cuda_stream,
            temp_storage.data_ptr()
        )
    
    # Measure
    times = []
    for _ in range(n_measure):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        cub.DeviceRadixSort.SortPairs(
            d_keys_obj, d_values_obj, n, 0, end_bit,
            torch.cuda.current_stream().cuda_stream,
            temp_storage.data_ptr()
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    times = np.array(times)
    theoretical_passes = (end_bit + 3) // 4
    
    return {
        "n": n,
        "end_bit": end_bit,
        "sort_bits": end_bit,
        "label": f"CUB radix sort ({end_bit}-bit)",
        "times_ms": {
            "mean": float(times.mean()),
            "median": float(np.median(times)),
            "std": float(times.std()),
            "min": float(times.min()),
            "max": float(times.max()),
        },
        "temp_storage_bytes": temp_bytes,
        "theoretical_passes": theoretical_passes,
    }


def main():
    torch.cuda.synchronize()
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    
    # First, let's check if we can access CUB radix sort
    # gsplat's radix_sort_double_buffer is compiled and available
    from gsplat.cuda._backend import _C
    print(f"gsplat _C loaded: {_C is not None}")
    
    all_results = {"device": device_name, "scenarios": []}
    
    configs = [
        (1_000_000, "1M"),
        (5_000_000, "5M"),
        (10_000_000, "10M"),
        (176_000_000, "176M"),  # tile16 scenario
    ]
    
    end_bits_configs = [
        (BASELINE_END_BIT, "baseline_47bit"),
        (C1_END_BIT, "c1_31bit"),
    ]
    
    for n, n_label in configs:
        print(f"\n{'='*60}")
        print(f"N = {n:,} ({n_label})")
        print(f"{'='*60}")
        
        # Generate test data
        print(f"  Generating {n:,} realistic keys...")
        keys = generate_random_keys(n)
        print(f"  Keys shape: {keys.shape}, dtype: {keys.dtype}")
        
        for end_bit, config_label in end_bits_configs:
            passes = (end_bit + 3) // 4
            print(f"\n  --- {config_label} (end_bit={end_bit}, {passes} passes) ---")
            result = measure_sort_time(keys, end_bit)
            result["config"] = config_label
            result["n_label"] = n_label
            all_results["scenarios"].append(result)
            
            t = result["times_ms"]
            print(f"    Sort time: {t['median']:.3f} ± {t['std']:.3f} ms (median ± std)")
            print(f"    Range: [{t['min']:.3f}, {t['max']:.3f}] ms")
            if "temp_storage_bytes" in result:
                print(f"    Temp storage: {result['temp_storage_bytes'] / 1e9:.2f} GB")
            print(f"    Theoretical passes: {result['theoretical_passes']}")
        
        # Compute speedup
        if len(all_results["scenarios"]) >= 2:
            r1 = all_results["scenarios"][-2]
            r2 = all_results["scenarios"][-1]
            speedup = r1["times_ms"]["median"] / r2["times_ms"]["median"]
            print(f"\n  --- Speedup (C1 vs Baseline): {speedup:.3f}x ---")
    
    # Memory traffic estimation
    print(f"\n{'='*60}")
    print(f"MEMORY TRAFFIC ESTIMATION")
    print(f"{'='*60}")
    # Each pass reads/writes 12B per element (8B key + 4B value in double buffer)
    print(f"  {'N':>12s} | {'Baseline passes':>15s} | {'C1 passes':>12s} | {'Baseline traffic':>18s} | {'C1 traffic':>13s} | {'Reduction':>10s}")
    for n, n_label in configs:
        baseline_traffic = n * 12 * BASELINE_END_BIT / 4  # bytes
        c1_traffic = n * 12 * C1_END_BIT / 4
        # Actually: each pass = 12 * N bytes (read + write of key+value)
        baseline_traffic = n * 12 * ((BASELINE_END_BIT + 3) // 4)
        c1_traffic = n * 12 * ((C1_END_BIT + 3) // 4)
        reduction = (baseline_traffic - c1_traffic) / baseline_traffic * 100
        print(f"  {n:>12,} | {((BASELINE_END_BIT+3)//4):>13d} passes | {((C1_END_BIT+3)//4):>10d} passes | {baseline_traffic/1e9:>16.2f} GB | {c1_traffic/1e9:>11.2f} GB | {reduction:>8.1f}%")
    
    # Save results
    output_file = OUTPUT_DIR / "c1_sort_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
