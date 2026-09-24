#!/usr/bin/env python3
"""
Phase 16-C1: Depth ordering preservation with upper-16-bit depth encoding
for CUB radix sort key compression.

Background
----------
The gsplat rasterizer builds a 64-bit *intersection ID* per Gaussian-tile
intersection in ``IntersectTile.cu``::

    int32_t depth_i32 = *(int32_t *)&(depths[idx]);          // IEEE 754 bitcast
    int64_t depth_id_enc = static_cast<uint32_t>(depth_i32); // zero-extend

    isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;

CUB's ``DeviceRadixSort::SortPairs`` sorts the lower ``32 + tile_n_bits +
image_n_bits`` bits.  If we replace the depth field with only its **upper 16
bits**::

    depth_16bit = depth_uint32 >> 16;   // keep only MSB 16 bits

    compressed_id = (image_id << (16 + tile_n_bits))
                  | (tile_id  << 16)
                  | depth_16bit;

we save 16 bits — 4 fewer radix-sort passes (8 → 4).  But does this preserve
ordering?

**Key insight — monotonicity of right-shift:**:
For positive float32 values, the IEEE 754 encoding is lexicographic::

    depth_a < depth_b   <=>   uint32_cast(depth_a) < uint32_cast(depth_b)

Right-shift is monotonic over unsigned integers::

    a < b   =>   (a >> k) <= (b >> k)   for any k >= 0

Therefore::

    depth_a < depth_b   =>   compressed(depth_a) <= compressed(depth_b)

**Zero inversions are possible**.  The only effect of compression is **ties**:
two Gaussians with different depths can land in the same sort-key bucket when
their IEEE bit patterns differ only in the lower 16 bits (mantissa bits 15:0).

This script measures:
  1. **Tie rate**: fraction of (i,j) pairs whose depth difference is too small
     to change the upper 16 bits.
  2. **Bucket statistics**: per-bucket depth span, element count.
  3. **Depth resolution**: the minimum relative depth change needed to
     advance to the next compressed key, across the depth range.
  4. **Real scene data**: Euclidean depths from Mip-NeRF 360 checkpoints
     (bicycle, garden, room).
  5. **Tile-local tie rates**: the per-tile tie probability that affects
     actual GPU rendering.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ieee_bitcast_uint32(arr: np.ndarray) -> np.ndarray:
    """Reinterpret float32 → uint32 (same bit pattern)."""
    return arr.view(np.uint32)


def float32_from_uint32(arr: np.ndarray) -> np.ndarray:
    """Reinterpret uint32 → float32."""
    return arr.view(np.float32)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_depth_bucket_at(depth_value: float) -> dict:
    """
    At a given depth, compute the float32 bucket spanned by a single
    upper-16-bit key (i.e. all float32 values sharing the same MSB 16 bits).

    Returns dict with bucket start, end, width, and relative width.
    """
    d = np.float32(depth_value)
    u32 = int(ieee_bitcast_uint32(np.array([d], dtype=np.float32))[0])
    bucket_lo_uint = u32 & 0xFFFF0000        # zero out lower 16 bits
    bucket_hi_uint = bucket_lo_uint | 0xFFFF  # maximum in this bucket
    lo_f = float(float32_from_uint32(np.array([bucket_lo_uint], dtype=np.uint32))[0])
    hi_f = float(float32_from_uint32(np.array([bucket_hi_uint], dtype=np.uint32))[0])
    width = hi_f - lo_f
    return {
        "depth": float(d),
        "uint32_hex": f"0x{u32:08X}",
        "upper16_hex": f"{(u32 >> 16):04X}",
        "upper16": int(u32 >> 16),
        "bucket_start": lo_f,
        "bucket_end": hi_f,
        "bucket_width": width,
        "relative_width": width / float(d),
        "relative_width_pct": 100.0 * width / float(d),
    }


def analyze_ordering(
    depths: np.ndarray,
    scene_label: str = "unknown",
) -> dict:
    """
    Analyze depth ordering when the sort key uses upper-16-bit depth encoding.

    Confirms zero inversions (theoretical guarantee) and measures ties.

    Parameters
    ----------
    depths : (N,) float32 array of per-Gaussian depths (positive).
    scene_label : str

    Returns
    -------
    dict of metrics.
    """
    n = len(depths)
    assert depths.min() > 0, "Depths must be positive"

    depth_uint32 = ieee_bitcast_uint32(depths)
    depth_upper16 = depth_uint32 >> 16

    # ---- Confirm zero inversions ----
    baseline_keys = depth_uint32.astype(np.uint64)
    compressed_keys = depth_upper16.astype(np.uint64)

    baseline_order = np.argsort(baseline_keys, kind="stable")
    compressed_order = np.argsort(compressed_keys, kind="stable")

    # Within each tie bucket (same compressed key), elements can be in any
    # relative order; this is NOT an inversion — the sort keys are equal.
    # A TRUE inversion would mean compressed_key_a < compressed_key_b BUT
    # depth_a > depth_b, which is impossible by the monotonicity proof.
    #
    # We verify: for every pair of compressed_order with different keys,
    # the earlier element is not deeper than the later element.
    sorted_keys = compressed_keys[compressed_order]
    sorted_by_compressed = depths[compressed_order]
    # Group by compressed key and check that max depth of bucket i <=
    # min depth of bucket j for all i < j.
    unique_keys_sorted = np.unique(sorted_keys)
    true_inversions = 0
    bucket_max = np.zeros(len(unique_keys_sorted), dtype=np.float32)
    bucket_min = np.zeros(len(unique_keys_sorted), dtype=np.float32)
    for bi, k in enumerate(unique_keys_sorted):
        mask = sorted_keys == k
        bucket_max[bi] = sorted_by_compressed[mask].max()
        bucket_min[bi] = sorted_by_compressed[mask].min()
    for bi in range(len(unique_keys_sorted) - 1):
        if bucket_max[bi] > bucket_min[bi + 1]:
            true_inversions += 1
    inversions_found = int(true_inversions)

    # ---- Rank displacement ----
    baseline_rank = np.empty(n, dtype=np.int64)
    baseline_rank[baseline_order] = np.arange(n)
    compressed_rank = np.empty(n, dtype=np.int64)
    compressed_rank[compressed_order] = np.arange(n)
    rank_disp = np.abs(compressed_rank - baseline_rank)
    max_rank_disp = int(rank_disp.max())
    mean_rank_disp = float(rank_disp.mean())

    # ---- Bucket (tie) analysis ----
    unique_keys, inverse, counts = np.unique(
        compressed_keys, return_inverse=True, return_counts=True
    )
    n_unique = len(unique_keys)
    n_tied = int((counts > 1).sum())

    # Per-bucket depth range
    bucket_mins = np.empty(n_unique, dtype=np.float32)
    bucket_maxs = np.empty(n_unique, dtype=np.float32)
    for b in range(n_unique):
        mask = inverse == b
        bucket_mins[b] = depths[mask].min()
        bucket_maxs[b] = depths[mask].max()
    bucket_span = bucket_maxs - bucket_mins

    # Global tie rate
    tied_elements = int(counts[counts > 1].sum())
    tie_fraction_global = tied_elements / n if n > 0 else 0.0

    # Pairwise tie rate (fraction of unordered pairs that collide)
    total_pairs = n * (n - 1) // 2
    tied_pairs = int(sum(c * (c - 1) // 2 for c in counts if c > 1))
    tie_pair_ratio = tied_pairs / total_pairs if total_pairs > 0 else 0.0

    # ---- Resolution at sample depths ----
    sample_depths = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    bucket_info = [analyze_depth_bucket_at(d) for d in sample_depths]

    # ---- Depth-range coverage ----
    depth_min_f = float(depths.min())
    depth_max_f = float(depths.max())
    exponent_min = int((depth_uint32.min() >> 23) & 0xFF)
    exponent_max = int((depth_uint32.max() >> 23) & 0xFF)

    result = {
        "meta": {
            "scene": scene_label,
            "n_elements": int(n),
            "depth_range": [depth_min_f, depth_max_f],
            "depth_mean": float(depths.mean()),
            "depth_std": float(depths.std()),
            "exponent_range": [exponent_min, exponent_max],
        },
        "ordering_integrity": {
            "proof": (
                "depth_a < depth_b => uint32(depth_a) < uint32(depth_b) (IEEE 754 "
                "monotonicity for positive floats).  Right-shift is monotonic: "
                "a < b => (a>>16) <= (b>>16).  Therefore compressed(depth_a) <= "
                "compressed(depth_b).  Zero inversions are possible."
            ),
            "n_inversions_possible": 0,
            "n_inversions_confirmed": int(inversions_found),
        },
        "rank_displacement": {
            "max": max_rank_disp,
            "mean": mean_rank_disp,
        },
        "ties": {
            "n_unique_keys": int(n_unique),
            "n_multi_element_buckets": int(n_tied),
            "tied_elements": tied_elements,
            "tie_element_fraction": float(tie_fraction_global),
            "tied_pairs": tied_pairs,
            "total_pairs": total_pairs,
            "tie_pair_ratio": float(tie_pair_ratio),
        },
        "bucket_depth_spans": {
            "min": float(bucket_span.min()) if len(bucket_span) > 0 else 0.0,
            "max": float(bucket_span.max()) if len(bucket_span) > 0 else 0.0,
            "mean": float(bucket_span.mean()) if len(bucket_span) > 0 else 0.0,
            "nonzero_span_buckets": int((bucket_span > 0).sum()),
            "total_buckets": int(n_unique),
        },
        "bucket_fill": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
        },
        "resolution": {
            "description": (
                "Upper 16 bits of float32 = sign(1) | exponent(8) | mantissa_top(7). "
                "Relative precision ~2^(-7) ≈ 0.78 %.  All float32 values sharing "
                "the same MSB 16 bits map to one bucket."
            ),
            "relative_precision_estimate": 2.0 ** (-7),
            "mantissa_bits_preserved": 7,
            "baseline_depth_bits": 32,
            "compressed_depth_bits": 16,
        },
        "sample_buckets": bucket_info,
    }

    return result


# ---------------------------------------------------------------------------
# Depth distribution generators
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_log_uniform(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Log-uniform depths in [0.01, 100]."""
    if rng is None:
        rng = _rng(42)
    log_depths = rng.uniform(math.log(0.01), math.log(100.0), int(n))
    return np.exp(log_depths).astype(np.float32)


def generate_clustered(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """80 % at ~5 m (wall), outliers near/far."""
    if rng is None:
        rng = _rng(43)
    n_wall = int(n * 0.8)
    n_rest = n - n_wall
    wall = rng.normal(5.0, 0.5, n_wall)
    close = rng.uniform(0.1, 1.0, n_rest // 2)
    far   = rng.uniform(20.0, 80.0, n_rest - n_rest // 2)
    depths = np.concatenate([wall, close, far])
    rng.shuffle(depths)
    return np.abs(depths).astype(np.float32)


def generate_coplanar(n: int, value: float = 5.0) -> np.ndarray:
    """All Gaussians at exactly the same depth."""
    return np.full(int(n), value, dtype=np.float32)


def generate_very_close(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Depths differing by ~1e-6 (worst-case for ties)."""
    if rng is None:
        rng = _rng(44)
    base = 10.0
    eps = rng.uniform(-1e-6, 1e-6, int(n))
    return (base + eps).astype(np.float32)


def generate_mixture(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Three Gaussian clusters (near 1m, mid 10m, far 50m)."""
    if rng is None:
        rng = _rng(45)
    n1 = int(n * 0.3)
    n2 = int(n * 0.4)
    n3 = int(n) - n1 - n2
    c1 = rng.normal(1.0, 0.2, n1)
    c2 = rng.normal(10.0, 1.0, n2)
    c3 = rng.normal(50.0, 5.0, n3)
    depths = np.concatenate([c1, c2, c3])
    rng.shuffle(depths)
    return np.abs(depths).astype(np.float32)


def generate_extreme_range(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Wide range [0.001, 1000]."""
    if rng is None:
        rng = _rng(46)
    log_depths = rng.uniform(math.log(0.001), math.log(1000.0), int(n))
    return np.exp(log_depths).astype(np.float32)


# ---------------------------------------------------------------------------
# Real scene depth extraction
# ---------------------------------------------------------------------------

def extract_real_depths(scene_name: str) -> np.ndarray | None:
    """
    Load a Mip-NeRF 360 PLY checkpoint and compute Euclidean depth from
    the first camera's position.

    Returns (N,) float32 or None on failure.
    """
    ply_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "point_cloud.ply"
    if not ply_path.exists():
        return None

    try:
        from benchmark_framework.scene import load_ply
        from benchmark_framework.cameras import load_cameras_from_json
    except ImportError:
        return None

    scene = load_ply(str(ply_path), device="cpu")
    xyz = scene["xyz"].numpy()

    cam_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "cameras.json"
    if cam_path.exists():
        import torch
        cameras = load_cameras_from_json(str(cam_path), device="cpu")
        cam = cameras[0]
        cc = cam.camera_center
        if isinstance(cc, torch.Tensor):
            cc = cc.numpy()
        if isinstance(cc, np.ndarray) and cc.shape == (3,):
            delta = xyz - cc[None, :]
        else:
            delta = xyz
    else:
        delta = xyz

    depths = np.linalg.norm(delta, axis=1).astype(np.float32)
    depths = np.clip(depths, 1e-6, None)
    return depths


# ---------------------------------------------------------------------------
# Tile-local tie simulation
# ---------------------------------------------------------------------------

def simulate_tile_ties(
    depths: np.ndarray,
    n_tiles_sim: int = 1000,
    gaussians_per_tile: int = 128,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Emulate tile-local sorting: each tile sees a subset of Gaussians and
    sorts them by the compressed key.  Measure the tie rate per tile.
    """
    if rng is None:
        rng = _rng(99)
    n_total = len(depths)
    tile_tie_scores = []
    tile_tied_pair_ratios = []

    for _ in range(n_tiles_sim):
        idx = rng.choice(n_total, gaussians_per_tile, replace=False)
        tile_depths = depths[idx]
        tile_u32 = ieee_bitcast_uint32(tile_depths)
        tile_upper = tile_u32 >> 16

        _, counts_tile = np.unique(tile_upper, return_counts=True)
        tied_elems = int((counts_tile > 1).sum())
        tile_tie_scores.append(tied_elems / gaussians_per_tile)

        tp = int(sum(c * (c - 1) // 2 for c in counts_tile if c > 1))
        total_tp = gaussians_per_tile * (gaussians_per_tile - 1) // 2
        tile_tied_pair_ratios.append(tp / total_tp if total_tp > 0 else 0.0)

    return {
        "n_tiles_simulated": n_tiles_sim,
        "gaussians_per_tile": gaussians_per_tile,
        "element_tie_rate": {
            "mean": float(np.mean(tile_tie_scores)),
            "max": float(np.max(tile_tie_scores)),
            "p95": float(np.percentile(tile_tie_scores, 95)),
        },
        "pair_tie_rate": {
            "mean": float(np.mean(tile_tied_pair_ratios)),
            "max": float(np.max(tile_tied_pair_ratios)),
            "p95": float(np.percentile(tile_tied_pair_ratios, 95)),
        },
    }


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_all_analyses() -> list[dict]:
    """Run ordering analysis for all distributions and real scenes."""
    results = []
    N = 1_000_000

    SYNTECTIC_TESTS = [
        ("log_uniform",   "Log-uniform [0.01, 100]",   lambda: generate_log_uniform(N)),
        ("clustered",     "Clustered (wall-like)",     lambda: generate_clustered(N)),
        ("coplanar",      "Co-planar (identical)",     lambda: generate_coplanar(N)),
        ("very_close",    "Very close (1e-6 delta)",   lambda: generate_very_close(N)),
        ("mixture",       "3-cluster mixture",          lambda: generate_mixture(N)),
        ("extreme_range", "Extreme range [0.001, 1k]", lambda: generate_extreme_range(N)),
    ]

    print("=" * 72)
    print("  1. Synthetic depth distributions  (N = {:,})".format(N))
    print("=" * 72)

    for name, label, gen_fn in SYNTECTIC_TESTS:
        print(f"\n  >>> {label}")
        t0 = time.perf_counter()
        depths = gen_fn()
        t_gen = time.perf_counter() - t0
        print(f"      Generated {len(depths):,} depths in {t_gen*1000:.1f} ms")

        t0 = time.perf_counter()
        res = analyze_ordering(depths, scene_label=name)
        t_ana = time.perf_counter() - t0
        ties = res["ties"]
        bc = res["bucket_fill"]
        print(f"      Analyzed in {t_ana*1000:.1f} ms")
        print(f"      Depth range: [{res['meta']['depth_range'][0]:.4f}, "
              f"{res['meta']['depth_range'][1]:.4f}]")
        print(f"      Unique keys: {ties['n_unique_keys']:,}  "
              f"Tied buckets: {ties['n_multi_element_buckets']:,}")
        print(f"      Tied pairs: {ties['tied_pairs']:,} / {ties['total_pairs']:,}  "
              f"({ties['tie_pair_ratio']:.2e})")
        print(f"      Bucket fill: max={bc['max']:,}  mean={bc['mean']:.1f}")

        # Tile simulation
        tile = simulate_tile_ties(depths)
        res["tile_local"] = tile
        print(f"      Per-tile element ties: mean={tile['element_tie_rate']['mean']*100:.3f}%  "
              f"p95={tile['element_tie_rate']['p95']*100:.3f}%")

        results.append(res)

    # ---- Real scenes ----
    print("\n" + "=" * 72)
    print("  2. Real Mip-NeRF 360 scenes")
    print("=" * 72)

    for scene_name in ["bicycle", "garden", "room"]:
        ply_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "point_cloud.ply"
        if not ply_path.exists():
            print(f"\n  >>> {scene_name}  — PLY not found, skipping")
            continue

        print(f"\n  >>> {scene_name}")
        t0 = time.perf_counter()
        depths = extract_real_depths(scene_name)
        if depths is None:
            print("      Failed to load")
            continue
        t_ld = time.perf_counter() - t0
        print(f"      Loaded {len(depths):,} depths in {t_ld*1000:.1f} ms")
        print(f"      Depth range: [{depths.min():.4f}, {depths.max():.4f}]  "
              f"mean={depths.mean():.4f}  std={depths.std():.4f}")

        # Sample if needed
        if len(depths) > 2_000_000:
            rng_s = _rng(99)
            idx = rng_s.choice(len(depths), 2_000_000, replace=False)
            depths = depths[idx]
            print(f"      (sampled to {len(depths):,} for speed)")

        t0 = time.perf_counter()
        res = analyze_ordering(depths, scene_label=scene_name)
        t_ana = time.perf_counter() - t0
        ties = res["ties"]
        bc = res["bucket_fill"]
        print(f"      Analyzed in {t_ana*1000:.1f} ms")
        print(f"      Unique keys: {ties['n_unique_keys']:,}  "
              f"Tied buckets: {ties['n_multi_element_buckets']:,}")
        print(f"      Tied pairs: {ties['tied_pairs']:,} / {ties['total_pairs']:,}  "
              f"({ties['tie_pair_ratio']:.2e})")
        print(f"      Bucket fill: max={bc['max']:,}  mean={bc['mean']:.1f}")

        tile = simulate_tile_ties(depths)
        res["tile_local"] = tile
        print(f"      Per-tile element ties: mean={tile['element_tie_rate']['mean']*100:.3f}%  "
              f"p95={tile['element_tie_rate']['p95']*100:.3f}%")

        results.append(res)

    return results


def summarize(results: list[dict]) -> dict:
    rows = []
    for r in results:
        rows.append({
            "scene": r["meta"]["scene"],
            "n": r["meta"]["n_elements"],
            "tie_pair_ratio": r["ties"]["tie_pair_ratio"],
            "tie_element_fraction": r["ties"]["tie_element_fraction"],
            "max_rank_disp": r["rank_displacement"]["max"],
            "mean_rank_disp": r["rank_displacement"]["mean"],
            "n_unique_keys": r["ties"]["n_unique_keys"],
        })
    worst = max(rows, key=lambda x: x["tie_pair_ratio"])
    best  = min(rows, key=lambda x: x["tie_pair_ratio"])
    s = {
        "n_scenarios": len(rows),
        "scenarios": rows,
        "worst_case": worst,
        "best_case": best,
    }
    return s


def save(results: list[dict], summary: dict) -> Path:
    out_dir = REPO_ROOT / "results" / "epic05" / "phase16"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "c1_ordering.json"

    # Add sample bucket resolution table (from first result)
    sample_buckets = results[0].get("sample_buckets", []) if results else []

    payload = {
        "title": "Phase 16-C1: Depth ordering with upper-16-bit key compression",
        "description": (
            "Confirms zero inversions (theoretical guarantee via IEEE 754 "
            "monotonicity + right-shift monotonicity) when compressing the "
            "depth sort key from 32 bits to 16 bits.  Measures tie rates "
            "(Gaussians with indistinguishable sort keys) across synthetic "
            "and real depth distributions."
        ),
        "encoding": {
            "baseline": "isect_id = (image_id << (32+tile_n_bits)) | (tile_id<<32) | depth_uint32",
            "compressed": "isect_id = (image_id << (16+tile_n_bits)) | (tile_id<<16) | (depth_uint32>>16)",
            "baseline_sort_bits": "32 + tile_n_bits + image_n_bits",
            "compressed_sort_bits": "16 + tile_n_bits + image_n_bits",
            "cub_passes_saved": 4,
            "relative_precision": 2.0 ** (-7),
        },
        "sample_bucket_resolution": sample_buckets,
        "analyses": results,
        "summary": summary,
        "generated_by": "phase16_c1_ordering_analysis.py",
    }

    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return out_path


def print_resolution_table(sample_buckets: list[dict]) -> None:
    print("\n--- Float32 → Upper-16-Bit Bucket Resolution ---")
    print(f"  {'Depth':>8s} | {'Upper16':>6s} | {'Bucket Lo':>10s} | {'Bucket Hi':>10s} | "
          f"{'Width':>10s} | {'Rel %':>7s}")
    print(f"  {'-'*8} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*7}")
    for b in sample_buckets:
        print(f"  {b['depth']:>8.3f} | 0x{b['upper16_hex']:>4s}  | {b['bucket_start']:>10.6f} | "
              f"{b['bucket_end']:>10.6f} | {b['bucket_width']:>10.2e} | {b['relative_width_pct']:>6.3f}%")


def main() -> int:
    print("=" * 72)
    print("  Phase 16-C1: Depth ordering preservation analysis")
    print("  Upper-16-bit depth encoding for CUB radix sort key compression")
    print("=" * 72)

    t_start = time.perf_counter()

    results = run_all_analyses()
    summary = summarize(results)
    out_path = save(results, summary)

    # Resolution table
    if results and "sample_buckets" in results[0]:
        print_resolution_table(results[0]["sample_buckets"])

    elapsed = time.perf_counter() - t_start

    print(f"\n{'=' * 72}")
    print(f"  Done in {elapsed:.1f} s")
    print(f"  Scenarios: {summary['n_scenarios']}")
    if summary.get("worst_case"):
        w = summary["worst_case"]
        print(f"  Worst tie rate: {w['scene']}  "
              f"(tie_pair_ratio={w['tie_pair_ratio']:.2e}, "
              f"tie_elem_frac={w['tie_element_fraction']*100:.3f}%)")
    if summary.get("best_case"):
        b = summary["best_case"]
        print(f"  Best tie rate:  {b['scene']}  "
              f"(tie_pair_ratio={b['tie_pair_ratio']:.2e})")
    print(f"  Saved: {out_path}")
    print(f"{'=' * 72}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
