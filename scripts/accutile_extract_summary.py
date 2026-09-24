#!/usr/bin/env python3
"""Extract correctness summary from saved JSON results."""
import json, sys, glob

for path in sorted(glob.glob("/tmp/accutile_a100_results/accutile_*.json")):
    if "all_scenes" in path:
        continue
    with open(path) as f:
        d = json.load(f)
    scene = d.get("scene", "?")
    corr = d.get("correctness", {})
    bench = d.get("benchmark", {})
    print(f"\n=== {scene} ===")
    print(f"Gaussians: {corr.get('gaussian_count')}, Resolution: {corr.get('resolution')}, SH: {corr.get('sh_degree')}")
    fwd = corr.get("forward", {})
    print(f"Forward RGB: max_abs={fwd.get('rgb',{}).get('max_abs')}, rel_L2={fwd.get('rgb',{}).get('relative_L2')}")
    print(f"Forward alpha: max_abs={fwd.get('alpha',{}).get('max_abs')}, rel_L2={fwd.get('alpha',{}).get('relative_L2')}")
    print(f"Forward depth: max_abs={fwd.get('depth',{}).get('max_abs')}, rel_L2={fwd.get('depth',{}).get('relative_L2')}")
    bwd = corr.get("backward", {})
    for k in ("means", "quats", "scales", "opacities", "colors"):
        v = bwd.get(k, {})
        print(f"Backward {k}: max_abs={v.get('max_abs')}, mean_abs={v.get('mean_abs')}, rel_L2={v.get('relative_L2')}, nan={v.get('nan_count_a')}/{v.get('nan_count_b')}, inf={v.get('inf_count_a')}/{v.get('inf_count_b')}")
    isect = corr.get("intersections", {})
    print(f"Intersections: N_AABB={isect.get('N_AABB')}, N_AccuTile={isect.get('N_AccuTile')}, reduction={isect.get('reduction')}")
    if bench:
        b1 = bench.get("B1", {})
        b1a = bench.get("B1A", {})
        print(f"B1: fwd={b1.get('forward',{}).get('mean_ms'):.2f}ms, bwd={b1.get('backward',{}).get('mean_ms'):.2f}ms, total={b1.get('total',{}).get('mean_ms'):.2f}ms")
        print(f"B1A: fwd={b1a.get('forward',{}).get('mean_ms'):.2f}ms, bwd={b1a.get('backward',{}).get('mean_ms'):.2f}ms, total={b1a.get('total',{}).get('mean_ms'):.2f}ms")
        print(f"Speedup: fwd={bench.get('speedup_forward',0):.4f}, bwd={bench.get('speedup_backward',0):.4f}, total={bench.get('speedup_total',0):.4f}")
