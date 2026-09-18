#!/usr/bin/env python3
"""Summarize all R6 profiling results."""
import json, glob, os, sys

prof_dir = "/mnt/storage_pool/liaoyuanjun/r6_profiling"

print("=== R6-1 Backward Decomposition ===")
for f in sorted(glob.glob(f"{prof_dir}/r6_1_*.json")):
    if "smoke" in f:
        continue
    d = json.load(open(f))
    name = os.path.basename(f).replace("r6_1_", "").replace(".json", "")
    kd = d["kernel_decomposition"]
    print(f"  {name:20s}: N={d['N_total']:>8d} r_touch={d['r_touch']:.3f} "
          f"T_bwd={d['T_bwd_ms']['mean']:.1f}ms "
          f"raster={kd['raster_bwd']['per_iter_ms']:.1f}ms "
          f"memset={kd['memset_zero']['per_iter_ms']:.2f}ms({kd['T_zero_pct_bwd']:.1f}%) "
          f"T_iter={d['T_iter_ms']['mean']:.1f}ms")

print("\n=== R6-4 Warp Duplicate ===")
for f in sorted(glob.glob(f"{prof_dir}/r6_4_*.json")):
    d = json.load(open(f))
    name = os.path.basename(f).replace("r6_4_", "").replace(".json", "")
    agg = d["aggregate"]
    print(f"  {name:20s}: R_atomic={agg['R_atomic_mean']:.2f}±{agg['R_atomic_std']:.2f} "
          f"tiles/G={agg['tiles_per_gauss_mean']:.1f} "
          f"warps/G={agg['warps_per_gauss_mean']:.1f} "
          f"n_isects={agg['n_isects_mean']:.0f} N={d['N_total']}")

print("\n=== R6-5 Fusion Oracle ===")
for f in sorted(glob.glob(f"{prof_dir}/r6_5_*.json")):
    d = json.load(open(f))
    name = os.path.basename(f).replace("r6_5_", "").replace(".json", "")
    print(f"  {name:20s}: T_opt={d['T_optimizer_ms']['mean']:.2f}ms "
          f"({d['T_optimizer_pct_iter']:.1f}% T_iter) "
          f"T_C_cons={d['T_C_conservative_ms']:.2f}ms "
          f"({d['T_C_conservative_pct_iter']:.1f}% T_iter) "
          f"avoid_traffic={d['avoidable_grad_traffic_bytes']/1e6:.0f}MB")
