#!/usr/bin/env python3
import json, sys
d = json.load(open(sys.argv[1]))
for m in ("baseline", "b0", "b1"):
    r = d[m]
    print(f"\n=== {m} ===")
    for k in sorted(r.keys()):
        if k.startswith("T_") and isinstance(r[k], dict):
            print(f"  {k}: mean={r[k]['mean']:.3f}ms std={r[k]['std']:.3f}ms p95={r[k]['p95']:.3f}ms")
        elif k in ("metadata_MB", "n_rows", "speedup_bwd_pct", "speedup_iter_pct",
                    "peak_memory_MB", "memory_delta_MB", "n_gaussians"):
            print(f"  {k}: {r[k]}")
