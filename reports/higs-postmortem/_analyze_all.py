#!/usr/bin/env python3
"""Analyze all result JSONs: print per-method counts and a sample file's structure."""
import json, glob, os
from collections import defaultdict

roots = [
    r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-all\results",
    r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-paper\results",
]
for root in roots:
    if not os.path.isdir(root):
        print(f"[missing] {root}")
        continue
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    print(f"\n### {root}: {len(files)} files")
    per = defaultdict(list)
    for f in files:
        base = os.path.basename(f)
        try:
            with open(f) as fh:
                j = json.load(fh)
        except Exception as e:
            print(f"  parse error {base}: {e}")
            continue
        parts = base.split("--")
        wl = parts[0]
        method = parts[1] if len(parts) > 1 else "?"
        per[(wl, method)].append(base)
        if len(per[(wl, method)]) == 1:
            print(f"  sample: {base}")
            print(f"    method={j.get('method')}, scene={j.get('scene')}, seed={j.get('seed')}, status={j.get('status')}, hw={j.get('hardware')}")
            perf = j.get("performance", {})
            print(f"    performance keys: {sorted(perf.keys())}")
            print(f"      wall_time_seconds={perf.get('wall_time_seconds')}")
            print(f"      time_to_quality_seconds={perf.get('time_to_quality_seconds')}")
            print(f"    resources keys: {sorted(j.get('resources', {}).keys())}")
            res = j.get("resources", {})
            print(f"      peak_gpu_memory_mib={res.get('peak_gpu_memory_mib')}")
            print(f"      final_gaussian_count={res.get('final_gaussian_count')}")
            qc = j.get("quality", {})
            print(f"    quality keys: {sorted(qc.keys())}")
            qc_curve = j.get("quality_curve")
            if isinstance(qc_curve, list) and qc_curve:
                print(f"    quality_curve len={len(qc_curve)}; first={json.dumps(qc_curve[0])}")
    for (wl, method), bases in sorted(per.items()):
        print(f"  {wl} / {method}: {len(bases)}")
