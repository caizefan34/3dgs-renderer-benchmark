#!/usr/bin/env python3
"""Check progress of C31 runs by reading result files."""
from __future__ import annotations
import json, sys
from pathlib import Path

results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/phase-c31")
files = sorted(results_dir.glob("c31_*.json"))
if not files:
    print("No C31 result files found.")
    sys.exit(1)

for f in files:
    try:
        d = json.load(open(f))
        phase = d.get("phase", f.stem)
        ti = d.get("t_iter_ms", {})
        if ti:
            print(f"  {f.name}: T_iter={ti.get('mean', '?'):>7.2f}ms ± {ti.get('std', '?'):.2f}  "
                  f"(steps={d.get('measured', d.get('measured_steps', '?'))})")
        elif "results" in d:
            r = d["results"]
            for k, v in r.items():
                if isinstance(v, dict) and "mean" in v:
                    print(f"  {f.name}: {k}_mean={v['mean']:.3f}")
                    break
            else:
                print(f"  {f.name}: {phase} ({d.get('verdict','?')})")
        else:
            print(f"  {f.name}: {phase} ({d.get('verdict','?')})")
    except Exception as e:
        print(f"  {f.name}: ERROR - {e}")
