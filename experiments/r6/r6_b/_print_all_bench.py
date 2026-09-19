#!/usr/bin/env python3
"""Print comprehensive summary of all 9 A100 benchmark results."""
import json, sys
from pathlib import Path

bench_dir = Path(sys.argv[1])
files = sorted(bench_dir.glob("*.json"))

print(f"{'Scene':<12} {'Iter':>5} {'N':>8} {'Mode':>8} {'T_bwd':>8} {'T_iter':>8} {'T_prep':>8} {'T_scat':>8} {'T_clear':>8} {'MetaMB':>7} {'Δiter%':>7}")
print("-" * 110)

all_results = {}
for f in files:
    d = json.loads(f.read_text())
    scene = d["baseline"]["scene"]
    ckpt = d["baseline"]["ckpt_iter"]
    n_g = d["baseline"]["n_gaussians"]
    all_results[f.stem] = d
    for mode in ("baseline", "b0", "b1"):
        r = d[mode]
        t_bwd = r["T_bwd_ms"]["mean"]
        t_iter = r["T_iter_ms"]["mean"]
        t_prep = r.get("T_prepare_ms", {}).get("mean", "-")
        t_scat = r.get("T_scatter_ms", {}).get("mean", "-")
        t_clear = r.get("T_clear_ms", {}).get("mean", "-")
        meta = r.get("metadata_MB", "-")
        speedup = r.get("speedup_iter_pct", 0)
        t_prep_s = f"{t_prep:.3f}" if isinstance(t_prep, float) else "-"
        t_scat_s = f"{t_scat:.3f}" if isinstance(t_scat, float) else "-"
        t_clear_s = f"{t_clear:.3f}" if isinstance(t_clear, float) else "-"
        meta_s = f"{meta:.2f}" if isinstance(meta, (int, float)) else "-"
        speedup_s = f"{speedup:+.2f}" if speedup != 0 else "-"
        print(f"{scene:<12} {ckpt:>5} {n_g:>8} {mode:>8} {t_bwd:>8.2f} {t_iter:>8.2f} {t_prep_s:>8} {t_scat_s:>8} {t_clear_s:>8} {meta_s:>7} {speedup_s:>7}")

print("\n\n=== Summary ===")
b0_speedups = []
b1_speedups = []
for f in files:
    d = json.loads(f.read_text())
    stem = f.stem
    b0_s = d["b0"].get("speedup_iter_pct", 0)
    b1_s = d["b1"].get("speedup_iter_pct", 0)
    b0_speedups.append(b0_s)
    b1_speedups.append(b1_s)
    print(f"  {stem:<20} B0: {b0_s:+.2f}%  B1: {b1_s:+.2f}%")

import numpy as np
print(f"\n  B0 mean Δiter: {np.mean(b0_speedups):+.2f}%  (range: {min(b0_speedups):+.2f}% to {max(b0_speedups):+.2f}%)")
print(f"  B1 mean Δiter: {np.mean(b1_speedups):+.2f}%  (range: {min(b1_speedups):+.2f}% to {max(b1_speedups):+.2f}%)")
print(f"  B0 negative: {sum(1 for s in b0_speedups if s < 0)}/9")
print(f"  B1 negative: {sum(1 for s in b1_speedups if s < 0)}/9")
