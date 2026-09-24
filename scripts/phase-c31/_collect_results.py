#!/usr/bin/env python3
"""Collect C31-A DDP results and print summary table."""
from __future__ import annotations
import json
from pathlib import Path

results_dir = Path("results/phase-c31")

print("=== C31-A DDP Results ===")
for n in [1, 2, 4, 8]:
    f = results_dir / f"c31_a_n{n}.json"
    try:
        d = json.load(open(f))
        ti = d["t_iter_ms"]
        comm = d.get("comm_overhead_ms_est", "?")
        speedup = d.get("speedup_vs_baseline", "N/A")
        eff = d.get("scaling_efficiency_pct", "N/A")
        print(f"  N={d['world_size']}: T_iter={ti['mean']:.2f}+-{ti['std']:.2f}ms  "
              f"fwd={d['fwd_gpu_ms']['mean']:.2f}  "
              f"bwd={d['bwd_gpu_ms']['mean']:.2f}  "
              f"opt={d['opt_gpu_ms']['mean']:.2f}  "
              f"speedup={speedup}  eff={eff}%  "
              f"comm={comm}ms  "
              f"psnr={d['final_psnr']:.2f}")
    except FileNotFoundError:
        print(f"  N={n}: file not found")
    except Exception as e:
        print(f"  N={n}: ERROR - {e}")

print()
print("Summary table:")
print(f"{'N':>5} {'T_iter':>10} {'Fwd':>8} {'Bwd':>8} {'Opt':>8} {'Speedup':>8} {'Eff':>6} {'Comm':>8}")
print("-" * 65)
for n in [1, 2, 4, 8]:
    f = results_dir / f"c31_a_n{n}.json"
    try:
        d = json.load(open(f))
        ti = d["t_iter_ms"]
        sp = d.get("speedup_vs_baseline", 0)
        eff = d.get("scaling_efficiency_pct", 0)
        comm = d.get("comm_overhead_ms_est", 0)
        print(f"{n:>5} {ti['mean']:>8.2f}ms {d['fwd_gpu_ms']['mean']:>7.2f}ms "
              f"{d['bwd_gpu_ms']['mean']:>7.2f}ms {d['opt_gpu_ms']['mean']:>7.2f}ms "
              f"{sp:>7.3f}x {eff:>5.1f}% {comm:>7.3f}ms")
    except Exception as e:
        print(f"{n:>5} ERROR {e}")
