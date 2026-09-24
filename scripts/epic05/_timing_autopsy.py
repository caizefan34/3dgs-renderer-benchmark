#!/usr/bin/env python3
"""Deep analysis of tile20 timing — total time breakdown."""
import json
from collections import Counter

with open("results/epic05/phase7/phase7_room_t20_results.json") as f:
    t20 = json.load(f)
with open("results/epic05/phase7/phase7_room_30k_v2_16_results.json") as f:
    t16 = json.load(f)

ml20 = t20["metrics_log"]
ml16 = t16["metrics_log"]

# Total wall time breakdown
print("=== Total Wall Time ===")
print(f"  tile16: {t16['total_wall_seconds']:.0f}s = {t16['total_wall_minutes']:.1f} min")
print(f"  tile20: {t20['total_wall_seconds']:.0f}s = {t20['total_wall_minutes']:.1f} min")
print(f"  tile32: check")
print()

# Sum of measured iteration_ms
sum_it20 = sum(m["iteration_ms"] for m in ml20) / 1000  # convert to seconds
sum_it16 = sum(m["iteration_ms"] for m in ml16) / 1000
print(f"=== Measured vs Total Time ===")
print(f"  tile16: sum(iteration_ms)={sum_it16:.0f}s, total={t16['total_wall_seconds']:.0f}s, overhead={t16['total_wall_seconds']-sum_it16:.0f}s")
print(f"  tile20: sum(iteration_ms)={sum_it20:.0f}s, total={t20['total_wall_seconds']:.0f}s, overhead={t20['total_wall_seconds']-sum_it20:.0f}s")

# The iteration_ms is the time for the LOGGED iteration (every 10th step)
# Each entry covers one step's timing
# wall_time should be ~ sum(iteration_ms * 10) + checkpoint + init

# Breakdown tile20 timing by step range
print()
print("=== tile20 Step Time Histogram ===")
buckets = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, 5000), (5000, 30000), (30000, 60000), (60000, 160000)]
for lo, hi in buckets:
    count = sum(1 for m in ml20 if lo <= m["iteration_ms"] < hi)
    if count > 0:
        avg_t = sum(m["iteration_ms"] for m in ml20 if lo <= m["iteration_ms"] < hi) / count
        total = sum(m["iteration_ms"] for m in ml20 if lo <= m["iteration_ms"] < hi)
        print(f"  [{lo:>5},{hi:>6})ms: {count:>4} entries, avg={avg_t:>8.1f}ms, total={total/1000:>8.0f}s")

# Are the slow steps around specific iteration ranges?
print()
print("=== tile20 Slow Steps by Region ===")
for m in ml20:
    if m["iteration_ms"] > 20000:
        d = m.get("cloned",0) + m.get("split",0) + m.get("pruned",0)
        print(f"  Step {m['iteration']:>5}: {m['iteration_ms']/1000:>6.1f}s fwd={m.get('fwd_ms',0)/1000:.1f}s bwd={m.get('bwd_ms',0)/1000:.1f}s opt={m.get('opt_ms',0)/1000:.1f}s topo={m.get('topology_ms',0)/1000:.1f}s clone={m.get('cloned',0):>4} split={m.get('split',0):>4} prune={m.get('pruned',0):>4}")

# Non-densification steady state
print()
print("=== tile20 Non-Denf Steady State ===")
nod20 = [m for m in ml20 if m.get("cloned",0)+m.get("split",0)+m.get("pruned",0) == 0]
fwd = [m["fwd_ms"] for m in nod20 if m["fwd_ms"] > 0]
bwd = [m["bwd_ms"] for m in nod20 if m["bwd_ms"] > 0]
opt = [m["opt_ms"] for m in nod20 if m["opt_ms"] > 0]
print(f"  {len(nod20)} entries")
print(f"  Fwd:  min={min(fwd):.1f} median={sorted(fwd)[len(fwd)//2]:.1f} mean={sum(fwd)/len(fwd):.1f} max={max(fwd):.1f} ms")
print(f"  Bwd:  min={min(bwd):.1f} median={sorted(bwd)[len(bwd)//2]:.1f} mean={sum(bwd)/len(bwd):.1f} max={max(bwd):.1f} ms")
print(f"  Opt:  min={min(opt):.1f} median={sorted(opt)[len(opt)//2]:.1f} mean={sum(opt)/len(opt):.1f} max={max(opt):.1f} ms")

# Check tile16 timing histogram for comparison
print()
print("=== tile16 Step Time Histogram ===")
for lo, hi in buckets:
    count = sum(1 for m in ml16 if lo <= m["iteration_ms"] < hi)
    if count > 0:
        total = sum(m["iteration_ms"] for m in ml16 if lo <= m["iteration_ms"] < hi)
        print(f"  [{lo:>5},{hi:>6})ms: {count:>4} entries, total={total/1000:>8.0f}s")
