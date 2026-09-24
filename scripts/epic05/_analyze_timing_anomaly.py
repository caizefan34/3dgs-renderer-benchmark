#!/usr/bin/env python3
"""Analyze tile20 timing anomalies - densification steps."""
import json

with open("results/epic05/phase7/phase7_room_t20_results.json") as f:
    t20 = json.load(f)
with open("results/epic05/phase7/phase7_room_30k_v2_16_results.json") as f:
    t16 = json.load(f)

ml20 = t20["metrics_log"]
ml16 = t16["metrics_log"]

# Find slowest tile20 steps with breakdown
print("tile20 - Top 15 slowest steps with breakdown")
print(f"{'Step':<8} {'Total(ms)':<12} {'Fwd(ms)':<10} {'Bwd(ms)':<10} {'Opt(ms)':<10} {'Topo(ms)':<10} {'Clone':<8} {'Split':<8} {'Prune':<8}")
slow20 = sorted(ml20, key=lambda m: m["iteration_ms"], reverse=True)[:15]
for m in slow20:
    print(f"{m['iteration']:<8} {m['iteration_ms']:<12.1f} {m.get('fwd_ms',0):<10.1f} {m.get('bwd_ms',0):<10.1f} {m.get('opt_ms',0):<10.1f} {m.get('topology_ms',0):<10.1f} {m.get('cloned',0):<8} {m.get('split',0):<8} {m.get('pruned',0):<8}")

print()
print("tile16 v2 - Top 15 slowest steps")
print(f"{'Step':<8} {'Total(ms)':<12} {'Clone':<8} {'Split':<8} {'Prune':<8}")
slow16 = sorted(ml16, key=lambda m: m["iteration_ms"], reverse=True)[:15]
for m in slow16:
    print(f"{m['iteration']:<8} {m['iteration_ms']:<12.1f} {m.get('cloned',0):<8} {m.get('split',0):<8} {m.get('pruned',0):<8}")

# Summary
print()
print("tile20 fwd/bwd/opt breakdown (non-densification steps)")
steady20 = [m for m in ml20 if m.get("cloned",0)+m.get("split",0)+m.get("pruned",0) == 0 and m["iteration_ms"] < 500]
if steady20:
    avg_fwd = sum(m["fwd_ms"] for m in steady20)/len(steady20)
    avg_bwd = sum(m["bwd_ms"] for m in steady20)/len(steady20)
    avg_opt = sum(m["opt_ms"] for m in steady20)/len(steady20)
    avg_topo = sum(m["topology_ms"] for m in steady20)/len(steady20)
    print(f"  Non-denf steps: {len(steady20)}")
    print(f"  Avg fwd: {avg_fwd:.1f}ms ({100*avg_fwd/(avg_fwd+avg_bwd+avg_opt+avg_topo):.0f}%)")
    print(f"  Avg bwd: {avg_bwd:.1f}ms ({100*avg_bwd/(avg_fwd+avg_bwd+avg_opt+avg_topo):.0f}%)")
    print(f"  Avg opt: {avg_opt:.1f}ms ({100*avg_opt/(avg_fwd+avg_bwd+avg_opt+avg_topo):.0f}%)")
    print(f"  Avg topo: {avg_topo:.1f}ms ({100*avg_topo/(avg_fwd+avg_bwd+avg_opt+avg_topo):.0f}%)")
    print(f"  Avg total: {avg_fwd+avg_bwd+avg_opt+avg_topo:.1f}ms")
