#!/usr/bin/env python3
"""Compare densification between tile16 and tile20."""
import json

with open('results/epic05/phase7/phase7_room_t20_results.json') as f:
    t20 = json.load(f)
with open('results/epic05/phase7/phase7_room_30k_v2_16_results.json') as f:
    t16v2 = json.load(f)

ml20 = t20['metrics_log']
ml16 = t16v2['metrics_log']

c20 = sum(m.get('cloned',0) for m in ml20)
s20 = sum(m.get('split',0) for m in ml20)
p20 = sum(m.get('pruned',0) for m in ml20)
c16 = sum(m.get('cloned',0) for m in ml16)
s16 = sum(m.get('split',0) for m in ml16)
p16 = sum(m.get('pruned',0) for m in ml16)
de20 = sum(1 for m in ml20 if m.get('cloned',0)+m.get('split',0)+m.get('pruned',0) > 0)
de16 = sum(1 for m in ml16 if m.get('cloned',0)+m.get('split',0)+m.get('pruned',0) > 0)

print('=' * 60)
print('Densification Comparison')
print('=' * 60)
print(f'{"Metric":<25} {"tile16_v2":<14} {"tile20"}')
print(f'{"Denf events":<25} {de16:<14} {de20}')
print(f'{"Total cloned":<25} {c16:<14,} {c20:,}')
print(f'{"Total split":<25} {s16:<14,} {s20:,}')
print(f'{"Total pruned":<25} {p16:<14,} {p20:,}')
print(f'{"Final Gs":<25} {ml16[-1]["num_gaussians"]:<14,} {ml20[-1]["num_gaussians"]:,}')
print(f'{"Best PSNR":<25} {max(m["psnr"] for m in ml16):<14.2f} {max(m["psnr"] for m in ml20):.2f}')

# Timing comparison - steady state (no densification)
steady16 = [m for m in ml16 if m['iteration_ms'] < 5000 and m['iteration_ms'] > 0]
steady20 = [m for m in ml20 if m['iteration_ms'] < 5000 and m['iteration_ms'] > 0]
if steady16 and steady20:
    print()
    print('=' * 60)
    print('Steady-State Timing (non-densification steps)')
    print('=' * 60)
    print(f'{"Metric":<25} {"tile16_v2":<14} {"tile20"}')
    fwd16 = sum(m['fwd_ms'] for m in steady16)/len(steady16)
    bwd16 = sum(m['bwd_ms'] for m in steady16)/len(steady16)
    opt16 = sum(m['opt_ms'] for m in steady16)/len(steady16)
    topo16 = sum(m['topology_ms'] for m in steady16)/len(steady16)
    fwd20 = sum(m['fwd_ms'] for m in steady20)/len(steady20)
    bwd20 = sum(m['bwd_ms'] for m in steady20)/len(steady20)
    opt20 = sum(m['opt_ms'] for m in steady20)/len(steady20)
    topo20 = sum(m['topology_ms'] for m in steady20)/len(steady20)
    print(f'{"Fwd (ms)":<25} {fwd16:<14.2f} {fwd20:.2f}')
    print(f'{"Bwd (ms)":<25} {bwd16:<14.2f} {bwd20:.2f}')
    print(f'{"Opt (ms)":<25} {opt16:<14.2f} {opt20:.2f}')
    print(f'{"Topo (ms)":<25} {topo16:<14.2f} {topo20:.2f}')
    print(f'{"Total (ms)":<25} {fwd16+bwd16+opt16+topo16:<14.2f} {fwd20+bwd20+opt20+topo20:.2f}')

# Total time comparison
print()
print('=' * 60)
print('Wall Time Comparison')
print('=' * 60)
print(f'{"Metric":<25} {"tile16_v2":<14} {"tile20"}')
print(f'{"Wall (min)":<25} {t16v2["total_wall_minutes"]:<14.1f} {t20["total_wall_minutes"]:.1f}')
print(f'{"Speedup":<25} {1.0:<14.2f} {t16v2["total_wall_minutes"]/t20["total_wall_minutes"]:.2f}')
