#!/usr/bin/env python3
"""Deep analysis of tile20 training vs tile16 training dynamics."""
import json

with open('results/epic05/phase7/phase7_room_t20_results.json') as f:
    t20 = json.load(f)
with open('results/epic05/phase10a/phase10a_m2_full_training_results.json') as f:
    t16 = json.load(f)

m20 = t20['metrics_log']
m16_packed = t16['results']['packed']['metrics_log']

# Densification analysis
d20 = [m for m in m20 if m.get('cloned',0)+m.get('split',0)+m.get('pruned',0) > 0]
d16 = [m for m in m16_packed if m.get('cloned',0)+m.get('split',0)+m.get('pruned',0) > 0]

print('=== Densification Event Comparison ===')
print(f'{"Metric":<35} {"tile16 (P10A)":<15} {"tile20":<15}')
print(f'{"Total denf events":<35} {len(d16):<15} {len(d20):<15}')
print(f'{"Total cloned":<35} {sum(m["cloned"] for m in d16):<15,} {sum(m["cloned"] for m in d20):<15,}')
print(f'{"Total split":<35} {sum(m["split"] for m in d16):<15,} {sum(m["split"] for m in d20):<15,}')
print(f'{"Total pruned":<35} {sum(m["pruned"] for m in d16):<15,} {sum(m["pruned"] for m in d20):<15,}')
print(f'{"Avg clone per event":<35} {sum(m["cloned"] for m in d16)/len(d16):<15.1f} {sum(m["cloned"] for m in d20)/len(d20):<15.1f}')
print(f'{"Steady-state fwd (ms)":<35} {sum(m["fwd_ms"] for m in m16_packed if m["fwd_ms"]<100)/sum(1 for m in m16_packed if m["fwd_ms"]<100):<15.1f} {sum(m["fwd_ms"] for m in m20 if m["fwd_ms"]<100)/sum(1 for m in m20 if m["fwd_ms"]<100):<15.1f}')
print(f'{"Steady-state bwd (ms)":<35} {sum(m["bwd_ms"] for m in m16_packed if m["bwd_ms"]<200)/sum(1 for m in m16_packed if m["bwd_ms"]<200):<15.1f} {sum(m["bwd_ms"] for m in m20 if m["bwd_ms"]<200)/sum(1 for m in m20 if m["bwd_ms"]<200):<15.1f}')

# PSNR trajectory comparison
print()
print('=== PSNR Trajectory ===')
p20 = [(m['iteration'], m['psnr']) for m in m20 if m['iteration'] % 500 == 0]
p16 = [(m['iteration'], m['psnr']) for m in m16_packed if m['iteration'] % 500 == 0]
print(f'{"Step":<8} {"t16 PSNR":<12} {"t20 PSNR":<12}')
for s20, s16 in zip(p20[:20], p16[:20]):
    print(f'{s20[0]:<8} {s16[1]:<12.2f} {s20[1]:<12.2f}')

# Gaussian count trajectory
print()
print('=== Gaussian Count Trajectory ===')
g20 = [(m['iteration'], m['num_gaussians']) for m in m20 if m['iteration'] % 500 == 0]
g16 = [(m['iteration'], m['num_gaussians']) for m in m16_packed if m['iteration'] % 500 == 0]
print(f'{"Step":<8} {"t16 Gs":<14} {"t20 Gs":<14}')
for s20, s16 in zip(g20, g16):
    print(f'{s20[0]:<8} {s16[1]:<14,} {s20[1]:<14,}')
