#!/usr/bin/env python3
"""Analyze room tile20 30K training results."""
import json

with open('results/epic05/phase7/phase7_room_t20_results.json') as f:
    t20 = json.load(f)

metrics = t20['metrics_log']
best_psnr = max(m['psnr'] for m in metrics)
final_m = metrics[-1]

# Steady-state: steps where iteration_ms < 5000 (not in heavy densification)
steady = [m for m in metrics if m['iteration_ms'] < 5000 and m['iteration_ms'] > 0]
if steady:
    avg_fwd = sum(m['fwd_ms'] for m in steady) / len(steady)
    avg_bwd = sum(m['bwd_ms'] for m in steady) / len(steady)
    avg_opt = sum(m['opt_ms'] for m in steady) / len(steady)
else:
    avg_fwd = avg_bwd = avg_opt = 0

total_clone = sum(m.get('cloned', 0) for m in metrics)
total_split = sum(m.get('split', 0) for m in metrics)
total_prune = sum(m.get('pruned', 0) for m in metrics)
avg_all = sum(m['iteration_ms'] for m in metrics) / len(metrics)

print('=' * 60)
print('Room tile20 30K Training Results')
print('=' * 60)
print(f'Wall time: {t20["total_wall_minutes"]:.1f} min ({t20["total_wall_seconds"]/3600:.1f} hr)')
print(f'Best PSNR: {best_psnr:.2f} dB')
print(f'Final PSNR: {final_m["psnr"]:.2f} dB')
print(f'Initial Gs: {t20["milestones"]["initial_gaussian_count"]:,}')
print(f'Final Gs: {t20["milestones"]["final_gaussian_count"]:,}')
print(f'Iter/s: {t20["iterations_per_second"]:.2f}')
print(f'Avg fwd (steady): {avg_fwd:.1f} ms')
print(f'Avg bwd (steady): {avg_bwd:.1f} ms')
print(f'Avg opt (steady): {avg_opt:.1f} ms')
print(f'Avg iter time (all): {avg_all:.1f} ms')
print(f'Peak VRAM: {max(m["peak_memory_mb"] for m in metrics if m["peak_memory_mb"] or 0):.0f} MB')
print(f'Total cloned: {total_clone:,}')
print(f'Total split: {total_split:,}')
print(f'Total pruned: {total_prune:,}')
print()

# Compare with Phase 7 tile16 and tile32
print('=' * 60)
print('Comparison with tile16 and tile32 (Phase 7)')
print('=' * 60)
print(f'{"Metric":<25} {"tile16":<12} {"tile20":<12} {"tile32":<12}')
print(f'{"Wall (min)":<25} {150.3:<12.1f} {t20["total_wall_minutes"]:<12.1f} {94.8:<12.1f}')
print(f'{"Best PSNR (dB)":<25} {29.27:<12.2f} {best_psnr:<12.2f} {29.39:<12.2f}')
print(f'{"Iter/s":<25} {3.33:<12.2f} {t20["iterations_per_second"]:<12.2f} {5.27:<12.2f}')
print(f'{"Speedup vs t16":<25} {1.00:<12.2f} {(150.3/t20["total_wall_minutes"]):<12.2f} {1.58:<12.2f}')
