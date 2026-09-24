#!/usr/bin/env python3
"""Analyze tile20 training vs tile16 training."""
import json

with open('results/epic05/phase7/phase7_room_t20_results.json') as f:
    t20 = json.load(f)
with open('results/epic05/phase10a/phase10a_m2_full_training_results.json') as f:
    t16 = json.load(f)

print('=== Phase10A structure ===')
for k, v in t16['results'].items():
    print(f'  {k}: time={v["total_time_min"]}min psnr={v["best_psnr"]}dB ips={v["iter_per_sec"]}')
    if 'metrics_log' in v:
        ml = v['metrics_log']
        print(f'    metrics_log: {len(ml)} entries')

print()
print('=== Phase7 tile20 ===')
print(f'  time={t20["total_wall_minutes"]}min psnr={t20["milestones"]["best_psnr"]}dB')
ml20 = t20['metrics_log']
print(f'  metrics_log: {len(ml20)} entries')
print(f'  First: {ml20[0]}')
print(f'  Last: {ml20[-1]}')
