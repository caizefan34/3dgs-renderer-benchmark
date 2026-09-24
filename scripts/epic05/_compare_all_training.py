#!/usr/bin/env python3
"""Full training comparison."""
import json

files = {
    "tile16": "results/epic05/phase7/phase7_room_30k_v2_16_results.json",
    "tile20": "results/epic05/phase7/phase7_room_t20_results.json",
    "tile32": "results/epic05/phase7/phase7_room_30k_v2_t32_32_results.json",
}

data = {}
for name, path in files.items():
    with open(path) as f:
        data[name] = json.load(f)

print("Wall Time Comparison")
print("-" * 50)
for name in ["tile16", "tile20", "tile32"]:
    d = data[name]
    print(f"  {name}: {d['total_wall_seconds']:.0f}s = {d['total_wall_minutes']:.1f} min")

print()
print("PSNR Comparison")
print("-" * 50)
for name in ["tile16", "tile20", "tile32"]:
    d = data[name]
    print(f"  {name}: best PSNR = {d['milestones']['best_psnr']:.2f} dB")

print()
print("Gaussian Count")
print("-" * 50)
for name in ["tile16", "tile20", "tile32"]:
    d = data[name]
    print(f"  {name}: init={d['milestones']['initial_gaussian_count']:,} final={d['milestones']['final_gaussian_count']:,}")

print()
print("Iter/s")
print("-" * 50)
for name in ["tile16", "tile20", "tile32"]:
    d = data[name]
    print(f"  {name}: {d['iterations_per_second']:.2f}")

# Timing distribution for tile20
ml20 = data["tile20"]["metrics_log"]
times20 = [m["iteration_ms"] for m in ml20 if m["iteration_ms"] > 0]
print()
print("tile20 Step Time Distribution")
print("-" * 50)
print(f"  Min: {min(times20):.0f} ms")
print(f"  Median: {sorted(times20)[len(times20)//2]:.0f} ms")
print(f"  Mean: {sum(times20)/len(times20):.0f} ms")
print(f"  Max: {max(times20):.0f} ms")
slow20 = [t for t in times20 if t > 1000]
print(f"  Steps > 1s: {len(slow20)} / {len(times20)} ({100*len(slow20)/len(times20):.1f}%)")
very_slow = [t for t in times20 if t > 10000]
print(f"  Steps > 10s: {len(very_slow)} / {len(times20)} ({100*len(very_slow)/len(times20):.1f}%)")

# Timing distribution for tile16 (no fwd_ms breakdown)
ml16 = data["tile16"]["metrics_log"]
times16 = [m["iteration_ms"] for m in ml16 if m["iteration_ms"] > 0]
print()
print("tile16 Step Time Distribution")
print("-" * 50)
print(f"  Min: {min(times16):.0f} ms")
print(f"  Median: {sorted(times16)[len(times16)//2]:.0f} ms")
print(f"  Mean: {sum(times16)/len(times16):.0f} ms")
print(f"  Max: {max(times16):.0f} ms")
slow16 = [t for t in times16 if t > 1000]
print(f"  Steps > 1s: {len(slow16)} / {len(times16)} ({100*len(slow16)/len(times16):.1f}%)")
