"""
Compare C1 vs Baseline benchmark results.
"""
import json

with open(r'C:\Users\36570\3dgs-renderer-benchmark\results\epic05\phase16\c1_benchmark_c1_result.json') as f:
    c1 = json.load(f)
with open(r'C:\Users\36570\3dgs-renderer-benchmark\results\epic05\phase16\c1_benchmark_baseline_result.json') as f:
    bl = json.load(f)

print("=== C1 vs BASELINE COMPARISON ===")
print(f"{'Label':>25s} {'N_isects':>12s} {'Baseline(ms)':>15s} {'C1(ms)':>12s} {'Diff':>10s}")
print("-" * 75)
for s1, s2 in zip(c1['scenarios'], bl['scenarios']):
    label = s1['label']
    ni = f"{s1['n_isects']:,}"
    bl_t = s2['forward_time_ms']['median']
    c1_t = s1['forward_time_ms']['median']
    diff = c1_t - bl_t
    pct = (diff / bl_t) * 100
    print(f'{label:>25s} {ni:>12s} {bl_t:>12.3f}ms {c1_t:>11.3f}ms {diff:>+7.3f}ms ({pct:+.1f}%)')

print()
# Also compute ratio
ni_vals = [s1['n_isects'] for s1 in c1['scenarios']]
bl_vals = [s['forward_time_ms']['median'] for s in bl['scenarios']]
c1_vals = [s['forward_time_ms']['median'] for s in c1['scenarios']]

for i in range(len(ni_vals)):
    speedup = 1 - c1_vals[i] / bl_vals[i]
    print(f"  {ni_vals[i]:,} isects: speedup = {speedup*100:+.1f}%")
