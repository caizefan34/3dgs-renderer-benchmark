import json

# C49 G_opt concentration summary
with open("results/reference_v1/r0.1/c49_gopt_concentration.json") as f:
    d = json.load(f)
print("=== C49 G_opt (mature summary) ===")
for k,v in d["mature_summary"].items():
    print(f"  {k}: mean={v['mean']:.4f} median={v['median']:.4f}")

# C49 G_dens
with open("results/reference_v1/r0.1/c49_gdens_concentration.json") as f:
    d = json.load(f)
print("\n=== C49 G_dens (mature summary) ===")
for k,v in d["mature_summary"].items():
    print(f"  {k}: mean={v['mean']:.4f} median={v['median']:.4f}")

# C50 lag-1 G_opt overall
with open("results/reference_v1/r0.1/c50_true_lag1_gopt.json") as f:
    d = json.load(f)
print("\n=== C50 G_opt lag-1 (overall) ===")
for k,v in d["overall"].items():
    print(f"  {k}: mean={v['mean']:.4f} median={v['median']:.4f} p10={v['p10']:.4f} p90={v['p90']:.4f}")

# C50 topology stratified
with open("results/reference_v1/r0.1/c50_topology_stratified.json") as f:
    d = json.load(f)
print("\n=== C50 topology stratified (pearson + top10_jaccard) ===")
for cat, metrics in d.items():
    p = metrics.get("pearson", {})
    j = metrics.get("top10_jaccard", {})
    print(f"  {cat}: pearson_mean={p.get('mean',0):.4f} pearson_n={p.get('n',0)} jaccard10_mean={j.get('mean',0):.4f}")

# C50 camera conditioned
with open("results/reference_v1/r0.1/c50_camera_conditioned.json") as f:
    d = json.load(f)
print("\n=== C50 camera conditioned (pearson) ===")
for cat, data in d.get("by_center_distance", {}).items():
    p = data.get("metrics", {}).get("pearson", {})
    print(f"  {cat} ({data['threshold']}): pearson_mean={p.get('mean',0):.4f} n={p.get('n',0)}")

# C53 workload
with open("results/reference_v1/r0.1/c53_true_lag1_workload.json") as f:
    d = json.load(f)
print("\n=== C53 workload lag-1 (overall) ===")
for k,v in d["overall"].items():
    print(f"  {k}: mean={v['mean']:.4f} median={v['median']:.4f}")

# C53 camera conditioned
with open("results/reference_v1/r0.1/c53_camera_conditioned.json") as f:
    d = json.load(f)
print("\n=== C53 camera conditioned (pearson) ===")
for cat, data in d.get("by_center_distance", {}).items():
    p = data.get("metrics", {}).get("pearson", {})
    print(f"  {cat} ({data['threshold']}): pearson_mean={p.get('mean',0):.4f} n={p.get('n',0)}")

# G_opt vs G_dens correlation
with open("results/reference_v1/r0.1/gopt_gdens_correlation.json") as f:
    d = json.load(f)
print("\n=== G_opt vs G_dens correlation ===")
for k,v in d.items():
    print(f"  {k}: {v}")
