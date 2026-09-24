import json

with open("results/reference_v1/r0.1/c50_true_lag1_gopt.json") as f:
    d = json.load(f)
print("=== C50 G_opt per-window ===")
for w in ["2000","5000","10000","14000"]:
    if w in d["per_window"]:
        pw = d["per_window"][w]
        p = pw.get("pearson",{}).get("median",0)
        j = pw.get("top50_jaccard",{}).get("median",0)
        j10 = pw.get("top10_jaccard",{}).get("median",0)
        n = pw.get("pearson",{}).get("n",0)
        print(f"  window {w}: pearson={p:.4f} top10_jaccard={j10:.4f} top50_jaccard={j:.4f} n={n}")

with open("results/reference_v1/r0.1/c49_gopt_concentration.json") as f:
    d = json.load(f)
print("\n=== C49 G_opt per-window (gini, top10, top50) ===")
for w in ["2000","5000","10000","14000"]:
    if w in d["per_window"]:
        pw = d["per_window"][w]
        # average over iterations in this window
        ginis = [v.get("gini",0) for v in pw.values()]
        t10 = [v.get("top10_mass",0) for v in pw.values()]
        t50 = [v.get("top50_mass",0) for v in pw.values()]
        print(f"  window {w}: gini={sum(ginis)/len(ginis):.4f} top10={sum(t10)/len(t10):.4f} top50={sum(t50)/len(t50):.4f}")

# Provenance
with open("results/reference_v1/r0.1/window_2000/provenance.json") as f:
    p = json.load(f)
print(f"\n=== Provenance (window 2000) ===")
for k,v in p.items():
    print(f"  {k}: {v}")
