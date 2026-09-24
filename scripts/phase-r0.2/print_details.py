import json

# Backward profile
with open("results/reference_v1/r0.2/backward_profile.json") as f:
    p = json.load(f)
print("=== Backward Profile ===")
print(f"N={p['N_gaussians']}, SH degree={p['sh_degree']}")
for phase, s in p["timings_ms"].items():
    print(f"  {phase}: mean={s['mean']:.2f}ms median={s['median']:.2f}ms")
print(f"\nEstimated breakdown:")
for k, v in p["estimated_breakdown"].items():
    if isinstance(v, float):
        print(f"  {k}: {v:.2f}ms")
print(f"\nAmdahl ceilings:")
for k, v in p["amdahl_ceilings"].items():
    print(f"  {k}: {100*v['value']:.1f}% — {v['description']}")

# Final decision
with open("results/reference_v1/r0.2/final_decision.json") as f:
    d = json.load(f)
print(f"\n=== Final Decision ===")
print(f"Decision: {d['decision']}")
print(f"Reason: {d['reason']}")

# Per-window coverage
with open("results/reference_v1/r0.2/predictive_gradient_mass_coverage.json") as f:
    cov = json.load(f)
print(f"\n=== Per-window Coverage (G_opt_total K=50) ===")
for w in ["2000","5000","10000","14000"]:
    if w in cov["per_window"].get("gopt_total", {}):
        pw = cov["per_window"]["gopt_total"][w]
        print(f"  window {w}: oracle={pw['oracle_k50']['median']:.4f} prev={pw['previous_k50']['median']:.4f} rand={pw['random_k50']['median']:.4f}")

# Per-window G_dens→G_opt
with open("results/reference_v1/r0.2/gdens_gopt_per_gaussian.json") as f:
    dg = json.load(f)
print(f"\n=== Per-window G_dens↔G_opt ===")
for w in ["2000","5000","10000","14000"]:
    if w in dg["per_window"]:
        pw = dg["per_window"][w]
        print(f"  window {w}: pearson={pw['pearson_gdens_gopt_total']['median']:.4f} spearman={pw['spearman_gdens_gopt_total']['median']:.4f} top50_gdens_covers_gopt={pw['top50_gdens_covers_gopt_total']['median']:.4f}")

# Topology events
for w in ["2000","5000","10000","14000"]:
    with open(f"results/reference_v1/r0.2/window_{w}/topology_events.json") as f:
        te = json.load(f)
    print(f"\nWindow {w} topology events: {len(te['events'])}")
    for e in te["events"]:
        print(f"  iter {e['iteration']}: N {e['N_before']}→{e['N_after']} clone={e['cloned']} split={e['split']} pruned={e['pruned_total']} ids {e['ids_before']}→{e['ids_after']} next_id={e['next_gaussian_id']}")
