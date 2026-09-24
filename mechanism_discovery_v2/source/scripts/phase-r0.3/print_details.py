import json

# Final decision
with open("results/reference_v1/r0.3/final_decision.json") as f:
    d = json.load(f)
print("=== Final Decision ===")
print(f"Decision: {d['decision']}")
print(f"Reason: {d['reason']}")
print(f"Best signal: {d['best_signal']}")
print(f"E2E saving estimate K=50: {d['e2e_saving_estimate_k50']:.1f}%")

# Masked backward scaling detail
with open("results/reference_v1/r0.3/masked_backward_scaling.json") as f:
    mbs = json.load(f)
print(f"\n=== Masked Backward Scaling (N={mbs['N_gaussians']}) ===")
for key, val in mbs["retention_results"].items():
    if "f_k" in val:
        print(f"  {key}: backward={val['backward_total_ms']['median']:.2f}ms f(K)={val['f_k']:.3f} saving={val['backward_saving_pct']:.1f}%")
print(f"  no_mask_reference: backward={mbs['retention_results']['no_mask_reference']['backward_total_ms']['median']:.2f}ms")

# Per-window visible fraction
with open("results/reference_v1/r0.3/final_decision.json") as f:
    d = json.load(f)
vf = d["visible_fraction_overall"]
print(f"\n=== Visible Fraction ===")
print(f"  Overall: mean={vf['mean']:.4f} median={vf['median']:.4f}")

# Per-parameter correlations for opacity
with open("results/reference_v1/r0.3/visible_conditional_correlations.json") as f:
    corr = json.load(f)
print(f"\n=== Opacity vs per-parameter G_opt (Spearman median) ===")
for gopt_name in ["total_norm", "xyz_norm", "opacity_abs", "scale_norm", "rot_norm", "shs_norm"]:
    key = f"opacity_vs_{gopt_name}"
    if key in corr["overall"]:
        s = corr["overall"][key]["spearman"]["median"]
        p = corr["overall"][key]["pearson"]["median"]
        print(f"  opacity vs {gopt_name}: Pearson={p:.4f} Spearman={s:.4f}")

# Per-window opacity coverage K=50%
with open("results/reference_v1/r0.3/visible_gradient_mass_coverage.json") as f:
    vc = json.load(f)
print(f"\n=== Per-window Opacity Coverage K=50% ===")
for w in ["2000","5000","10000","14000"]:
    if w in vc["per_window"].get("opacity", {}):
        pw = vc["per_window"]["opacity"][w]
        print(f"  window {w}: coverage={pw['k50']['median']:.4f} oracle={pw['oracle_k50']['median']:.4f} random={pw['random_k50']['median']:.4f}")
