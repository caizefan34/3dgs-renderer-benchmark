import json
with open("results/reference_v1/r0.3/visible_gradient_mass_coverage.json") as f:
    vc = json.load(f)
print("Per-window opacity K=50%:")
for w in ["2000","5000","10000","14000"]:
    if w in vc.get("per_window",{}).get("opacity",{}):
        pw = vc["per_window"]["opacity"][w]
        print(f"  window {w}: coverage={pw['k50']['median']:.4f} oracle={pw['oracle_k50']['median']:.4f} random={pw['random_k50']['median']:.4f}")
print("\nPer-window tiles_per_gauss K=50%:")
for w in ["2000","5000","10000","14000"]:
    if w in vc.get("per_window",{}).get("tiles_per_gauss",{}):
        pw = vc["per_window"]["tiles_per_gauss"][w]
        print(f"  window {w}: coverage={pw['k50']['median']:.4f}")
print("\nPer-window projected_radius K=50%:")
for w in ["2000","5000","10000","14000"]:
    if w in vc.get("per_window",{}).get("projected_radius",{}):
        pw = vc["per_window"]["projected_radius"][w]
        print(f"  window {w}: coverage={pw['k50']['median']:.4f}")
