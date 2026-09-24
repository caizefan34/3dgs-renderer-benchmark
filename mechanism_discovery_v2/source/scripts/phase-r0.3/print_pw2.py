import json
with open("results/reference_v1/r0.3/visible_gradient_mass_coverage.json") as f:
    vc = json.load(f)
print("Top-level keys:", list(vc.keys())[:5])
print("per_window keys:", list(vc.get("per_window",{}).keys())[:5])
print("overall keys:", list(vc.get("overall",{}).keys())[:5])
# Print opacity overall K=50
if "opacity" in vc.get("overall",{}):
    o = vc["overall"]["opacity"]
    print(f"\nOpacity overall K=50: {o.get('k50',{})}")
    print(f"Opacity overall oracle_k50: {o.get('oracle_k50',{})}")
    print(f"Opacity overall random_k50: {o.get('random_k50',{})}")

# Print all signals overall K=50
for sig in ["tiles_per_gauss", "projected_radius", "opacity"]:
    if sig in vc.get("overall",{}):
        o = vc["overall"][sig]
        k50 = o.get("k50", {})
        or50 = o.get("oracle_k50", {})
        r50 = o.get("random_k50", {})
        print(f"\n{sig} K=50: coverage={k50.get('median',0):.4f} oracle={or50.get('median',0):.4f} random={r50.get('median',0):.4f}")
        k32 = o.get("k32", {})
        or32 = o.get("oracle_k32", {})
        r32 = o.get("random_k32", {})
        print(f"{sig} K=32: coverage={k32.get('median',0):.4f} oracle={or32.get('median',0):.4f} random={r32.get('median',0):.4f}")
        k10 = o.get("k10", {})
        or10 = o.get("oracle_k10", {})
        r10 = o.get("random_k10", {})
        print(f"{sig} K=10: coverage={k10.get('median',0):.4f} oracle={or10.get('median',0):.4f} random={r10.get('median',0):.4f}")
