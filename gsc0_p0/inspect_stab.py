import json
s = json.load(open("/root/gsc0_out/GSC0_SUMMARY.json"))
for scene in ["room","garden","bicycle"]:
    r = s["per_scene"][scene]
    st = r["stability_mature_window"]
    ndc = r["N"]["densify_complete"]; nm = r["N"]["mature"]
    print("===== %s (N_dc=%d, N_mat=%d) =====" % (scene, ndc, nm))
    print("  composite stable_fraction (<=1e-2): %.5f  stable_count=%d" % (st["stable_fraction"], st["stable_count"]))
    for a in ["xyz","scaling","rotation","shs","opacity","composite_max"]:
        d = st.get(a, {})
        print("  %-14s: median=%.3e p90=%.3e p99=%.3e" % (a, d["median"], d["p90"], d["p99"]))
    print("  per-attr stable_fraction:", {k: round(v,4) for k,v in st["per_attr_stable_fraction"].items()})
    print()