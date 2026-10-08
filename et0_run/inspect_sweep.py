import json
d = json.load(open("/root/epic05_moonshot/et0/full_room10000/ET0_SWEEP_room_10000.json"))
for p in d["probes"][:3]:
    print(f"=== cam {p['cam']} L0={p['L_src_theta0']:.5f} gd={p['gd']:.4e} ===")
    print(f"{'a':>5} {'L_src':>9} {'real':>9} {'E':>8} {'sigF':>5} {'C_tile':>8} {'C_ord':>8} {'C':>8} {'H':>11}")
    for r in p["rows"]:
        print(f"{r['alpha']:>5} {r['L_src']:>9.5f} {r['real']:>9.5f} {r['E']:>8.4f} {str(r['sign_failure']):>5} {r['C_tile']:>8.4f} {r['C_order']:>8.4f} {r['C']:>8.4f} {r['H']:>11.4e}")
    print(f"labels(break)={[1 if (r['E']>=0.5 or r['sign_failure']) else 0 for r in p['rows']]}")
