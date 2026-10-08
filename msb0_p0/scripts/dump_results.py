import json
for scene in ["room","garden","bicycle"]:
    try:
        d = json.load(open(f"/root/msb0_p0/out/msb0_replay_{scene}.json"))
    except FileNotFoundError:
        print(scene, "MISSING"); continue
    print("=====", scene, "n_buckets", d["n_buckets"], "tile", d["chosen_tile"], "pos", d["tile_positions_replayed"])
    for mode in ["M0","M1","M2","M3","M0_vs_autograd"]:
        if mode not in d["fidelity"]: continue
        f = d["fidelity"][mode]
        line = "  " + mode + " "
        for grp in ["dL_dcolor","dL_dalpha","dL_dconic","dL_dmean2d","dL_dopacity"]:
            if grp in f:
                line += f"{grp}: cos={f[grp]['cos']:.6f} rel={f[grp]['rel_l2']:.2e} | "
        print(line)
    print("  gate4_M0", d["gate4_pass_M0"])
