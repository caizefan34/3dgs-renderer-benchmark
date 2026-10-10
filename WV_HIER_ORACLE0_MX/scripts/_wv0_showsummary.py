import torch, json
s = torch.load("/dev/shm/wv0_out/test_state.pt", weights_only=False)
print("state", s["state_id"], "N", s["N"], "cams", s["n_cams"], "hw", s["cam_hw"])
print("work vis", s["work"]["visible_gaussians"])
for G in ["16","8","4","1"]:
    w=s["work"]["per_G"][G]
    print(f"  G{G}: dense_isects={w['dense_isects']:,} n_tiles={w['n_tiles']:,}")
print("=== conditions (means over masks & cams) ===")
for key,c in s["conditions"].items():
    print(f"\n{key}: sel_frac={c['sel_frac']:.3f} sel_px={c['sel_px_mean']:.0f}")
    for gname,gv in c["groups"].items():
        print(f"  {gname:9s} nmse_mean={gv['nmse']['mean']:.4f} cos_mean={gv['cos']['mean']:.4f} "
              f"relL2_mean={gv['rel_l2']['mean']:.4f} var={gv['variance']:.3e} bias_relL2={gv['bias_rel_l2']:.4f}")
