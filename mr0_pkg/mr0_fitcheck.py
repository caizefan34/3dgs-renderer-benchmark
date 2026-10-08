import sys, json, torch, numpy as np
sys.path.insert(0, "/root/mr0_pkg")
import mr0_common as M
import mr0_cell as CC
gj = json.load(open("/root/mr0_out_test2/room@10000_groups.json"))
ds = M.C.load_dataset("room", device="cuda")
m, _, ident, _, _ = M.C.load_model("room", 10000, device="cuda")
sh = gj["sh"]
fits = {}
for rec in json.load(open("/root/mr0_out_test2/room@10000_fits_0_8.json")):
    fits[rec["gid"]] = rec
isect_cams = [ds.get_camera(ci) for ci in gj["isect_cams"][:4]]
print("gid  fit_cams[:4]                          cos16 cos8 cos4  rmse16")
for gi in range(8):
    fc = fits[gi]["K_fits"]
    raw16 = fc["16"]["raw"]; mem16 = fc["16"]["members"]
    red, gh, mh = CC._intersect_reduction(m, ds, raw16, mem16, isect_cams, sh, "cuda")
    print(f"{gi:3d}  {str(fits[gi].get('fit_cams', fc['16'].get('members'))[:4]):34s} "
          f"{fc['16']['response_cos']:.3f} {fc['8']['response_cos']:.3f} {fc['4']['response_cos']:.3f}  "
          f"{fc['16']['rgb_rmse']:.4f}  red={red:.3f}", flush=True)
print("FITCHECK_DONE")
