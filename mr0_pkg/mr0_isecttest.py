import sys, json, torch
sys.path.insert(0, "/root/mr0_pkg")
import mr0_common as M
import mr0_cell as CC
gj = json.load(open("/root/mr0_out_test/room@10000_groups.json"))
ds = M.C.load_dataset("room", device="cuda")
m, _, ident, _, _ = M.C.load_model("room", 10000, device="cuda")
sh = gj["sh"]
fits = {}
for rec in json.load(open("/root/mr0_out_test/room@10000_fits_0_8.json")):
    fits[rec["gid"]] = rec
isect_cams = [ds.get_camera(ci) for ci in gj["isect_cams"][:4]]
for gi in (0, 1, 2):
    gid = gi
    if gid not in fits or "16" not in fits[gid]["K_fits"]:
        print("no fit", gi); continue
    raw = fits[gid]["K_fits"]["16"]["raw"]
    mem = fits[gid]["K_fits"]["16"]["members"]
    red, gh, mh = CC._intersect_reduction(m, ds, raw, mem, isect_cams, sh, "cuda")
    print(f"g{gid}: K=16 g_pairs={gh:.0f} m_pairs={mh:.0f} reduction={red:.4f}", flush=True)
print("ISECT_TEST_DONE")
