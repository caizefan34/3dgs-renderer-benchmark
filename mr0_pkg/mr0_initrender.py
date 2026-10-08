import sys, json, torch
sys.path.insert(0, "/root/mr0_pkg")
import mr0_common as M
gj = json.load(open("/root/mr0_out_test/room@10000_groups.json"))
ds = M.C.load_dataset("room", device="cuda")
m, _, ident, _, _ = M.C.load_model("room", 10000, device="cuda")
sh = gj["sh"]
fits = {}
for rec in json.load(open("/root/mr0_out_test/room@10000_fits_0_8.json")):
    fits[rec["gid"]] = rec
for gi in (0, 1, 2):
    mem = fits[gi]["K_fits"]["16"]["members"]
    idx = torch.tensor(mem, device="cuda")
    g = M.leaves_from_model(m, idx)
    mp = M.macro_moment_init(g)
    with torch.no_grad():
        me, qu, sc, op, co = M.activated(mp)
        print(f"g{gi} INIT: scale={sc.flatten().tolist()} opac={float(op.flatten()[0]):.4f} "
              f"mean={me.flatten().tolist()}", flush=True)
    cam = ds.get_camera(gj["fit_pool"][0])
    fr, al = M.isolated_group_render(me, qu, sc, op, co, [cam], sh)
    print(f"   g{gi} INIT render: alpha_sum={float(al.sum()):.3f} alpha_px={int((al>0).sum())} "
          f"rgb_sum={float(fr.sum()):.4f}", flush=True)
print("INITRENDER_DONE")
