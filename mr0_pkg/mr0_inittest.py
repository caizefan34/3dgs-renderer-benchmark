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
    mem = fits[gi]["K_fits"]["16"]["members"]
    idx = torch.tensor(mem, device="cuda")
    g = M.leaves_from_model(m, idx)
    T = M.tangents(g)
    fit_cams = [ds.get_camera(ci) for ci in gj["fit_pool"][:8]]
    Rg = M.response_exact(g, fit_cams, sh, T).detach()
    fit, info = M.fit_macro(g, fit_cams, sh, T, Rg, n_steps=8, eps=1e-3)
    red, gh, mh = CC._intersect_reduction(m, ds, fit, mem, isect_cams, sh, "cuda")
    print(f"g{gi}: NEW resp_cos={info['response_cos']:.3f} rgb_rmse={info['rgb_rmse']:.5f} "
          f"alpha_rmse={info['alpha_rmse']:.5f} | g_pairs={gh:.0f} m_pairs={mh:.0f} reduction={red:.4f}", flush=True)
print("INITTEST_DONE")
