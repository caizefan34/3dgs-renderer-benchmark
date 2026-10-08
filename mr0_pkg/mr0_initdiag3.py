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
for gi in (0,1,2):
    mem = fits[gi]["K_fits"]["16"]["members"]
    idx = torch.tensor(mem, device="cuda")
    g = M.leaves_from_model(m, idx)
    with torch.no_grad():
        me,qu,sc,op,co = M.activated(g)
        rms = (me - (op[:,None]*me).sum(0)/op.sum()).pow(2).sum(-1).sqrt()
        print(f"g{gi}: K={len(mem)} opac_mean={float(op.mean()):.4f} opac_max={float(op.max()):.4f} "
              f"combined={float(1-(1-op).prod()):.4f} ext_max={float(rms.max()):.4f} "
              f"scale_mean={float(torch.exp(g['scales']).mean()):.5f} "
              f"saved_resp_cos={fits[gi]['K_fits']['16']['response_cos']:.3f} "
              f"saved_rgb_rmse={fits[gi]['K_fits']['16']['rgb_rmse']:.4f}", flush=True)
print("DIAG_DONE")
