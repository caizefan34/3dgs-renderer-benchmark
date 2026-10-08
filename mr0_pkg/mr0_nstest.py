import sys, json, torch
sys.path.insert(0, "/root/mr0_pkg")
import mr0_common as M
gj = json.load(open("/root/mr0_out_test2/room@10000_groups.json"))
ds = M.C.load_dataset("room", device="cuda")
m, _, ident, _, _ = M.C.load_model("room", 10000, device="cuda")
sh = gj["sh"]
for gi in (0, 1, 3, 5, 7):
    fc = gj["groups"][gi]["fit_cams"]
    idx = torch.tensor(gj["groups"][gi]["members"], device="cuda")
    g = M.leaves_from_model(m, idx)
    T = M.tangents(g)
    cams = [ds.get_camera(ci) for ci in fc]
    Rg = M.response_exact(g, cams, sh, T).detach()
    row = []
    for ns in (3, 5, 8):
        fit, info = M.fit_macro(g, cams, sh, T, Rg, n_steps=ns, eps=1e-3)
        row.append(f"ns{ns}:cos{info['response_cos']:.3f}")
    print(f"g{gi}: " + "  ".join(row), flush=True)
print("NS_DONE")
