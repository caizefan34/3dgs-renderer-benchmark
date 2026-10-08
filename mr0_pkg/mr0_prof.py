import sys, time, torch
sys.path.insert(0, "/root/mr0_pkg")
import mr0_common as M
import mr0_cell as CC
gj = json.load(open("/root/mr0_out_test2/room@10000_groups.json")) if False else None
import json
gj = json.load(open("/root/mr0_out_test2/room@10000_groups.json"))
ds = M.C.load_dataset("room", device="cuda")
m, _, ident, _, _ = M.C.load_model("room", 10000, device="cuda")
sh = gj["sh"]
mem = gj["groups"][1]["members"]
idx = torch.tensor(mem, device="cuda")
g = M.leaves_from_model(m, idx)
fc = gj["groups"][1]["fit_cams"]
cams = [ds.get_camera(ci) for ci in fc]
T = M.tangents(g)
def sync(): torch.cuda.synchronize()
# time isolated render
sync(); t0=time.perf_counter()
for _ in range(5):
    me,qu,sc,op,co = M.activated(g)
    fr,al = M.isolated_group_render(me,qu,sc,op,co,cams,sh)
sync(); print("isolated_render(8cam):", (time.perf_counter()-t0)/5*1000, "ms")
# time response_exact (12 autograd perturbations)
sync(); t0=time.perf_counter()
Rg = M.response_exact(g, cams, sh, T)
sync(); print("response_exact:", (time.perf_counter()-t0)*1000, "ms")
# time one _render_summary
sync(); t0=time.perf_counter()
for _ in range(5):
    s = M._render_summary({k: g[k] for k in g}, cams, sh)
sync(); print("_render_summary(8cam):", (time.perf_counter()-t0)/5*1000, "ms")
# full fit_macro timing
sync(); t0=time.perf_counter()
fit, info = M.fit_macro(g, cams, sh, T, Rg, n_steps=8, eps=1e-3)
sync(); print("fit_macro(8steps):", (time.perf_counter()-t0), "s cos=", info["response_cos"])
print("PROFILE_DONE")
