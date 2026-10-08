#!/usr/bin/env python
"""DPGR0 correctness gate vs DPC0 reference. Usage: dpgr0_correct.py <gpu> <cell>"""
import sys, os, json, datetime
sys.path.insert(0, "/root/epic05_dpgr0"); sys.path.insert(0, "/root/epic05_dpc0"); sys.path.insert(0, "/root/epic05_n1")
import numpy as np, torch, n1_core as C, dpc0_common as U

gpu = int(sys.argv[1]); cell = sys.argv[2]
torch.cuda.set_device(gpu); torch.manual_seed(0)
scene, it = U.cell_scene_iter(cell)
U.bootstrap("c0")
model, ds = U.setup_state(scene, it)
cc = U.cell_cameras(scene)
sources = cc["sources"]
PN = U.PARAM_NAMES

DPGR0_SO = ("/root/.cache/torch_extensions/py310_cu128/"
            "dpgr0_gaussian_render_inference_scene_cuda/"
            "dpgr0_gaussian_render_inference_scene_cuda.so")
import importlib.util
spec = importlib.util.spec_from_file_location("dpgr0_gaussian_render_inference_scene_cuda", DPGR0_SO)
DPGR0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(DPGR0)
DPGR0_BWD = DPGR0.higs_rasterize_backward_dpgr0
DPC_BWD = U.load_dpc().higs_rasterize_backward

exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]

def raw_flat(params):
    out = []
    for p in params:
        g = p.grad
        out.append(np.zeros(p.numel(), np.float64) if g is None
                   else g.detach().to("cpu", torch.float32).numpy().astype(np.float64).ravel())
    return np.concatenate(out)

def cos_relL2(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")
    return cos, float(np.linalg.norm(a - b) / max(nb, 1e-300))

rows = []
for src in sources:
    cam_idx = int(src)
    exp.higs_rasterize_backward = DPC_BWD
    L, params, img, gt, cam = U.render_loss(model, ds, cam_idx)
    gd = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for p, gr in zip(params, gd):
        p.grad = None if gr is None else gr.detach().clone()
    ref = raw_flat(params)
    exp.higs_rasterize_backward = DPGR0_BWD
    gg = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for p, gr in zip(params, gg):
        p.grad = None if gr is None else gr.detach().clone()
    cand = raw_flat(params)
    cos, rl2 = cos_relL2(ref, cand)
    nan_inf = int(np.count_nonzero(~np.isfinite(cand)))
    rows.append({"src": int(src), "dpc_cos": cos, "dpc_rel_l2": rl2, "nan_inf": nan_inf})
    print(f"src={src} cos={cos:.8f} relL2={rl2:.3e} nan_inf={nan_inf}")

cos_all = np.array([r["dpc_cos"] for r in rows])
rl2_all = np.array([r["dpc_rel_l2"] for r in rows])
nan_inf_total = int(sum(r["nan_inf"] for r in rows))
out = {
    "gate": "DPGR0_CORRECTNESS_D1",
    "cell": cell, "gpu": gpu,
    "n_sources": len(rows),
    "dpc_cos_median": float(np.median(cos_all)),
    "dpc_cos_p10": float(np.percentile(cos_all, 10)),
    "dpc_rel_l2_median": float(np.median(rl2_all)),
    "dpc_rel_l2_p90": float(np.percentile(rl2_all, 90)),
    "nan_inf_count": nan_inf_total,
    "rows": rows,
    "pass_d1": bool(np.median(cos_all) >= 0.999999 and np.percentile(cos_all, 10) >= 0.999995
                    and np.median(rl2_all) <= 5e-5 and np.percentile(rl2_all, 90) <= 1e-4
                    and nan_inf_total == 0),
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
}
os.makedirs("/root/epic05_dpgr0/artifacts", exist_ok=True)
with open("/root/epic05_dpgr0/artifacts/DPGR0_CORRECTNESS.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


