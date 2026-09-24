import sys
import torch
from f9_0_loader import load_scene
from gsplat.experimental.render.functional.gaussian_inference import _higs_dynamic_forward

scene, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m, q, s, o, sh, vm, K, w, h, _ = load_scene(scene)
for x in (m, q, s, o, sh):
    x.requires_grad_(True)
torch.manual_seed(seed)
r = _higs_dynamic_forward(
    m, q, s, o, sh, viewmats=vm, Ks=K, width=w, height=h, sh_degree=3,
    backward_mode="higs_native", enable_culling=True, camera_model="pinhole",
    render_mode="RGB", eps2d=.3,
)
p = r["densification_info"]["means2d"]
p.retain_grad()
nf, na = torch.randn_like(r["frame"]), torch.randn_like(r["alpha"])
((r["frame"] * nf).sum() + (r["alpha"] * na).sum()).backward()
torch.save({
    "means": m.grad.cpu(), "quats": q.grad.cpu(), "scales": s.grad.cpu(),
    "opacities": o.grad.cpu(), "sh": sh.grad.cpu(), "means2d": p.grad.cpu(),
    "radii": r["densification_info"]["radii"].cpu(),
    "visible_ids": r["densification_info"]["visible_gaussian_ids"].cpu(),
}, out)
