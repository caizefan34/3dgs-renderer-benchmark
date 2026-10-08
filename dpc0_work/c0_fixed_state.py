#!/usr/bin/env python3
"""Confirm C0 backward determinism with FIXED captured state (render once,
backward n times). Also test: does re-rendering the SAME view give identical
captured state? (forward determinism check via render_alphas/last_ids hash).
Usage: c0_fixed_state.py <scene> <iter> <cam> <n>
"""
import sys, importlib.util, hashlib
sys.path.insert(0, "/root/epic05_n1")
import torch
import n1_core as C

DPC_SO = ("/root/.cache/torch_extensions/py310_cu128/"
          "dpc0_gaussian_render_inference_scene_cuda/"
          "dpc0_gaussian_render_inference_scene_cuda.so")
C.bootstrap_c0()
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
spec = importlib.util.spec_from_file_location("dpc0_gaussian_render_inference_scene_cuda", DPC_SO)
dpc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dpc)
orig_bwd = exp.higs_rasterize_backward

scene = sys.argv[1]; it = int(sys.argv[2]); cam_i = int(sys.argv[3]); n = int(sys.argv[4])
torch.cuda.set_device(0); torch.manual_seed(0)
ckpt = C.load_checkpoint(scene, it); model = C.build_model(ckpt); C.scene_reset()
ds = C.load_scene_dataset(scene); cam, gt = ds.get_item(cam_i)
params = [getattr(model, nm) for nm in C.PARAM_NAMES]
GRAD_ORDER = C.PARAM_NAMES
fused = C.make_fused_loss()

def th(t):
    if t is None: return b"NULL"
    if torch.is_tensor(t): return t.detach().to("cpu",torch.float32).numpy().tobytes()
    return str(t).encode()

IMMUT_IN = ("means2d","conics","colors_eval","opacities","tile_offsets","flatten_ids",
            "render_alphas","last_ids","viewmats","Ks","visible_ids")

# ---- forward determinism: render the SAME view n times, hash captured state via
#      a one-off backward wrapper (we only need the inputs) ----
calls = []
def wrapped(**kw):
    h_in = hashlib.sha256()
    for k in IMMUT_IN:
        h_in.update(k.encode()); h_in.update(th(kw.get(k)))
    out = dpc.higs_rasterize_backward(**kw)
    h_out = hashlib.sha256(); h_out.update(th(out[-1]))
    for k in ("grad_means","grad_quats","grad_scales","grad_opacities","grad_colors"):
        h_out.update(th(kw.get(k)))
    calls.append((h_in.hexdigest(), h_out.hexdigest()))
    return out
exp.higs_rasterize_backward = wrapped

# forward determinism: render n times, each followed by 1 backward (to capture state)
print("=== forward determinism (render same view n times, hash captured state) ===", flush=True)
fwd_hashes = []
for r in range(n):
    img, _, _ = C.render_view(model, cam, model.active_sh_degree)
    L = fused(img, gt)
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for nm, gr in zip(GRAD_ORDER, grads):
        getattr(model, nm).grad = None if gr is None else gr.detach().clone()
    fwd_hashes.append(calls[-1][0])
print("  unique forward-state hashes=" + str(len(set(fwd_hashes))) + "/" + str(n), flush=True)

# backward determinism: render ONCE, backward n times
img, _, _ = C.render_view(model, cam, model.active_sh_degree)
L = fused(img, gt)
calls.clear()
for r in range(n):
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for nm, gr in zip(GRAD_ORDER, grads):
        getattr(model, nm).grad = None if gr is None else gr.detach().clone()
print("=== DPC backward determinism (fixed state, n=" + str(n) + ") ===", flush=True)
print("  unique out=" + str(len(set(x[1] for x in calls))) + "/" + str(n), flush=True)

# C0 backward determinism: render ONCE, backward n times
exp.higs_rasterize_backward = orig_bwd
img, _, _ = C.render_view(model, cam, model.active_sh_degree)
L = fused(img, gt)
c0_hashes = []
for r in range(n):
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    h = hashlib.sha256()
    for gr in grads:
        h.update(th(gr))
    c0_hashes.append(h.hexdigest())
    for nm, gr in zip(GRAD_ORDER, grads):
        getattr(model, nm).grad = None if gr is None else gr.detach().clone()
print("=== C0 backward determinism (fixed state, n=" + str(n) + ") ===", flush=True)
print("  unique out=" + str(len(set(c0_hashes))) + "/" + str(n), flush=True)
print("[c0_fixed] DONE", flush=True)
