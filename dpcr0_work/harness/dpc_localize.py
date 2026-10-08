#!/usr/bin/env python3
"""Decisive: render ONCE, backward n times. In the SAME run, hash BOTH the
kernel master-grad buffers (inside wrapper) AND the autograd param grads
(after torch.autograd.grad). Pinpoints where non-determinism enters.
Usage: dpc_localize.py <scene> <iter> <cam> <n>
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

scene = sys.argv[1]; it = int(sys.argv[2]); cam_i = int(sys.argv[3]); n = int(sys.argv[4])
torch.cuda.set_device(0); torch.manual_seed(0)
ckpt = C.load_checkpoint(scene, it); model = C.build_model(ckpt); C.scene_reset()
ds = C.load_scene_dataset(scene); cam, gt = ds.get_item(cam_i)
params = [getattr(model, nm) for nm in C.PARAM_NAMES]
GRAD_ORDER = C.PARAM_NAMES
fused = C.make_fused_loss()

img, _, _ = C.render_view(model, cam, model.active_sh_degree)
L = fused(img, gt)

def th(t):
    if t is None: return b"NULL"
    if torch.is_tensor(t): return t.detach().to("cpu",torch.float32).numpy().tobytes()
    return str(t).encode()

kernel_hashes = []
param_hashes = []
def wrapped(**kw):
    out = dpc.higs_rasterize_backward(**kw)
    h = hashlib.sha256()
    for k in ("grad_means","grad_quats","grad_scales","grad_opacities","grad_colors"):
        h.update(th(kw.get(k)))
    kernel_hashes.append(h.hexdigest())
    return out
exp.higs_rasterize_backward = wrapped

for r in range(n):
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    h = hashlib.sha256()
    for gr in grads:
        h.update(th(gr))
    param_hashes.append(h.hexdigest())
    for nm, gr in zip(GRAD_ORDER, grads):
        getattr(model, nm).grad = None if gr is None else gr.detach().clone()

print("=== render ONCE, backward n times ===", flush=True)
print("kernel master-grads unique=" + str(len(set(kernel_hashes))) + "/" + str(n), flush=True)
print("autograd param-grads  unique=" + str(len(set(param_hashes))) + "/" + str(n), flush=True)
if len(set(kernel_hashes)) == 1 and len(set(param_hashes)) > 1:
    print(">>> NON-DETERMINISM IS IN AUTOGRAD ASSEMBLY (not the DPC kernel)", flush=True)
elif len(set(kernel_hashes)) > 1:
    print(">>> NON-DETERMINISM IS IN THE DPC KERNEL", flush=True)
else:
    print(">>> DPC FULLY DETERMINISTIC (kernel + autograd)", flush=True)
print("[localize] DONE", flush=True)
