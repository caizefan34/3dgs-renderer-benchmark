#!/usr/bin/env python3
"""DPC0 determinism smoke: end-to-end forward+backward, DPC vs C0, n repeats.
Usage: dpc_det_smoke.py <scene> <iter> <cam> [n]
"""
import sys, os, importlib.util, hashlib
sys.path.insert(0, "/root/epic05_n1")
import numpy as np
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

scene = sys.argv[1]; it = int(sys.argv[2]); cam_i = int(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 5
torch.cuda.set_device(0); torch.manual_seed(0)

ckpt = C.load_checkpoint(scene, it)
model = C.build_model(ckpt)
C.scene_reset()
ds = C.load_scene_dataset(scene)
cam, gt = ds.get_item(cam_i)
params = [getattr(model, nm) for nm in C.PARAM_NAMES]
GRAD_ORDER = C.PARAM_NAMES
fused = C.make_fused_loss()

def hash_grads():
    h = hashlib.sha256()
    for nm in GRAD_ORDER:
        t = getattr(model, nm).grad
        h.update(b"NULL" if t is None else t.detach().to("cpu",torch.float32).numpy().tobytes())
    return h.hexdigest()

def one_pass():
    img, _, _ = C.render_view(model, cam, model.active_sh_degree)
    L = fused(img, gt)
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for nm, gr in zip(GRAD_ORDER, grads):
        getattr(model, nm).grad = None if gr is None else gr.detach().clone()
    return hash_grads()

one_pass()
model.optimizer.zero_grad(set_to_none=True)

for tag, mod in (("DPC", dpc.higs_rasterize_backward), ("C0", orig_bwd)):
    exp.higs_rasterize_backward = mod
    hs = [one_pass() for _ in range(n)]
    u = sorted(set(hs))
    print("[det] " + tag + " n=" + str(n) + " unique_hashes=" + str(len(u)) + " deterministic=" + str(len(u)==1), flush=True)
    print("[det] " + tag + " first=" + hs[0][:16], flush=True)
    model.optimizer.zero_grad(set_to_none=True)
print("[det] DONE", flush=True)
