#!/usr/bin/env python3
"""Diagnose backward nondeterminism: with a FIXED captured state, run backward n
times. Copy grads to CPU fp64 immediately. Report per-group relative spread.
Also test: does calling the kernel DIRECTLY (no autograd) vary?
Usage: dpc_diag_det.py <scene> <iter> <cam> <n>
"""
import sys, importlib.util, hashlib
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

scene = sys.argv[1]; it = int(sys.argv[2]); cam_i = int(sys.argv[3]); n = int(sys.argv[4])
torch.cuda.set_device(0); torch.manual_seed(0)
ckpt = C.load_checkpoint(scene, it); model = C.build_model(ckpt); C.scene_reset()
ds = C.load_scene_dataset(scene); cam, gt = ds.get_item(cam_i)
params = [getattr(model, nm) for nm in C.PARAM_NAMES]
GRAD_ORDER = C.PARAM_NAMES
fused = C.make_fused_loss()

img, _, _ = C.render_view(model, cam, model.active_sh_degree)
L = fused(img, gt)

exp.higs_rasterize_backward = dpc.higs_rasterize_backward
snaps = []
for r in range(n):
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    cpu = [None if gr is None else gr.detach().to("cpu", torch.float64).numpy() for gr in grads]
    snaps.append(cpu)

print("=== DPC per-group spread across " + str(n) + " runs (fixed captured state) ===", flush=True)
for gi, nm in enumerate(GRAD_ORDER):
    a0 = snaps[0][gi]
    if a0 is None:
        print(nm + ": None", flush=True); continue
    norms = [np.linalg.norm(s[gi].astype(np.float64).ravel()) for s in snaps]
    # max pairwise relL2 vs first
    rels = [np.linalg.norm(s[gi].astype(np.float64).ravel() - a0.ravel()) / (np.linalg.norm(a0.ravel())+1e-30) for s in snaps]
    print(nm + " max_relL2_vs_first=" + format(float(max(rels)),".3e") + " norm=" + format(float(norms[0]),".4e"), flush=True)

# also: hash the raw cuda grad tensor bytes (before cpu copy) to see if copy is the issue
exp.higs_rasterize_backward = dpc.higs_rasterize_backward
hs = []
for r in range(n):
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    h = hashlib.sha256()
    for gr in grads:
        h.update(b"NULL" if gr is None else gr.detach().to("cpu",torch.float32).numpy().tobytes())
    hs.append(h.hexdigest())
print("raw cuda-grad hashes unique=" + str(len(set(hs))) + "/" + str(n), flush=True)
print("[diag] DONE", flush=True)
