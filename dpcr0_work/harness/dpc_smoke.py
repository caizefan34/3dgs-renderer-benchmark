#!/usr/bin/env python3
"""DPC0 smoke test: verify DPC backward loads under gsplat env (py3.10/torch2.9.1)
and produces gradients matching C0 on a single camera.
Usage: dpc_smoke.py <scene> <iter> <cam>
"""
import sys, os, importlib.util, datetime
sys.path.insert(0, "/root/epic05_n1")
import numpy as np
import torch
import n1_core as C

DPC_SO = ("/root/.cache/torch_extensions/py310_cu128/"
          "dpc0_gaussian_render_inference_scene_cuda/"
          "dpc0_gaussian_render_inference_scene_cuda.so")

info = C.bootstrap_c0()
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
spec = importlib.util.spec_from_file_location("dpc0_gaussian_render_inference_scene_cuda", DPC_SO)
dpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpc)
assert hasattr(dpc, "higs_rasterize_backward"), "DPC .so missing higs_rasterize_backward"
orig_bwd = exp.higs_rasterize_backward
print("[smoke] DPC .so loaded:", DPC_SO, flush=True)
print("[smoke] C0 bwd module:", getattr(orig_bwd, "__module__", orig_bwd), flush=True)

scene = sys.argv[1] if len(sys.argv) > 1 else "room"
it    = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
cam_i = int(sys.argv[3]) if len(sys.argv) > 3 else 5

torch.cuda.set_device(0)
torch.manual_seed(0)

ckpt = C.load_checkpoint(scene, it)
model = C.build_model(ckpt)
C.scene_reset()
ds = C.load_scene_dataset(scene)
cam, gt = ds.get_item(cam_i)

img, _, _ = C.render_view(model, cam, model.active_sh_degree)
b = img.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
gg = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
fused = C.make_fused_loss()
L = fused(img, gt)
params = [getattr(model, n) for n in C.PARAM_NAMES]
GRAD_ORDER = C.PARAM_NAMES

def run(tag):
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    out = []
    for name, gr in zip(GRAD_ORDER, grads):
        t = None if gr is None else gr.detach().to("cpu", torch.float32).numpy()
        out.append(t)
    finite = all(bool(np.all(np.isfinite(a))) for a in out if a is not None)
    print("[smoke] " + tag + " finite=" + str(finite), flush=True)
    return out

def cmp(gA, gB):
    cos_all, rl2_all = [], []
    for name, a, bb in zip(GRAD_ORDER, gA, gB):
        if a is None or bb is None:
            print("  " + name + ": None"); continue
        af = a.astype(np.float64).ravel(); bf = bb.astype(np.float64).ravel()
        na = np.linalg.norm(af); nb = np.linalg.norm(bf)
        cos = float(np.dot(af, bf) / (na * nb + 1e-30))
        rl2 = float(np.linalg.norm(af - bf) / (na + 1e-30))
        cos_all.append(cos); rl2_all.append(rl2)
        print("  " + name + " cos=" + format(cos,".8f") + " relL2=" + format(rl2,".3e") + " |a|=" + format(na,".4e") + " |b|=" + format(nb,".4e"), flush=True)
    print("  MEAN cos=" + format(float(np.mean(cos_all)),".8f") + " relL2=" + format(float(np.mean(rl2_all)),".3e"), flush=True)

torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
model.optimizer.zero_grad(set_to_none=True)

exp.higs_rasterize_backward = dpc.higs_rasterize_backward
g_dpc = run("DPC")
exp.higs_rasterize_backward = orig_bwd
g_c0 = run("C0")

print("[smoke] === DPC vs C0 ===", flush=True)
cmp(g_dpc, g_c0)
print("[smoke] DONE", flush=True)
