#!/usr/bin/env python3
"""DPC0 cross-A100 determinism: one DPC backward on a given GPU for cell source 0.
Run as a fresh process per GPU to get a clean CUDA context. Prints a JSON line.
Usage: dpc0_cross.py <gpu> <scene> <iter>
"""
import sys, os, json, hashlib, importlib.util
import torch
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C

DPC_SO = ("/root/.cache/torch_extensions/py310_cu128/"
          "dpc0_gaussian_render_inference_scene_cuda/"
          "dpc0_gaussian_render_inference_scene_cuda.so")

gpu = int(sys.argv[1]); scene = sys.argv[2]; it = int(sys.argv[3])
C.bootstrap_c0()
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
_spec = importlib.util.spec_from_file_location("dpc0_gaussian_render_inference_scene_cuda", DPC_SO)
dpc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dpc)
exp.higs_rasterize_backward = dpc.higs_rasterize_backward

torch.cuda.set_device(gpu); torch.manual_seed(0)
ckpt = C.load_checkpoint(scene, it)
model = C.build_model(ckpt)
C.scene_reset()
ds = C.load_scene_dataset(scene)
n = len(ds._valid_indices)
cam_i = int(round(n * 0.5 / 8))
cam, gt = ds.get_item(cam_i)
img, _, _ = C.render_view(model, cam, model.active_sh_degree)
fused = C.make_fused_loss()
L = fused(img, gt)
params = [getattr(model, nm) for nm in C.PARAM_NAMES]
grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
h = hashlib.sha256()
for gr in grads:
    h.update(b"NULL" if gr is None else gr.detach().to("cpu", torch.float32).numpy().tobytes())
print(json.dumps({"gpu": gpu, "scene": scene, "iter": it, "cam": cam_i, "hash": h.hexdigest()}), flush=True)
