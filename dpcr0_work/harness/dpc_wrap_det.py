#!/usr/bin/env python3
"""Definitive DPC determinism diagnostic: wrap higs_rasterize_backward to hash
inputs (pre-call) and outputs (post-call, incl in-place grad_ buffers) per call.
Render ONCE (fixed state), call backward n times.
Usage: dpc_wrap_det.py <scene> <iter> <cam> <n>
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

calls = []
def wrapped(**kw):
    h_in = hashlib.sha256()
    for k in ("means2d","conics","colors_eval","opacities","tile_offsets","flatten_ids",
              "render_alphas","last_ids","v_render_colors","v_render_alphas","viewmats","Ks","visible_ids"):
        h_in.update(k.encode()); h_in.update(th(kw.get(k)))
    out = dpc.higs_rasterize_backward(**kw)
    h_out = hashlib.sha256()
    for t in out:
        h_out.update(th(t))
    for k in ("grad_means","grad_quats","grad_scales","grad_opacities","grad_colors"):
        h_out.update(th(kw.get(k)))
    calls.append((h_in.hexdigest()[:12], h_out.hexdigest()[:12]))
    return out

exp.higs_rasterize_backward = wrapped

img, _, _ = C.render_view(model, cam, model.active_sh_degree)
L = fused(img, gt)
for r in range(n):
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for nm, gr in zip(GRAD_ORDER, grads):
        getattr(model, nm).grad = None if gr is None else gr.detach().clone()

print("=== DPC kernel call input/output hashes (fixed captured state) ===", flush=True)
for i,(hi,ho) in enumerate(calls):
    print("  call " + str(i) + " in=" + hi + " out=" + ho, flush=True)
uniq_in = set(x[0] for x in calls); uniq_out = set(x[1] for x in calls)
print("unique inputs=" + str(len(uniq_in)) + "/" + str(n) + " unique outputs=" + str(len(uniq_out)) + "/" + str(n), flush=True)
print("[wrap] DONE", flush=True)
