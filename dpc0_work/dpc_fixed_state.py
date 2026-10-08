#!/usr/bin/env python3
"""Render ONCE, call backward n times on the SAME graph. Hash FULL captured state
(immutable inputs only, read BEFORE the call) + immutable outputs (v_means2d_abs)
to isolate backward determinism from forward non-determinism.
Usage: dpc_fixed_state.py <scene> <iter> <cam> <n>
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

IMMUT_IN = ("means2d","conics","colors_eval","opacities","tile_offsets","flatten_ids",
            "render_alphas","last_ids","viewmats","Ks","visible_ids")

def th(t):
    if t is None: return b"NULL"
    if torch.is_tensor(t): return t.detach().to("cpu",torch.float32).numpy().tobytes()
    return str(t).encode()

calls = []
def wrapped(**kw):
    h_in = hashlib.sha256()
    for k in IMMUT_IN:
        h_in.update(k.encode()); h_in.update(th(kw.get(k)))
    out = dpc.higs_rasterize_backward(**kw)
    # immutable output: v_means2d_abs (last element); grad buffers are in-place-mutated
    h_out = hashlib.sha256(); h_out.update(th(out[-1]))
    # also hash the master grad buffers AFTER the call (these are the real outputs)
    for k in ("grad_means","grad_quats","grad_scales","grad_opacities","grad_colors"):
        h_out.update(th(kw.get(k)))
    calls.append((h_in.hexdigest(), h_out.hexdigest()))
    return out

exp.higs_rasterize_backward = wrapped
for r in range(n):
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for nm, gr in zip(GRAD_ORDER, grads):
        getattr(model, nm).grad = None if gr is None else gr.detach().clone()

print("=== FIXED captured state: immutable-input / (absgrad+master-grads) hashes ===", flush=True)
for i,(hi,ho) in enumerate(calls):
    print("  call " + str(i) + " in=" + hi[:16] + " out=" + ho[:16], flush=True)
print("unique in=" + str(len(set(x[0] for x in calls))) + "/" + str(n) +
      " unique out=" + str(len(set(x[1] for x in calls))) + "/" + str(n), flush=True)
print("[fixed_state] DONE", flush=True)
