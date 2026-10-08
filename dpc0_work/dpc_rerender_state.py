#!/usr/bin/env python3
"""Resolve the re-rendering discrepancy: re-render the SAME view n times; in each
pass hash (a) captured state at backward ENTRY and (b) param grads at exit.
If state drifts -> forward/cache artifact (out of DPC backward scope).
If state stable but grads drift -> contradicts dpc_localize (bug).
Usage: dpc_rerender_state.py <scene> <iter> <cam> <n>
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

def th(t):
    if t is None: return b"NULL"
    if torch.is_tensor(t): return t.detach().to("cpu",torch.float32).numpy().tobytes()
    return str(t).encode()

STATE_KEYS = ("means2d","conics","colors_eval","opacities","tile_offsets","flatten_ids",
              "render_alphas","last_ids","visible_ids")
state_hashes = []
param_hashes = []
def wrapped(**kw):
    h = hashlib.sha256()
    for k in STATE_KEYS:
        h.update(k.encode()); h.update(th(kw.get(k)))
    state_hashes.append(h.hexdigest())
    out = dpc.higs_rasterize_backward(**kw)
    return out
exp.higs_rasterize_backward = wrapped

for r in range(n):
    img, _, _ = C.render_view(model, cam, model.active_sh_degree)
    L = fused(img, gt)
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    h = hashlib.sha256()
    for gr in grads:
        h.update(th(gr))
    param_hashes.append(h.hexdigest())
    for nm, gr in zip(GRAD_ORDER, grads):
        getattr(model, nm).grad = None if gr is None else gr.detach().clone()

print("=== re-render same view " + str(n) + "x (state at entry / param grads at exit) ===", flush=True)
for i in range(n):
    print("  pass " + str(i) + " state=" + state_hashes[i][:16] + " param=" + param_hashes[i][:16], flush=True)
print("unique state=" + str(len(set(state_hashes))) + "/" + str(n) +
      "  unique param=" + str(len(set(param_hashes))) + "/" + str(n), flush=True)
print("[rerender_state] DONE", flush=True)
