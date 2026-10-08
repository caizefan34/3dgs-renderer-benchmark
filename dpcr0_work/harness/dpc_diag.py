import sys, os, hashlib, importlib.util
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
DPC_SO = ("/root/.cache/torch_extensions/py310_cu128/"
          "dpc0_gaussian_render_inference_scene_cuda/"
          "dpc0_gaussian_render_inference_scene_cuda.so")
C.bootstrap_c0()
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
spec = importlib.util.spec_from_file_location("dpc0_gaussian_render_inference_scene_cuda", DPC_SO)
dpc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dpc)
exp.higs_rasterize_backward = dpc.higs_rasterize_backward
import torch
torch.cuda.set_device(0); torch.manual_seed(0)
scene, it, cam_idx = "room", 10000, 5
ckpt = C.load_checkpoint(scene, it); model = C.build_model(ckpt); C.scene_reset()
ds = C.load_scene_dataset(scene); cam, gt = ds.get_item(cam_idx)
img, _, _ = C.render_view(model, cam, model.active_sh_degree)
b = img.unsqueeze(0).permute(0,3,1,2).contiguous(); g = gt.unsqueeze(0).permute(0,3,1,2).contiguous()
fused = C.make_fused_loss(); L = fused(img, gt)
params = [getattr(model, n) for n in C.PARAM_NAMES]
GRAD_ORDER = C.PARAM_NAMES
def arrs():
    return [getattr(model,n).grad.detach().to("cpu",torch.float32).numpy() for n in GRAD_ORDER]
# warmup
torch.autograd.grad(L, params, retain_graph=True); model.optimizer.zero_grad(set_to_none=True)
NREP=4
per = {n:[] for n in GRAD_ORDER}
for r in range(NREP):
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for p,gr in zip(params,grads): p.grad = None if gr is None else gr.detach().clone()
    for n,a in zip(GRAD_ORDER, arrs()):
        per[n].append(hashlib.sha256(a.tobytes()).hexdigest())
print("=== per-param unique hashes over", NREP, "DPC runs ===")
for n in GRAD_ORDER:
    u = sorted(set(per[n]))
    print(f"{n:12s} unique={len(u)}/{NREP}  {'DET' if len(u)==1 else 'NONDET'}")