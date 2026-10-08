#!/usr/bin/env python3
"""DPC0 determinism/correctness harness: uses the DPC0 backward (monkey-patched
over the C0 extension) and hashes raster-gradient output bytes run-to-run.

Usage: dpc0_run.py <tag> <n_repeat> <gpu> [scene] [iter] [cam] [mode]
mode: hash (default) | once (single run + relL2 vs C0)
"""
import sys, os, hashlib, json, datetime, importlib.util
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C

DPC_SO = ("/root/.cache/torch_extensions/py310_cu128/"
          "dpc0_gaussian_render_inference_scene_cuda/"
          "dpc0_gaussian_render_inference_scene_cuda.so")

info = C.bootstrap_c0()
exp = sys.modules["experimental_gaussian_render_inference_scene_cuda"]
spec = importlib.util.spec_from_file_location("dpc0_gaussian_render_inference_scene_cuda", DPC_SO)
dpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpc)
assert hasattr(dpc, "higs_rasterize_backward")
# Monkey-patch: route the native backward through the DPC0 implementation.
exp.higs_rasterize_backward = dpc.higs_rasterize_backward
import torch

tag      = sys.argv[1] if len(sys.argv) > 1 else "dpc"
n_repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 32
gpu      = int(sys.argv[3]) if len(sys.argv) > 3 else 0
scene    = sys.argv[4] if len(sys.argv) > 4 else "room"
it       = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
cam_idx  = int(sys.argv[6]) if len(sys.argv) > 6 else 5
mode     = sys.argv[7] if len(sys.argv) > 7 else "hash"

torch.cuda.set_device(gpu)
torch.manual_seed(0)

ckpt = C.load_checkpoint(scene, it)
model = C.build_model(ckpt)
C.scene_reset()
ds = C.load_scene_dataset(scene)
cam, gt = ds.get_item(cam_idx)

img, _, _ = C.render_view(model, cam, model.active_sh_degree)
b = img.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
g = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
fused = C.make_fused_loss()
L = fused(img, gt)
params = [getattr(model, n) for n in C.PARAM_NAMES]
GRAD_ORDER = ("_xyz", "_shs", "_scaling", "_rotation", "_opacity")

def grab_grads():
    out = []
    for name in GRAD_ORDER:
        t = getattr(model, name).grad
        out.append(None if t is None else t.detach().to("cpu", torch.float32).numpy())
    return out

def hash_grads():
    h = hashlib.sha256()
    for arr in grab_grads():
        h.update(b"NULL" if arr is None else arr.tobytes())
    return h.hexdigest()

# warmup (also compiles/caches any lazy state)
torch.autograd.grad(L, params, retain_graph=True)
model.optimizer.zero_grad(set_to_none=True)

if mode == "once":
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for p, gr in zip(params, grads):
        p.grad = None if gr is None else gr.detach().clone()
    arrs = grab_grads()
    finite = all(bool(np.all(np.isfinite(a))) for a in arrs if a is not None) if (np:=__import__("numpy")) else True
    print(json.dumps({"tag":tag,"gpu":gpu,"scene":scene,"iter":it,"cam":cam_idx,
        "finite":finite,
        "norms":{n: (float(a.astype('f8').ravel()@0+a.std()) if a is not None else None) for n,a in zip(GRAD_ORDER,arrs)},
        "hash":hash_grads()}, indent=2))
    sys.exit(0)

hashes = []
for r in range(n_repeat):
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for p, gr in zip(params, grads):
        p.grad = None if gr is None else gr.detach().clone()
    hashes.append(hash_grads())

uniq = sorted(set(hashes))
out = {"tag":tag,"gpu":gpu,"scene":scene,"iter":it,"cam":cam_idx,
       "n_repeat":n_repeat,"n_unique_hashes":len(uniq),
       "deterministic":len(uniq)==1,"first_hash":hashes[0],
       "timestamp":datetime.datetime.utcnow().isoformat()+"Z"}
print(json.dumps({k:v for k,v in out.items() if k!="all_hashes"},indent=2))
os.makedirs("/root/epic05_dpc0/runs",exist_ok=True)
with open(f"/root/epic05_dpc0/runs/{tag}_gpu{gpu}_{scene}_{it}_cam{cam_idx}.json","w") as f:
    json.dump(out,f,indent=2)