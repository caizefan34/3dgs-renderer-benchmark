#!/usr/bin/env python3
"""DPC0 determinism harness: hash raster-gradient output bytes, run-to-run.

Usage: dpc0_hash_grads.py <tag> <n_repeat> <gpu> [scene] [iter] [cam]
Repeats the identical native backward n_repeat times on one GPU, hashing the
master gradient tensors byte-for-byte each run. Reports unique-hash count.
"""
import sys, os, hashlib, json, datetime
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
C.bootstrap_c0()
import torch

tag      = sys.argv[1] if len(sys.argv) > 1 else "c0"
n_repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 32
gpu      = int(sys.argv[3]) if len(sys.argv) > 3 else 0
scene    = sys.argv[4] if len(sys.argv) > 4 else "room"
it       = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
cam_idx  = int(sys.argv[6]) if len(sys.argv) > 6 else 5

torch.cuda.set_device(gpu)
torch.manual_seed(0)

ckpt = C.load_checkpoint(scene, it)
model = C.build_model(ckpt)
C.scene_reset()
ds = C.load_scene_dataset(scene)
cam, gt = ds.get_item(cam_idx)

# Build the graph once, then re-run grad() on the SAME graph each repeat.
# retain_graph=True keeps the captured native-backward state alive; the
# native backward recomputes from the saved forward state each call.
img, _, _ = C.render_view(model, cam, model.active_sh_degree)
b = img.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
g = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
fused = C.make_fused_loss()
L = fused(img, gt)
params = [getattr(model, n) for n in C.PARAM_NAMES]

GRAD_ORDER = ("_xyz", "_shs", "_scaling", "_rotation", "_opacity")

def hash_grads():
    h = hashlib.sha256()
    # order: xyz, shs, scaling, rotation, opacity (stable)
    for name in GRAD_ORDER:
        t = getattr(model, name).grad
        if t is None:
            h.update(b"NULL"); continue
        h.update(t.detach().to("cpu", torch.float32).numpy().tobytes())
    return h.hexdigest()

# warmup
torch.autograd.grad(L, params, retain_graph=True)
model.optimizer.zero_grad(set_to_none=True)

hashes = []
for r in range(n_repeat):
    grads = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
    for p, gr in zip(params, grads):
        if gr is None:
            p.grad = None
        else:
            p.grad = gr.detach().clone()
    hashes.append(hash_grads())

uniq = sorted(set(hashes))
out = {
    "tag": tag, "gpu": gpu, "scene": scene, "iter": it, "cam": cam_idx,
    "n_repeat": n_repeat, "n_unique_hashes": len(uniq),
    "deterministic": len(uniq) == 1,
    "first_hash": hashes[0], "all_hashes": hashes,
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
}
print(json.dumps({k: v for k, v in out.items() if k != "all_hashes"}, indent=2))
os.makedirs("/root/epic05_dpc0/runs", exist_ok=True)
with open(f"/root/epic05_dpc0/runs/{tag}_gpu{gpu}_{scene}_{it}_cam{cam_idx}.json", "w") as f:
    json.dump(out, f, indent=2)
