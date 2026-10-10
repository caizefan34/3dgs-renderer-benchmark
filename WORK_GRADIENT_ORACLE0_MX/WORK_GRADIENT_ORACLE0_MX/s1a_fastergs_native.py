#!/usr/bin/env python3
"""Decisive render test: use FasterGS's OWN _C.forward (the trainer's rasterizer)
with the bicycle 5K checkpoint + benchmark camera. If this matches GT, the
checkpoint is valid and the gsplat-HiGS mismatch is the renderer, not the data."""
import os, sys, json, math, time, ctypes
import numpy as np
from PIL import Image
import torch
# Preload torch libs for _C ABI
_tlib = os.path.dirname(torch.__file__)
for _lib in ["libc10.so","libtorch.so","libtorch_cpu.so","libtorch_python.so","libtorch_cuda.so","libc10_cuda.so"]:
    _p = os.path.join(_tlib, _lib)
    if os.path.exists(_p):
        try: ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        except Exception: pass

FGB = "/mnt/storage_pool/liaoyuanjun/fused_profile0/faster-gaussian-splatting/FasterGSFusedCudaBackend"
dev = "cuda:6"
CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
SCENE = "/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle"

# Try importing the FasterGS _C backend
sys.path.insert(0, FGB)
try:
    from FasterGSFusedCudaBackend import _C
    print("FasterGS _C imported OK; ops:", [a for a in dir(_C) if not a.startswith('__')][:20])
except Exception as e:
    import traceback; traceback.print_exc()
    print("FasterGS _C import FAILED:", e)
    sys.exit(0)

import torch
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
N = g["gaussians._means"].shape[0]; sh_degree=3
active_bases = (sh_degree+1)**2  # 16

# Raw (unactivated) params as FasterGS expects (it applies activations internally)
means = g["gaussians._means"].to(dev).float()
scales_raw = g["gaussians._scales"].to(dev).float()
rots_raw = g["gaussians._rotations"].to(dev).float()
opac_raw = g["gaussians._opacities"].to(dev).float()
sh0 = g["gaussians._sh_coefficients_0"].to(dev).float()
shrest = g["gaussians._sh_coefficients_rest"].to(dev).float()
print(f"N={N} means{tuple(means.shape)} sh0{tuple(sh0.shape)} shrest{tuple(shrest.shape)}")

# moments (second-order) - check if in checkpoint
moment_keys = [k for k in blob["model_state_dict"] if "moment" in k.lower()]
print("moment keys in ckpt:", moment_keys[:10])
for mk in moment_keys:
    print(f"  {mk}: {tuple(blob['model_state_dict'][mk].shape)}")

# Camera id=0, benchmark convention
cams = json.load(open(f"{SCENE}/eval_cameras.json"))
c = cams[0]; scale=0.25
W,H = int(c["width"]*scale), int(c["height"]*scale)
R = np.asarray(c["rotation"],dtype=np.float64); p = np.asarray(c["position"],dtype=np.float64)
Rw2c = R.T
w2c = np.eye(4); w2c[:3,:3]=Rw2c; w2c[:3,3] = -Rw2c @ p
cam_position = p  # camera center in world
w2c_t = torch.tensor(w2c, dtype=torch.float32, device=dev).unsqueeze(0)
cam_pos_t = torch.tensor(cam_position, dtype=torch.float32, device=dev).unsqueeze(0)
bg = torch.zeros(3, dtype=torch.float32, device=dev)

# Dummy moments (zeros) if not required for forward-only render
def zlike(t): return torch.zeros_like(t)
moments = [zlike(means), zlike(scales_raw), zlike(rots_raw), zlike(opac_raw), zlike(sh0), zlike(shrest)]
densify_info = torch.zeros(N, dtype=torch.float32, device=dev)

try:
    out = _C.forward(
        means, scales_raw, rots_raw, opac_raw, sh0, shrest,
        *moments,
        densify_info,
        (w2c_t, cam_pos_t, bg, active_bases, W, H,
         float(c["fx"])*scale, float(c["fy"])*scale, (W-1)/2.0, (H-1)/2.0,
         0.2, 10000.0, 0.00016, 0),
    )
    image = out[0]
    print("forward OK; image shape", tuple(image.shape), "range", image.min().item(), image.max().item())
    img = image.detach().squeeze(0).cpu().numpy()
    if img.shape[0] in (1,3): img = img.transpose(1,2,0)
    img = np.clip(img,0,1)
    gt = Image.open(f"{SCENE}/eval_images/{c['img_name']}.JPG").convert("RGB").resize((W,H),Image.LANCZOS)
    gt_np = np.asarray(gt,dtype=np.float32)/255.0
    corr = np.corrcoef(img.ravel(), gt_np.ravel())[0,1]
    mse = ((img-gt_np)**2).mean()
    print(f"=== FASTERGS NATIVE RENDER: corr={corr:.4f} mse={mse:.4f} ===")
    Image.fromarray((np.concatenate([img,gt_np],axis=1)*255).astype(np.uint8)).save("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/fastergs_native_sbs.png")
    print("saved fastergs_native_sbs.png")
except Exception as e:
    import traceback; traceback.print_exc()
    print("FasterGS native forward FAILED:", e)
