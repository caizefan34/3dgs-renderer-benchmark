#!/usr/bin/env python3
"""Render sanity test: bicycle 5K ckpt + real camera id=0 + GT comparison.
Confirms camera convention & color convention produce a recognizable render."""
import os, sys, math, json, time
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("PYTHONNOUSERSITE","1")
import torch, numpy as np
from PIL import Image

dev = "cuda:6"   # least-loaded GPU
CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
SCENE = "/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle"

blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
N = g["gaussians._means"].shape[0]
sh_degree = 3
print(f"N={N} iter={blob.get('num_iterations_trained')} sh={sh_degree}")

means = g["gaussians._means"].to(dev).float()
quats_raw = g["gaussians._rotations"].to(dev).float()
scales_raw = g["gaussians._scales"].to(dev).float()
opac_raw = g["gaussians._opacities"].to(dev).float()
sh0 = g["gaussians._sh_coefficients_0"].to(dev).float()
shrest = g["gaussians._sh_coefficients_rest"].to(dev).float()
colors = torch.cat([sh0, shrest], dim=1).contiguous()
quats = quats_raw / quats_raw.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales = torch.exp(scales_raw)
opac = torch.sigmoid(opac_raw).squeeze(-1)
for t in (means,quats,scales,opac,colors): t.requires_grad_(True)
print("activated: scales median", scales.median().item(), "opac mean", opac.mean().item())

# Camera id=0 with benchmark convention (c2w rotation, position=camera center)
cams = json.load(open(f"{SCENE}/eval_cameras.json"))
c = cams[0]
scale = 0.25
W, H = int(c["width"]*scale), int(c["height"]*scale)
R = np.asarray(c["rotation"], dtype=np.float64); p = np.asarray(c["position"], dtype=np.float64)
Rw2c = R.T
vm = np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3] = -Rw2c @ p
K = np.array([[float(c["fx"])*scale,0,(W-1)/2.0],[0,float(c["fy"])*scale,(H-1)/2.0],[0,0,1.0]],dtype=np.float64)
viewmats = torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)
Ks = torch.tensor(K,dtype=torch.float32,device=dev).unsqueeze(0)
C=1; tile=16; th=math.ceil(H/tile); tw=math.ceil(W/tile)
print(f"cam0 {c['img_name']} render {W}x{H} tiles {th}x{tw}")

# GT
gt_path = f"{SCENE}/eval_images/{c['img_name']}.JPG"
gt = Image.open(gt_path).convert("RGB").resize((W,H), Image.LANCZOS)
gt_t = torch.tensor(np.asarray(gt,dtype=np.float32)/255.0, device=dev).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn

# Subsample for memory (full 2.4M at 1235x821 too big for quick test)
sub = 400000
idx = torch.randperm(N, device=dev)[:sub]
m_s,q_s,sc_s,o_s,col_s = [t[idx].contiguous() for t in (means,quats,scales,opac,colors)]
for t in (m_s,q_s,sc_s,o_s,col_s): t.requires_grad_(True)

torch.cuda.synchronize(); t0=time.time()
out = R_fn(m_s,q_s,sc_s,o_s,col_s, sh_degree=sh_degree,
           viewmats=viewmats, Ks=Ks, width=W, height=H, tile_size=tile,
           near_plane=0.2, far_plane=10000.0,
           background=torch.zeros(3,device=dev), enable_culling=False, use_higs_culling=False)
torch.cuda.synchronize(); t1=time.time()
frame = out["frame"]  # [1,H,W,3]
print(f"render time {t1-t0:.3f}s n_isects={out.get('metadata',{}).get('n_isects')} shape={tuple(frame.shape)}")

# frame [1,H,W,3] -> [1,3,H,W]
img = frame.permute(0,3,1,2).clamp(0,1)
mse = (img-gt_t).pow(2).mean().item()
psnr = -10*np.log10(max(mse,1e-12))
# correlation per channel
a = img.detach().cpu().numpy().ravel(); b = gt_t.cpu().numpy().ravel()
corr = np.corrcoef(a,b)[0,1]
# SSIM quick (use gsplat? skip - use simple)
print(f"=== SANITY: mse={mse:.5f} psnr={psnr:.2f}dB corr={corr:.4f} ===")
print(f"render mean={img.mean().item():.4f} std={img.std().item():.4f}")
print(f"GT     mean={gt_t.mean().item():.4f} std={gt_t.std().item():.4f}")

# Save side-by-side
img_np = (img.detach().squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
gt_np = gt_t.squeeze(0).permute(1,2,0).cpu().numpy()
Image.fromarray(np.concatenate([img_np, gt_np], axis=1)).save("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/sanity_cam0.png")
print("saved sanity_cam0.png (left=render, right=GT)")
