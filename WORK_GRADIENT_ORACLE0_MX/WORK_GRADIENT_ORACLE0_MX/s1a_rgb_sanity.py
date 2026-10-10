#!/usr/bin/env python3
"""Decisive RGB render sanity test: bicycle 5K checkpoint, real training camera.
Confirms color/activation convention matches training. Reports PSNR vs GT."""
import os, sys, glob, math, json, time
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
import torch, numpy as np
from PIL import Image

dev = torch.device("cuda:6")  # least-loaded GPU
torch.manual_seed(0)

CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
N = g["gaussians._means"].shape[0]
sh_degree = 3

# gsplat activations (match FasterGS Model.py exactly)
means = g["gaussians._means"].to(dev).float()
quats = g["gaussians._rotations"].to(dev).float()
quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
scales = g["gaussians._scales"].to(dev).float().exp()
opac = g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)  # [N]
colors = torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),
                    g["gaussians._sh_coefficients_rest"].to(dev).float()], dim=1).contiguous()  # [N,16,3]

print(f"N={N} sh_degree={sh_degree}")
print(f"means range [{means.min():.2f},{means.max():.2f}]  scales exp range [{scales.min():.4f},{scales.max():.4f}]")
print(f"opac sigmoid mean {opac.mean():.3f} range [{opac.min():.3f},{opac.max():.3f}]")
print(f"DC(sh0) range [{colors[:,0].min():.2f},{colors[:,0].max():.2f}] mean {colors[:,0].mean():.3f}")

# Camera 0 from eval_cameras.json, scaled to a moderate res for the sanity render
import json as _json
cams = _json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
W, H = 792, 528   # 0.25 * (4946/... ) -> use ~0.16 scale for speed; keep aspect
W, H = 792, 528
c = cams[0]
scale = W / float(c["width"])
R = np.asarray(c["rotation"], dtype=np.float64); p = np.asarray(c["position"], dtype=np.float64)
Rw2c = R.T
vm = np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
K = np.array([[float(c["fx"])*scale,0,(W-1)/2.0],[0,float(c["fy"])*scale,(H-1)/2.0],[0,0,1.0]], dtype=np.float64)
viewmats = torch.tensor(vm, dtype=torch.float32, device=dev).unsqueeze(0)
Ks = torch.tensor(K, dtype=torch.float32, device=dev).unsqueeze(0)
print(f"\ncamera0 {c['img_name']} render {W}x{H} fx={c['fx']*scale:.1f} fy={c['fy']*scale:.1f}")

# GT image (eval_images has real JPEGs; images/ are git-lfs stubs)
gt_path = None
for sub in ("eval_images","images"):
    for ext in (".JPG",".jpg",".png",".jpeg"):
        cand = f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
        if os.path.exists(cand) and os.path.getsize(cand) > 1000:
            gt_path = cand; break
    if gt_path: break
print("GT:", gt_path)
gt = Image.open(gt_path).convert("RGB").resize((W,H), Image.LANCZOS)
gt_t = torch.tensor(np.asarray(gt,dtype=np.float32)/255.0, device=dev).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]

# Render via HiGS dynamic (sh_degree=3)
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn
for t in (means,quats,scales,opac,colors): t.requires_grad_(False)
torch.cuda.synchronize()
t0=time.time()
out = R_fn(means,quats,scales,opac,colors, sh_degree=sh_degree,
           viewmats=viewmats, Ks=Ks, width=W, height=H, tile_size=16,
           near_plane=0.2, far_plane=10000.0,
           background=torch.zeros(3,device=dev), enable_culling=False, use_higs_culling=False)
torch.cuda.synchronize()
render = out["frame"]  # [1,H,W,3]
meta = out.get("metadata",{}) or {}
print(f"render in {(time.time()-t0)*1000:.0f}ms  n_isects={meta.get('n_isects')} shape={tuple(render.shape)}")
render = render.permute(0,3,1,2).clamp(0,1)  # [1,3,H,W]

# Metrics
mse = (render - gt_t).pow(2).mean().item()
psnr = -10*np.log10(max(mse,1e-12))
mae = (render - gt_t).abs().mean().item()
print(f"\n=== RGB SANITY (camera0) ===")
print(f"PSNR={psnr:.2f} dB  MSE={mse:.5f}  MAE={mae:.4f}")
print(f"render mean={render.mean().item():.3f} range=[{render.min().item():.3f},{render.max().item():.3f}]")
print(f"GT     mean={gt_t.mean().item():.3f} range=[{gt_t.min().item():.3f},{gt_t.max().item():.3f}]")
# color correlation per channel
for i,ch in enumerate("RGB"):
    r=render[0,i].cpu().numpy().ravel(); gg=gt_t[0,i].cpu().numpy().ravel()
    corr=np.corrcoef(r,gg)[0,1] if r.std()>0 and gg.std()>0 else 0.0
    print(f"  {ch}: render mean {r.mean():.3f} vs GT {gg.mean():.3f}  corr={corr:.3f}")

verdict = "PASS_COLOR_OK" if (psnr > 12 and render.mean().item()>0.05 and mae<0.4) else "SUSPECT"
print(f"\nVERDICT: {verdict}")
with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S1A_RGB_SANITY.json","w") as f:
    json.dump({"N":N,"camera":c["img_name"],"wh":[W,H],"psnr":psnr,"mse":mse,"mae":mae,
               "render_mean":render.mean().item(),"gt_mean":gt_t.mean().item(),
               "n_isects":meta.get('n_isects'),"verdict":verdict}, f, indent=2)
print("saved S1A_RGB_SANITY.json")
