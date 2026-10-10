#!/usr/bin/env python3
"""Check scale distribution and render at native training resolution."""
import os, sys, json
import torch, numpy as np
from PIL import Image
dev = torch.device("cuda:6")
CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
scales_raw = g["gaussians._scales"].to(dev).float()
scales = scales_raw.exp()
means = g["gaussians._means"].to(dev).float()
N = means.shape[0]
print("=== scale distribution (exp) ===")
sv = scales.flatten().cpu().numpy()
for q in [1,5,10,25,50,75,90,95,99]:
    print(f"  p{q}: {np.percentile(sv,q):.5f}")
print(f"  max {sv.max():.4f} mean {sv.mean():.5f}")
# max extent per gaussian (3x scale)
max_ext = scales.max(dim=1).values * 2  # diameter
print(f"  median diameter {max_ext.median():.4f}  p90 {max_ext.quantile(0.9):.4f}")

# Render at native training res (0.25 * 4946 = 1236, but cap pixels)
cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
c = cams[0]
# native-ish: 0.25 scale
for sf, tag in [(0.25, "native0.25"), (0.5, "half")]:
    w2 = int(c["width"]*sf); h2 = int(c["height"]*sf)
    if w2*h2 > 900000:
        s = (900000/(c["width"]*c["height"]))**0.5; w2=int(c["width"]*s); h2=int(c["height"]*s); sf=s
    scale = w2/float(c["width"])
    R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
    Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    vm=torch.tensor(vm,dtype=torch.float32,device=dev)
    K=torch.tensor([[c["fx"]*sf,0,(w2-1)/2.0],[0,c["fy"]*sf,(h2-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev)
    # GT at this res
    gt=None
    for sub in ("eval_images","images"):
        for ext in (".JPG",".jpg",".png"):
            pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
            if os.path.exists(pp) and os.path.getsize(pp)>1000:
                gt=Image.open(pp).convert("RGB").resize((w2,h2),Image.LANCZOS); break
        if gt is not None: break
    gt_t=torch.tensor(np.asarray(gt,np.float32)/255.,device=dev).permute(2,0,1).unsqueeze(0)
    quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
    colors=torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
    from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn
    out=R_fn(means,quats,scales,opac,colors,sh_degree=3,viewmats=vm.unsqueeze(0),Ks=K.unsqueeze(0),
             width=w2,height=h2,tile_size=16,near_plane=0.2,far_plane=10000.0,
             background=torch.zeros(3,device=dev),enable_culling=True,use_higs_culling=False)
    rend=out["frame"].permute(0,3,1,2).clamp(0,1)
    mse=(rend-gt_t).pow(2).mean().item(); psnr=-10*np.log10(max(mse,1e-12))
    corr=np.corrcoef(rend[0].cpu().numpy().ravel(),gt_t[0].cpu().numpy().ravel())[0,1]
    # median gaussian pixel radius
    import math
    fx_pix = c["fx"]*sf
    med_diam_pix = (max_ext.median().item()) * fx_pix / 1.0  # rough: diameter * focal
    print(f"\n[{tag} {w2}x{h2}] PSNR={psnr:.2f} corr={corr:.3f} rend_mean={rend.mean():.3f}")
    print(f"  median gaussian diameter ~{med_diam_pix:.2f} px (focal {fx_pix:.0f})")
    if tag=="native0.25":
        Image.fromarray((rend[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/s1a_render_native.png")
        Image.fromarray((gt_t[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/s1a_gt_native.png")
