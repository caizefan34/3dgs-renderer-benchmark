#!/usr/bin/env python3
"""Save visual render vs GT for camera 0, plus test principal-point variants."""
import os, sys, json, math
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
import torch, numpy as np
from PIL import Image
dev = torch.device("cuda:6"); torch.manual_seed(0)
CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means = g["gaussians._means"].to(dev).float()
quats = g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales = g["gaussians._scales"].to(dev).float().exp()
opac = g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
colors = torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
W,H = 396,264
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn
c = cams[0]
# Try multiple render resolutions & principal points
variants = []
for scale_factor, tag in [(W/float(c["width"]), "json_scale"), (0.5, "half"), (1.0, "full_res")]:
    w2 = int(c["width"]*scale_factor); h2 = int(c["height"]*scale_factor)
    if w2*h2 > 400000:  # cap
        s = (400000/(c["width"]*c["height"]))**0.5
        w2=int(c["width"]*s); h2=int(c["height"]*s); scale_factor=s
    variants.append((tag, w2, h2, scale_factor))
print("variants:", variants)
def render(c, w2, h2, sf, cx_off=0.0, cy_off=0.0):
    scale=sf
    R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
    Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    vm=torch.tensor(vm,dtype=torch.float32,device=dev)
    K=torch.tensor([[float(c["fx"])*scale,0,(w2-1)/2.0+cx_off],[0,float(c["fy"])*scale,(h2-1)/2.0+cy_off],[0,0,1.0]],dtype=torch.float32,device=dev)
    out=R_fn(means,quats,scales,opac,colors,sh_degree=3,viewmats=vm.unsqueeze(0),Ks=K.unsqueeze(0),
             width=w2,height=h2,tile_size=16,near_plane=0.01,far_plane=1e10,
             background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    return out["frame"]
# GT at 396x264
gt=None
for sub in ("eval_images","images"):
    for ext in (".JPG",".jpg",".png"):
        pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
        if os.path.exists(pp) and os.path.getsize(pp)>1000:
            gt=Image.open(pp).convert("RGB").resize((W,H),Image.LANCZOS); break
    if gt is not None: break
gt_t=torch.tensor(np.asarray(gt,np.float32)/255.,device=dev).permute(2,0,1).unsqueeze(0)
rend = render(c, W, H, W/float(c["width"])).permute(0,3,1,2).clamp(0,1)
Image.fromarray((rend[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/s1a_render_c0.png")
Image.fromarray((gt_t[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/s1a_gt_c0.png")
mse=(rend-gt_t).pow(2).mean().item(); psnr=-10*np.log10(max(mse,1e-12))
corr=np.corrcoef(rend[0].cpu().numpy().ravel(),gt_t[0].cpu().numpy().ravel())[0,1]
print(f"\ncamera0 base: PSNR={psnr:.2f} corr={corr:.3f}")
# save a downscaled full-res render for visual
tag,w2,h2,sf = variants[-1]
print(f"rendering full-res variant {tag} {w2}x{h2}")
r2 = render(c, w2, h2, sf)
Image.fromarray((r2[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)).save(f"/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/s1a_render_{tag}.png")
print("saved s1a_render_c0.png, s1a_gt_c0.png, s1a_render_full_res.png")
