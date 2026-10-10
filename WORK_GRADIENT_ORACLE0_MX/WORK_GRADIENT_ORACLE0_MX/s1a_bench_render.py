#!/usr/bin/env python3
"""Use benchmark's OWN reference render path (_std_ll_forward) on the checkpoint.
Isolates whether the issue is my render call or the checkpoint/camera data."""
import os, sys, json, math
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
import torch, numpy as np
from PIL import Image
dev = torch.device("cuda:6"); torch.manual_seed(0)

# --- benchmark reference render path (copied from run_higs_train_benchmark.py) ---
def _std_ll_forward(means, quats, scales, opacities, colors, viewmats, Ks, width, height, sh_degree, radius_clip=0.0):
    from gsplat.cuda._wrapper import (fully_fused_projection, isect_tiles, isect_offset_encode, _make_lazy_cuda_func)
    from gsplat.rendering import _maybe_evaluate_sh
    C = viewmats.shape[-3]; N = means.shape[-2]; tile_size = 16
    tile_width = math.ceil(width / tile_size); tile_height = math.ceil(height / tile_size)
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means=means.contiguous(), covars=None, quats=quats.contiguous(), scales=scales.contiguous(),
        viewmats=viewmats, Ks=Ks, width=width, height=height, eps2d=0.3, near_plane=0.01,
        far_plane=1e10, radius_clip=radius_clip, packed=False, calc_compensations=False, camera_model="pinhole")
    opacities_bc = torch.broadcast_to(opacities[..., None, :], (1, C, N)).contiguous()
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, packed=False, n_images=C,
        image_ids=None, gaussian_ids=None, conics=conics, opacities=opacities_bc)
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height).reshape((1, C, tile_height, tile_width))
    colors_eval = _maybe_evaluate_sh(sh_degree, colors, means, radii, viewmats, (1,), C, N, True).contiguous()
    bg_kernel = torch.zeros((1, C, 3), device=means.device)
    render_colors, render_alphas, _absgrad, last_ids = (_make_lazy_cuda_func("rasterize_to_pixels_3dgs"))(
        means2d.contiguous(), conics.contiguous(), colors_eval.contiguous(), opacities_bc.contiguous(),
        bg_kernel, None, width, height, tile_size, isect_offsets.contiguous(), flatten_ids.contiguous(),
        False, False)
    return render_colors, render_alphas

def load_ckpt(ckpt):
    blob=torch.load(ckpt,map_location="cpu",weights_only=False)
    g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
    means=g["gaussians._means"].to(dev).float()
    quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    scales=g["gaussians._scales"].to(dev).float().exp()
    opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
    colors=torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
    return means,quats,scales,opac,colors

cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
c=cams[0]
for sf, tag in [(0.25,"native0.25"), (0.08,"small0.08")]:
    w2=int(c["width"]*sf); h2=int(c["height"]*sf)
    scale=sf
    R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
    Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    viewmats=torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0).unsqueeze(0)  # [1,C,4,4]
    Ks=torch.tensor([[c["fx"]*sf,0,(w2-1)/2.0],[0,c["fy"]*sf,(h2-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev).unsqueeze(0).unsqueeze(0)  # [1,C,3,3]
    # GT
    gt=None
    for sub in ("eval_images","images"):
        for ext in (".JPG",".jpg",".png"):
            pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
            if os.path.exists(pp) and os.path.getsize(pp)>1000:
                gt=Image.open(pp).convert("RGB").resize((w2,h2),Image.LANCZOS); break
        if gt is not None: break
    gt_t=torch.tensor(np.asarray(gt,np.float32)/255.,device=dev).permute(2,0,1).unsqueeze(0)
    for ckpt_tag, ckpt in [("5K","/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"),
                           ("10K","/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0010000.pt")]:
        means,quats,scales,opac,colors=load_ckpt(ckpt)
        # colors for _maybe_evaluate_sh: [C, N, K, 3]? benchmark passes colors as [N,K,3] and sh_degree
        # _maybe_evaluate_sh(sh_degree, colors, means, radii, viewmats, (1,), C, N, True) - colors [N,K,3]
        try:
            rc, ra = _std_ll_forward(means.unsqueeze(0), quats.unsqueeze(0), scales.unsqueeze(0), opac.unsqueeze(0), colors, viewmats, Ks, w2, h2, 3)
            rend = rc.reshape(1, h2, w2, 3).permute(0,3,1,2).clamp(0,1)
        except Exception as e:
            print(f"[{tag} {ckpt_tag}] render error: {type(e).__name__}: {e}")
            continue
        mse=(rend-gt_t).pow(2).mean().item(); psnr=-10*np.log10(max(mse,1e-12))
        corr=np.corrcoef(rend[0].cpu().numpy().ravel(),gt_t[0].cpu().numpy().ravel())[0,1]
        print(f"[{tag} {w2}x{h2} {ckpt_tag}] PSNR={psnr:.2f} corr={corr:.3f} rend_mean={rend.mean():.3f} alpha_mean={ra.mean():.3f}")
        if tag=="native0.25" and ckpt_tag=="10K":
            Image.fromarray((rend[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/s1a_bench_render_10K.png")
