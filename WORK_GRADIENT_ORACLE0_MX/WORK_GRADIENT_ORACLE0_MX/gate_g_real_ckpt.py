#!/usr/bin/env python3
"""Gate G: real FasterGSFused bicycle checkpoint compatibility test.
Loads 5K checkpoint, extracts SH/opacity/scale, sets up a real camera,
and runs full/sparse HiGS forward+backward. Reports convention mismatches
(notably PCA color space) without claiming oracle validity."""
import os, sys, glob, math, json
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
import torch, numpy as np

CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
dev = "cuda:0"

blob = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = blob["model_state_dict"]
g = {k: v for k, v in sd.items() if k.startswith("gaussians.")}
print("=== checkpoint keys ===")
for k in sorted(g.keys()):
    print(f"  {k}: {tuple(g[k].shape)} {g[k].dtype}")
print("top-level:", blob.get("model_name"), "iter:", blob.get("num_iterations_trained"), "sh_degree:", blob.get("SH_DEGREE"))

N = g["gaussians._means"].shape[0]
sh_degree = 3
K = (sh_degree+1)**2   # 16
means = g["gaussians._means"].to(dev).float()
quats_raw = g["gaussians._rotations"].to(dev).float()
scales_raw = g["gaussians._scales"].to(dev).float()
opac_raw = g["gaussians._opacities"].to(dev).float()   # [N,1]
sh0 = g["gaussians._sh_coefficients_0"].to(dev).float()      # [N,1,3]
shrest = g["gaussians._sh_coefficients_rest"].to(dev).float() # [N,15,3]
# gsplat SH layout: colors [N, K, 3] = cat(sh0, shrest) dim=1
colors = torch.cat([sh0, shrest], dim=1).contiguous()  # [N,16,3]
quats = quats_raw / quats_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
scales = torch.exp(scales_raw)          # gsplat stores log-scales
opac = torch.sigmoid(opac_raw).squeeze(-1)  # [N] activated opacity

print("\n=== activated ranges (sanity) ===")
print(f"means range: [{means.min():.2f}, {means.max():.2f}]")
print(f"scales exp range: [{scales.min():.4f}, {scales.max():.4f}] median {scales.median():.4f}")
print(f"opac sigmoid range: [{opac.min():.3f}, {opac.max():.3f}] mean {opac.mean():.3f}")
print(f"colors[sh0=DC] range: [{colors[:,0].min():.2f}, {colors[:,0].max():.2f}]")
print(f"DC mean (should be ~0.5 if RGB-centered): {colors[:,0].mean().item():.3f}")

# Real camera (id=0) from bicycle eval_cameras.json, scaled to 0.25 like training
import json
cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
c = cams[0]
W, H = 4946//4, 3286//4   # IMAGE_SCALE_FACTOR 0.25 -> but keep modest for smoke
W, H = min(W, 640), min(H, 424)  # cap for smoke memory
scale = W / float(c["width"])
R = np.asarray(c["rotation"], dtype=np.float64); p = np.asarray(c["position"], dtype=np.float64)
Rw2c = R.T
vm = np.eye(4); vm[:3,:3] = Rw2c; vm[:3,3] = -Rw2c @ p
K = np.array([[float(c["fx"])*scale,0,(W-1)/2.0],[0,float(c["fy"])*scale,(H-1)/2.0],[0,0,1.0]], dtype=np.float64)
viewmats = torch.tensor(vm, dtype=torch.float32, device=dev).unsqueeze(0)
Ks = torch.tensor(K, dtype=torch.float32, device=dev).unsqueeze(0)
C = 1
tile = 16
th, tw = math.ceil(H/tile), math.ceil(W/tile)
print(f"\ncamera0: {c['img_name']} render {W}x{H}, tiles {th}x{tw}={th*tw}")

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn

# Subsample gaussians for smoke memory (real N=2.4M too big for quick smoke)
sub = 300000
idx = torch.randperm(N, device=dev)[:sub]
def pack(t, dim_keep=True):
    return t[idx].contiguous()
m_s, q_s, sc_s, o_s, col_s = pack(means), pack(quats), pack(scales), pack(opac), pack(colors)
for t in (m_s,q_s,sc_s,o_s,col_s): t.requires_grad_(True)

def run(tag, tile_mask):
    for t in (m_s,q_s,sc_s,o_s,col_s):
        if t.grad is not None: t.grad=None
    out = R_fn(m_s,q_s,sc_s,o_s,col_s, sh_degree=sh_degree,
               viewmats=viewmats, Ks=Ks, width=W, height=H, tile_size=tile,
               tile_mask=tile_mask, near_plane=0.2, far_plane=10000.0,
               background=torch.zeros(3,device=dev), enable_culling=False, use_higs_culling=False)
    frame = out["frame"]
    torch.cuda.synchronize()
    loss = frame.float().abs().mean()
    loss.backward(); torch.cuda.synchronize()
    gnz={n:int((t.grad.abs()>1e-10).sum().item()) if t.grad is not None else -1
         for n,t in zip(["means","quats","scales","opac","colors"],[m_s,q_s,sc_s,o_s,col_s])}
    print(f"[{tag}] loss={loss.item():.4f} finite={torch.isfinite(frame).all().item()} "
          f"n_isects={out.get('metadata',{}).get('n_isects')} sampled={out.get('metadata',{}).get('sampled_tile_ratio')}")
    print(f"       grad_nonzero(sub={sub}): {gnz}")
    return {"ok":True,"loss":loss.item(),"n_isects":out.get('metadata',{}).get('n_isects'),"gnz":gnz}

res_full = run("full", None)
mask = torch.zeros((C,th,tw),dtype=torch.bool,device=dev); mask.reshape(C,-1)[:,::4]=True
res_sparse = run("sparse", mask)
print("\nNOTE: colors are in raw SH space; APPLY_PCA=true in training means DC may be")
print("PCA-rotated. Render validity as oracle requires PCA inverse - recorded as caveat.")

with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S0R2_REAL_CKPT_RESULT.json","w") as fp:
    json.dump({"N_full":N,"sub":sub,"sh_degree":sh_degree,"camera":c["img_name"],
               "render_wh":[W,H],"full":res_full,"sparse":res_sparse,
               "pca_caveat":"training_config APPLY_PCA=true"}, fp, indent=2)
print("saved.")
