#!/usr/bin/env python3
"""S1A-2: Real GPU cost calibration for HiGS sparse tile execution.

Same ckpt/camera/kernel/loss. Measure full vs sparse (p=0.25, 0.40):
fwd, backward, F+B. Record n_isects, sampled_tile_ratio. CUDA events + wall time.
>=30 warmup, >=100 timing, 3 paired repeats, alternating order.
Render-quality-independent (kernel cost only).
"""
import os, sys, json, math, time
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
import torch, numpy as np
from PIL import Image
dev = torch.device("cuda:6"); torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False

CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means_all = g["gaussians._means"].to(dev).float()
quats_all = g["gaussians._rotations"].to(dev).float(); quats_all=quats_all/quats_all.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales_all = g["gaussians._scales"].to(dev).float().exp()
opac_all = g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
colors_all = torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),
                        g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()

cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
c = cams[0]
W, H = 396, 264
tile = 16
th, tw = math.ceil(H/tile), math.ceil(W/tile)
num_tiles = th*tw
sf = W/float(c["width"])
R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
viewmats=torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)
Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev).unsqueeze(0)

gt=None
for sub in ("eval_images","images"):
    for ext in (".JPG",".jpg",".png"):
        pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
        if os.path.exists(pp) and os.path.getsize(pp)>1000:
            gt=Image.open(pp).convert("RGB").resize((W,H),Image.LANCZOS); break
    if gt is not None: break
gt_t=torch.tensor(np.asarray(gt,np.float32)/255.,device=dev).permute(2,0,1).unsqueeze(0)

# on-screen gaussians
cam_pts = (torch.tensor(Rw2c,dtype=torch.float32,device=dev) @ means_all.T).T + torch.tensor(-Rw2c@p,dtype=torch.float32,device=dev)
z = cam_pts[:,2]
uv = (Ks[0] @ cam_pts.T).T
u = uv[:,0]/uv[:,2].clamp_min(1e-6); v = uv[:,1]/uv[:,2].clamp_min(1e-6)
onscreen = (z>0.2)&(z<10000)&(u>=-8)&(u<W+8)&(v>=-8)&(v<H+8)
idx = onscreen.nonzero(as_tuple=True)[0]
N_TARGET = 150000
if idx.numel() > N_TARGET:
    sel = torch.randperm(idx.numel(), device=dev)[:N_TARGET]; idx = idx[sel]
idx = idx.sort().values; N = idx.numel()
means = means_all[idx].detach().requires_grad_(True)
quats = quats_all[idx].detach().requires_grad_(True)
scales = scales_all[idx].detach().requires_grad_(True)
opac = opac_all[idx].detach().requires_grad_(True)
colors = colors_all[idx].detach().requires_grad_(True)
params = [means,quats,scales,opac,colors]

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn

def make_mask(ratio, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    m = torch.zeros(th, tw, dtype=torch.bool, device=dev)
    k = max(1, int(num_tiles*ratio))
    sel = torch.randperm(num_tiles, generator=gen, device=dev)[:k]
    flat = torch.zeros(num_tiles, dtype=torch.bool, device=dev)
    flat[sel] = True
    m = flat.view(th, tw)
    return m.unsqueeze(0)  # [1, th, tw]

def render(mask=None):
    kw = dict(sh_degree=3,viewmats=viewmats,Ks=Ks,width=W,height=H,tile_size=tile,
              near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),
              enable_culling=False,use_higs_culling=False)
    if mask is not None:
        kw["tile_mask"] = mask
    out = R_fn(means,quats,scales,opac,colors, **kw)
    return out

def loss_fn(out):
    frame = out["frame"]
    return (frame[0]-gt_t[0].permute(1,2,0)).abs().sum()/(3*H*W)

# warmup
for _ in range(30):
    out = render(); l = loss_fn(out); l.backward()
    means.grad=None; quats.grad=None; scales.grad=None; opac.grad=None; colors.grad=None
torch.cuda.synchronize()

def time_cfg(mask, n_rep=3, n_iter=100):
    cfg = {"fwd_ms":[], "bwd_ms":[], "fb_ms":[], "n_isects":[], "tile_ratio":[]}
    for rep in range(n_rep):
        # fwd timing
        s_e=torch.cuda.Event(True); e_e=torch.cuda.Event(True)
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(n_iter):
            s_e.record(); out=render(mask); e_e.record()
        torch.cuda.synchronize(); t1=time.perf_counter()
        fwd_gpu = s_e.elapsed_time(e_e)/n_iter
        fwd_wall = (t1-t0)/n_iter*1000
        # fwd+bwd timing
        s_e2=torch.cuda.Event(True); e_e2=torch.cuda.Event(True)
        torch.cuda.synchronize(); t2=time.perf_counter()
        for _ in range(n_iter):
            s_e2.record(); out=render(mask); l=loss_fn(out); l.backward(); e_e2.record()
        torch.cuda.synchronize(); t3=time.perf_counter()
        fb_gpu = s_e2.elapsed_time(e_e2)/n_iter
        fb_wall = (t3-t2)/n_iter*1000
        bwd_gpu = fb_gpu - fwd_gpu
        cfg["fwd_ms"].append(fwd_gpu); cfg["bwd_ms"].append(bwd_gpu); cfg["fb_ms"].append(fb_gpu)
        cfg["n_isects"].append(int(out["metadata"]["n_isects"]))
        cfg["tile_ratio"].append(float(out["metadata"]["sampled_tile_ratio"]))
        means.grad=None; quats.grad=None; scales.grad=None; opac.grad=None; colors.grad=None
    # aggregate (median across reps)
    cfg["fwd_ms"]=float(np.median(cfg["fwd_ms"]))
    cfg["bwd_ms"]=float(np.median(cfg["bwd_ms"]))
    cfg["fb_ms"]=float(np.median(cfg["fb_ms"]))
    cfg["n_isects"]=int(np.median(cfg["n_isects"]))
    cfg["tile_ratio"]=float(np.median(cfg["tile_ratio"]))
    cfg["wall_fwd_ms"]=fwd_wall; cfg["wall_fb_ms"]=fb_wall
    return cfg

print("=== S1A-2 cost calibration (3 paired repeats, alternating) ===")
# alternating order across repeats
results = {}
configs = [("full", None), ("sparse_p25", make_mask(0.25)), ("sparse_p40", make_mask(0.40))]
# do 3 paired repeats, alternating order
for rep in range(3):
    order = configs if rep%2==0 else list(reversed(configs))
    for name, mask in order:
        r = time_cfg(mask, n_rep=1, n_iter=100)
        key = f"{name}_rep{rep}"
        results[key] = r
        print(f"[rep{rep}] {name:12s} fwd={r['fwd_ms']:.3f}ms bwd={r['bwd_ms']:.3f}ms F+B={r['fb_ms']:.3f}ms n_isects={r['n_isects']} tile_ratio={r['tile_ratio']:.3f}")

# aggregate per config
agg = {}
for name, _ in configs:
    rs = [results[f"{name}_rep{rep}"] for rep in range(3)]
    agg[name] = {
        "fwd_ms_median":float(np.median([r["fwd_ms"] for r in rs])),
        "bwd_ms_median":float(np.median([r["bwd_ms"] for r in rs])),
        "fb_ms_median":float(np.median([r["fb_ms"] for r in rs])),
        "n_isects_median":int(np.median([r["n_isects"] for r in rs])),
        "tile_ratio_median":float(np.median([r["tile_ratio"] for r in rs])),
        "fb_ms_reps":[r["fb_ms"] for r in rs],
    }
full_fb = agg["full"]["fb_ms_median"]
for name in ["sparse_p25","sparse_p40"]:
    sp = full_fb / agg[name]["fb_ms_median"]
    isect_red = 1 - agg[name]["n_isects_median"]/agg["full"]["n_isects_median"]
    agg[name]["speedup_F+B"] = float(sp)
    agg[name]["isect_reduction"] = float(isect_red)
    print(f"[{name}] speedup(F+B)={sp:.3f}x  isect_reduction={isect_red:.3f}")

result = {
  "schema":"S1A_COST_CALIBRATION",
  "checkpoint":"bicycle 0005000.pt","camera":c["img_name"],
  "resolution":[W,H],"tile":tile,"num_tiles":num_tiles,"N_gaussians":N,
  "warmup":30,"timing_iters":100,"paired_repeats":3,
  "aggregate":agg,
  "note":"HiGS sparse tile kernel cost vs full; render-quality-independent (kernel cost only)",
}
with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S1A_COST_CALIBRATION.json","w") as f:
    json.dump(result,f,indent=2,default=str)
print("saved S1A_COST_CALIBRATION.json")
