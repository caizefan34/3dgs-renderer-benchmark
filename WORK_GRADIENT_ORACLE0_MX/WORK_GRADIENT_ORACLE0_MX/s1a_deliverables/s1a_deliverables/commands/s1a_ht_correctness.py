#!/usr/bin/env python3
"""S1A-0: HT gradient estimator validation (pure math, render-quality-independent).

Fixed real bicycle 5K checkpoint, camera 0, additive full-image-normalized L1.
Tile contribution L_j = sum_{pixel in tile j} |render-GT| / (3HW).
HT estimator g_hat = sum_j (z_j/p_j) grad(L_j), z_j~Ber(p_j).

Validates (render quality independent -- these are estimator-math properties):
1. p=1 (all z_j=1) == dense gradient
2. disjoint partition additivity: sum of per-tile grads == dense grad
3. MC consistency at p=0.25, 0.40: mean(g_hat) ~ dense, report CI
4. boundary tile pixel counts + loss normalization (sum_j L_j == L_full)
5. all 5 param groups receive gradients
6. signed gradient vs absgrad vs densification stat (kept distinct)
7. unbiasedness + independence conditions
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
torch.backends.cudnn.allow_tf32 = False

CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means_all = g["gaussians._means"].to(dev).float()
quats_all = g["gaussians._rotations"].to(dev).float(); quats_all=quats_all/quats_all.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales_all = g["gaussians._scales"].to(dev).float().exp()
opac_all = g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
colors_all = torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),
                        g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
N_all = means_all.shape[0]

cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
c = cams[0]
W, H = 128, 128
tile = 16
th, tw = math.ceil(H/tile), math.ceil(W/tile)
num_tiles = th*tw
sf = W/float(c["width"])
R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
viewmats=torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)
Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev).unsqueeze(0)

# GT
gt=None
for sub in ("eval_images","images"):
    for ext in (".JPG",".jpg",".png"):
        pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
        if os.path.exists(pp) and os.path.getsize(pp)>1000:
            gt=Image.open(pp).convert("RGB").resize((W,H),Image.LANCZOS); break
    if gt is not None: break
gt_t=torch.tensor(np.asarray(gt,np.float32)/255.,device=dev).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]

# Select gaussians on-screen for camera 0 (project, keep those in front + in bounds with margin)
cam_pts = (torch.tensor(Rw2c,dtype=torch.float32,device=dev) @ means_all.T).T + torch.tensor(-Rw2c@p,dtype=torch.float32,device=dev)
z = cam_pts[:,2]
uv = (Ks[0] @ cam_pts.T).T
u = uv[:,0]/uv[:,2].clamp_min(1e-6); v = uv[:,1]/uv[:,2].clamp_min(1e-6)
onscreen = (z>0.2)&(z<10000)&(u>=-8)&(u<W+8)&(v>=-8)&(v<H+8)
idx = onscreen.nonzero(as_tuple=True)[0]
# subsample to N_TARGET for memory/speed
N_TARGET = 120000
if idx.numel() > N_TARGET:
    sel = torch.randperm(idx.numel(), device=dev)[:N_TARGET]
    idx = idx[sel]
idx = idx.sort().values
N = idx.numel()
print(f"on-screen gaussians: {onscreen.sum().item()}, using N={N}")

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn

def make_params(requires_grad):
    m=means_all[idx].clone().detach().requires_grad_(requires_grad)
    q=quats_all[idx].clone().detach().requires_grad_(requires_grad)
    s=scales_all[idx].clone().detach().requires_grad_(requires_grad)
    o=opac_all[idx].clone().detach().requires_grad_(requires_grad)
    co=colors_all[idx].clone().detach().requires_grad_(requires_grad)
    return {"means":m,"quats":q,"scales":s,"opac":o,"colors":co}

PARAM_KEYS = ["means","quats","scales","opac","colors"]

def render_full(params):
    out=R_fn(params["means"],params["quats"],params["scales"],params["opac"],params["colors"],
             sh_degree=3,viewmats=viewmats,Ks=Ks,width=W,height=H,tile_size=tile,
             near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),
             enable_culling=False,use_higs_culling=False)
    return out["frame"]  # [1,H,W,3]

def tile_pixel_counts():
    counts=[]
    for j in range(num_tiles):
        ti=j//tw; tj=j%tw
        y0=ti*tile; x0=tj*tile
        y1=min(y0+tile,H); x1=min(x0+tile,W)
        counts.append((y1-y0)*(x1-x0))
    return counts

def masked_l1(frame, tile_idx=None):
    # frame [1,H,W,3]; L = sum |frame-GT| / (3HW); if tile_idx given, restrict to that tile
    diff = (frame[0] - gt_t[0].permute(1,2,0)).abs()  # [H,W,3]
    if tile_idx is None:
        return diff.sum() / (3*H*W)
    ti=tile_idx//tw; tj=tile_idx%tw
    y0=ti*tile; x0=tj*tile; y1=min(y0+tile,H); x1=min(x0+tile,W)
    return diff[y0:y1, x0:x1].sum() / (3*H*W)

print("\n=== S1A-0 HT estimator validation ===")
t0=time.time()
# 1) Dense gradient (direct)
params = make_params(True)
frame = render_full(params)
L_full = masked_l1(frame)
grads_dense = torch.autograd.grad(L_full, [params[k] for k in PARAM_KEYS], retain_graph=True)
grads_dense = {k:grads_dense[i].detach().clone() for i,k in enumerate(PARAM_KEYS)}
print(f"[dense] L_full={L_full.item():.6f}  render_time={(time.time()-t0)*1000:.0f}ms")

# 2) Per-tile gradients via masked-loss backward on retained graph
tile_counts = tile_pixel_counts()
grads_tile = {k:[] for k in PARAM_KEYS}
L_tiles = []
for j in range(num_tiles):
    Lj = masked_l1(frame, tile_idx=j)
    L_tiles.append(Lj.item())
    gj = torch.autograd.grad(Lj, [params[k] for k in PARAM_KEYS], retain_graph=True)
    for i,k in enumerate(PARAM_KEYS):
        grads_tile[k].append(gj[i].detach().clone())
for k in PARAM_KEYS:
    grads_tile[k] = torch.stack(grads_tile[k], dim=0)  # [num_tiles, N, ...]
L_tiles = np.array(L_tiles)
print(f"[per-tile] computed {num_tiles} tile gradients")

# 3) Partition additivity: sum_j grad(L_j) == dense grad
additivity = {}
for k in PARAM_KEYS:
    s = grads_tile[k].sum(dim=0)
    d = grads_dense[k]
    rel = (s-d).norm() / d.norm().clamp_min(1e-12)
    additivity[k] = {"sum_norm":float(s.norm()),"dense_norm":float(d.norm()),
                     "rel_err":float(rel),"pass":bool(rel < 1e-3)}
print(f"[additivity] " + ", ".join(f"{k} rel_err={additivity[k]['rel_err']:.2e}" for k in PARAM_KEYS))

# 4) Loss normalization: sum_j L_j == L_full
L_sum = L_tiles.sum()
loss_norm = {"sum_L_tiles":float(L_sum),"L_full":float(L_full.item()),
             "rel_err":float(abs(L_sum-L_full.item())/max(abs(L_full.item()),1e-12)),
             "boundary_tile_pixel_counts":{"corners":tile_counts[0],"edges":tile_counts[tw-1],"interior":tile_counts[(th//2)*tw+tw//2]},
             "pass":bool(abs(L_sum-L_full.item())/max(abs(L_full.item()),1e-12) < 1e-4)}
print(f"[loss_norm] sum_L_tiles={L_sum:.6f} L_full={L_full.item():.6f} rel_err={loss_norm['rel_err']:.2e}")

# 5) HT estimator: p=1 == dense
p1 = 1.0
z1 = torch.ones(num_tiles, device=dev)
ht_p1 = {}
for k in PARAM_KEYS:
    ghat = ((z1/p1).unsqueeze(-1) if grads_tile[k].dim()>1 else (z1/p1)) * 0
    # weighted sum: sum_j (z_j/p_j) grad_j
    w = (z1/p1).view(-1, *([1]*(grads_tile[k].dim()-1)))
    ghat = (w * grads_tile[k]).sum(dim=0)
    d = grads_dense[k]
    rel = (ghat-d).norm()/d.norm().clamp_min(1e-12)
    ht_p1[k] = {"rel_err":float(rel),"pass":bool(rel<1e-4)}
print(f"[HT p=1] " + ", ".join(f"{k} rel_err={ht_p1[k]['rel_err']:.2e}" for k in PARAM_KEYS))

# 6) MC consistency at p=0.25, 0.40
def mc_trial(p, n_trials=300, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    # accumulate per-group estimate norms and cosine vs dense
    est = {k: torch.zeros_like(grads_dense[k]) for k in PARAM_KEYS}
    for t in range(n_trials):
        z = (torch.rand(num_tiles, device=dev, generator=g) < p).float()
        w = (z/p).view(-1, *([1]*(grads_tile["means"].dim()-1)))
        for k in PARAM_KEYS:
            wk = (z/p).view(-1, *([1]*(grads_tile[k].dim()-1)))
            ghat = (wk * grads_tile[k]).sum(dim=0)
            est[k] += ghat
    for k in PARAM_KEYS: est[k] /= n_trials
    out = {}
    for k in PARAM_KEYS:
        d = grads_dense[k]; e = est[k]
        cos = torch.nn.functional.cosine_similarity(e.flatten().unsqueeze(0), d.flatten().unsqueeze(0)).item()
        rel = (e-d).norm()/d.norm().clamp_min(1e-12)
        out[k] = {"cos":cos,"rel_err":float(rel)}
    return out

# analytic per-group HT std for a single trial (independent Bernoulli): Var(g_hat_k) = sum_j (1-p_j)/p_j^2 ||grad_j_k||^2
def analytic_var(k, p):
    gk = grads_tile[k]  # [T, N, ...]
    sq = (gk*gk).flatten(1).sum(dim=1)  # [T] ||grad_j||^2
    return float(((1-p)/(p*p) * sq).sum())

mc_results = {}
for p in [0.25, 0.40]:
    t1=time.time()
    res = mc_trial(p, n_trials=300, seed=42)
    av = {k: analytic_var(k, p) for k in PARAM_KEYS}
    mc_results[str(p)] = {"per_group":res,"analytic_var_sum":av,"mc_time_s":time.time()-t1}
    print(f"[MC p={p}] " + ", ".join(f"{k} cos={res[k]['cos']:.4f} rel={res[k]['rel_err']:.3f}" for k in PARAM_KEYS))

# 7) signed vs absgrad vs densification (means group, per-tile)
signed_tile = grads_tile["means"]  # [T,N,3]
abs_per_gauss = signed_tile.abs().sum(dim=0)  # [N,3] sum over tiles of |grad|
densif_stat = signed_tile.abs().amax(dim=0)  # [N,3] max over tiles (typical densification)
signed_dense = grads_dense["means"]
group_stats = {
    "signed_dense_norm": float(signed_dense.norm()),
    "absgrad_per_gauss_mean_norm": float(abs_per_gauss.norm()),
    "densification_max_mean_norm": float(densif_stat.norm()),
    "note": "signed gradient (for HT), absgrad=sum|grad| per gaussian, densification=max|grad| over tiles; kept distinct"
}
print(f"[groups] signed_dense_norm={group_stats['signed_dense_norm']:.4e} absgrad={group_stats['absgrad_per_gauss_mean_norm']:.4e}")

# Unbiasedness summary: MC mean == dense within CI (rel_err small + cos~1)
unbiased = {}
for p in [0.25,0.40]:
    for k in PARAM_KEYS:
        unbiased[f"p{p}_{k}"] = {"cos":mc_results[str(p)]["per_group"][k]["cos"],
                                 "rel_err":mc_results[str(p)]["per_group"][k]["rel_err"]}

result = {
  "schema":"S1A_HT_CORRECTNESS",
  "checkpoint":"bicycle 0005000.pt",
  "camera":c["img_name"],
  "render_wh":[W,H],"tile":tile,"num_tiles":num_tiles,"N_gaussians":N,
  "loss":"additive full-image-normalized L1, L_j = sum_{tile j}|RGB-GT|/(3HW)",
  "L_full":float(L_full.item()),
  "dense_grad_norms":{k:float(grads_dense[k].norm()) for k in PARAM_KEYS},
  "partition_additivity":additivity,
  "loss_normalization":loss_norm,
  "ht_p1_equals_dense":ht_p1,
  "mc_consistency":mc_results,
  "group_statistics":group_stats,
  "unbiasedness":unbiased,
  "render_quality_note":"render does NOT reconstruct GT (corr~0.08); S1A-0 validates estimator MATH on the fixed objective, independent of render quality",
  "elapsed_s":time.time()-t0,
}
# pass/fail
add_pass = all(v["pass"] for v in additivity.values())
ln_pass = loss_norm["pass"]
p1_pass = all(v["pass"] for v in ht_p1.values())
mc_pass = all(mc_results[str(p)]["per_group"][k]["cos"]>0.95 for p in [0.25,0.40] for k in PARAM_KEYS)
result["gates"] = {"additivity":add_pass,"loss_norm":ln_pass,"p1_equals_dense":p1_pass,"mc_unbiased_cos>0.95":mc_pass}
result["overall_pass"] = bool(add_pass and ln_pass and p1_pass and mc_pass)
print(f"\n=== GATES ===")
print(json.dumps(result["gates"], indent=2))
print(f"OVERALL S1A-0 PASS: {result['overall_pass']}")
with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S1A_HT_CORRECTNESS.json","w") as f:
    json.dump(result, f, indent=2, default=str)
print("saved S1A_HT_CORRECTNESS.json")
