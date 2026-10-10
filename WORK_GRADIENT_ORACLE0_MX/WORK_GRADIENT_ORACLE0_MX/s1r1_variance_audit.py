#!/usr/bin/env python3
"""S1A-R1 Phase 1: HT variance formula audit.

CORRECT HT variance (independent Bernoulli, fixed additive objective):
    E||g_hat - g||^2 = sum_j [(1-p_j)/p_j] ||g_j||^2
Original S1A-1 misused (1-p_j)/p_j^2 (extra 1/p_j factor).

Recomputes per-tile gradients (cached norms), recomputes U/E/G/O analytic NMSE
with BOTH formulas, MC cross-check on means group, saves norms for reuse.
"""
import os, sys, json, math, time
import numpy as np
import torch
from PIL import Image
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
dev = torch.device("cuda:6"); torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
D = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT = f"{D}/s1a_r1_deliverables"
os.makedirs(OUT, exist_ok=True)

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
W, H = 128, 128; tile = 16
th, tw = math.ceil(H/tile), math.ceil(W/tile); num_tiles = th*tw
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

cam_pts = (torch.tensor(Rw2c,dtype=torch.float32,device=dev) @ means_all.T).T + torch.tensor(-Rw2c@p,dtype=torch.float32,device=dev)
z = cam_pts[:,2]; uv = (Ks[0] @ cam_pts.T).T
u = uv[:,0]/uv[:,2].clamp_min(1e-6); v = uv[:,1]/uv[:,2].clamp_min(1e-6)
onscreen = (z>0.2)&(z<10000)&(u>=-8)&(u<W+8)&(v>=-8)&(v<H+8)
idx = onscreen.nonzero(as_tuple=True)[0]
N_TARGET = 120000
if idx.numel() > N_TARGET:
    sel = torch.randperm(idx.numel(), device=dev)[:N_TARGET]; idx = idx[sel]
idx = idx.sort().values; N = idx.numel()

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn
def make_params(rg):
    return {"means":means_all[idx].clone().detach().requires_grad_(True),
            "quats":quats_all[idx].clone().detach().requires_grad_(True),
            "scales":scales_all[idx].clone().detach().requires_grad_(True),
            "opac":opac_all[idx].clone().detach().requires_grad_(True),
            "colors":colors_all[idx].clone().detach().requires_grad_(True)}
PARAM_KEYS = ["means","quats","scales","opac","colors"]
def render(params):
    return R_fn(params["means"],params["quats"],params["scales"],params["opac"],params["colors"],
                sh_degree=3,viewmats=viewmats,Ks=Ks,width=W,height=H,tile_size=tile,
                near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),
                enable_culling=False,use_higs_culling=False)["frame"]
def tile_pixel_count(j):
    ti=j//tw; tj=j%tw; y0=ti*tile; x0=tj*tile
    return (min(y0+tile,H)-y0)*(min(x0+tile,W)-x0)
pixel_counts = np.array([tile_pixel_count(j) for j in range(num_tiles)])

print("=== S1A-R1 Phase 1: HT variance formula audit ===")
t0=time.time()
params = make_params(True)
frame = render(params)
diff = (frame[0]-gt_t[0].permute(1,2,0)).abs()
L_full = diff.sum()/(3*H*W)
g_full = torch.autograd.grad(L_full, [params[k] for k in PARAM_KEYS], retain_graph=True)
g_full = {k:g_full[i].detach().clone() for i,k in enumerate(PARAM_KEYS)}
g_full_norm = {k:float(g_full[k].norm()) for k in PARAM_KEYS}

# per-tile gradient vectors (store means-group vectors for MC; norms for all)
gnorm = {k: np.zeros(num_tiles) for k in PARAM_KEYS}
means_vecs = []  # [num_tiles, N, 3]
for j in range(num_tiles):
    ti=j//tw; tj=j%tw; y0=ti*tile; x0=tj*tile
    Lj = diff[y0:y0+tile, x0:x0+tile].sum()/(3*H*W)
    gj = torch.autograd.grad(Lj, [params[k] for k in PARAM_KEYS], retain_graph=True)
    for i,k in enumerate(PARAM_KEYS):
        gnorm[k][j] = gj[i].norm().item()
    means_vecs.append(gj[0].detach().clone())
    if j % 16 == 0: print(f"  tile {j}/{num_tiles}")
means_vecs = torch.stack(means_vecs, dim=0)  # [T, N, 3]
# cache norms to disk
np.savez(f"{OUT}/pertile_grad_norms_bicycle5K_cam0.npz",
         **{k:gnorm[k] for k in PARAM_KEYS}, pixel_counts=pixel_counts, N=N, num_tiles=num_tiles)
print(f"[cache] saved per-tile norms -> pertile_grad_norms_bicycle5K_cam0.npz  ({time.time()-t0:.0f}s)")

# per-tile L1 error |L_j|
L_tiles = np.zeros(num_tiles)
for j in range(num_tiles):
    ti=j//tw; tj=j%tw; y0=ti*tile; x0=tj*tile
    L_tiles[j] = diff[y0:y0+tile, x0:x0+tile].sum().item()/(3*H*W)

# cost model from S1A-2
C_full, C_p25 = 0.050, 0.044
C_var = (C_full - C_p25)/(1.0-0.25); C_fixed = C_full - C_var
c_j = pixel_counts / pixel_counts.mean()

def normalize_to_budget(w, p):
    w = np.asarray(w, np.float64); pj = p * w / (w.mean()+1e-12)
    return np.clip(pj, 1e-3, 1.0)

def var_correct(pj, gnorm_k):  # (1-p)/p  [CORRECT]
    pj=np.asarray(pj); return float((((1-pj)/pj)*(gnorm_k**2)).sum())
def var_wrong(pj, gnorm_k):    # (1-p)/p^2 [ORIGINAL, WRONG]
    pj=np.asarray(pj); return float((((1-pj)/(pj*pj))*(gnorm_k**2)).sum())

# MC variance cross-check on means group
def mc_var(pj, n_trials=400, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    gmeans = means_vecs  # [T,N,3]
    g_dense = g_full["means"]  # [N,3]
    acc = torch.zeros((), device=dev)
    for _ in range(n_trials):
        z = (torch.rand(num_tiles, device=dev, generator=gen) < torch.tensor(pj,device=dev)).float()
        w = (z/torch.tensor(pj,device=dev)).view(-1,1,1)
        ghat = (w*gmeans).sum(dim=0)
        acc += (ghat-g_dense).norm()**2
    return float(acc/n_trials)

results = {}
for p in [0.25, 0.40]:
    pj_U = np.full(num_tiles, p)
    pj_E = normalize_to_budget(L_tiles, p)
    pj_G = normalize_to_budget(gnorm['means'], p)
    pj_O = normalize_to_budget(np.array([gnorm['means'][j]/math.sqrt(c_j[j]) for j in range(num_tiles)]), p)
    strats = {"U_uniform":pj_U,"E_error":pj_E,"G_grad_only":pj_G,"O_cost_aware":pj_O}
    results[str(p)] = {}
    for sname,pj in strats.items():
        var_c = {k:var_correct(pj,gnorm[k]) for k in PARAM_KEYS}
        var_w = {k:var_wrong(pj,gnorm[k]) for k in PARAM_KEYS}
        nmse_c = {k:var_c[k]/(g_full_norm[k]**2) for k in PARAM_KEYS}
        nmse_w = {k:var_w[k]/(g_full_norm[k]**2) for k in PARAM_KEYS}
        results[str(p)][sname] = {
            "nmse_correct":nmse_c, "nmse_wrong":nmse_w,
            "primary_nmse_correct":nmse_c["means"], "primary_nmse_wrong":nmse_w["means"],
            "pj_summary":{"min":float(pj.min()),"max":float(pj.max()),"mean":float(pj.mean()),"sum":float(pj.sum())}
        }
    # MC cross-check (means group) for U and G
    mc_U = mc_var(pj_U, n_trials=300, seed=1)
    mc_G = mc_var(pj_G, n_trials=300, seed=2)
    results[str(p)]["_mc_variance_means"] = {
        "U_analytic_correct":var_correct(pj_U,gnorm['means']),
        "U_mc":mc_U,
        "G_analytic_correct":var_correct(pj_G,gnorm['means']),
        "G_mc":mc_G,
        "U_analytic_wrong":var_wrong(pj_U,gnorm['means']),
        "G_analytic_wrong":var_wrong(pj_G,gnorm['means']),
    }
    print(f"\n[p={p}] primary NMSE means-group:")
    for sname in ["U_uniform","E_error","G_grad_only","O_cost_aware"]:
        r=results[str(p)][sname]
        print(f"  {sname:14s} correct={r['primary_nmse_correct']:.4f}  wrong={r['primary_nmse_wrong']:.4f}")
    mc=results[str(p)]["_mc_variance_means"]
    print(f"  MC check (means): U analytic_correct={mc['U_analytic_correct']:.4e} mc={mc['U_mc']:.4e} | G analytic_correct={mc['G_analytic_correct']:.4e} mc={mc['G_mc']:.4e}")

# improvement of O/G vs U with CORRECT formula
improvements = {}
for p in [0.25,0.40]:
    base_c = results[str(p)]["U_uniform"]["primary_nmse_correct"]
    base_w = results[str(p)]["U_uniform"]["primary_nmse_wrong"]
    improvements[str(p)] = {}
    for sname in ["E_error","G_grad_only","O_cost_aware"]:
        improvements[str(p)][sname] = {
            "nmse_reduction_correct": 1 - results[str(p)][sname]["primary_nmse_correct"]/base_c,
            "nmse_reduction_wrong": 1 - results[str(p)][sname]["primary_nmse_wrong"]/base_w,
        }
    print(f"\n[p={p}] NMSE reduction vs U (correct vs wrong formula):")
    for sname in ["E_error","G_grad_only","O_cost_aware"]:
        im=improvements[str(p)][sname]
        print(f"  {sname:14s} correct={im['nmse_reduction_correct']*100:+.1f}%  wrong={im['nmse_reduction_wrong']*100:+.1f}%")

result = {
  "schema":"S1A_R1_VARIANCE_ERRATUM",
  "phase":1,
  "formula_correct":"E||g_hat-g||^2 = sum_j [(1-p_j)/p_j] ||g_j||^2",
  "formula_original_wrong":"sum_j [(1-p_j)/p_j^2] ||g_j||^2 (extra 1/p_j factor)",
  "checkpoint":"bicycle 0005000.pt","camera":c["img_name"],"render_wh":[W,H],"num_tiles":num_tiles,"N":N,
  "dense_grad_norms":g_full_norm,
  "cost_model":{"C_fixed_ms":C_fixed,"C_var_ms":C_var,"fixed_overhead_fraction":C_fixed/C_full},
  "results":results,"improvements_vs_U":improvements,
  "mc_cross_check":"means group, 300 trials, analytic vs MC variance",
  "note":"render sanity FAILED; DERIVED estimator-efficiency math; does NOT support GO",
}
json.dump(result, open(f"{OUT}/S1A_R1_VARIANCE_ERRATUM.json","w"), indent=2, default=str)
print(f"\nsaved S1A_R1_VARIANCE_ERRATUM.json  ({time.time()-t0:.0f}s)")
