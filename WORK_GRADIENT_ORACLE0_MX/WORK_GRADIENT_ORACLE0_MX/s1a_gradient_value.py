#!/usr/bin/env python3
"""S1A-1: SUBSET_ORACLE cost-aware gradient analysis (DERIVED estimator math).

Small pre-frozen tile analysis subset. Per-tile real gradient contribution +
per-group norms. Compares U / E / O inclusion strategies at p=0.25, 0.40.
Analytic HT variance (independent Bernoulli + fixed objective).

LABEL: DERIVED. Render sanity FAILED (checkpoint does not reconstruct GT with
available gsplat-HiGS renderer). This is estimator-efficiency math on the fixed
objective; it does NOT establish training value and does NOT support a GO.
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
W, H = 128, 128
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

# on-screen gaussians subset
cam_pts = (torch.tensor(Rw2c,dtype=torch.float32,device=dev) @ means_all.T).T + torch.tensor(-Rw2c@p,dtype=torch.float32,device=dev)
z = cam_pts[:,2]
uv = (Ks[0] @ cam_pts.T).T
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

print("=== S1A-1 SUBSET_ORACLE (DERIVED) ===")
params = make_params(True)
frame = render(params)
diff = (frame[0]-gt_t[0].permute(1,2,0)).abs()  # [H,W,3]
# per-tile L1 error |L_j| (normalized)
L_tiles = np.zeros(num_tiles)
for j in range(num_tiles):
    ti=j//tw; tj=j%tw; y0=ti*tile; x0=tj*tile
    L_tiles[j] = diff[y0:y0+tile, x0:x0+tile].sum().item()/(3*H*W)
# per-tile gradient norms (per group)
gnorm = {k: np.zeros(num_tiles) for k in PARAM_KEYS}
for j in range(num_tiles):
    ti=j//tw; tj=j%tw; y0=ti*tile; x0=tj*tile
    Lj = diff[y0:y0+tile, x0:x0+tile].sum()/(3*H*W)
    gj = torch.autograd.grad(Lj, [params[k] for k in PARAM_KEYS], retain_graph=True)
    for i,k in enumerate(PARAM_KEYS):
        gnorm[k][j] = gj[i].norm().item()
    if j==0:
        L_full = diff.sum()/(3*H*W)
        g_full = torch.autograd.grad(L_full, [params[k] for k in PARAM_KEYS], retain_graph=True)
        g_full_norm = {k:g_full[i].norm().item() for i,k in enumerate(PARAM_KEYS)}
print(f"[subset] N={N}, tiles={num_tiles}, L_full={L_full.item():.6f}")
print(f"[dense grad norms] " + ", ".join(f"{k}={g_full_norm[k]:.3e}" for k in PARAM_KEYS))

# cost model from S1A-2: C(ratio) = C_fixed + C_var*ratio (F+B ms)
# S1A-2 measured: full(1.0)=0.050, p40=0.047, p25=0.044
C_full, C_p40, C_p25 = 0.050, 0.047, 0.044
C_var = (C_full - C_p25)/(1.0-0.25)
C_fixed = C_full - C_var
print(f"[cost model] C_fixed={C_fixed:.4f}ms C_var={C_var:.4f}ms/ratio (fixed overhead {C_fixed/C_full*100:.0f}% of total)")
# per-tile cost proxy c_j: variable cost share proportional to pixel count
c_j = pixel_counts / pixel_counts.mean()

def normalize_to_budget(w, p):
    # p_j = p * w_j / mean(w), clamped to (0,1]
    w = np.asarray(w, np.float64)
    pj = p * w / (w.mean() + 1e-12)
    pj = np.clip(pj, 1e-3, 1.0)
    return pj

def analytic_var(pj, gnorm_k):
    # Var(g_hat) = sum_j (1-p_j)/p_j^2 ||g_j||^2
    pj = np.asarray(pj); return float((((1-pj)/(pj*pj)) * (gnorm_k**2)).sum())

results = {}
for p in [0.25, 0.40]:
    # strategies
    pj_U = np.full(num_tiles, p)
    pj_E = normalize_to_budget(L_tiles, p)            # error-guided
    pj_O = normalize_to_budget(np.array([math.sqrt(gnorm['means'][j]**2/ c_j[j]) for j in range(num_tiles)]), p)  # cost-aware oracle
    pj_G = normalize_to_budget(gnorm['means'], p)      # gradient-only (no cost)
    strats = {"U_uniform":pj_U, "E_error":pj_E, "O_cost_aware":pj_O, "G_grad_only":pj_G}
    results[str(p)] = {}
    for sname, pj in strats.items():
        var = {k: analytic_var(pj, gnorm[k]) for k in PARAM_KEYS}
        # aggregate means-group NMSE as primary; also all groups
        nmse = {k: var[k]/(g_full_norm[k]**2) for k in PARAM_KEYS}
        # primary aggregate (means group, the main geometry group)
        results[str(p)][sname] = {
            "inclusion_probs_summary": {"min":float(pj.min()),"max":float(pj.max()),"mean":float(pj.mean()),"sum":float(pj.sum())},
            "analytic_var":var, "nmse":nmse,
            "primary_nmse_means":nmse["means"],
        }
    # improvement of O vs U (NMSE reduction)
    base = results[str(p)]["U_uniform"]["primary_nmse_means"]
    for sname in ["E_error","O_cost_aware","G_grad_only"]:
        imp = 1 - results[str(p)][sname]["primary_nmse_means"]/base
        results[str(p)][sname]["nmse_reduction_vs_U"] = float(imp)
    print(f"\n[p={p}] primary NMSE (means group):")
    for sname in ["U_uniform","E_error","G_grad_only","O_cost_aware"]:
        r = results[str(p)][sname]
        print(f"  {sname:14s} NMSE={r['primary_nmse_means']:.4f}  pj[min/max]=({r['inclusion_probs_summary']['min']:.3f}/{r['inclusion_probs_summary']['max']:.3f})")

# gradient concentration (DERIVED): what fraction of ||g|| is in top-k tiles?
order = np.argsort(-gnorm['means'])
cum = np.cumsum(gnorm['means'][order]**2) / (gnorm['means']**2).sum()
concentration = {"top_10pct_tiles_capture_grad_sq":float(cum[max(0,int(0.1*num_tiles)-1)]),
                 "top_25pct_tiles_capture_grad_sq":float(cum[max(0,int(0.25*num_tiles)-1)]),
                 "note":"fraction of total ||g_means||^2 captured by top-k tiles (gradient concentration)"}
print(f"\n[concentration] {concentration}")

result = {
  "schema":"S1A_GRADIENT_VALUE",
  "label":"DERIVED",
  "subset_oracle":True,
  "checkpoint":"bicycle 0005000.pt","camera":c["img_name"],
  "render_wh":[W,H],"tile":tile,"num_tiles":num_tiles,"N_gaussians":N,
  "loss":"additive full-image-normalized L1",
  "dense_grad_norms":g_full_norm,
  "cost_model_from_S1A2":{"C_fixed_ms":C_fixed,"C_var_ms_per_ratio":C_var,
                          "fixed_overhead_fraction":C_fixed/C_full,
                          "note":"C(ratio)=C_fixed+C_var*ratio; per-tile cost NOT assumed additive (fixed projection/sort overhead dominates)"},
  "strategies":results,
  "gradient_concentration":concentration,
  "render_quality_note":"render sanity FAILED (corr~0.08); DERIVED estimator-efficiency math only, NOT training value, does NOT support GO",
  "analytic_variance_premise":"independent Bernoulli inclusion + fixed additive objective (holds for HT)",
}
with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S1A_GRADIENT_VALUE.json","w") as f:
    json.dump(result,f,indent=2,default=str)
# CSV
import csv
with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S1A_GRADIENT_VALUE.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["budget","strategy","primary_nmse_means","nmse_reduction_vs_U","pj_min","pj_max","pj_sum"])
    for p in [0.25,0.40]:
        for sname in ["U_uniform","E_error","G_grad_only","O_cost_aware"]:
            r=results[str(p)][sname]
            w.writerow([p,sname,f"{r['primary_nmse_means']:.6f}",
                        f"{r.get('nmse_reduction_vs_U',''):.4f}" if 'nmse_reduction_vs_U' in r else "",
                        f"{r['inclusion_probs_summary']['min']:.4f}",
                        f"{r['inclusion_probs_summary']['max']:.4f}",
                        f"{r['inclusion_probs_summary']['sum']:.2f}"])
print("\nsaved S1A_GRADIENT_VALUE.json/.csv")
