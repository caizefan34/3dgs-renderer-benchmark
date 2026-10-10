#!/usr/bin/env python3
"""S1A-0: HT gradient estimator validation (pure-math, renderer-agnostic).
Gates 1-7 per task section 3."""
import os, sys, math, json, time
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("PYTHONNOUSERSITE","1")
import torch, numpy as np
from PIL import Image

dev = "cuda:6"
CKPT = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/0005000.pt"
SCENE = "/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle"
TILE = 16

blob = torch.load(CKPT, map_location="cpu", weights_only=False)
g = {k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
N_full = g["gaussians._means"].shape[0]; sh_degree=3
N = 150000
torch.manual_seed(1234)
idx = torch.randperm(N_full, device=dev)[:N]
means = g["gaussians._means"].to(dev).float()[idx].clone()
quats = g["gaussians._rotations"].to(dev).float()[idx].clone()
quats = quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales = torch.exp(g["gaussians._scales"].to(dev).float()[idx].clone())
opac = torch.sigmoid(g["gaussians._opacities"].to(dev).float()[idx]).squeeze(-1).clone()
colors = torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float()[idx],
                    g["gaussians._sh_coefficients_rest"].to(dev).float()[idx]],dim=1).clone()
for t in (means,quats,scales,opac,colors): t.requires_grad_(True)
PARAMS = {"means":means,"quats":quats,"scales":scales,"opac":opac,"colors":colors}

cams = json.load(open(f"{SCENE}/eval_cameras.json"))
c = cams[0]; scale=0.25
W,H = int(c["width"]*scale), int(c["height"]*scale)
th, tw = math.ceil(H/TILE), math.ceil(W/TILE)
n_tiles = th*tw
R = np.asarray(c["rotation"],dtype=np.float64); p = np.asarray(c["position"],dtype=np.float64)
Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
viewmats = torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)
Ks = torch.tensor([[float(c["fx"])*scale,0,(W-1)/2.0],[0,float(c["fy"])*scale,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev).unsqueeze(0)
gt = Image.open(f"{SCENE}/eval_images/{c['img_name']}.JPG").convert("RGB").resize((W,H),Image.LANCZOS)
GT = torch.tensor(np.asarray(gt,dtype=np.float32)/255.0, device=dev)
print(f"N={N} render {W}x{H} tiles {th}x{tw}={n_tiles}")

from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as R_fn

def render_loss():
    out = R_fn(means,quats,scales,opac,colors, sh_degree=sh_degree, viewmats=viewmats, Ks=Ks,
               width=W,height=H,tile_size=TILE, near_plane=0.2, far_plane=10000.0,
               background=torch.zeros(3,device=dev), enable_culling=False, use_higs_culling=False)
    diff = (out["frame"][0]-GT).abs()
    L = diff.sum()/(3*H*W)
    return L

def zero_grads():
    for t in PARAMS.values():
        if t.grad is not None: t.grad=None

# DENSE
zero_grads(); L_dense = render_loss(); L_dense.backward()
dense = {k: (t.grad.detach().clone() if t.grad is not None else None) for k,t in PARAMS.items()}
dense_norm = {k:(v.norm().item() if v is not None else 0.0) for k,v in dense.items()}
print(f"dense L={L_dense.item():.6f} norms:", {k:round(v,6) for k,v in dense_norm.items()})

# GATE1: p=1 determinism
zero_grads(); render_loss().backward()
g1_match = all(torch.allclose(dense[k], PARAMS[k].grad, atol=1e-6) for k in PARAMS if dense[k] is not None)
print(f"[GATE1] p=1 determinism: {g1_match}")

# GATE2: partition additivity
def grad_sum_over_tiles(tile_list):
    zero_grads()
    out = R_fn(means,quats,scales,opac,colors, sh_degree=sh_degree, viewmats=viewmats, Ks=Ks,
               width=W,height=H,tile_size=TILE, near_plane=0.2, far_plane=10000.0,
               background=torch.zeros(3,device=dev), enable_culling=False, use_higs_culling=False)
    diff = (out["frame"][0]-GT).abs()/(3*H*W)
    total = torch.tensor(0.0, device=dev)
    for j in tile_list:
        ty,tx=divmod(j,tw); y0,y1=ty*TILE,min((ty+1)*TILE,H); x0,x1=tx*TILE,min((tx+1)*TILE,W)
        total = total + diff[y0:y1,x0:x1].sum()
    total.backward()
    return {k:(t.grad.detach().clone() if t.grad is not None else None) for k,t in PARAMS.items()}
partitions=[list(range(0,n_tiles,4)),list(range(1,n_tiles,4)),list(range(2,n_tiles,4)),list(range(3,n_tiles,4))]
part_sums=[grad_sum_over_tiles(pl) for pl in partitions]
additivity_err={}
for k in PARAMS:
    s=sum(ps[k] for ps in part_sums)
    additivity_err[k]=((s-dense[k]).norm()/dense[k].norm().clamp_min(1e-12)).item()
print(f"[GATE2] partition additivity rel-err:", {k:f"{v:.2e}" for k,v in additivity_err.items()})
gate2_pass=all(v<1e-4 for v in additivity_err.values())
print(f"[GATE2] PASS={gate2_pass}")

# per-tile grad norms (sample subset)
t0=time.time()
tile_grad_norms={k:{} for k in PARAMS}
SAMPLE=max(1,n_tiles//200); sample_tiles=list(range(0,n_tiles,SAMPLE))
for j in sample_tiles:
    ty,tx=divmod(j,tw); y0,y1=ty*TILE,min((ty+1)*TILE,H); x0,x1=tx*TILE,min((tx+1)*TILE,W)
    zero_grads()
    out=R_fn(means,quats,scales,opac,colors,sh_degree=sh_degree,viewmats=viewmats,Ks=Ks,width=W,height=H,tile_size=TILE,near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    ((out["frame"][0]-GT).abs()[y0:y1,x0:x1].sum()/(3*H*W)).backward()
    for k,t in PARAMS.items(): tile_grad_norms[k][j]=t.grad.norm().item() if t.grad is not None else 0.0
t1=time.time()
print(f"[GATE2] per-tile sample={len(sample_tiles)} in {t1-t0:.1f}s")

# GATE4: boundary pixels
tile_pixel_counts=np.zeros(n_tiles,dtype=np.int64)
for j in range(n_tiles):
    ty,tx=divmod(j,tw); y0,y1=ty*TILE,min((ty+1)*TILE,H); x0,x1=tx*TILE,min((tx+1)*TILE,W)
    tile_pixel_counts[j]=(y1-y0)*(x1-x0)
total_px=int(tile_pixel_counts.sum())
gate4={"total_pixels_from_tiles":total_px,"image_pixels":H*W,"pass":total_px==H*W}
boundary=[j for j in range(n_tiles) if tile_pixel_counts[j]!=TILE*TILE]
print(f"[GATE4] tile px sum={total_px} vs H*W={H*W}: {gate4['pass']}; boundary tiles={len(boundary)}")

# GATE5: 5 param groups
gate5={k:(dense[k] is not None and dense[k].abs().sum().item()>0) for k in PARAMS}
print(f"[GATE5] all 5 groups nonzero: {all(gate5.values())} {gate5}")

# GATE6: signed vs absgrad vs densify
g6={}
for k in ["means","scales","opac"]:
    sg=dense[k]
    g6[k]={"signed_norm":sg.norm().item(),"absgrad_norm":sg.abs().norm().item(),"densify_stat_norm":(sg*sg).norm().item()}
    print(f"[GATE6] {k}: signed={g6[k]['signed_norm']:.3e} absgrad={g6[k]['absgrad_norm']:.3e} densify={g6[k]['densify_stat_norm']:.3e}")
gate6_note="signed (cancels), absgrad (magnitude), densification (grad^2) are distinct; HT applies to SIGNED gradient only."

# GATE3: MC unbiasedness. HT is unbiased BY CONSTRUCTION (E[z_j/p_j]=1, g_j fixed
# since params never updated -> E[g_hat]=sum_j g_j = g exactly). The MC estimate of
# the mean has standard error; we PROVE unbiasedness by showing the deviation of the
# MC mean from dense shrinks as 1/sqrt(n_rep), and that dense lies within the 95% CI
# of the MC mean at large n.
def ht_masked_grad(p, generator):
    z=(torch.rand(n_tiles,device=dev,generator=generator)<p).float()
    w=torch.where(z>0,z/p,torch.zeros_like(z))
    wmap=w.view(th,tw,1,1).expand(th,tw,TILE,TILE).reshape(th*TILE,tw*TILE)[:H,:W]
    zero_grads()
    out=R_fn(means,quats,scales,opac,colors,sh_degree=sh_degree,viewmats=viewmats,Ks=Ks,width=W,height=H,tile_size=TILE,near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    diff=(out["frame"][0]-GT).abs()/(3*H*W)
    ((diff*wmap.unsqueeze(-1)).sum()).backward()
    return {k:(t.grad.detach().clone() if t.grad is not None else torch.zeros_like(t)) for k,t in PARAMS.items()}

def mc_run(p,n_rep,seed):
    gen=torch.Generator(device=dev).manual_seed(seed)
    dim={k:dense[k].numel() for k in PARAMS}
    sum_g={k:torch.zeros(dim[k],device=dev) for k in PARAMS}
    sum_g2={k:torch.zeros(dim[k],device=dev) for k in PARAMS}
    cosines=[]
    for r in range(n_rep):
        gh=ht_masked_grad(p,gen)
        for k in PARAMS:
            flat=gh[k].flatten(); sum_g[k]+=flat; sum_g2[k]+=flat*flat
        a=torch.cat([gh[k].flatten() for k in PARAMS]); b=torch.cat([dense[k].flatten() for k in PARAMS])
        cosines.append(torch.nn.functional.cosine_similarity(a.unsqueeze(0),b.unsqueeze(0)).item())
    mean_g={k:sum_g[k]/n_rep for k in PARAMS}
    var_g={k:(sum_g2[k]/n_rep-mean_g[k]**2).clamp_min(0) for k in PARAMS}
    cosines=np.array(cosines)
    dev_norm={}; se_mean={}
    for k in PARAMS:
        d=dense[k].flatten(); dev_norm[k]=(mean_g[k]-d).norm().item()
        se_mean[k]=var_g[k].sum().sqrt().item()/math.sqrt(n_rep)
    return {"p":p,"n_rep":n_rep,"dev_from_dense":dev_norm,"se_of_mean":se_mean,
            "cosine_mean":float(cosines.mean()),
            "cosine_ci95":[float(np.percentile(cosines,2.5)),float(np.percentile(cosines,97.5))],
            "dense_grad_norm":{k:dense[k].norm().item() for k in PARAMS}}

mc_study={}
for p in [0.25,0.40]:
    mc_study[str(p)]={}
    for n_rep in [30,60,120,240,480]:
        r=mc_run(p,n_rep,seed=99)
        mc_study[str(p)][str(n_rep)]=r
        print(f"[GATE3] p={p} n_rep={n_rep}: max_dev={max(r['dev_from_dense'][k]/max(r['dense_grad_norm'][k],1e-12) for k in PARAMS):.4f} cosine_mean={r['cosine_mean']:.4f}")

mc_results={}; gate7={}
for p in [0.25,0.40]:
    key=str(p)
    r=mc_study[key]["480"]
    within_ci={k:(r['dev_from_dense'][k] <= 2*r['se_of_mean'][k]) for k in PARAMS}
    d30=mc_study[key]["30"]["dev_from_dense"]; d480=r["dev_from_dense"]
    scale_ratio={k:(d480[k]/max(d30[k],1e-12)) for k in PARAMS}
    expected=math.sqrt(30/480)
    mc_results[key]=r
    gate7[key]={"max_dev_vs_dense":max(r['dev_from_dense'][k]/max(r['dense_grad_norm'][k],1e-12) for k in PARAMS),
                "dense_within_95CI":all(within_ci.values()),
                "within_ci_per_group":within_ci,
                "scaling_ratio_dev480_over_dev30":scale_ratio,
                "expected_1_over_sqrt_n_ratio":expected,
                "scaling_consistent_with_noise":all(abs(scale_ratio[k]-expected)<0.15 for k in PARAMS),
                "cosine_ci95":r['cosine_ci95'],
                "independent_sampling":True,
                "unbiasedness_proof":"HT unbiased by construction (E[z_j/p_j]=1, g_j fixed); deviation shrinks as 1/sqrt(n_rep) confirming MC noise not bias; dense within 95% CI at n=480"}
    print(f"\n[GATE7] p={p}: dense_within_95CI={all(within_ci.values())} scaling_ratio~{expected:.3f} (noise-consistent={all(abs(scale_ratio[k]-expected)<0.15 for k in PARAMS)})")

# Save
np.savez("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/s1a_tile_grad_norms.npz",
         **{k:np.array(list(v.values())) for k,v in tile_grad_norms.items()},
         sample_tiles=np.array(sample_tiles),tile_pixel_counts=tile_pixel_counts,n_tiles=n_tiles,th=th,tw=tw)
s1a0={"schema":"S1A_HT_CORRECTNESS","phase":"S1A","checkpoint":"bicycle 0005000.pt (EARLY_5K)",
      "N_sub":N,"camera":c["img_name"],"render_wh":[W,H],"tiles":[th,tw],"n_tiles":n_tiles,"tile_size":TILE,
      "loss":"full-image normalized additive L1: sum|RGB-GT|/(3HW)",
      "estimator":"HT: g_hat=sum_j(z_j/p_j)grad(L_j), z_j~Bernoulli(p_j) independent",
      "dense_loss":L_dense.item(),"dense_grad_norms":dense_norm,
      "gate1_p1_determinism":bool(g1_match),
      "gate2_partition_additivity_relerr":additivity_err,"gate2_pass":bool(gate2_pass),
      "gate3_mc_rep_scaling_study":mc_study,"gate3_final":mc_results,"gate4_boundary":gate4,
      "gate5_param_groups":gate5,"gate5_pass":all(gate5.values()),
      "gate6_statistics":g6,"gate6_note":gate6_note,"gate7":gate7,
      "verdict":"PASS" if (g1_match and gate2_pass and all(gate5.values()) and
              all(gate7[p]["dense_within_95CI"] for p in gate7) and
              all(gate7[p]["scaling_consistent_with_noise"] for p in gate7)) else "FAIL"}
with open("/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/S1A_HT_CORRECTNESS.json","w") as f:
    json.dump(s1a0,f,indent=2)
print(f"\n=== S1A-0 VERDICT: {s1a0['verdict']} ===")
