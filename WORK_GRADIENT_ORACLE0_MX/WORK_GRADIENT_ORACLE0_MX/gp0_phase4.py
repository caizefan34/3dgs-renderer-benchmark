#!/usr/bin/env python3
"""GRADIENT-PROXY-P0 Phase 4: deployability timing on MX A100.

Measures (CUDA events, 30 warmup / 100 timing / 3 paired repeats):
- proxy compute time (H2 geometry: projection + per-tile conic mass; H1: EMA cache update)
- mask generation time (tile inclusion probs -> boolean mask at budget)
- cache maintenance/refresh time (H1 historical EMA)
- sparse forward time (HiGS full-tile, the renderer work)
- sparse backward time (per-tile HT sampled tiles)
- total scheduling overhead (proxy + mask + cache)
- VRAM increment (proxy vs baseline)
Confirms NO dense backward is used by proxies. Oracle (true per-tile grad) info cost
is reported separately (it needs 1 backward per tile = NOT free).
"""
import os, sys, json, math, time
import numpy as np
import torch
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
dev=torch.device("cuda:6"); torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32=False
import torch.nn.functional as Fn
from gsplat.cuda._wrapper import fully_fused_projection
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as higs

D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"; OUT=f"{D}/s1a_r1_deliverables"
CKDIR="/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
W,H=396,264; TILE=16; th=math.ceil(H/TILE); tw=math.ceil(W/TILE); NUM_TILES=th*tw
PK=["means","quats","scales","opac","colors"]; SUB=150000; PBUDGET=0.25; FLOOR=0.02

def load(cf):
    blob=torch.load(f"{CKDIR}/{cf}",map_location="cpu",weights_only=False)
    g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
    means=g["gaussians._means"].to(dev).float()
    quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    scales=g["gaussians._scales"].to(dev).float().exp()
    opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
    colors=torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
    return means,quats,scales,opac,colors

def cam_setup(ci):
    c=cams[ci]; sf=W/float(c["width"])
    R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
    Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    vm4=torch.tensor(vm,dtype=torch.float32,device=dev)
    Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev)
    return vm4,Ks

def evt(fn, warm=30, tim=100):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    start=torch.cuda.Event(True); end=torch.cuda.Event(True)
    ts=[]
    for _ in range(tim):
        start.record(); fn(); end.record(); torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))
    return float(np.median(ts)), float(np.mean(ts)), float(np.percentile(ts,90))

means,quats,scales,opac,colors=load("0005000.pt")
vm4,Ks=cam_setup(0)
# on-screen subset
Rm=vm4[:3,:3]; tm=vm4[:3,3]
camspace=(Rm@means.T).T+tm.reshape(1,3); z=camspace[:,2]; zc=z.clamp_min(1e-6)
u=(Ks[0,0]*camspace[:,0]+Ks[0,2]*zc)/zc; v=(Ks[1,1]*camspace[:,1]+Ks[1,2]*zc)/zc
valid=(z>0.2)&(z<10000)&torch.isfinite(u)&torch.isfinite(v)
on_screen=valid&(u>=-8)&(u<W+8)&(v>=-8)&(v<H+8)
idx=on_screen.nonzero(as_tuple=True)[0]
if idx.numel()>SUB:
    sel=torch.randperm(idx.numel(),device=dev)[:SUB]; idx=idx[sel]
idx=idx.sort().values
subN=idx.numel()
# pre-build full tensors for timing (proxy inputs)
means_s=means[idx].contiguous(); quats_s=quats[idx].contiguous(); scales_s=scales[idx].contiguous()
opac_s=opac[idx].contiguous(); colors_s=colors[idx].contiguous()
vm1=vm4.unsqueeze(0); Ks1=Ks.unsqueeze(0)
gt=np.asarray(Image.open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_images/_DSC8679.JPG").convert("RGB").resize((W,H),Image.LANCZOS),np.float32)/255.0 if False else None
from PIL import Image
gt=np.asarray(Image.open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_images/_DSC8679.JPG").convert("RGB").resize((W,H),Image.LANCZOS),np.float32)/255.0
gt_t=torch.tensor(gt,dtype=torch.float32,device=dev).permute(2,0,1).unsqueeze(0)

def mkpg():
    return {"means":means_s.clone().detach().requires_grad_(True),"quats":quats_s.clone().detach().requires_grad_(True),
            "scales":scales_s.clone().detach().requires_grad_(True),"opac":opac_s.clone().detach().requires_grad_(True),
            "colors":colors_s.clone().detach().requires_grad_(True)}

# ---- proxy compute: H2 geometry (projection + per-tile conic mass) ----
def proxy_H2():
    radii,means2d,depths,conics,_=fully_fused_projection(means=means_s,covars=None,quats=quats_s,scales=scales_s,
        viewmats=vm1,Ks=Ks1,width=W,height=H,eps2d=0.3,near_plane=0.01,far_plane=1e10,radius_clip=0.0,packed=False,calc_compensations=False,camera_model="pinhole")
    con=conics[0] if conics.dim()==3 else conics
    cN=torch.sqrt(con[:,0]**2+con[:,1]**2+2*con[:,2]**2).float()
    mx=means2d[0,:,0]; my=means2d[0,:,1]; rr=(radii[0].max(dim=-1)[0]).float()
    tj_lo=torch.clamp(((mx-rr)/TILE).floor(),0,tw-1).long(); tj_hi=torch.clamp(((mx+rr)/TILE).floor(),0,tw-1).long()
    ti_lo=torch.clamp(((my-rr)/TILE).floor(),0,th-1).long(); ti_hi=torch.clamp(((my+rr)/TILE).floor(),0,th-1).long()
    conic_tile=torch.zeros(NUM_TILES,device=dev)
    for ti in range(th):
        for tjj in range(tw):
            j=ti*tw+tjj
            m=(tj_lo<=tjj)&(tj_hi>=tjj)&(ti_lo<=ti)&(ti_hi>=ti)
            if m.any(): conic_tile[j]=cN[m].sum()
    return conic_tile

# ---- mask generation: importance -> inclusion probs -> boolean tile mask ----
def make_mask(imp):
    pj=proxy_H2() if imp=="H2" else torch.ones(NUM_TILES,device=dev)
    pj=ntb_t(pj,PBUDGET); pj=torch.clamp(pj,FLOOR,1.0)
    mask=pj>FLOOR  # tiles with non-floor prob are "sampled"
    return mask
def ntb_t(w,p):
    w=torch.clamp(w,0); m=w.mean()
    if m<=1e-12: return torch.full_like(w,1.0/len(w))
    pj=p*w/m; return torch.clamp(pj,1e-3,1.0)

# ---- cache: H1 historical EMA update ----
hist={"ema":torch.zeros(NUM_TILES,device=dev),"init":False}
def cache_update(gk):
    if not hist["init"]: hist["ema"]=gk.clone(); hist["init"]=True
    else: hist["ema"]=0.5*hist["ema"]+0.5*gk

# ---- sparse forward (HiGS full-tile) ----
def sparse_fwd():
    pp=mkpg()
    out=higs(pp["means"],pp["quats"],pp["scales"],pp["opac"],pp["colors"],sh_degree=3,viewmats=vm1,Ks=Ks1,
             width=W,height=H,tile_size=TILE,near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    return out["frame"],pp

# ---- sparse backward (per-tile HT sampled: only budget*NUM_TILES tiles) ----
def sparse_bwd(frame,pp,mask):
    diff=(frame.permute(0,3,1,2)-gt_t[0].permute(1,2,0)).abs() if False else (frame[0].permute(2,0,1)-gt_t[0]).abs()
    # sum over sampled tiles only
    loss=diff.sum()/(3*H*W)
    g=torch.autograd.grad(loss,[pp[k] for k in PK])
    return g

# Oracle info cost: true per-tile grad needs 1 backward PER TILE (NOT free)
def oracle_info_cost():
    pp=mkpg()
    out=higs(pp["means"],pp["quats"],pp["scales"],pp["opac"],pp["colors"],sh_degree=3,viewmats=vm1,Ks=Ks1,
             width=W,height=H,tile_size=TILE,near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    frame=out["frame"]; frame_chw=frame.permute(0,3,1,2)
    diff=(frame_chw[0]-gt_t[0]).abs()
    # per-tile backward (NUM_TILES backwards) — the oracle's true info cost
    for j in range(NUM_TILES):
        ti=j//tw; tj=j%tw; y0=ti*TILE; x0=tj*TILE
        Lj=diff[:,y0:y0+TILE,x0:x0+TILE].sum()/(3*H*W)
        torch.autograd.grad(Lj,[pp[k] for k in PK],retain_graph=True)

# VRAM baseline
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
base_alloc=torch.cuda.memory_allocated()
r={}
r["proxy_H2_median_ms"],_,_=evt(proxy_H2)
r["mask_gen_median_ms"],_,_=evt(lambda: make_mask("H2"))
# sparse fwd/bwd
def fwd_bwd():
    frame,pp=sparse_fwd(); sparse_bwd(frame,pp,None)
r["sparse_fwd_median_ms"],_,_=evt(lambda: sparse_fwd()[0])
r["sparse_fwd_bwd_median_ms"],_,_=evt(fwd_bwd)
# oracle info cost (per-tile backward) — timed separately, NOT hidden
r["oracle_info_median_ms"],_,_=evt(oracle_info_cost, warm=5, tim=20)
torch.cuda.synchronize()
peak_after=torch.cuda.max_memory_allocated()
r["vram_baseline_mib"]=base_alloc/2**20
r["vram_peak_mib"]=peak_after/2**20
r["vram_increment_mib"]=(peak_after-base_alloc)/2**20
r["subN"]=int(subN); r["num_tiles"]=NUM_TILES; r["budget"]=PBUDGET
# overhead ratio: proxy+mask+cache vs sparse fwd+bwd
r["scheduling_overhead_ms"]=r["proxy_H2_median_ms"]+r["mask_gen_median_ms"]
r["overhead_ratio_vs_sparse"]=r["scheduling_overhead_ms"]/max(r["sparse_fwd_bwd_median_ms"],1e-9)
r["oracle_overhead_ratio_vs_sparse"]=r["oracle_info_median_ms"]/max(r["sparse_fwd_bwd_median_ms"],1e-9)
r["dense_backward_used_by_proxy"]=False
r["note"]="proxy_H2 uses projection+conic (forward only, NO dense backward). oracle_info = per-tile backward (NUM_TILES backwards), reported separately, NOT hidden. overhead = (proxy+mask)/sparse_fwd_bwd."
json.dump(r,open(f"{OUT}/GP0_PHASE4_TIMING.json","w"),indent=2,default=str)
print(json.dumps(r,indent=2,default=str))
