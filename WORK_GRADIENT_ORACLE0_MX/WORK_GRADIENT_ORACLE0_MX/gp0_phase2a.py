#!/usr/bin/env python3
"""GRADIENT-PROXY-P0 Phase 2a: compute per-tile TRUE gradient (HiGS path C) + H2
geometry-Jacobian forward quantities, for 9 combos (5K/10K/15K x cam0/1/2).

H2 uses ONLY forward info (gsplat projection for tile membership + opacity mass,
HiGS render for residual) — NO dense backward. True per-tile grad uses HiGS per-tile
backward (offline oracle G). Saves pertile stats to disk for Phase 3.
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
dev = torch.device("cuda:6"); torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
import torch.nn.functional as Fn
D = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT = f"{D}/s1a_r1_deliverables"

CKDIR = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
CKPTS = {"5K":"0005000.pt","10K":"0010000.pt","15K":"final.pt"}
cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
from gsplat.cuda._wrapper import fully_fused_projection
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as higs

W,H=396,264; TILE=16; th=math.ceil(H/TILE); tw=math.ceil(W/TILE); NUM_TILES=th*tw
PK=["means","quats","scales","opac","colors"]
SUB=150000  # cap for per-tile backward (memory)

def load(ckpt_file):
    blob=torch.load(f"{CKDIR}/{ckpt_file}",map_location="cpu",weights_only=False)
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
    return c,sf,vm4,Ks,vm,p

def get_gt(ci):
    c=cams[ci]
    for sub in ("eval_images","images"):
        for ext in (".JPG",".jpg",".png"):
            pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
            if os.path.exists(pp) and os.path.getsize(pp)>1000:
                return np.asarray(Image.open(pp).convert("RGB").resize((W,H),Image.LANCZOS),np.float32)/255.0, pp
    return None,None

def compute(ci):
    c,sf,vm4,Ks,vm,p=cam_setup(ci)
    gt,gt_path=get_gt(ci)
    gt_t=torch.tensor(gt,dtype=torch.float32,device=dev).permute(2,0,1).unsqueeze(0)
    R=vm4[:3,:3].clone(); t=vm4[:3,3].clone()
    return c,sf,vm4,Ks,vm,p,gt,gt_path,gt_t

def run_combo(ct, cf, ci):
    t0=time.time()
    means,quats,scales,opac,colors=load(cf)
    N=means.shape[0]
    c,sf,vm4,Ks,vm,p,gt,gt_path,gt_t=compute(ci)
    # visible subset
    cp=(R@means.T).T+t.reshape(1,3) if False else None
    Rm=vm4[:3,:3]; tm=vm4[:3,3]
    camspace=(Rm@means.T).T+tm.reshape(1,3)
    z=camspace[:,2]; zc=z.clamp_min(1e-6)
    u=(Ks[0,0]*camspace[:,0]+Ks[0,2]*zc)/zc
    v=(Ks[1,1]*camspace[:,1]+Ks[1,2]*zc)/zc
    valid=(z>0.2)&(z<10000)&torch.isfinite(u)&torch.isfinite(v)
    on_screen=valid&(u>=-8)&(u<W+8)&(v>=-8)&(v<H+8)
    vis_op=on_screen&(opac>0.01)
    idx=vis_op.nonzero(as_tuple=True)[0]
    if idx.numel()>SUB:
        sel=torch.randperm(idx.numel(),device=dev)[:SUB]; idx=idx[sel]
    idx=idx.sort().values
    subN=idx.numel()
    # H2 geometry: forward projection on FULL on-screen set (no backward)
    opac_full=opac[on_screen]
    radii,means2d,depths,conics,_=fully_fused_projection(means=means[on_screen].contiguous(),covars=None,
        quats=quats[on_screen].contiguous(),scales=scales[on_screen].contiguous(),viewmats=vm4.unsqueeze(0),Ks=Ks.unsqueeze(0),
        width=W,height=H,eps2d=0.3,near_plane=0.01,far_plane=1e10,radius_clip=0.0,packed=False,calc_compensations=False,camera_model="pinhole")
    print(f"  proj shapes means2d={tuple(means2d.shape)} radii={tuple(radii.shape)}",flush=True)
    m2d=means2d[0]; rad=radii[0]
    if rad.dim()>1: rad=rad.max(dim=-1)[0]  # [N,2]->[N]
    mx=m2d[:,0]; my=m2d[:,1]; rr=rad.float()
    tj_lo=torch.clamp(((mx-rr)/TILE).floor(),0,tw-1).long(); tj_hi=torch.clamp(((mx+rr)/TILE).floor(),0,tw-1).long()
    ti_lo=torch.clamp(((my-rr)/TILE).floor(),0,th-1).long(); ti_hi=torch.clamp(((my+rr)/TILE).floor(),0,th-1).long()
    opac_tile=torch.zeros(NUM_TILES,device=dev); opac2_tile=torch.zeros(NUM_TILES,device=dev); cnt_tile=torch.zeros(NUM_TILES,device=dev)
    conic_tile=torch.zeros(NUM_TILES,device=dev)
    of=opac_full
    # conic: [C,N,3] packed (conic_x, conic_y, conic_xy). per-gaussian conic norm = sqrt(a^2+b^2+2*ab^2/...)
    con=conics[0] if conics.dim()==3 else conics  # [N,3]
    cN=torch.sqrt(con[:,0]**2+con[:,1]**2+2*con[:,2]**2).float()  # per-gaussian footprint sharpness
    for ti in range(th):
        for tjj in range(tw):
            j=ti*tw+tjj
            m=(tj_lo<=tjj)&(tj_hi>=tjj)&(ti_lo<=ti)&(ti_hi>=ti)
            if m.any():
                opac_tile[j]=of[m].sum(); opac2_tile[j]=(of[m]**2).sum(); cnt_tile[j]=m.sum()
                conic_tile[j]=cN[m].sum()
    # HiGS full render (for residual) on subset
    def mkpg():
        return {"means":means[idx].clone().detach().requires_grad_(True),"quats":quats[idx].clone().detach().requires_grad_(True),
                "scales":scales[idx].clone().detach().requires_grad_(True),"opac":opac[idx].clone().detach().requires_grad_(True),
                "colors":colors[idx].clone().detach().requires_grad_(True)}
    pp=mkpg()
    out=higs(pp["means"],pp["quats"],pp["scales"],pp["opac"],pp["colors"],sh_degree=3,viewmats=vm4.unsqueeze(0),Ks=Ks.unsqueeze(0),
             width=W,height=H,tile_size=TILE,near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    frame=out["frame"]  # [C,H,W,3]
    frame_chw=frame.permute(0,3,1,2)  # [C,3,H,W]
    with torch.no_grad():
        rend=frame[0].detach().clamp(0,1).cpu().numpy()  # [H,W,3]
    # per-tile residual (mean |render-GT|) and per-tile TRUE gradient (HiGS backward)
    diff=(frame_chw[0]-gt_t[0]).abs()  # [3,H,W]
    residual_tile=torch.zeros(NUM_TILES,device=dev)
    for j in range(NUM_TILES):
        ti=j//tw; tj=j%tw; y0=ti*TILE; x0=tj*TILE
        residual_tile[j]=diff[:,y0:y0+TILE,x0:x0+TILE].sum()/(3*TILE*TILE)
    residual_tile=residual_tile.detach().cpu().numpy().astype(np.float32)
    gnorm={k:np.zeros(NUM_TILES) for k in PK}
    for j in range(NUM_TILES):
        ti=j//tw; tj=j%tw; y0=ti*TILE; x0=tj*TILE
        Lj=diff[:,y0:y0+TILE,x0:x0+TILE].sum()/(3*H*W)
        gj=torch.autograd.grad(Lj,[pp[k] for k in PK],retain_graph=True)
        for i,k in enumerate(PK): gnorm[k][j]=gj[i].norm().item()
    gfn={k:float(torch.autograd.grad(diff.sum()/(3*H*W),[pp[k] for k in PK],retain_graph=True)[i].norm()) for i,k in enumerate(PK)}
    rec={"ckpt":ct,"cam":ci,"N_total":int(N),"subN":int(subN),"on_screen_full":int(on_screen.sum()),
         "img_name":c["img_name"],"num_iters":None,"gt_path":gt_path,
         "render_mean":float(rend.mean()),"render_std":float(rend.std()),
         "residual_tile":residual_tile,
         "opac_tile":opac_tile.cpu().numpy().astype(np.float32),
         "opac2_tile":opac2_tile.cpu().numpy().astype(np.float32),
         "cnt_tile":cnt_tile.cpu().numpy().astype(np.float32),
         "conic_tile":conic_tile.cpu().numpy().astype(np.float32),
         "gnorm_means":gnorm["means"].astype(np.float32),"gnorm_quats":gnorm["quats"].astype(np.float32),
         "gnorm_scales":gnorm["scales"].astype(np.float32),"gnorm_opac":gnorm["opac"].astype(np.float32),
         "gnorm_colors":gnorm["colors"].astype(np.float32),
         "gfn":gfn,"seconds":time.time()-t0}
    return rec

allrec={}
for ct in CKPTS:
    for ci in [0,1,2]:
        print(f"=== {ct} cam{ci} ===",flush=True)
        r=run_combo(ct,CKPTS[ct],ci)
        allrec[f"{ct}_cam{ci}"]=r
        np.savez(f"{OUT}/gp0_pertile_{ct}_cam{ci}.npz",
                 residual_tile=r["residual_tile"],opac_tile=r["opac_tile"],opac2_tile=r["opac2_tile"],
                 cnt_tile=r["cnt_tile"],conic_tile=r["conic_tile"],
                 gnorm_means=r["gnorm_means"],gnorm_quats=r["gnorm_quats"],
                 gnorm_scales=r["gnorm_scales"],gnorm_opac=r["gnorm_opac"],gnorm_colors=r["gnorm_colors"])
        print(f"  subN={r['subN']} on_screen_full={r['on_screen_full']} gfn_means={r['gfn']['means']:.3f} t={r['seconds']:.1f}s",flush=True)
# save metadata (everything except arrays)
meta={k:{kk:vv for kk,vv in v.items() if not isinstance(vv,np.ndarray)} for k,v in allrec.items()}
json.dump(meta,open(f"{OUT}/gp0_pertile_meta.json","w"),indent=2,default=str)
print("saved gp0_pertile_meta.json + 9 npz")
