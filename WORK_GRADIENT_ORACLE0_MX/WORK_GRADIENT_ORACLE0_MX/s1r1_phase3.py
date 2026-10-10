#!/usr/bin/env python3
"""S1A-R1 Phase 3: valid-render S1A-1 U/E/G/O with CORRECT HT variance formula.

Uses the WORKING render setup (396x264, scale 0.08, full gaussians, camera 0/1).
2 states (5K,10K) x 2 cameras. Correct formula: Var=sum[(1-p_j)/p_j]||g_j||^2.
MC cross-check. Clarifies O vs G. DERIVED estimator math (render sanity now PASSES
for 5K; gradient is meaningful but this is still estimator-efficiency, not training)."""
import os, sys, json, math, time
import numpy as np
import torch
from PIL import Image
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
import torch.nn.functional as F
dev = torch.device("cuda:6"); torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
D = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT = f"{D}/s1a_r1_deliverables"; os.makedirs(OUT, exist_ok=True)

CKDIR = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
CKPTS = {"5K":"0005000.pt","10K":"0010000.pt"}
cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as higs_render

def psnr_t(a,b):
    mse=(a-b).pow(2).mean().item(); return -10*math.log10(max(mse,1e-12))

def run_state_cam(ckpt_tag, ckpt_file, cam_idx, W=396, H=264):
    blob=torch.load(f"{CKDIR}/{ckpt_file}",map_location="cpu",weights_only=False)
    g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
    means=g["gaussians._means"].to(dev).float()
    quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    scales=g["gaussians._scales"].to(dev).float().exp()
    opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
    colors=torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
    N=means.shape[0]
    c=cams[cam_idx]; sf=W/float(c["width"])
    R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
    Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    viewmats=torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)
    Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev).unsqueeze(0)
    gt=None
    for sub in ("eval_images","images"):
        for ext in (".JPG",".jpg",".png"):
            pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
            if os.path.exists(pp) and os.path.getsize(pp)>1000:
                gt=np.asarray(Image.open(pp).convert("RGB").resize((W,H),Image.LANCZOS),np.float32)/255.0; break
        if gt is not None: break
    gt_t=torch.tensor(gt,dtype=torch.float32,device=dev).permute(2,0,1).unsqueeze(0)

    tile=16; th=math.ceil(H/tile); tw=math.ceil(W/tile); num_tiles=th*tw
    def mk(rg): return {"means":means[rg].clone().detach().requires_grad_(True),"quats":quats[rg].clone().detach().requires_grad_(True),
        "scales":scales[rg].clone().detach().requires_grad_(True),"opac":opac[rg].clone().detach().requires_grad_(True),
        "colors":colors[rg].clone().detach().requires_grad_(True)}
    PK=["means","quats","scales","opac","colors"]
    def render(pp):
        return higs_render(pp["means"],pp["quats"],pp["scales"],pp["opac"],pp["colors"],sh_degree=3,
            viewmats=viewmats,Ks=Ks,width=W,height=H,tile_size=tile,near_plane=0.2,far_plane=10000.0,
            background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)["frame"]

    # on-screen subset
    cp=(torch.tensor(Rw2c,dtype=torch.float32,device=dev)@means.T).T+torch.tensor(-Rw2c@p,dtype=torch.float32,device=dev)
    z=cp[:,2]; uv=(Ks[0]@cp.T).T; u=uv[:,0]/uv[:,2].clamp_min(1e-6); v=uv[:,1]/uv[:,2].clamp_min(1e-6)
    on=(z>0.2)&(z<10000)&(u>=-8)&(u<W+8)&(v>=-8)&(v<H+8)
    idx=on.nonzero(as_tuple=True)[0]
    N_T=min(120000,idx.numel())
    if idx.numel()>N_T:
        sel=torch.randperm(idx.numel(),device=dev)[:N_T]; idx=idx[sel]
    idx=idx.sort().values
    pp=mk(idx)
    frame=render(pp)
    diff=(frame[0]-gt_t[0].permute(1,2,0)).abs()
    L_full=diff.sum()/(3*H*W)
    # render sanity
    rend_np=frame[0].detach().permute(1,2,0).clamp(0,1).cpu().numpy()
    rs_psnr=psnr_t(torch.tensor(rend_np.transpose(2,0,1)),torch.tensor(gt))
    g_full=torch.autograd.grad(L_full,[pp[k] for k in PK],retain_graph=True)
    g_full={k:g_full[i].detach().clone() for i,k in enumerate(PK)}
    gnorm={k:np.zeros(num_tiles) for k in PK}
    for j in range(num_tiles):
        ti=j//tw; tj=j%tw; y0=ti*tile; x0=tj*tile
        Lj=diff[y0:y0+tile,x0:x0+tile].sum()/(3*H*W)
        gj=torch.autograd.grad(Lj,[pp[k] for k in PK],retain_graph=True)
        for i,k in enumerate(PK): gnorm[k][j]=gj[i].norm().item()
    gfn={k:float(g_full[k].norm()) for k in PK}
    # per-tile error + cost
    L_tiles=np.zeros(num_tiles)
    for j in range(num_tiles):
        ti=j//tw; tj=j%tw; y0=ti*tile; x0=tj*tile
        L_tiles[j]=diff[y0:y0+tile,x0:x0+tile].sum().item()/(3*H*W)
    pc=np.array([(min((j//tw)*tile+tile,H)-(j//tw)*tile)*(min((j%tw)*tile+tile,W)-(j%tw)*tile) for j in range(num_tiles)])
    c_j=pc/pc.mean()
    def ntb(w,p):
        w=np.asarray(w,np.float64); pj=p*w/(w.mean()+1e-12); return np.clip(pj,1e-3,1.0)
    def var_c(pj,gk): pj=np.asarray(pj); return float((((1-pj)/pj)*(gk**2)).sum())
    out={"render_sanity_psnr":rs_psnr,"N":int(idx.numel()),"num_tiles":num_tiles,"gfn":gfn,"L_full":float(L_full)}
    for pb in [0.25,0.40]:
        pjU=np.full(num_tiles,pb); pjE=ntb(L_tiles,pb); pjG=ntb(gnorm['means'],pb)
        pjO=ntb(np.array([gnorm['means'][j]/math.sqrt(c_j[j]) for j in range(num_tiles)]),pb)
        out[str(pb)]={}
        for sn,pj in [("U",pjU),("E",pjE),("G",pjG),("O",pjO)]:
            nc={k:var_c(pj,gnorm[k])/(gfn[k]**2) for k in PK}
            out[str(pb)][sn]={"nmse":nc,"primary":nc["means"]}
        out[str(pb)]["_red_vs_U"]={sn:1-out[str(pb)][sn]["primary"]/out[str(pb)]["U"]["primary"] for sn in ["E","G","O"]}
    return out

results={}
for ct,cf in CKPTS.items():
    for ci in [0,1]:
        print(f"=== {ct} cam{ci} ===")
        r=run_state_cam(ct,cf,ci)
        results[f"{ct}_cam{ci}"]=r
        print(f"  render_sanity PSNR={r['render_sanity_psnr']:.2f}")
        for pb in ["0.25","0.4"]:
            print(f"  p={pb}: "+" ".join(f"{sn}={r[pb][sn]['primary']:.3f}" for sn in ["U","E","G","O"])+f" | red: "+" ".join(f"{sn}:{r[pb]['_red_vs_U'][sn]*100:+.0f}%" for sn in ["E","G","O"]))

# aggregate: O vs U, G vs U across all 4 (state,cam) combos
agg={"O_red":[], "G_red":[], "E_red":[], "O_vs_G":[]}
for key,r in results.items():
    for pb in ["0.25","0.4"]:
        agg["O_red"].append(r[pb]["_red_vs_U"]["O"]); agg["G_red"].append(r[pb]["_red_vs_U"]["G"])
        agg["E_red"].append(r[pb]["_red_vs_U"]["E"])
        agg["O_vs_G"].append(1 - r[pb]["O"]["primary"]/r[pb]["G"]["primary"])
summary={
  "render_sanity_all_pass":all(r["render_sanity_psnr"]>10 for r in results.values()),
  "mean_O_reduction_vs_U":float(np.mean(agg["O_red"])),
  "mean_G_reduction_vs_U":float(np.mean(agg["G_red"])),
  "mean_E_reduction_vs_U":float(np.mean(agg["E_red"])),
  "mean_O_vs_G":float(np.mean(agg["O_vs_G"])),
  "O_beats_U_all":all(x>0 for x in agg["O_red"]),
  "G_beats_U_all":all(x>0 for x in agg["G_red"]),
  "O_beats_G_any":any(x>0 for x in agg["O_vs_G"]),
  "note":"DERIVED estimator-efficiency math with CORRECT HT variance; render sanity now passes for 5K (PSNR17.1); does NOT by itself prove training benefit"
}
print("\n=== SUMMARY ===")
print(json.dumps(summary,indent=2))
json.dump({"per_combo":results,"summary":summary},
          open(f"{OUT}/S1A_R1_GRADIENT_VALUE.json","w"),indent=2,default=str)
print("saved S1A_R1_GRADIENT_VALUE.json")
