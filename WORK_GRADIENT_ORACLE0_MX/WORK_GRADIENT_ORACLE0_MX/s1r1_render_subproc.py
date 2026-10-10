#!/usr/bin/env python3
"""Run paths B (gsplat low-level) and C (HiGS full-tile) each in a subprocess.
Path A (native FasterGS _C) is BLOCKED: ABI mismatch with torch2.9/cu128 (illegal memory access)."""
import os, sys, json, math, subprocess, numpy as np
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
import torch
from PIL import Image
dev = torch.device("cuda:6"); torch.manual_seed(0)
D = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT = f"{D}/s1a_r1_deliverables"; IMGDIR=f"{OUT}/renders"; os.makedirs(IMGDIR,exist_ok=True)
CKDIR = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
CKPTS = {"5K":"0005000.pt","10K":"0010000.pt","15K":"final.pt"}
cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))

WORKER = r'''
import os,sys,json,math
import numpy as np, torch
from PIL import Image
BUILD_DIR="/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0,BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"]="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
dev=torch.device("cuda:6")
which=sys.argv[1]; ckpt=sys.argv[2]; cam_idx=int(sys.argv[3])
blob=torch.load(ckpt,map_location="cpu",weights_only=False)
g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means=g["gaussians._means"].to(dev).float()
quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales=g["gaussians._scales"].to(dev).float().exp()
opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
colors=torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
c=cams[cam_idx]
W,H=396,264; sf=W/float(c["width"])
R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
vm4=torch.tensor(vm,dtype=torch.float32,device=dev)
Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev)
# GT
gt=None
for sub in ("eval_images","images"):
    for ext in (".JPG",".jpg",".png"):
        pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
        if os.path.exists(pp) and os.path.getsize(pp)>1000:
            gt=np.asarray(Image.open(pp).convert("RGB").resize((W,H),Image.LANCZOS),np.float32)/255.0; break
    if gt is not None: break
vms=vm4.unsqueeze(0); Kss=Ks.unsqueeze(0)
meta={"N":int(means.shape[0]),"W":W,"H":H,"scale":sf,"fx":float(c["fx"]*sf),"fy":float(c["fy"]*sf),
      "cx":float((W-1)/2.0),"cy":float((H-1)/2.0),"cam_id":c["id"],"img_name":c["img_name"],
      "w2c":vm.tolist(),"cam_position":p.tolist()}
if which=="B":
    from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode
    from gsplat.rendering import _maybe_evaluate_sh
    from gsplat import rendering as _r
    radii,means2d,depths,conics,_=fully_fused_projection(means=means.contiguous(),covars=None,quats=quats.contiguous(),
        scales=scales.contiguous(),viewmats=vms,Ks=Kss,width=W,height=H,eps2d=0.3,near_plane=0.01,far_plane=1e10,
        radius_clip=0.0,packed=False,calc_compensations=False,camera_model="pinhole")
    opac_bc=opac.unsqueeze(0).contiguous()  # [1,N] for isect_tiles
    tw=math.ceil(W/16); th=math.ceil(H/16)
    _,isect_ids,flatten_ids=isect_tiles(means2d,radii,depths,16,tw,th,packed=False,n_images=1,image_ids=None,gaussian_ids=None,conics=conics,opacities=opac_bc)
    isect_offsets=isect_offset_encode(isect_ids,1,tw,th).reshape((1,1,th,tw))
    colors_eval=_maybe_evaluate_sh(3,colors,means,radii,vms,(1,),1,means.shape[0],True).contiguous()
    bg=torch.zeros((1,3),device=dev)
    opac_rast=opac.unsqueeze(0).contiguous()  # [1,N] for rasterizer
    fn=_r._make_lazy_cuda_func("rasterize_to_pixels_3dgs")
    rc,ra,_,_=fn(means2d.contiguous(),conics.contiguous(),colors_eval.contiguous(),opac_rast.contiguous(),bg,None,W,H,16,isect_offsets.contiguous(),flatten_ids.contiguous(),False,False)
    arr=rc.reshape(1,H,W,3).permute(0,3,1,2).clamp(0,1)[0].cpu().numpy().transpose(1,2,0)
    meta["alpha_mean"]=float(ra.mean()); meta["n_isects"]=int(isect_ids.numel())
elif which=="C":
    from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as higs
    out=higs(means,quats,scales,opac,colors,sh_degree=3,viewmats=vms,Ks=Kss,width=W,height=H,tile_size=16,
             near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    arr=out["frame"].permute(0,3,1,2).clamp(0,1)[0].cpu().numpy().transpose(1,2,0)
    meta["metadata"]={k:(v.item() if isinstance(v,torch.Tensor) and v.numel()==1 else (str(v) if isinstance(v,torch.Tensor) else (v if v is None or isinstance(v,(str,int,float,np.integer)) else float(v)))) for k,v in out["metadata"].items()} if out.get("metadata") else {}
def psnr(a,b):
    mse=(a.astype(np.float64)-b.astype(np.float64)).mean(); return float(-10*np.log10(max(mse,1e-12)))
meta["psnr_vs_gt"]=psnr(arr,gt); meta["mae_vs_gt"]=float(np.abs(arr.astype(np.float64)-gt).mean())
meta["arr_mean"]=float(arr.mean()); meta["arr_std"]=float(arr.std())
tag=os.path.basename(ckpt).replace(".pt","")
Image.fromarray((arr*255).astype(np.uint8)).save(f"{sys.argv[4]}/{tag}_{which}.png")
np.save(f"{sys.argv[4]}/{tag}_{which}.npy", arr.astype(np.float32))
print("META:"+json.dumps(meta))
'''

wpath=f"{D}/s1r1_worker.py"
open(wpath,"w").write(WORKER)
PY="/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python"
env=f"PYTHONNOUSERSITE=1 TMPDIR={D} TORCH_EXTENSIONS_DIR={D}/torch_ext XDG_CACHE_HOME={D}/xdg"
all_results=[]
for ckpt_tag, ckpt_file in CKPTS.items():
    ckpt=f"{CKDIR}/{ckpt_file}"
    for which in ["B","C"]:
        cmd=f"cd {D} && {env} {PY} {wpath} {which} {ckpt} 0 {IMGDIR} 2>&1"
        r=subprocess.run(["bash","-c",cmd],capture_output=True,text=True,timeout=600)
        meta=None
        for line in r.stdout.splitlines():
            if line.startswith("META:"): meta=json.loads(line[5:])
        err = r.stderr[-500:] if r.returncode!=0 else ""
        print(f"[{ckpt_tag} {which}] rc={r.returncode} psnr={meta.get('psnr_vs_gt') if meta else 'NA'} {('ERR:'+err) if err else ''}")
        all_results.append({"ckpt":ckpt_tag,"path":which,"rc":r.returncode,"meta":meta,"stderr":err})
# path A blocked
for ckpt_tag in CKPTS:
    all_results.append({"ckpt":ckpt_tag,"path":"A","rc":-1,"meta":None,
        "stderr":"BLOCKED: native FasterGS _C.forward illegal memory access (ABI mismatch with torch2.9.1+cu128; cannot rebuild per constraints)"})
json.dump(all_results, open(f"{OUT}/S1A_R1_RENDER_THREEWAY.json","w"), indent=2, default=str)
print("saved S1A_R1_RENDER_THREEWAY.json")
