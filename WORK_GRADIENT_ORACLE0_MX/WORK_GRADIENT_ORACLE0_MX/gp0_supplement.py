#!/usr/bin/env python3
"""GRADIENT-PROXY-P0 supplement: close Phase 1.1 (alpha MAE/max), Phase 4 (VRAM),
and source/binary identity. Alpha rendered per-combo in subprocess (avoid corruption)."""
import os, sys, json, subprocess, hashlib, numpy as np, torch
D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT=f"{D}/gp0_deliverables"
PY="/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python"
env=f"PYTHONNOUSERSITE=1 TMPDIR={D} TORCH_EXTENSIONS_DIR={D}/torch_ext XDG_CACHE_HOME={D}/xdg"
CKDIR="/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
CKPTS={"5K":"0005000.pt","10K":"0010000.pt","15K":"final.pt"}; CAMS=[0,1]; W,H=396,264

# ---- A. source / binary identity (inline, no render) ----
def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536),b""): h.update(b)
    return h.hexdigest()
so_path="/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda/gsplat_scene_cuda.so"
ident={"torch_version":torch.__version__,"cuda_version":torch.version.cuda,
       "python_version":sys.version.split()[0],
       "gsplat_so_path":so_path,"gsplat_so_sha256":sha256_file(so_path) if os.path.exists(so_path) else "MISSING",
       "gsplat_so_bytes":os.path.getsize(so_path) if os.path.exists(so_path) else None}
# git commit of gsplat source if available
for gitdir in ["/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/.git",
               "/home/liaoyuanjun/higs-13scene/.git"]:
    if os.path.isdir(gitdir):
        try:
            r=subprocess.run(["git","--git-dir",gitdir,"rev-parse","HEAD"],capture_output=True,text=True,timeout=10)
            if r.returncode==0: ident["gsplat_git"]=r.stdout.strip(); ident["gsplat_git_dir"]=gitdir; break
        except Exception: pass
# nvidia driver
try:
    r=subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],capture_output=True,text=True,timeout=10)
    ident["nvidia_driver"]=r.stdout.strip().split("\n")[6] if r.returncode==0 else None
except Exception: ident["nvidia_driver"]=None
json.dump(ident,open(f"{OUT}/GP0_SOURCE_IDENTITY.json","w"),indent=2)
print("SOURCE IDENTITY:",json.dumps(ident,indent=2))

# ---- B. alpha MAE/max per combo (subprocess) ----
WORKER=r'''
import os,sys,json,math
import numpy as np, torch
BUILD_DIR="/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0,BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"]="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
dev=torch.device("cuda:6")
from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode
from gsplat.rendering import _maybe_evaluate_sh
from gsplat import rendering as _r
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as higs
CKDIR="/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
CKPTS={"5K":"0005000.pt","10K":"0010000.pt","15K":"final.pt"}
cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
W,H=396,264
which=sys.argv[1]; ct=sys.argv[2]; ci=int(sys.argv[3]); OUTD=sys.argv[4]
blob=torch.load(f"{CKDIR}/{CKPTS[ct]}",map_location="cpu",weights_only=False)
g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means=g["gaussians._means"].to(dev).float()
quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales=g["gaussians._scales"].to(dev).float().exp()
opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
colors=torch.cat([g["gaussians._sh_coefficients_0"].to(dev).float(),g["gaussians._sh_coefficients_rest"].to(dev).float()],dim=1).contiguous()
c=cams[ci]; sf=W/float(c["width"])
R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
vm4=torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)
Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev).unsqueeze(0)
N=means.shape[0]
if which=="B":
    radii,means2d,depths,conics,_=fully_fused_projection(means=means.contiguous(),covars=None,quats=quats.contiguous(),scales=scales.contiguous(),viewmats=vm4,Ks=Ks,width=W,height=H,eps2d=0.3,near_plane=0.01,far_plane=1e10,radius_clip=0.0,packed=False,calc_compensations=False,camera_model="pinhole")
    opac_bc=opac.unsqueeze(0).contiguous(); tw=math.ceil(W/16); th=math.ceil(H/16)
    _,isect_ids,flatten_ids=isect_tiles(means2d,radii,depths,16,tw,th,packed=False,n_images=1,image_ids=None,gaussian_ids=None,conics=conics,opacities=opac_bc)
    isect_offsets=isect_offset_encode(isect_ids,1,tw,th).reshape((1,1,th,tw))
    colors_eval=_maybe_evaluate_sh(3,colors,means,radii,vm4,(1,),1,N,True).contiguous()
    bg=torch.zeros((1,3),device=dev); opac_rast=opac.unsqueeze(0).contiguous()
    fn=_r._make_lazy_cuda_func("rasterize_to_pixels_3dgs")
    rc,ra,_,_=fn(means2d.contiguous(),conics.contiguous(),colors_eval.contiguous(),opac_rast,bg,None,W,H,16,isect_offsets.contiguous(),flatten_ids.contiguous(),False,False)
    alpha=ra.detach().cpu().numpy().reshape(H,W)
else:
    out=higs(means,quats,scales,opac,colors,sh_degree=3,viewmats=vm4,Ks=Ks,width=W,height=H,tile_size=16,near_plane=0.2,far_plane=10000.0,background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    a=out.get("alpha")
    alpha=a.detach().cpu().numpy().reshape(H,W) if a is not None else None
np.save(f"{OUTD}/{ct}_{which}_cam{ci}_alpha.npy", alpha.astype(np.float32))
print("ALPHA_OK "+str(float(alpha.mean()) if alpha is not None else None))
'''
wpath=f"{D}/gp0_alpha_worker.py"; open(wpath,"w").write(WORKER)
alpha={}
for ct in CKPTS:
    for ci in CAMS:
        for which in ["B","C"]:
            cmd=f"cd {D} && {env} {PY} {wpath} {which} {ct} {ci} {OUT} 2>&1 | tail -2"
            r=subprocess.run(["bash","-c",cmd],capture_output=True,text=True,timeout=600)
            ok="ALPHA_OK" in r.stdout
            print(f"[{ct} {which} cam{ci} alpha] rc={r.returncode} ok={ok}",flush=True)
# aggregate alpha B vs C
for ct in CKPTS:
    for ci in CAMS:
        try:
            b=np.load(f"{OUT}/{ct}_B_cam{ci}_alpha.npy"); c=np.load(f"{OUT}/{ct}_C_cam{ci}_alpha.npy")
            alpha[f"{ct}_cam{ci}"]={"alpha_mae":float(np.abs(b.astype(np.float64)-c.astype(np.float64)).mean()),
                "alpha_maxerr":float(np.abs(b.astype(np.float64)-c.astype(np.float64)).max()),
                "B_alpha_mean":float(b.mean()),"C_alpha_mean":float(c.mean()),
                "B_alpha_std":float(b.std()),"C_alpha_std":float(c.std())}
        except Exception as e: alpha[f"{ct}_cam{ci}"]=f"ERR:{e}"
# merge into Phase1 JSON
p1=json.load(open(f"{OUT}/GP0_PHASE1_CONSISTENCY.json"))
p1["alpha_B_vs_C"]=alpha
p1["source_identity_ref"]="GP0_SOURCE_IDENTITY.json"
json.dump(p1,open(f"{OUT}/GP0_PHASE1_CONSISTENCY.json","w"),indent=2,default=str)
print("ALPHA B vs C:",json.dumps(alpha,indent=2))

# ---- C. VRAM increment (Phase 4) ----
VRAM=r'''
import os,sys,json,math
import numpy as np, torch
BUILD_DIR="/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0,BUILD_DIR)
os.environ["TORCH_EXTENSIONS_DIR"]="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
dev=torch.device("cuda:6")
from gsplat.cuda._wrapper import fully_fused_projection
CKDIR="/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
blob=torch.load(f"{CKDIR}/0005000.pt",map_location="cpu",weights_only=False)
g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
means=g["gaussians._means"].to(dev).float()
quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales=g["gaussians._scales"].to(dev).float().exp()
opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
import json as J
cams=J.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
c=cams[0]; W,H=396,264; sf=W/float(c["width"])
R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64); Rw2c=R.T
vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
vm4=torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)
Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev).unsqueeze(0)
Rm=vm4[0,:3,:3]; tm=vm4[0,:3,3]; cs=(Rm@means.T).T+tm.reshape(1,3); z=cs[:,2]; zc=z.clamp_min(1e-6)
u=(Ks[0,0,0]*cs[:,0]+Ks[0,0,2]*zc)/zc; v=(Ks[0,1,1]*cs[:,1]+Ks[0,1,2]*zc)/zc
valid=(z>0.2)&(z<10000)&torch.isfinite(u)&torch.isfinite(v)
on_screen=valid&(u>=-8)&(u<W+8)&(v>=-8)&(v<H+8)
idx=on_screen.nonzero(as_tuple=True)[0]
subN=min(150000,idx.numel()); idx=idx[:subN]
ms=means[idx].contiguous(); qs=quats[idx].contiguous(); ss=scales[idx].contiguous()
torch.cuda.set_device(6)
torch.cuda.reset_peak_memory_stats(6); torch.cuda.synchronize(6)
base=torch.cuda.memory_allocated(6)
radii,means2d,depths,conics,_=fully_fused_projection(means=ms,covars=None,quats=qs,scales=ss,viewmats=vm4,Ks=Ks,width=W,height=H,eps2d=0.3,near_plane=0.01,far_plane=1e10,radius_clip=0.0,packed=False,calc_compensations=False,camera_model="pinhole")
torch.cuda.synchronize(6)
peak=torch.cuda.max_memory_allocated(6)
print("VRAM_RESULT "+J.dumps({"subN":int(subN),"base_alloc_mib":base/2**20,"peak_alloc_mib":peak/2**20,"increment_mib":(peak-base)/2**20,"increment_pct":100*(peak-base)/max(base,1)}))
'''
vpath=f"{D}/gp0_vram_worker.py"; open(vpath,"w").write(VRAM)
cmd=f"cd {D} && {env} {PY} {vpath} 2>&1 | tail -3"
r=subprocess.run(["bash","-c",cmd],capture_output=True,text=True,timeout=600)
vram=None
for line in r.stdout.splitlines():
    if line.startswith("VRAM_RESULT"): vram=json.loads(line.split(" ",1)[1])
if vram:
    p4=json.load(open(f"{OUT}/GP0_PHASE4_TIMING.json"))
    p4["vram"]=vram
    json.dump(p4,open(f"{OUT}/GP0_PHASE4_TIMING.json","w"),indent=2,default=str)
    print("VRAM:",json.dumps(vram,indent=2))
else:
    print("VRAM FAILED:",r.stderr[-400:])
print("SUPPLEMENT DONE")
