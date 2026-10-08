import sys
sys.path.insert(0, "/root/epic05_n1"); sys.path.insert(0, "/root/epic05_frozen/higs_c0_final30k_worktree"); sys.path.insert(0, "/root/epic05_build_cache/scene/gsplat_scene_cuda"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch, numpy as np, torch.nn as nn
import n1_core as C
import cg0_prereg as P
import cg0_compiler as COM
import cg0_appearance as APP
from cg0_run import render_depth, build_compiled_model
C.bootstrap_c0()
scene="room"; it=15000
ds=C.load_scene_dataset(scene); n=len(ds)
split=P.build_split(n)
ck=C.load_checkpoint(scene,it); Ns=int(ck["xyz"].shape[0]); ss=float(ck["spatial_lr_scale"])
src_shd=int(ck["active_sh_degree"]); src_model=C.build_model(ck)
compile_idx=split["compile64"][:16]
dm={}
for ci in compile_idx:
    cam=ds.get_camera(ci); d,_=render_depth(src_model,cam,src_shd); dm[ci]=d
C.scene_reset()
def gt_get(i): return ds.get_gt_image(i)
cams_c=[ds.get_camera(i) for i in compile_idx]
cand=COM.generate_candidates(dm,cams_c,compile_idx,gt_get,"cuda")
X=cand["X"]; rgb=cand["rgb"]; cidx=cand["cidx"]
# subsample to ~600K for render speed
M=X.shape[0]; keepn=min(M,600000)
sel=torch.randperm(M,device="cuda")[:keepn]
Xs=X[sel]; rgbs=rgb[sel]; cids=cidx[sel]
# DC sh from candidate rgb (originating cam GT)
dc=(rgbs.clamp(0,1)-P.SH_DC_OFFSET)/P.C0
shs=torch.zeros(keepn,16,3,device="cuda"); shs[:,0,:]=dc
op=torch.full((keepn,),float(P.logit(0.5)),device="cuda")
cam=cams_c[0]; gt=gt_get(compile_idx[0]).clamp(0,1)
def corr(x,y):
    x=x-x.mean();y=y-y.mean();return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()+1e-12))
for sc in [0.002, 0.004, 0.008]:
    params={"mu":Xs,"scaling":torch.full((keepn,3),sc,device="cuda"),"rotation":torch.tensor([[1.,0,0],[0,1.,0],[0,0,1.]],device="cuda").unsqueeze(0).expand(keepn,3,3).contiguous(),"opacity_logit":op,"shs":shs}
    m=build_compiled_model(params,0,ss,"cuda")
    img,_,_=C.render_view(m,cam,0); C.scene_reset()
    a=img.clamp(0,1).mean(2).detach().cpu().numpy(); g=gt.mean(2).detach().cpu().numpy()
    print("CANDIDATE render scale=%.3f: mean=%.3f std=%.3f corr=%.4f MAE=%.4f"%(sc,a.mean(),a.std(),corr(a,g),np.abs(a-g).mean()))
