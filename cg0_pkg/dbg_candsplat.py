import sys
sys.path.insert(0, "/root/epic05_n1"); sys.path.insert(0, "/root/epic05_frozen/higs_c0_final30k_worktree"); sys.path.insert(0, "/root/epic05_build_cache/scene/gsplat_scene_cuda"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch, numpy as np
import n1_core as C
import cg0_prereg as P
import cg0_compiler as COM
import cg0_appearance as APP
from cg0_run import render_depth
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
X=cand["X"]
cam=cams_c[0]; H,W=cam.image_height,cam.image_width
gt=gt_get(compile_idx[0])
u,v,z,inf=APP._project(cam,X)
ui=u.round().long().clamp(0,W-1); vi=v.round().long().clamp(0,H-1)
inframe=inf&(ui>=0)&(ui<W)&(vi>=0)&(vi<H)
# splat cam0 GT at reprojected candidate pixels
splat=torch.full((H,W,3),0.0,device="cuda")
g=gt.clamp(0,1)
splat[vi[inframe],ui[inframe]]=g[vi[inframe],ui[inframe]]
sp=splat.mean(2).detach().cpu().numpy(); gg=g.mean(2).detach().cpu().numpy()
def corr(x,y):
    x=x-x.mean();y=y-y.mean();return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()+1e-12))
print("CANDIDATE splat (cam0 GT at reprojected cand pixels): corr=%.4f covered=%.3f n_cand_inframe=%d"%(corr(sp,gg),float((splat.sum(2)>0).float().mean()),int(inframe.sum())))
# also: depth check - reprojected z vs cam0 rendered depth
cd,_=render_depth(src_model,cam,src_shd); C.scene_reset()
Dp=cd[vi,ui]
consist=((Dp-z).abs()/z.clamp_min(1e-12)<=0.01)&inframe
print("candidate depth-consistency on cam0: frac=%.3f"%float(consist.float().mean()))
