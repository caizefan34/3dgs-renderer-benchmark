import sys
sys.path.insert(0, "/root/epic05_n1"); sys.path.insert(0, "/root/epic05_frozen/higs_c0_final30k_worktree"); sys.path.insert(0, "/root/epic05_build_cache/scene/gsplat_scene_cuda"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch, numpy as np
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
compile_idx=split["compile64"]
dm={}
for ci in compile_idx:
    cam=ds.get_camera(ci); d,_=render_depth(src_model,cam,src_shd); dm[ci]=d
C.scene_reset()
def gt_get(i): return ds.get_gt_image(i)
cams_c=[ds.get_camera(i) for i in compile_idx]
cand=COM.generate_candidates(dm,cams_c,compile_idx,gt_get,"cuda")
h,cnt,labels,order=COM.fuse_count(cand,Ns,ss)
clusters=COM.cluster_gaussians(cand,labels,order,h,ss)
shd_res=APP.compile_sh(cand,clusters,cams_c,compile_idx,[dm[i] for i in compile_idx],gt_get,"cuda")
shs=shd_res["shs"]
dc=(P.SH_DC_OFFSET+P.C0*shs[:,0,:]).clamp(0,1)  # [N,3]
mu=clusters["mu"]; N=mu.shape[0]
cam=ds.get_camera(split["calib8"][0]); gt=gt_get(split["calib8"][0])
H,W=cam.image_height,cam.image_width
u,v,z,inf=APP._project(cam,mu)
ui=u.round().long().clamp(0,W-1); vi=v.round().long().clamp(0,H-1)
# splat DC color at projected pixel (last write wins)
splat=torch.full((H,W,3),0.0,device="cuda")
splat[vi,ui]=dc
# depth consistency: does projected z match the source depth map of the ORIGINATING camera?
# use calib cam depth from source
cd,_=render_depth(src_model,cam,src_shd); C.scene_reset()
consist=((cd[vi,ui]-z).abs()/z.clamp_min(1e-12)<=0.01)&inf
print("calib-cam depth-consistency for compiled mu: frac=%.3f"%float(consist.float().mean()))
sp=splat.clamp(0,1).mean(2).detach().cpu().numpy(); g=gt.clamp(0,1).mean(2).detach().cpu().numpy()
def corr(x,y):
    x=x-x.mean();y=y-y.mean();return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()+1e-12))
print("SPLAT(no blend) corr(gt)=%.4f  covered_frac=%.3f"%(corr(sp,g),float((splat.sum(2)>0).float().mean())))
# splat only consistent
splat2=torch.full((H,W,3),0.0,device="cuda")
splat2[vi[consist],ui[consist]]=dc[consist]
sp2=splat2.clamp(0,1).mean(2).detach().cpu().numpy()
print("SPLAT(consistent only) corr(gt)=%.4f covered=%.3f"%(corr(sp2,g),float((splat2.sum(2)>0).float().mean())))
