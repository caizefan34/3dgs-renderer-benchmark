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
compile_idx=split["compile64"]
dm={}
for ci in compile_idx:
    cam=ds.get_camera(ci); d,_=render_depth(src_model,cam,src_shd); dm[ci]=d
C.scene_reset()
def gt_get(i): return ds.get_gt_image(i)
cams_c=[ds.get_camera(i) for i in compile_idx]
cand=COM.generate_candidates(dm,cams_c,compile_idx,gt_get,"cuda")
# 1. raw candidate depth consistency on ORIGINATING camera
X=cand["X"]; cidx=cand["cidx"]; pix=cand["pix"]
consist_orig=[]
for k,ci in enumerate(compile_idx):
    m=cidx==ci
    if int(m.sum())==0: continue
    cam=cams_c[k]; D=dm[ci]
    u,v,z,inf=APP._project(cam,X[m])
    ui=u.round().long().clamp(0,cam.image_width-1); vi=v.round().long().clamp(0,cam.image_height-1)
    Dp=D[vi,ui]
    c=((Dp-z).abs()/z.clamp_min(1e-12)<=0.01)&inf
    consist_orig.append(float(c.float().mean()))
print("1. raw candidate depth-consistency on ORIGINATING cam: mean=%.3f"%np.mean(consist_orig))
h,cnt,labels,order=COM.fuse_count(cand,Ns,ss)
clusters=COM.cluster_gaussians(cand,labels,order,h,ss)
mu=clusters["mu"]
# 2. cluster mean depth consistency on originating cam (use first compile cam that sees it)
cam0=cams_c[0]; D0=dm[compile_idx[0]]
u,v,z,inf=APP._project(cam0,mu)
ui=u.round().long().clamp(0,cam0.image_width-1); vi=v.round().long().clamp(0,cam0.image_height-1)
Dp=D0[vi,ui]
c0=((Dp-z).abs()/z.clamp_min(1e-12)<=0.01)&inf
print("2. cluster-mean depth-consistency on cam0: frac=%.3f (in-frame frac=%.3f)"%(float(c0.float().mean()),float(inf.float().mean())))
# 3. distance from cluster mean to nearest candidate in same cluster (spread)
Xo=cand["X"][order]; lab=labels
spread=[]
for cid in range(clusters["n_clust"]):
    m=lab==cid
    if int(m.sum())>0:
        d=(Xo[m]-mu[cid]).norm(dim=1)
        spread.append(float(d.median()))
spread=np.array(spread)
print("3. cluster spread (median dist to mean): median=%.4f p90=%.4f"%(np.median(spread),np.percentile(spread,90)))
# 4. same-camera splat: splat DC sampled on cam0 onto cam0
shd_res=APP.compile_sh(cand,clusters,cams_c,compile_idx,[dm[i] for i in compile_idx],gt_get,"cuda")
dc=(P.SH_DC_OFFSET+P.C0*shd_res["shs"][:,0,:]).clamp(0,1)
H,W=cam0.image_height,cam0.image_width
splat=torch.full((H,W,3),0.0,device="cuda")
splat[vi,ui]=dc  # splat at cam0 projection
sp=splat.clamp(0,1).mean(2).detach().cpu().numpy(); g=gt_get(compile_idx[0]).clamp(0,1).mean(2).detach().cpu().numpy()
def corr(x,y):
    x=x-x.mean();y=y-y.mean();return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()+1e-12))
print("4. same-camera splat corr(gt)=%.4f covered=%.3f"%(corr(sp,g),float((splat.sum(2)>0).float().mean())))
