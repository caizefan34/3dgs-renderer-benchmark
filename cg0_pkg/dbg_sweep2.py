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
shs=shd_res["shs"]; n_obs=shd_res["n_obs"]; op=APP.analytic_opacity(n_obs)
# source render sanity on a calib cam
cam=ds.get_camera(split["calib8"][0]); gt=gt_get(split["calib8"][0])
img,_,_=C.render_view(src_model,cam,src_shd); C.scene_reset()
a=img.clamp(0,1).mean(2).detach().cpu().numpy(); g=gt.clamp(0,1).mean(2).detach().cpu().numpy()
def corr(x,y):
    x=x-x.mean();y=y-y.mean(); return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()+1e-12))
print("SOURCE corr(gt)=%.4f std=%.3f"%(corr(a,g),a.std()))
# scale sweep (deg0)
for f in [0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0]:
    params={"mu":clusters["mu"],"scaling":clusters["scaling"]*f,"rotation":clusters["rotation"],"opacity_logit":op,"shs":shs.clone()}
    shs0=shs.clone(); shs0[:,1:,:]=0.0; params["shs"]=shs0
    m=build_compiled_model(params,0,ss,"cuda")
    img,_,_=C.render_view(m,cam,0); C.scene_reset()
    a=img.clamp(0,1).mean(2).detach().cpu().numpy()
    print("f=%.2f render mean=%.3f std=%.3f corr(gt)=%.4f MAE=%.4f"%(f,a.mean(),a.std(),corr(a,g),np.abs(a-g).mean()))
