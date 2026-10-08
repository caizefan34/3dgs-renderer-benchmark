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
compile_idx=split["compile64"][:16]
dm={}
for ci in compile_idx:
    cam=ds.get_camera(ci); d,_=render_depth(src_model,cam,src_shd); dm[ci]=d
C.scene_reset()
def gt_get(i): return ds.get_gt_image(i)
cams_c=[ds.get_camera(i) for i in compile_idx]
cand=COM.generate_candidates(dm,cams_c,compile_idx,gt_get,"cuda")
shd_res=APP.compile_sh(cand,None,cams_c,compile_idx,[dm[i] for i in compile_idx],gt_get,"cuda") if False else None
# need clusters for sh; do per-h
cam=cams_c[0]; gt=gt_get(compile_idx[0]).clamp(0,1)
def corr(x,y):
    x=x-x.mean();y=y-y.mean();return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()+1e-12))
X=cand["X"]; nn=cand["n"]
base_order=torch.argsort(cand["cidx"]*100000 + cand["pix"][:,0]*1000 + cand["pix"][:,1], stable=True)
Xb=X[base_order]; nb=nn[base_order]
for h in [0.335, 0.15, 0.08, 0.04, 0.02]:
    key=COM._cluster_ids(Xb,nb,h)
    uk,inv=torch.unique(key,return_inverse=True)
    labels=inv; order=base_order
    clusters=COM.cluster_gaussians(cand,labels,order,h,ss)
    Nc=clusters["n_clust"]
    # recompute sh for these clusters
    shd_res=APP.compile_sh(cand,clusters,cams_c,compile_idx,[dm[i] for i in compile_idx],gt_get,"cuda")
    shs=shd_res["shs"]; op=APP.analytic_opacity(shd_res["n_obs"])
    for f in [0.2]:
        params={"mu":clusters["mu"],"scaling":clusters["scaling"]*f,"rotation":clusters["rotation"],"opacity_logit":op,"shs":shs}
        m=build_compiled_model(params,0,ss,"cuda")
        img,_,_=C.render_view(m,cam,0); C.scene_reset()
        a=img.clamp(0,1).mean(2).detach().cpu().numpy(); g=gt.mean(2).detach().cpu().numpy()
        ev=clusters["scaling"].max(dim=1).values
        print("h=%.3f f=%.2f N=%d maxev_med=%.4f render mean=%.3f std=%.3f corr=%.4f"%(h,f,Nc,float(ev.median()),a.mean(),a.std(),corr(a,g)))
