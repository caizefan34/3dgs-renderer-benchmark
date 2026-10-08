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
src_shd=int(ck["active_sh_degree"])
compile_idx=split["compile64"]
dm={}
for ci in compile_idx:
    cam=ds.get_camera(ci); d,_=render_depth(C.build_model(ck),cam,src_shd); dm[ci]=d
C.scene_reset()
def gt_get(i): return ds.get_gt_image(i)
cams_c=[ds.get_camera(i) for i in compile_idx]
cand=COM.generate_candidates(dm,cams_c,compile_idx,gt_get,"cuda")
h,cnt,labels,order=COM.fuse_count(cand,Ns,ss)
clusters=COM.cluster_gaussians(cand,labels,order,h,ss)
shd_res=APP.compile_sh(cand,clusters,cams_c,compile_idx,[dm[i] for i in compile_idx],gt_get,"cuda")
shs=shd_res["shs"]; n_obs=shd_res["n_obs"]
op=APP.analytic_opacity(n_obs)
print("N_clust",clusters["n_clust"])
print("n_obs: frac0=%.3f median=%.1f mean=%.2f"%(float((n_obs==0).float().mean()), float(n_obs.median()), float(n_obs.mean())))
dc=(P.SH_DC_OFFSET+P.C0*shs[:,0,:]).clamp(0,1)
print("DC rgb: mean=%.3f std=%.3f min=%.3f max=%.3f"%(float(dc.mean()),float(dc.std()),float(dc.min()),float(dc.max())))
ev=clusters["scaling"].max(dim=1).values
print("scale max-ev: median=%.4f p90=%.4f max=%.4f"%(float(ev.median()),float(ev.kthvalue(int(ev.numel()*0.9)).values),float(ev.max())))
print("opacity: mean=%.3f min=%.3f max=%.3f frac>0.9=%.3f"%(float(op.sigmoid().mean()),float(op.sigmoid().min()),float(op.sigmoid().max()),float((op.sigmoid()>0.9).float().mean())))
# render DC-only vs full on calib camera
cam=ds.get_camera(split["calib8"][0]); gt=gt_get(split["calib8"][0])
for deg,sh in [(0,shs.clone()),(3,shs)]:
    sh2=sh.clone()
    if deg==0:
        sh2[:,1:,:]=0.0
    params={"mu":clusters["mu"],"scaling":clusters["scaling"],"rotation":clusters["rotation"],"opacity_logit":op,"shs":sh2}
    m=build_compiled_model(params,deg,ss,"cuda")
    img,_,_=C.render_view(m,cam,deg); C.scene_reset()
    a=img.clamp(0,1).mean(2).detach().cpu().numpy(); g=gt.clamp(0,1).mean(2).detach().cpu().numpy()
    def corr(x,y):
        x=x-x.mean();y=y-y.mean(); return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()+1e-12))
    print("deg=%d render mean=%.3f std=%.3f corr(gt)=%.4f MAE=%.4f"%(deg,a.mean(),a.std(),corr(a,g),np.abs(a-g).mean()))
