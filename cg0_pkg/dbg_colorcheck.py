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
dc=(P.SH_DC_OFFSET+P.C0*shs[:,0,:]).clamp(0,1)
cam=ds.get_camera(compile_idx[0]); gt=gt_get(compile_idx[0])
# project cluster centers into compile0
u,v,z,inf=APP._project(cam, clusters["mu"])
ui=u.round().long().clamp(0,1919); vi=v.round().long().clamp(0,1079)
sel=inf.nonzero(as_tuple=False).reshape(-1)
gtpix=gt[vi[sel], ui[sel]]
dcc=dc[sel]
diff=(dcc-gtpix).abs().mean(1)
print("in-frame clusters:", sel.numel(), "of", clusters["n_clust"])
print("DC vs GT-at-projection: mean_abs_diff", diff.mean().item(), "median", diff.median().item(), "frac<0.1", (diff<0.1).float().mean().item())
# also: how many compile0's own candidates are depth-consistent (sanity)
print("compile0 depth mean", dm[compile_idx[0]].mean().item())
