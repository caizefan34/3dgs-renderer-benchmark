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
# candidates per cluster
cpc=torch.bincount(labels)
print("candidates-per-cluster: mean", cpc.float().mean().item(), "median", cpc.float().median().item(), "max", cpc.max().item())
# normal orientation: cluster normal vs view-to-camera for compile0
cam=ds.get_camera(compile_idx[0])
nrm=clusters["n"]
vdir=APP.F.normalize(clusters["mu"]-cam.camera_center.unsqueeze(0),dim=1,eps=1e-12)
dot=(nrm*(-vdir)).sum(1)
print("compile0 cluster n.(-vdir): mean", dot.mean().item(), "frac>0", (dot>0).float().mean().item(), "frac>0.5", (dot>0.5).float().mean().item())
# project and depth-consistency pass rate for compile0
u,v,z,inf=APP._project(cam, clusters["mu"])
ui=u.round().long().clamp(0,1919); vi=v.round().long().clamp(0,1079)
zi=z
Dpix=dm[compile_idx[0]][vi,ui]
consist=(Dpix-zi).abs()/zi.clamp_min(1e-12)<=0.01
print("compile0 in-frame", int(inf.sum()), "depth-consistent", int(consist.sum()), "pass-rate", (consist[inf].float().mean().item() if inf.any() else 0))

