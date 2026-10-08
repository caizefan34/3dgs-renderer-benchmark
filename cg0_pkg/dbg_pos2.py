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
compile_idx=split["compile64"][:8]
dm={}
for ci in compile_idx:
    cam=ds.get_camera(ci); d,_=render_depth(src_model,cam,src_shd); dm[ci]=d
C.scene_reset()
def gt_get(i): return ds.get_gt_image(i)
cams_c=[ds.get_camera(i) for i in compile_idx]
cand=COM.generate_candidates(dm,cams_c,compile_idx,gt_get,"cuda")
X=cand["X"]  # candidate world positions [M,3]
print("M candidates:",X.shape[0])
# source xyz
src_xyz=src_model.get_xyz.detach()
print("src_xyz:",src_xyz.shape, "scene_scale=",ss)
# subsample src for kNN
S=src_xyz[::20]  # ~42K
print("src subsample:",S.shape)
# compute min distance from each candidate to nearest src point (chunked)
d2=[]
for i in range(0,X.shape[0],200000):
    xb=X[i:i+200000]
    # [xb,1,3]-[1,S,3]
    dist=torch.cdist(xb,S)
    d2.append(dist.min(dim=1).values)
dmin=torch.cat(d2)
print("cand->nearest-src dist: median=%.5f mean=%.5f p90=%.5f max=%.5f"%(float(dmin.median()),float(dmin.mean()),float(dmin.kthvalue(int(dmin.numel()*0.9)).values),float(dmin.max())))
# also src->candidate (reverse)
d2r=[]
for i in range(0,S.shape[0],10000):
    sb=S[i:i+10000]
    dist=torch.cdist(sb,X[::7])
    d2r.append(dist.min(dim=1).values)
dminr=torch.cat(d2r)
print("src->nearest-cand dist: median=%.5f mean=%.5f"%(float(dminr.median()),float(dminr.mean())))
