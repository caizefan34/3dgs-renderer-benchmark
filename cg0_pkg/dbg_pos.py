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
# check candidate X and cluster mu near source surface
xyz=ck["xyz"].cuda()
X=cand["X"]
# sample 2000 candidates, find nearest source dist via chunked cdist
idx=torch.randperm(X.shape[0])[:2000]
Xs=X[idx]
d2=torch.cdist(Xs, xyz[::20])  # subsample source for speed
nn=d2.min(dim=1).values
print("candidate nearest-src dist (subsampled): median", nn.median().item(), "mean", nn.mean().item(), "p90", torch.quantile(nn,0.9).item())
mu=clusters["mu"]
idx2=torch.randperm(mu.shape[0])[:2000]
mus=mu[idx2]
d2b=torch.cdist(mus, xyz[::20])
nnb=d2b.min(dim=1).values
print("cluster mu nearest-src dist: median", nnb.median().item(), "mean", nnb.mean().item(), "p90", torch.quantile(nnb,0.9).item())
print("source subsample spacing (nn within source):")
d2c=torch.cdist(xyz[::200][:500], xyz[::20])
d2c.fill_diagonal_(1e9)
print("  source nn median", d2c.min(dim=1).values.median().item())
