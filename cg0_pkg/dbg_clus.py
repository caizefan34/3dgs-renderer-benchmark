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
h,cnt,labels,order=COM.fuse_count(cand,Ns,ss)
ncl=int(labels.max().item())+1
cidx=cand["cidx"][order]
cam_per=torch.bincount((labels*100000+cidx)//100000, minlength=ncl)
X=cand["X"][order]
cnt2=torch.bincount(labels, minlength=ncl).clamp_min(1).unsqueeze(1)
mu_l=torch.zeros(ncl,3,device="cuda"); mu_l.index_add_(0,labels,X); mu_l=mu_l/cnt2
d2=(X-mu_l[labels]).norm(dim=1)
spread=torch.zeros(ncl,device="cuda"); spread.index_add_(0,labels,d2); spread=spread/cnt2.squeeze(1)
sp=spread.detach().cpu().numpy()
cam=cams_c[0]
u,v,z,inf=APP._project(cam, mu_l)
ui=u.round().long().clamp(0,1919); vi=v.round().long().clamp(0,1079)
Dpix=dm[compile_idx[0]][vi,ui]
consist=((Dpix-z).abs()/z.clamp_min(1e-12)<=0.01)
single=(cam_per==1)
print("SINGLE clusters", int(single.sum()), "consistent", int((consist&single).sum()), "rate", float(consist[single].float().mean()) if single.any() else 0.0)
print("MULTI clusters", int((~single).sum()), "consistent", int((consist&(~single)).sum()), "rate", float(consist[~single].float().mean()) if (~single).any() else 0.0)
print("spread median", float(np.median(sp)), "p90", float(np.quantile(sp,0.9)), "scene_scale", ss)
