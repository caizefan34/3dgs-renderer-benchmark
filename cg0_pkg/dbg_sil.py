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
cam=ds.get_camera(compile_idx[0]); gt=ds.get_gt_image(compile_idx[0])
# white silhouette: tiny uniform scale, white DC, high opacity
N=clusters["n_clust"]
white_dc=(1.0-P.SH_DC_OFFSET)/P.C0
shs_white=torch.zeros(N,16,3,device="cuda"); shs_white[:,0,:]=white_dc
params={"mu":clusters["mu"],"scaling":torch.full((N,3),0.02,device="cuda"),"rotation":clusters["rotation"],"opacity_logit":torch.full((N,),3.0,device="cuda"),"shs":shs_white}
m=build_compiled_model(params,0,ss,"cuda")
img,_,_=C.render_view(m,cam,0)
C.scene_reset()
# source silhouette
imgs,_,_=C.render_view(src_model,cam,src_shd)
sil_src=(imgs.mean(-1)>0.4).float().cpu().numpy()
sil_cmp=(img.mean(-1)>0.4).float().cpu().numpy()
inter=np.logical_and(sil_src>0,sil_cmp>0).sum(); union=np.logical_or(sil_src>0,sil_cmp>0).sum()
iou=inter/union
print(f"silhouette IoU={iou:.3f} src_cov={sil_src.mean():.3f} cmp_cov={sil_cmp.mean():.3f}")
# point-level corr of silhouettes
a=sil_cmp.flatten(); b=sil_src.flatten()
print("silhouette corr", np.corrcoef(a,b)[0,1])
# also depth-based: render compiled with cluster mean depth color? skip
