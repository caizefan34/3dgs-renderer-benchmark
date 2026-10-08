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
shs=shd_res["shs"]; op_logit=APP.analytic_opacity(shd_res["n_obs"])
cam=ds.get_camera(compile_idx[0]); gt=ds.get_gt_image(compile_idx[0])
params={"mu":clusters["mu"],"scaling":clusters["scaling"],"rotation":clusters["rotation"],"opacity_logit":op_logit,"shs":shs}
m=build_compiled_model(params,3,ss,"cuda")
img,rad,m2d=C.render_view(m,cam,3)
rf=rad.float()
covered=(rf>0).float().mean().item()
print(f"compiled: covered_frac={covered:.3f} rad_mean={rf.mean().item():.1f} rad_p99={torch.quantile(rf,0.99).item():.1f} rad_max={rf.max().item():.0f}")
print(f"compiled img mean={img.mean().item():.3f} std={img.std().item():.3f}")
# source coverage
C.scene_reset()
imgs,rads,_=C.render_view(src_model,cam,src_shd)
rfs=rads.float()
print(f"source: covered_frac={(rfs>0).float().mean().item():.3f} rad_mean={rfs.mean().item():.1f} rad_p99={torch.quantile(rfs,0.99).item():.1f} rad_max={rfs.max().item():.0f}")
print(f"source img mean={imgs.mean().item():.3f} std={imgs.std().item():.3f}")
# correlation
a=img.detach().cpu().numpy().flatten(); g=gt.cpu().numpy().flatten()
print("corr compiled-gt", np.corrcoef(a,g)[0,1])
# per-channel bias
print("compiled-gt mean diff per ch", (img.detach().cpu().mean(0).numpy()-gt.cpu().mean(0).numpy()).tolist())
C.scene_reset()
