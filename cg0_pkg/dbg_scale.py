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
srce=ck["scaling"].exp(); print("SRC actual scaling mean", srce.mean(0).tolist(), "q", torch.quantile(srce,0.5,dim=0).tolist())
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
for f in [1.0, 0.3, 0.1, 0.03, 0.01]:
    params={"mu":clusters["mu"],"scaling":clusters["scaling"]*f,"rotation":clusters["rotation"],"opacity_logit":op_logit,"shs":shs}
    m=build_compiled_model(params,3,ss,"cuda")
    img,rad,m2d=C.render_view(m,cam,3)
    mse=torch.mean((img-gt)**2); psnr=float(20*torch.log10(1.0/torch.sqrt(mse)))
    print(f"scale*={f} psnr={psnr:.2f} img_mean={img.mean().item():.3f} range[{img.min().item():.3f},{img.max().item():.3f}] max_radius={float(rad.float().max()):.0f}")
    C.scene_reset()


