import sys, math, time
sys.path.insert(0, "/root/epic05_n1")
sys.path.insert(0, "/root/epic05_frozen/higs_c0_final30k_worktree")
sys.path.insert(0, "/root/epic05_build_cache/scene/gsplat_scene_cuda")
sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7")
sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch, numpy as np
import n1_core as C
import cg0_prereg as P
import cg0_compiler as COM
import cg0_appearance as APP
from cg0_run import render_depth, build_compiled_model, eval_set
C.bootstrap_c0()
scene="room"; it=15000
ds=C.load_scene_dataset(scene); n=len(ds)
split=P.build_split(n)
ck=C.load_checkpoint(scene,it); Ns=int(ck["xyz"].shape[0]); ss=float(ck["spatial_lr_scale"])
src_shd=int(ck["active_sh_degree"]); src_model=C.build_model(ck); C.scene_reset()
compile_idx=split["compile64"]
dm={}
for ci in compile_idx:
    cam=ds.get_camera(ci); d,_=render_depth(src_model,cam,src_shd); dm[ci]=d
C.scene_reset()
def gt_get(i): return ds.get_gt_image(i)
cams_c=[ds.get_camera(i) for i in compile_idx]
t0=time.perf_counter()
cand=COM.generate_candidates(dm,cams_c,compile_idx,gt_get,"cuda")
print("M candidates", cand["M"], "gen_s", round(time.perf_counter()-t0,2))
h,cnt,labels,order=COM.fuse_count(cand,Ns,ss)
clusters=COM.cluster_gaussians(cand,labels,order,h,ss)
print("N_compiled", clusters["n_clust"], "h", h, "count_ratio", clusters["n_clust"]/Ns)
shd_res=APP.compile_sh(cand,clusters,cams_c,compile_idx,[dm[i] for i in compile_idx],gt_get,"cuda")
shs=shd_res["shs"]
print("sh degree hist", torch.bincount(shd_res["degree"]).cpu().tolist(), "fallback_frac", float(shd_res["fallback"].float().mean()))
op_logit=APP.analytic_opacity(shd_res["n_obs"])
print("opacity_logit mean", op_logit.mean().item(), "min", op_logit.min().item(), "max", op_logit.max().item())
# pick s=1,b=0 (no calib) to test raw
scaling=clusters["scaling"]; rotation=clusters["rotation"]; op=op_logit
params={"mu":clusters["mu"],"scaling":scaling,"rotation":rotation,"opacity_logit":op,"shs":shs}
m0=build_compiled_model(params,3,ss,"cuda")
# render on compile cam, calib cam, trainval cam, holdout cam
for tag,idx in [("compile0",compile_idx[0]),("calib0",split["calib8"][0]),("trainval0",split["trainval16"][0]),("holdout0",split["holdout16"][0])]:
    cam=ds.get_camera(idx); gt=ds.get_gt_image(idx)
    img,rad,m2d=C.render_view(m0,cam,3)
    mse=torch.mean((img-gt)**2); psnr=float(20*torch.log10(1.0/torch.sqrt(mse)))
    print(f"{tag} idx={idx} psnr={psnr:.2f} img_range[{img.min().item():.3f},{img.max().item():.3f}] gt_range[{gt.min().item():.3f},{gt.max().item():.3f}] mean_radius={float(rad.float().mean()):.2f}")

