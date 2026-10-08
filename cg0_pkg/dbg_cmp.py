import sys
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
from cg0_run import render_depth, build_compiled_model
from gaussian_model import GaussianModel
C.bootstrap_c0()
scene="room"; it=15000
ds=C.load_scene_dataset(scene); n=len(ds)
split=P.build_split(n)
ck=C.load_checkpoint(scene,it); Ns=int(ck["xyz"].shape[0]); ss=float(ck["spatial_lr_scale"])
src_shd=int(ck["active_sh_degree"]); src_model=C.build_model(ck)
print("src opacity mean (sigmoid)", torch.sigmoid(ck["opacity"]).mean().item(), "src scaling mean", ck["scaling"].mean(0).tolist())
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
print("compiled opacity mean (sigmoid)", torch.sigmoid(op_logit).mean().item(), "compiled scaling mean", clusters["scaling"].mean(0).tolist())
cam=ds.get_camera(compile_idx[0]); gt=ds.get_gt_image(compile_idx[0])
img_s,_,_=C.render_view(src_model,cam,src_shd)
mse=torch.mean((img_s-gt)**2); print("SOURCE compile0 psnr", float(20*torch.log10(1.0/torch.sqrt(mse))), "img_mean", img_s.mean().item(), "range", img_s.min().item(), img_s.max().item())
params={"mu":clusters["mu"],"scaling":clusters["scaling"],"rotation":clusters["rotation"],"opacity_logit":op_logit,"shs":shs}
m=build_compiled_model(params,3,ss,"cuda")
img_c,_,_=C.render_view(m,cam,3)
mse=torch.mean((img_c-gt)**2); print("COMPILED compile0 psnr", float(20*torch.log10(1.0/torch.sqrt(mse))), "img_mean", img_c.mean().item(), "range", img_c.min().item(), img_c.max().item())
try:
    import imageio.v2 as imageio
    imageio.imwrite("/root/epic05_cg0/dbg_src.png",(img_s.clamp(0,1).cpu().numpy()*255).astype(np.uint8))
    imageio.imwrite("/root/epic05_cg0/dbg_cmp.png",(img_c.clamp(0,1).cpu().numpy()*255).astype(np.uint8))
    imageio.imwrite("/root/epic05_cg0/dbg_gt.png",(gt.cpu().numpy()*255).astype(np.uint8))
    print("saved pngs")
except Exception as e:
    print("imgio fail", e)
    torch.save({"src":img_s.detach().cpu(),"cmp":img_c.detach().cpu(),"gt":gt.detach().cpu()}, "/root/epic05_cg0/dbg_imgs.pt")
C.scene_reset()
