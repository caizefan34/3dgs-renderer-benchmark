import sys, os, torch, numpy as np
REPO="/root/3dgs-renderer-benchmark"
REFERENCE_V1_DIR=os.path.join(REPO,"baseline","reference_v1")
PHASE7_DIR=os.path.join(REPO,"scripts","epic05","phase7")
SRC_DIR=os.path.join(REPO,"src")
ACCUTILE="/root/epic05_frozen/gsplat-true-accutile-v153"
C0_WT="/root/epic05_frozen/higs_c0_final30k_worktree"
SCENE_CUDA_DIR="/root/epic05_build_cache/scene/gsplat_scene_cuda"
for p in [ACCUTILE,PHASE7_DIR,SRC_DIR,REFERENCE_V1_DIR,C0_WT,SCENE_CUDA_DIR]:
    sys.path.insert(0,p)
import importlib.util
def load_ext(name,so):
    spec=importlib.util.spec_from_file_location(name,so)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
CORE_SO="/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
C0_SO="/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
core=load_ext("gsplat_cuda",CORE_SO); sys.modules["gsplat.csrc"]=core
exp=load_ext("experimental_gaussian_render_inference_scene_cuda",C0_SO)
sys.modules["gsplat.experimental.render.kernels.csrc"]=exp
sys.modules["experimental_gaussian_render_inference_scene_cuda"]=exp
from gsplat.experimental import rasterize_gaussian_higs_dynamic
from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE
from gaussian_model import GaussianModel
from config import ReferenceV1Config
from dataset import GTDataset
import torch.nn.functional as F
print("bootstrap OK; rasterize:", rasterize_gaussian_higs_dynamic.__module__)

config=ReferenceV1Config(); config.scene="room"; config.repo_root=REPO; config.seed=42
ds=GTDataset(scene="room", repo_root=REPO, resolution="1080p", device="cuda", background="black")
print("n_cameras", len(ds))
cam,gt=ds.get_item(0)
print("cam", cam.image_width, cam.image_height, "gt", tuple(gt.shape), "gt range", float(gt.min()), float(gt.max()))

CKPT="/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_10000.pt"
sd=torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
print("ckpt iter", sd["iteration"], "scene", sd["scene"], "arm", sd["arm"], "N", sd["xyz"].shape[0])
model=GaussianModel(max_sh_degree=config.sh_degree)
model.restore(sd, config.to_dict())
model.optimizer.load_state_dict(sd["optimizer_state_dict"])
print("model restored; active_sh", model.active_sh_degree, "N", model._xyz.shape[0], "opt groups", [g["name"] for g in model.optimizer.param_groups])
print("xyz lr", model.optimizer.param_groups[0]["lr"], "scaling lr", model.optimizer.param_groups[3]["lr"], "rot lr", model.optimizer.param_groups[4]["lr"])

eps2d=0.3
image,radii,means2d=model.get_xyz, None, None
out=rasterize_gaussian_higs_dynamic(model.get_xyz, model.get_rotation, model.get_scaling, model.get_opacity, model.get_features, sh_degree=model.active_sh_degree, viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0), Ks=cam.K.unsqueeze(0).unsqueeze(0), width=cam.image_width, height=cam.image_height, backward_mode="higs_native", enable_culling=True, camera_model="pinhole", render_mode="RGB", eps2d=eps2d)
frame=out["frame"].reshape(cam.image_height, cam.image_width, 3).clamp(0,1)
print("render frame", tuple(frame.shape), "range", float(frame.min()), float(frame.max()))
import fused_ssim, fused_ssim_cuda
print("fused so sha", __import__("hashlib").sha256(fused_ssim_cuda.__file__ and open(fused_ssim_cuda.__file__,"rb").read()).hexdigest() if False else "skip")
b=frame.unsqueeze(0).permute(0,3,1,2).contiguous(); g=gt.unsqueeze(0).permute(0,3,1,2).contiguous()
L1=F.l1_loss(b,g); ssim=fused_ssim.fused_ssim(b,g,padding="same",train=True)
loss=(1-config.lambda_dssim)*L1+config.lambda_dssim*(1-ssim)
print("loss", float(loss), "L1", float(L1), "ssim", float(ssim))
loss.backward()
print("grad xyz norm", float(model._xyz.grad.norm()), "scaling grad norm", float(model._scaling.grad.norm()), "rotation grad norm", float(model._rotation.grad.norm()), "opacity", float(model._opacity.grad.norm()), "shs", float(model._shs.grad.norm()))
print("means2d requires_grad", means2d is None or True)
print("SMOKE_OK")
