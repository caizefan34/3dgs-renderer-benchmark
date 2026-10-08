import os, sys, struct
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
os.environ["HIGS_PX_RUNTIME"]="2"; os.environ["HIGS_BWD_SCALAR_ADJOINT"]="scalar_adjoint"
os.environ["HIGS_BWD_H8_MR"]="1"; os.environ["HIGS_BWD_ABSGRAD"]="1"; os.environ.pop("HIGS_DISABLE_F9",None)
import torch
C0_WT="/root/epic05_frozen/higs_c0_final30k_worktree"; SCENE_CUDA_DIR="/root/epic05_build_cache/scene/gsplat_scene_cuda"
for p in (C0_WT, SCENE_CUDA_DIR): sys.path.insert(0,p)
import importlib.util
def load_ext(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); sys.modules[name]=m; return m
load_ext("gsplat_cuda","/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"); sys.modules["gsplat.csrc"]=sys.modules["gsplat_cuda"]
load_ext("experimental_gaussian_render_inference_scene_cuda","/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"); sys.modules["gsplat.experimental.render.kernels.csrc"]=sys.modules["experimental_gaussian_render_inference_scene_cuda"]
from gsplat.experimental import rasterize_gaussian_higs_dynamic
from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE
sys.path.insert(0,"/root/3dgs-renderer-benchmark/scripts/epic05/phase7"); sys.path.insert(0,"/root/3dgs-renderer-benchmark/src"); sys.path.insert(0,"/root/3dgs-renderer-benchmark/baseline/reference_v1")
from gaussian_model import GaussianModel
from colmap_reader import read_points3D_binary, sfm_to_pcd_data
from dataset import GTDataset
torch.manual_seed(42)
ds=GTDataset(scene="room", repo_root="/root/3dgs-renderer-benchmark", resolution="1080p", device="cuda", background="black")
cam,gt=ds.get_item(0)
W,H=cam.image_width,cam.image_height
sfm="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room/sparse/0/points3D.bin"
pts=read_points3D_binary(sfm); pcd=sfm_to_pcd_data(pts, sh_degree=0)
m=GaussianModel(max_sh_degree=0); m.create_from_pcd(pcd, spatial_lr_scale=1.0)
print("N", m._xyz.shape[0], "W",W,"H",H)
m.active_sh_degree=0
try: _HIGS_DYNAMIC_SCENE.mark_dirty()
except: pass
out=rasterize_gaussian_higs_dynamic(m.get_xyz,m.get_rotation,m.get_scaling,m.get_opacity,m.get_features,
    sh_degree=0, viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0), Ks=cam.K.unsqueeze(0).unsqueeze(0),
    width=W,height=H,tile_size=16,backward_mode="higs_native",enable_culling=True,camera_model="pinhole",render_mode="RGB")
info=out["densification_info"]; means2d=info["means2d"]; radii=info["radii"]
print("means2d shape",tuple(means2d.shape),"min",round(float(means2d.min()),3),"max",round(float(means2d.max()),3))
print("radii shape",tuple(radii.shape),"min",float(radii.min()),"max",float(radii.max()))
vis=(radii>0)
md=means2d[0][vis[0]] if means2d.dim()==3 else means2d[vis]
print("visible means2d sample:", md[:5].tolist())
print("cam.K[0]:", [round(float(x),2) for x in cam.K[0].tolist()])
print("cam.viewmatrix[0][:3,:]:", [[round(float(x),3) for x in cam.viewmatrix[0][:3,j].tolist()] for j in range(4)])