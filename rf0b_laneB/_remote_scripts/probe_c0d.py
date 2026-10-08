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
from config import ReferenceV1Config
from dataset import GTDataset
torch.manual_seed(42)
cfg=ReferenceV1Config(); cfg.repo_root="/root/3dgs-renderer-benchmark"; cfg.iterations=30000
ds=GTDataset(scene="room", repo_root=cfg.repo_root, resolution="1080p", device="cuda", background="black")
cam,gt=ds.get_item(0)
W,H=cam.image_width,cam.image_height
print("W,H",W,H)
sfm="/root/3dgs-renderer-benchmark/data/official/mipnerf360/room/sparse/0/points3D.bin"
data=open(sfm,"rb").read(); off=0; n=struct.unpack("<Q",data[off:off+8])[0]; off+=8
xyz=[]
for i in range(n):
    off+=8; x,y,z=struct.unpack("<ddd",data[off:off+24]); off+=24; off+=3; off+=8
    m=struct.unpack("<Q",data[off:off+8])[0]; off+=8+2*m*4
    xyz.append((x,y,z))
xyz=torch.tensor(xyz[:50000],dtype=torch.float32,device="cuda")
shs=torch.zeros((xyz.shape[0],3,1),dtype=torch.float32,device="cuda"); shs[:,0,0]=1.0
scaling=torch.log(torch.full((xyz.shape[0],3),0.01,device="cuda"))
rot=torch.zeros((xyz.shape[0],4),device="cuda"); rot[:,0]=1.0
op=torch.zeros((xyz.shape[0],1),device="cuda"); op[:,0]=-1.0
m=GaussianModel(max_sh_degree=3)
m._xyz=torch.nn.Parameter(xyz.requires_grad_(True)); m._shs=torch.nn.Parameter(shs.requires_grad_(True))
m._scaling=torch.nn.Parameter(scaling.requires_grad_(True)); m._rotation=torch.nn.Parameter(rot.requires_grad_(True)); m._opacity=torch.nn.Parameter(op.requires_grad_(True))
m.active_sh_degree=0
out=rasterize_gaussian_higs_dynamic(m.get_xyz,m.get_rotation,m.get_scaling,m.get_opacity,m.get_features,
    sh_degree=0, viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0), Ks=cam.K.unsqueeze(0).unsqueeze(0),
    width=W,height=H,tile_size=16,backward_mode="higs_native",enable_culling=True,camera_model="pinhole",render_mode="RGB")
info=out["densification_info"]; means2d=info["means2d"]; radii=info["radii"]
print("means2d shape",tuple(means2d.shape),"min",float(means2d.min()),"max",float(means2d.max()))
print("radii shape",tuple(radii.shape),"min",float(radii.min()),"max",float(radii.max()))
img=out["frame"]; print("img shape",tuple(img.shape))
loss=img.float().mean(); loss.backward()
print("means2d.grad", None if means2d.grad is None else (tuple(means2d.grad.shape), round(float(means2d.grad.abs().max()),6)))
ag=getattr(means2d,"absgrad",None)
print("absgrad", None if ag is None else (tuple(ag.shape), round(float(ag.abs().max()),6)))
print("viewmatrix range", float(cam.viewmatrix.min()), float(cam.viewmatrix.max()))
print("K[0]", cam.K[0].tolist())