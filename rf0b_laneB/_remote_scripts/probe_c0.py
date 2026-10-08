import os, sys, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
os.environ["HIGS_PX_RUNTIME"]="2"; os.environ["HIGS_BWD_SCALAR_ADJOINT"]="scalar_adjoint"
os.environ["HIGS_BWD_H8_MR"]="1"; os.environ["HIGS_BWD_ABSGRAD"]="1"; os.environ.pop("HIGS_DISABLE_F9",None)
import torch, torch.nn.functional as F
C0_WT="/root/epic05_frozen/higs_c0_final30k_worktree"; SCENE_CUDA_DIR="/root/epic05_build_cache/scene/gsplat_scene_cuda"
for p in (C0_WT, SCENE_CUDA_DIR): sys.path.insert(0,p)
import importlib.util
def load_ext(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); sys.modules[name]=m; return m
load_ext("gsplat_cuda","/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"); sys.modules["gsplat.csrc"]=sys.modules["gsplat_cuda"]
load_ext("exp_cuda","/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"); sys.modules["gsplat.experimental.render.kernels.csrc"]=sys.modules["exp_cuda"]
from gsplat.experimental import rasterize_gaussian_higs_dynamic
from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE
sys.path.insert(0,"/root/epic05_p12/scripts/epic05/phase7"); sys.path.insert(0,"/root/3dgs-renderer-benchmark/src"); sys.path.insert(0,"/root/3dgs-renderer-benchmark/baseline/reference_v1")
from gaussian_model import GaussianModel
from config import ReferenceV1Config
from dataset import GTDataset
torch.manual_seed(42)
cfg=ReferenceV1Config(); cfg.repo_root="/root/3dgs-renderer-benchmark"; cfg.iterations=30000
ds=GTDataset(scene="room", repo_root=cfg.repo_root, resolution="1080p", device="cuda", background="black")
cam,gt=ds.get_item(0)
print("cam", cam.image_width, cam.image_height, "gt", tuple(gt.shape))
import struct
def read_pts(p):
    with open(p,"rb") as f:
        data=f.read()
    # minimal points3D.bin parse
    off=0; n=struct.unpack("<Q",data[off:off+8])[0]; off+=8
    xyzs=[]
    for i in range(n):
        pid=data[off:off+8]; off+=8
        x,y,z=struct.unpack("<ddd",data[off:off+24]); off+=24
        off+=3*8+3*4+3*4  # rgb + error? points3D: x,y,z,r,g,b,err,3+ image*2
        # actually points3D record: id(8) xyz(24) rgb(12) err(8) then track (2+2*m)
        off-=3*8+3*4+3*4
        # redo properly below
        break
    return n
npts=112627
m=GaussianModel(max_sh_degree=cfg.sh_degree)
# build from dataset pcd via trainer path: use sfm_to_pcd? simpler: create_from_sfm
# Instead replicate: use GTDataset's pcd if present
print("has get_pcd:", hasattr(ds,"get_pcd"), "attrs:", [a for a in dir(ds) if "pcd" in a.lower() or "points" in a.lower() or "sfm" in a.lower()])