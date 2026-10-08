import os, sys, math, torch
os.environ["HIGS_PX_RUNTIME"]="2"; os.environ["HIGS_BWD_SCALAR_ADJOINT"]="scalar_adjoint"
os.environ["HIGS_BWD_H8_MR"]="1"; os.environ["HIGS_BWD_ABSGRAD"]="1"
os.environ.pop("HIGS_DISABLE_F9", None)
torch.cuda.set_device(7); os.environ["CUDA_VISIBLE_DEVICES"]="7"
C0_WT="/root/epic05_frozen/higs_c0_final30k_worktree"; SCENE="/root/epic05_build_cache/scene/gsplat_scene_cuda"
C0_SO="/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO="/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
REPO="/root/3dgs-renderer-benchmark"
import importlib.util
def lm(n,s):
    spec=importlib.util.spec_from_file_location(n,s); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.modules[n]=m; return m
sys.path.insert(0,C0_WT); sys.path.insert(0,SCENE)
core=lm("gsplat_cuda",CORE_SO); sys.modules["gsplat.csrc"]=core
exp=lm("experimental_gaussian_render_inference_scene_cuda",C0_SO); sys.modules["gsplat.experimental.render.kernels.csrc"]=exp
sys.path.insert(0,REPO+"/scripts/epic05/phase7"); sys.path.insert(0,REPO+"/src"); sys.path.insert(0,REPO+"/baseline/reference_v1")
from gsplat.cuda._wrapper import fully_fused_projection
from dataset import GTDataset
def gmem(t):
    print(f"  [{t}] alloc={torch.cuda.memory_allocated()/1e9:.3f}GiB", flush=True)
ds=GTDataset(scene="room",repo_root=REPO,resolution="1080p",device="cuda",background="black")
cam=ds.get_camera(19); W,H=cam.image_width,cam.image_height; ts=16
tw,th=math.ceil(W/ts),math.ceil(H/ts)
o=torch.load("/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_25000.pt",map_location="cpu",weights_only=False)
N=o["xyz"].shape[0]
m={k:o[k].to("cuda").float().contiguous() for k in ("xyz","shs","scaling","rotation","opacity")}
print("N",N,"W",W,"H",H,"tw",tw,"th",th,"n_tiles",tw*th,flush=True)
viewmats=cam.viewmatrix.unsqueeze(0); Ks=cam.K.unsqueeze(0)
# F9 OFF path: fully_fused_projection (non-gatherless)
r=fully_fused_projection(means=m["xyz"].contiguous(),covars=None,quats=m["rotation"].contiguous(),
    scales=m["scaling"].contiguous(),viewmats=viewmats,Ks=Ks,width=W,height=H,eps2d=0.3,
    near_plane=0.01,far_plane=1e10,radius_clip=0.0,packed=False,calc_compensations=False,camera_model="pinhole")
radii,means2d,depths,conics,comp = r
radii=radii.reshape(1,1,N,2)
print("radii",tuple(radii.shape),flush=True)
vis=(radii[...,0]>0)&(radii[...,1]>0)
visN=int(vis.sum())
print("N_visible (radii>0):",visN,flush=True)
gmem("after fully_fused_projection")
# estimate intersect_tile memory: ifecount = visN * (tw*th+1) uint32
ifecount=visN*(tw*th+1)
print(f"estimated ifecount={ifecount} uint32 = {ifecount*4/1e9:.2f} GiB", flush=True)
print("DONE", flush=True)
