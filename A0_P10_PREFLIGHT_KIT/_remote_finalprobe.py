import os, sys, gc, torch
os.environ["HIGS_PX_RUNTIME"]="2"; os.environ["HIGS_BWD_SCALAR_ADJOINT"]="scalar_adjoint"
os.environ["HIGS_BWD_H8_MR"]="1"; os.environ["HIGS_BWD_ABSGRAD"]="1"
os.environ.pop("HIGS_DISABLE_F9", None)
torch.cuda.set_device(0); os.environ["CUDA_VISIBLE_DEVICES"]="0"
C0_WT="/root/epic05_frozen/higs_c0_final30k_worktree"; SCENE="/root/epic05_build_cache/scene/gsplat_scene_cuda"
C0_SO="/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO="/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"; REPO="/root/3dgs-renderer-benchmark"
import importlib.util
def lm(n,s):
    spec=importlib.util.spec_from_file_location(n,s); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.modules[n]=m; return m
sys.path.insert(0,C0_WT); sys.path.insert(0,SCENE)
core=lm("gsplat_cuda",CORE_SO); sys.modules["gsplat.csrc"]=core
exp=lm("experimental_gaussian_render_inference_scene_cuda",C0_SO); sys.modules["gsplat.experimental.render.kernels.csrc"]=exp
sys.path.insert(0,REPO+"/scripts/epic05/phase7"); sys.path.insert(0,REPO+"/src"); sys.path.insert(0,REPO+"/baseline/reference_v1")
from gsplat.experimental import rasterize_gaussian_higs_dynamic
from benchmark_framework import load_cameras_from_json, resize_cameras
def mem(tag):
    f=os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader").read().strip().split("\n")[0]
    print(f"  [{tag}] {f}", flush=True)
print("=== 1080p exact-OOM capture (FROZEN recipe, read-only) ===", flush=True)
mem("before"); torch.cuda.empty_cache()
cams = resize_cameras(load_cameras_from_json(REPO+"/data/official/mipnerf360/room/cameras.json", device="cpu"), 1920, 1080)
cam = cams[19]; vm = cam.viewmatrix.to("cuda"); K = cam.K.to("cuda")
o = torch.load("/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_25000.pt", map_location="cpu", weights_only=False)
m = {k: o[k].to("cuda").float().contiguous().requires_grad_(True) for k in ("xyz","shs","scaling","rotation","opacity")}
mem("after params")
try:
    out = rasterize_gaussian_higs_dynamic(
        m["xyz"], m["rotation"], m["scaling"], m["opacity"], m["shs"], sh_degree=3,
        viewmats=vm.unsqueeze(0).unsqueeze(0), Ks=K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True, camera_model="pinhole",
        render_mode="RGB", eps2d=0.3)
    mem("after render")
    print("UNEXPECTED: 1080p render succeeded", flush=True)
except torch.OutOfMemoryError as e:
    mem("after OOM")
    print("EXACT OOM: " + str(e), flush=True)
print("DONE", flush=True)
