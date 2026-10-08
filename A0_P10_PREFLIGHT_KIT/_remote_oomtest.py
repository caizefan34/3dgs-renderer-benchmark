"""Empirical: does the frozen C0 render fit on a free A100 at 720p (expect OK) vs
1080p (expect OOM)? Isolates the OOM as capacity, not a bug. FROZEN renderer +
FROZEN C0 env + FROZEN checkpoint (read-only). Sequential, full free in between."""
import os, sys, math, gc, torch
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
o = torch.load("/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_25000.pt", map_location="cpu", weights_only=False)
def try_render(res):
    for _ in range(3): torch.cuda.empty_cache(); gc.collect()
    W,H = (1280,720) if res=="720p" else (1920,1080)
    cams = resize_cameras(load_cameras_from_json(REPO+"/data/official/mipnerf360/room/cameras.json", device="cpu"), W, H)
    cam = cams[19]; vm = cam.viewmatrix.to("cuda"); K = cam.K.to("cuda")
    print(f"  {res}: cam19 img={getattr(cam,'img_name',None)} W={cam.image_width} H={cam.image_height}", flush=True)
    m = {k: o[k].to("cuda").float().contiguous().requires_grad_(True) for k in ("xyz","shs","scaling","rotation","opacity")}
    try:
        out = rasterize_gaussian_higs_dynamic(
            m["xyz"], m["rotation"], m["scaling"], m["opacity"], m["shs"], sh_degree=3,
            viewmats=vm.unsqueeze(0).unsqueeze(0), Ks=K.unsqueeze(0).unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            backward_mode="higs_native", enable_culling=True, camera_model="pinhole",
            render_mode="RGB", eps2d=0.3)
        peak = torch.cuda.max_memory_allocated()/1e9
        rd = out["densification_info"]["radii"]
        nvis = int((rd > 0).sum())
        print(f"  {res}: RENDER OK peak_alloc={peak:.2f}GiB n_visible={nvis} radii_shape={tuple(rd.shape)} frame={tuple(out['frame'].shape)}", flush=True)
        del out; del m; torch.cuda.empty_cache(); gc.collect(); return True
    except torch.OutOfMemoryError as e:
        print(f"  {res}: OOM ({str(e)[:80]}...)", flush=True)
        del m; torch.cuda.empty_cache(); gc.collect(); return False
print("=== 720p test ===", flush=True); ok720 = try_render("720p")
print("=== 1080p test ===", flush=True); ok1080 = try_render("1080p")
print(f"RESULT: 720p_ok={ok720} 1080p_ok={ok1080}", flush=True)
