"""Minimal probe: reproduce the frozen C0 render of room@25K cam19 and measure GPU
memory + intermediate tensor sizes at each step, to isolate the 71 GiB OOM."""
import os, sys, torch
os.environ["HIGS_PX_RUNTIME"] = "2"
os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
os.environ["HIGS_BWD_H8_MR"] = "1"
os.environ["HIGS_BWD_ABSGRAD"] = "1"
os.environ.pop("HIGS_DISABLE_F9", None)
torch.cuda.set_device(7)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
C0_SO = "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
PHASE7_DIR = "/root/3dgs-renderer-benchmark/scripts/epic05/phase7"
SRC_DIR = "/root/3dgs-renderer-benchmark/src"
REF = "/root/3dgs-renderer-benchmark/baseline/reference_v1"
REPO = "/root/3dgs-renderer-benchmark"
import importlib.util
def load_mod(name, so):
    spec = importlib.util.spec_from_file_location(name, so)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); sys.modules[name]=mod; return mod
sys.path.insert(0, C0_WT); sys.path.insert(0, SCENE_CUDA_DIR)
core = load_mod("gsplat_cuda", CORE_SO); sys.modules["gsplat.csrc"]=core
exp = load_mod("experimental_gaussian_render_inference_scene_cuda", C0_SO)
sys.modules["gsplat.experimental.render.kernels.csrc"]=exp
sys.path.insert(0, PHASE7_DIR); sys.path.insert(0, SRC_DIR); sys.path.insert(0, REF)
from gsplat.experimental import rasterize_gaussian_higs_dynamic
from dataset import GTDataset

def gmem(tag):
    f = torch.cuda.memory_allocated()/1e9; r = torch.cuda.memory_reserved()/1e9
    print(f"  [{tag}] alloc={f:.3f}GiB reserved={r:.3f}GiB", flush=True)
def shape_mb(t, tag):
    if t is None: return
    print(f"  [{tag}] {tuple(t.shape)} {t.dtype} {t.numel()*t.element_size()/1e6:.1f}MiB", flush=True)

ds = GTDataset(scene="room", repo_root=REPO, resolution="1080p", device="cuda", background="black")
gmem("after dataset")
cam = ds.get_camera(19)
print("cam img_name", getattr(cam,"img_name",None) or getattr(cam,"image_name",None),
      "W", cam.image_width, "H", cam.image_height, "K", tuple(cam.K.shape), "vm", tuple(cam.viewmatrix.shape))

o = torch.load("/root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_25000.pt", map_location="cpu", weights_only=False)
gmem("after ckpt load cpu")
m = {k: o[k].to("cuda").float().contiguous().requires_grad_(True) for k in ("xyz","shs","scaling","rotation","opacity")}
for k,t in m.items(): shape_mb(t, "param "+k)
gmem("after params to cuda")
N = m["xyz"].shape[0]
print("N =", N, "active_sh_degree", o.get("active_sh_degree"), flush=True)

# render WITHOUT grad first (cheaper) to isolate intersect_tile cost
m2 = {k: m[k].detach() for k in m}
out = rasterize_gaussian_higs_dynamic(
    m2["xyz"], m2["rotation"], m2["scaling"], m2["opacity"], m2["shs"],
    sh_degree=o.get("active_sh_degree",3),
    viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0), Ks=cam.K.unsqueeze(0).unsqueeze(0),
    width=cam.image_width, height=cam.image_height,
    backward_mode="higs_native", enable_culling=True, camera_model="pinhole",
    render_mode="RGB", eps2d=0.3)
gmem("after render (no-grad)")
info = out["densification_info"]
print("densification_info keys:", list(info.keys()), flush=True)
for k,v in info.items():
    if torch.is_tensor(v): shape_mb(v, "info."+k)
    else: print(f"  [info.{k}] {type(v)} {getattr(v,'shape',v)}", flush=True)
print("DONE no-grad render", flush=True)
