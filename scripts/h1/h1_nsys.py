#!/usr/bin/env python3
"""Run B1 and B2 under nsys for representative cameras — produces .nsys-rep traces."""
import os, sys, subprocess, json

H = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"
CACHE = os.path.expanduser("~/.cache/torch_extensions/py310_cu128")
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage_pool/3dgs-renderer-benchmark/repo/artifacts/h1-clean-profile"
NSYS_DIR = os.path.join(OUT_DIR, "nsys")
os.makedirs(NSYS_DIR, exist_ok=True)

PY = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python"
NSYS = "/usr/lib/x86_64-linux-gnu/nsight-systems/target-linux-x64/nsys"

# 6 traces: train cam0 B1/B2, room cam0 B1/B2, bicycle cam0 B1/B2
traces = [
    ("train", 0, "B1"), ("train", 0, "B2"),
    ("room", 0, "B1"), ("room", 0, "B2"),
    ("bicycle", 0, "B1"), ("bicycle", 0, "B2"),
]

PROFILE_SCRIPT = os.path.join(os.path.dirname(__file__), "h1_kernel_inventory.py")

# We will create a minimal profiling wrapper that runs only one method
WRAPPER = r'''
import sys, os, json, math, torch
H = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"
CACHE = os.path.expanduser("~/.cache/torch_extensions/py310_cu128")
sys.path.insert(0, H)
sys.path.insert(0, os.path.join(CACHE, "gsplat_scene_cuda"))
os.environ["PYTHONNOUSERSITE"] = "1"

SCENE = sys.argv[1]
CAM_IDX = int(sys.argv[2])
METHOD = sys.argv[3]
WARMUP = int(sys.argv[4]) if len(sys.argv) > 4 else 3
MEASURE = int(sys.argv[5]) if len(sys.argv) > 5 else 5

SCENES = {
    "train": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/point_cloud/iteration_30000/point_cloud.ply",
              "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/cameras.json"),
    "room": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
             "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json"),
    "bicycle": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
                "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json"),
}
from plyfile import PlyData
import numpy as np
device = torch.device("cuda:0")
ply_path, cams_path = SCENES[SCENE]
ply = PlyData.read(ply_path)
v = ply["vertex"]
N = len(v)
def arr(n): return torch.tensor(v[n], dtype=torch.float32, device=device)
means = torch.stack([arr("x"), arr("y"), arr("z")], dim=-1)
sh0 = torch.stack([arr("f_dc_0"), arr("f_dc_1"), arr("f_dc_2")], dim=-1).unsqueeze(1)
opacities = torch.sigmoid(arr("opacity"))
scales = torch.stack([arr("scale_0"), arr("scale_1"), arr("scale_2")], dim=-1)
quats = torch.stack([arr("rot_0"), arr("rot_1"), arr("rot_2"), arr("rot_3")], dim=-1)
K_SH = 16
f_rest = []
for i in range(1, K_SH):
    f_rest.append(torch.stack([arr("f_rest_%d" % (3*(i-1)+j)) for j in range(3)], dim=-1))
f_rest = torch.stack(f_rest, dim=1)
sh = torch.zeros(N, K_SH, 3, dtype=torch.float32, device=device)
sh[:, 0] = sh0.squeeze(1)
sh[:, 1:] = f_rest

cams = json.load(open(cams_path))
cam = cams[CAM_IDX]
R = np.asarray(cam["rotation"], dtype=np.float64)
p = np.asarray(cam["position"], dtype=np.float64)
Rw2c = R.T
vm = np.eye(4); vm[:3,:3] = Rw2c; vm[:3,3] = -Rw2c @ p
width = 1024 if SCENE == "train" else 2048
height = 570 if SCENE == "train" else 1365
scale = width / float(cam["width"])
K = np.array([[float(cam["fx"])*scale, 0, (width-1)/2], [0, float(cam["fy"])*scale, (height-1)/2], [0,0,1]])
vm = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
K = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode, _make_lazy_cuda_func
from gsplat.rendering import _maybe_evaluate_sh
from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER
from gsplat.experimental import rasterize_gaussian_higs_frozen

SH_DEGREE = 3
def b1():
    with torch.no_grad():
        radii, means2d, depths, conics, _ = fully_fused_projection(
            means=means.unsqueeze(0).contiguous(), covars=None, quats=quats.unsqueeze(0).contiguous(),
            scales=scales.unsqueeze(0).contiguous(), viewmats=vm, Ks=K, width=width, height=height,
            eps2d=0.3, packed=False, calc_compensations=False, camera_model="pinhole")
        C = 1; Nloc = means.shape[0]
        colors_eval = _maybe_evaluate_sh(SH_DEGREE, sh, means.unsqueeze(0), radii, vm, (1,), C, Nloc, True).contiguous()
        opac_bc = torch.broadcast_to(opacities.unsqueeze(0)[..., None, :], (1, C, Nloc)).contiguous()
        ts = 16; tw = math.ceil(width/ts); th = math.ceil(height/ts)
        _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, ts, tw, th, packed=False, n_images=C, conics=conics, opacities=opac_bc)
        isect_offsets = isect_offset_encode(isect_ids, C, tw, th).reshape((1, C, th, tw))
        bg = torch.zeros((1, C, 3), device=device)
        rc, ra, _, _ = _make_lazy_cuda_func("rasterize_to_pixels_3dgs")(means2d.contiguous(), conics.contiguous(), colors_eval.contiguous(), opac_bc.contiguous(), bg, None, width, height, ts, isect_offsets.contiguous(), flatten_ids.contiguous(), False, False)

def b2():
    _HIGS_FROZEN_TRACKER.reset()
    h = create_higs_renderer(means, quats, scales, opacities, sh, sh_degree=SH_DEGREE)
    with torch.no_grad():
        res = rasterize_gaussian_higs_frozen(means, quats, scales, opacities, sh, backward_mode="higs_native", scene=h, freeze_topology=True, viewmats=vm, Ks=K, width=width, height=height, sh_degree=SH_DEGREE, use_higs_culling=True, tile_sampling_ratio=1.0)
    h.release()

fn = b1 if METHOD == "B1" else b2
for _ in range(WARMUP):
    fn()
torch.cuda.synchronize()
for i in range(MEASURE):
    torch.cuda.nvtx.range_push("iter_%d" % i)
    fn()
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print("done %s %s cam%d" % (SCENE, METHOD, CAM_IDX))
'''

wrapper_path = os.path.join(NSYS_DIR, "_nsys_wrapper.py")
with open(wrapper_path, "w") as f:
    f.write(WRAPPER)

for scene, cam, method in traces:
    tag = "%s_cam%d_%s" % (scene, cam, method)
    rep = os.path.join(NSYS_DIR, tag + ".nsys-rep")
    cmd = [NSYS, "profile", "--stats=true", "--force-overwrite=true",
           "-o", rep, "-t", "cuda,nvtx",
           PY, wrapper_path, scene, str(cam), method, "3", "5"]
    print("Running: %s" % tag)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print("  ERROR: %s" % r.stderr[-200:])
    else:
        print("  OK: %s" % rep)
print("All nsys traces done.")