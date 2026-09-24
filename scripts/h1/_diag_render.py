import os, sys, json, torch
import numpy as np
from plyfile import PlyData

H = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"
sys.path.insert(0, H)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from gsplat.rendering import rasterization

device = torch.device("cuda:0")
ply_path = "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply"
ply = PlyData.read(ply_path)
v = ply["vertex"]
N = len(v)
print("N =", N)
print("PLY properties:", [p.name for p in ply["vertex"].properties[:20]])

# Opacity distribution
op_raw = np.array(v["opacity"])
op_sigmoid = 1.0 / (1.0 + np.exp(-op_raw))
print("\n=== Opacity (raw) ===")
print("  min=%.4f, max=%.4f, mean=%.4f" % (op_raw.min(), op_raw.max(), op_raw.mean()))
print("  percentiles [1,5,25,50,75,95,99]:", [np.percentile(op_raw, p) for p in [1,5,25,50,75,95,99]])
print("  sigmoid: min=%.6f, max=%.6f, mean=%.4f" % (op_sigmoid.min(), op_sigmoid.max(), op_sigmoid.mean()))
print("  sigmoid < 0.01: %d (%.1f%%)" % ((op_sigmoid < 0.01).sum(), 100*(op_sigmoid < 0.01).mean()))
print("  sigmoid < 0.001: %d (%.1f%%)" % ((op_sigmoid < 0.001).sum(), 100*(op_sigmoid < 0.001).mean()))

# Scale distribution
s0 = np.array(v["scale_0"])
s1 = np.array(v["scale_1"])
s2 = np.array(v["scale_2"])
print("\n=== Scales (raw log-scale) ===")
print("  scale_0: min=%.4f, max=%.4f, mean=%.4f" % (s0.min(), s0.max(), s0.mean()))
print("  scale_1: min=%.4f, max=%.4f, mean=%.4f" % (s1.min(), s1.max(), s1.mean()))
print("  scale_2: min=%.4f, max=%.4f, mean=%.4f" % (s2.min(), s2.max(), s2.mean()))
print("  exp(scale_0): min=%.6f, max=%.6f" % (np.exp(s0).min(), np.exp(s0).max()))

# SH DC
dc0 = np.array(v["f_dc_0"])
print("\n=== SH DC ===")
print("  f_dc_0: min=%.4f, max=%.4f, mean=%.4f" % (dc0.min(), dc0.max(), dc0.mean()))

# Means bounding box
x = np.array(v["x"]); y = np.array(v["y"]); z = np.array(v["z"])
print("\n=== Means ===")
print("  x: [%.2f, %.2f], y: [%.2f, %.2f], z: [%.2f, %.2f]" % (x.min(), x.max(), y.min(), y.max(), z.min(), z.max()))

# Now render and check actual contribution
print("\n=== Render test ===")
cams = json.load(open("/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json"))
cam = cams[0]
R = np.asarray(cam["rotation"], dtype=np.float64)
p = np.asarray(cam["position"], dtype=np.float64)
Rw2c = R.T
vm = np.eye(4); vm[:3,:3] = Rw2c; vm[:3,3] = -Rw2c @ p
width = 2048; height = 1365
scale_f = width / float(cam["width"])
K = np.array([[float(cam["fx"])*scale_f, 0, (width-1)/2],
              [0, float(cam["fy"])*scale_f, (height-1)/2], [0,0,1]], dtype=np.float64)
vm_t = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
K_t = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

def arr_t(name): return torch.tensor(v[name], dtype=torch.float32, device=device)
means = torch.stack([arr_t("x"), arr_t("y"), arr_t("z")], dim=-1)
quats = torch.stack([arr_t("rot_0"), arr_t("rot_1"), arr_t("rot_2"), arr_t("rot_3")], dim=-1)
scales = torch.stack([arr_t("scale_0"), arr_t("scale_1"), arr_t("scale_2")], dim=-1)
opacities = torch.sigmoid(arr_t("opacity"))
K_SH = 16
sh0 = torch.stack([arr_t("f_dc_0"), arr_t("f_dc_1"), arr_t("f_dc_2")], dim=-1).unsqueeze(1)
f_rest = []
for i in range(1, K_SH):
    f_rest.append(torch.stack([arr_t("f_rest_%d" % (3*(i-1)+j)) for j in range(3)], dim=-1))
f_rest = torch.stack(f_rest, dim=1)
sh = torch.zeros(N, K_SH, 3, dtype=torch.float32, device=device)
sh[:, 0] = sh0.squeeze(1)
sh[:, 1:] = f_rest

out = rasterization(
    means=means.unsqueeze(0), quats=quats.unsqueeze(0), scales=scales.unsqueeze(0),
    opacities=opacities.unsqueeze(0), colors=sh,
    viewmats=vm_t, Ks=K_t, width=width, height=height,
    sh_degree=3, packed=False, radius_clip=0.0,
)
render = out[0]  # [1, 1, H, W, 3]
alpha = out[1]   # [1, 1, H, W, 1]
print("  render shape:", render.shape)
print("  render: min=%.6f, max=%.6f, mean=%.6f" % (render.min().item(), render.max().item(), render.mean().item()))
print("  alpha: min=%.6f, max=%.6f, mean=%.6f" % (alpha.min().item(), alpha.max().item(), alpha.mean().item()))
print("  alpha > 0.01 pixels: %d / %d (%.1f%%)" % ((alpha > 0.01).sum().item(), alpha.numel(), 100*(alpha > 0.01).float().mean().item()))
print("  alpha > 0.0 pixels: %d / %d" % ((alpha > 0.0).sum().item(), alpha.numel()))

# Check info dict for visibility info
if len(out) > 2 and isinstance(out[2], dict):
    info = out[2]
    print("\n  Info keys:", list(info.keys()))
    for k, val in info.items():
        if isinstance(val, torch.Tensor):
            print("    %s: shape=%s, dtype=%s" % (k, val.shape, val.dtype))
        elif isinstance(val, dict):
            print("    %s: dict with keys %s" % (k, list(val.keys())[:10]))
        else:
            print("    %s: %s" % (k, type(val).__name__))