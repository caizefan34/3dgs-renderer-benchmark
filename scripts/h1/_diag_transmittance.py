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

# Render with requires_grad to check gradients
m = means.detach().clone().requires_grad_(True)
q = quats.detach().clone().requires_grad_(True)
s = scales.detach().clone().requires_grad_(True)
o = opacities.detach().clone().requires_grad_(True)
c = sh.detach().clone().requires_grad_(True)

out = rasterization(
    means=m.unsqueeze(0), quats=q.unsqueeze(0), scales=s.unsqueeze(0),
    opacities=o.unsqueeze(0), colors=c,
    viewmats=vm_t, Ks=K_t, width=width, height=height,
    sh_degree=3, packed=False, radius_clip=0.0,
)
render = out[0]
alpha = out[1]
info = out[2]

# Check radii (which Gaussians are visible)
radii = info["radii"][0, 0]  # [N, 2]
n_visible = (radii > 0).any(dim=-1).sum().item()
print("Gaussians with radii > 0: %d / %d" % (n_visible, N))

# Check depths and conics
depths = info["depths"][0, 0]  # [N]
conics = info["conics"][0, 0]  # [N, 3]
print("Depths: min=%.4f, max=%.4f, mean=%.4f" % (depths.min().item(), depths.max().item(), depths.mean().item()))
print("Conics det: min=%.6f, max=%.6f" % (conics[:, 0].min().item(), conics[:, 0].max().item()))

# Check tiles_per_gauss
tiles_per_gauss = info["tiles_per_gauss"][0, 0]  # [N]
print("Tiles per Gaussian: min=%d, max=%d, mean=%.1f" % (tiles_per_gauss.min().item(), tiles_per_gauss.max().item(), tiles_per_gauss.float().mean().item()))
print("Gaussians with 0 tiles: %d" % (tiles_per_gauss == 0).sum().item())

# Total intersections
n_isect = info["isect_ids"].numel()
print("Total intersections: %d" % n_isect)

# Now do backward with a simple loss (sum of all pixels = upstream gradient of 1.0)
# This is the simplest possible upstream gradient — every pixel gets gradient 1.0
loss = render.sum()
loss.backward()

means_grad = m.grad  # [N, 3]
opac_grad = o.grad  # [N]

# Check gradient sparsity at different thresholds
print("\n=== Gradient sparsity (means) ===")
for thresh in [1e-6, 1e-8, 1e-10, 1e-15, 1e-20, 1e-30, 0.0]:
    n_nz = (means_grad.abs() > thresh).sum().item()
    n_nz_gauss = (means_grad.abs().reshape(N, -1) > thresh).any(dim=-1).sum().item()
    print("  thresh > %.1e: %d elements, %d Gaussians" % (thresh, n_nz, n_nz_gauss))

# Check exact zero
n_exact_zero = (means_grad == 0).sum().item()
n_exact_zero_gauss = (means_grad.reshape(N, -1) == 0).all(dim=-1).sum().item()
print("  exact zero: %d elements, %d Gaussians" % (n_exact_zero, n_exact_zero_gauss))

# Check the non-zero Gaussians
nz_mask = (means_grad.abs() > 1e-10).any(dim=-1)
nz_indices = nz_mask.nonzero().squeeze(-1)
print("\n=== Nonzero gradient Gaussians ===")
for idx in nz_indices:
    i = idx.item()
    print("  Gaussian %d: depth=%.2f, opacity=%.4f, radii=%s, means_grad=[%.6e, %.6e, %.6e]" % (
        i, depths[i].item(), opacities[i].item(), radii[i].tolist(),
        means_grad[i, 0].item(), means_grad[i, 1].item(), means_grad[i, 2].item()))

# Check opacity gradient
print("\n=== Opacity gradient sparsity ===")
for thresh in [1e-6, 1e-8, 1e-10, 1e-15, 1e-20, 0.0]:
    n_nz = (opac_grad.abs() > thresh).sum().item()
    print("  thresh > %.1e: %d Gaussians" % (thresh, n_nz))

# Hypothesis: transmittance underflow. Check by rendering with reduced opacity.
print("\n=== Opacity reduction test ===")
for op_scale in [1.0, 0.5, 0.1, 0.01]:
    m2 = means.detach().clone().requires_grad_(True)
    o2 = (opacities * op_scale).detach().clone().requires_grad_(True)
    out2 = rasterization(
        means=m2.unsqueeze(0), quats=quats.unsqueeze(0), scales=scales.unsqueeze(0),
        opacities=o2.unsqueeze(0), colors=sh,
        viewmats=vm_t, Ks=K_t, width=width, height=height,
        sh_degree=3, packed=False, radius_clip=0.0,
    )
    r2 = out2[0]
    a2 = out2[1]
    loss2 = r2.sum()
    loss2.backward()
    mg = m2.grad
    n_nz = (mg.abs().reshape(N, -1) > 1e-10).any(dim=-1).sum().item()
    n_exact0 = (mg.reshape(N, -1) == 0).all(dim=-1).sum().item()
    print("  op_scale=%.2f: alpha_mean=%.4f, nz_gaussians=%d, exact_zero=%d" % (
        op_scale, a2.mean().item(), n_nz, n_exact0))
    del m2, o2, out2, r2, a2, loss2, mg
    torch.cuda.empty_cache()