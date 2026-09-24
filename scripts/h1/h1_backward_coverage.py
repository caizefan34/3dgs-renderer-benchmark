#!/usr/bin/env python3
"""H1 Backward Coverage Sanity — P1/P2/P3 with real training loss (L1 + DSSIM).

Tests whether the previous ~6 nonzero-gradient Gaussians was a harness artifact
(random zero-mean upstream gradient) or a real coverage issue.

Uses the actual training loss: loss = 0.8 * L1 + 0.2 * (1 - SSIM)
with GT images from the room scene.
"""
import os, sys, json, csv, math, gc
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

H = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"
CACHE = os.path.expanduser("~/.cache/torch_extensions/py310_cu128")
sys.path.insert(0, H)
sys.path.insert(0, os.path.join(CACHE, "gsplat_scene_cuda"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from plyfile import PlyData
from gsplat.rendering import rasterization
from gsplat.experimental import rasterize_gaussian_higs_frozen
from gsplat.experimental.render.functional.gaussian_inference import (
    create_higs_renderer, _HIGS_FROZEN_TRACKER, _HigsAutogradFunction,
)

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage_pool/3dgs-renderer-benchmark/repo/artifacts/h1-clean-profile"
SH_DEGREE = 3
LAMBDA_DSSIM = 0.2
NZ_THRESH = 1e-10  # nonzero threshold

# ---- SepSSIM (from baseline/reference_v1/trainer.py) ----
class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + self.C1) * (2 * spt + self.C2) / \
                   ((mu_p2 + mu_t2 + self.C1) * (sp2 + st2 + self.C2))
        return 1.0 - ssim_map.mean()

SCENE_CFG = {
    "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
    "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
    "gt_dir": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/datasets/mipnerf360/room/images",
    "native_w": 3114, "native_h": 2075,
}

def load_ply(path, device):
    ply = PlyData.read(path)
    v = ply["vertex"]
    N = len(v)
    def arr(name): return torch.tensor(v[name], dtype=torch.float32, device=device)
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
    return means, quats, scales, opacities, sh

def load_gt_image(gt_dir, img_name, width, height, device):
    """Load and resize GT image to target resolution."""
    # Try .JPG then .jpg then .png
    for ext in [".JPG", ".jpg", ".png"]:
        path = os.path.join(gt_dir, img_name + ext)
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            img = img.resize((width, height), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
            return torch.tensor(arr, device=device)  # [H, W, 3]
    raise FileNotFoundError("GT image not found: %s" % img_name)

def make_viewmat_K(cam, width, height, device):
    R = np.asarray(cam["rotation"], dtype=np.float64)
    p = np.asarray(cam["position"], dtype=np.float64)
    Rw2c = R.T
    vm = np.eye(4); vm[:3,:3] = Rw2c; vm[:3,3] = -Rw2c @ p
    scale = width / float(cam["width"])
    K = np.array([[float(cam["fx"])*scale, 0, (width-1)/2],
                  [0, float(cam["fy"])*scale, (height-1)/2], [0,0,1]], dtype=np.float64)
    vm = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    K = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    return vm, K

def grad_metrics(ref, cand, name):
    if ref is None or cand is None:
        return {"name": name, "status": "MISSING"}
    if ref.shape != cand.shape:
        return {"name": name, "status": "SHAPE_MISMATCH",
                "ref_shape": list(ref.shape), "cand_shape": list(cand.shape)}
    ref_f = ref.flatten().float()
    cand_f = cand.flatten().float()
    diff = ref_f - cand_f
    return {
        "name": name, "status": "OK",
        "ref_norm": float(ref_f.norm().item()),
        "cand_norm": float(cand_f.norm().item()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "relative_l2": float(diff.norm().item() / max(ref_f.norm().item(), 1e-12)),
        "cosine": float(F.cosine_similarity(ref_f.unsqueeze(0), cand_f.unsqueeze(0)).item()),
        "nonzero_ref": int((ref_f.abs() > NZ_THRESH).sum().item()),
        "nonzero_cand": int((cand_f.abs() > NZ_THRESH).sum().item()),
        "zero_nonzero_disagreement": int(((ref_f.abs() > NZ_THRESH) != (cand_f.abs() > NZ_THRESH)).sum().item()),
        "exact_zero_ref": int((ref_f == 0).sum().item()),
        "exact_zero_cand": int((cand_f == 0).sum().item()),
        "NaN": int((ref_f.isnan() | cand_f.isnan()).sum().item()),
        "Inf": int((ref_f.isinf() | cand_f.isinf()).sum().item()),
        "n_elements": int(ref_f.numel()),
    }

def grad_support(grad, name, N_total, N_visible):
    """Measure gradient support for one parameter family."""
    if grad is None:
        return {"name": name, "status": "MISSING"}
    g = grad.float()
    n_elements = g.numel()
    n_exact_zero = int((g == 0).sum().item())
    n_nonzero = int((g.abs() > NZ_THRESH).sum().item())
    # Count nonzero Gaussians (for per-Gaussian params)
    if g.dim() >= 1 and n_elements % N_total == 0:
        g_per_gauss = g.reshape(N_total, -1)
        n_nz_gaussians = int((g_per_gauss.abs() > NZ_THRESH).any(dim=-1).sum().item())
    else:
        n_nz_gaussians = -1
    return {
        "name": name,
        "N_total": N_total,
        "N_visible": N_visible,
        "N_nonzero_elements": n_nonzero,
        "N_exact_zero_elements": n_exact_zero,
        "N_nonzero_gaussians": n_nz_gaussians,
        "nonzero_over_total": float(n_nz_gaussians / max(N_total, 1)) if n_nz_gaussians >= 0 else -1,
        "nonzero_over_visible": float(n_nz_gaussians / max(N_visible, 1)) if n_nz_gaussians >= 0 and N_visible > 0 else -1,
        "grad_norm": float(g.norm().item()),
        "grad_max_abs": float(g.abs().max().item()),
        "grad_mean_abs": float(g.abs().mean().item()),
    }

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

device = torch.device("cuda:0")
ssim_fn = SepSSIM(device=device)

# Resolution: capped at 2048 long side
nw, nh = SCENE_CFG["native_w"], SCENE_CFG["native_h"]
max_long = 2048
scale = max_long / max(nw, nh)
width = int(round(nw * scale))
height = int(round(nh * scale))
print("Resolution: %dx%d (native %dx%d, scale=%.4f)" % (width, height, nw, nh, scale))

# Load scene
print("Loading PLY...")
means, quats, scales, opacities, sh = load_ply(SCENE_CFG["ply"], device)
N = means.shape[0]
print("  N=%d Gaussians" % N)

# Load cameras
cams = json.load(open(SCENE_CFG["cams"]))
n_cams = len(cams)
cam_indices = [0, n_cams // 2]
print("Cameras: %s (of %d)" % (cam_indices, n_cams))

param_names = ["means", "quats", "scales", "opacities", "sh"]
all_results = {}

for cam_idx in cam_indices:
    cam = cams[cam_idx]
    img_name = cam.get("img_name", cam.get("id", str(cam_idx)))
    print("\n%s" % "="*60)
    print("Camera %d (%s)" % (cam_idx, img_name))
    print("="*60)

    vm, K = make_viewmat_K(cam, width, height, device)
    gt = load_gt_image(SCENE_CFG["gt_dir"], img_name, width, height, device)
    print("  GT loaded: %s, shape=%s, range=[%.3f, %.3f]" % (img_name, tuple(gt.shape), gt.min().item(), gt.max().item()))

    cam_results = {}

    # ================================================================
    # P1: B1 clean gsplat backward (rasterization packed=False)
    # ================================================================
    print("\n--- P1: B1 packed=False ---")
    m = means.detach().clone().requires_grad_(True)
    q = quats.detach().clone().requires_grad_(True)
    s = scales.detach().clone().requires_grad_(True)
    o = opacities.detach().clone().requires_grad_(True)
    c = sh.detach().clone().requires_grad_(True)
    out = rasterization(
        means=m.unsqueeze(0), quats=q.unsqueeze(0), scales=s.unsqueeze(0),
        opacities=o.unsqueeze(0), colors=c,
        viewmats=vm, Ks=K, width=width, height=height,
        sh_degree=SH_DEGREE, packed=False, radius_clip=0.0,
    )
    r_out = out[0]  # [1, C, H, W, 3]
    a_out = out[1]  # [1, C, H, W, 1]
    img = r_out[0, 0]  # [H, W, 3]
    # Loss: 0.8 * L1 + 0.2 * (1 - SSIM)
    L1 = F.l1_loss(img, gt)
    dssim = ssim_fn(img, gt)
    loss = (1.0 - LAMBDA_DSSIM) * L1 + LAMBDA_DSSIM * dssim
    print("  L1=%.6f, dssim=%.6f, loss=%.6f" % (L1.item(), dssim.item(), loss.item()))
    loss.backward()
    p1_grads = {k: g.detach().cpu().clone() for k, g in
                [("means", m.grad), ("quats", q.grad), ("scales", s.grad),
                 ("opacities", o.grad), ("sh", c.grad)]}
    p1_render = img.detach().cpu().clone()
    del m, q, s, o, c, out, r_out, a_out, img, L1, dssim, loss
    cleanup()

    # ================================================================
    # P2: B2 HiGS forward + gsplat_recompute backward
    # ================================================================
    print("\n--- P2: B2 gsplat_recompute ---")
    m = means.detach().clone().requires_grad_(True)
    q = quats.detach().clone().requires_grad_(True)
    s = scales.detach().clone().requires_grad_(True)
    o = opacities.detach().clone().requires_grad_(True)
    c = sh.detach().clone().requires_grad_(True)
    _HIGS_FROZEN_TRACKER.reset()
    h = create_higs_renderer(m, q, s, o, c, sh_degree=SH_DEGREE)
    res = rasterize_gaussian_higs_frozen(
        m, q, s, o, c,
        backward_mode="gsplat_recompute", scene=h, freeze_topology=True,
        viewmats=vm, Ks=K, width=width, height=height,
        sh_degree=SH_DEGREE, use_higs_culling=True, radius_clip=0.0,
        tile_sampling_ratio=1.0,
    )
    r_out = res["frame"]  # [1, H, W, 3]
    a_out = res["alpha"]  # [1, H, W, 1]
    img = r_out[0]  # [H, W, 3]
    L1 = F.l1_loss(img, gt)
    dssim = ssim_fn(img, gt)
    loss = (1.0 - LAMBDA_DSSIM) * L1 + LAMBDA_DSSIM * dssim
    print("  L1=%.6f, dssim=%.6f, loss=%.6f" % (L1.item(), dssim.item(), loss.item()))
    loss.backward()
    p2_grads = {k: g.detach().cpu().clone() for k, g in
                [("means", m.grad), ("quats", q.grad), ("scales", s.grad),
                 ("opacities", o.grad), ("sh", c.grad)]}
    p2_render = img.detach().cpu().clone()
    meta_p2 = dict(_HigsAutogradFunction.last_forward_metadata)
    p2_vis_ids = meta_p2.get("visible_gaussian_ids")
    if p2_vis_ids is not None:
        p2_vis_ids = p2_vis_ids.cpu().clone()
        n_vis = p2_vis_ids.numel()
    else:
        n_vis = N
    print("  B2 visible: %d / %d (culling_ratio=%.3f)" % (n_vis, N, meta_p2.get("culling_ratio", -1)))
    h.release()
    del m, q, s, o, c, h, res, r_out, a_out, img, L1, dssim, loss
    cleanup()

    # ================================================================
    # P3: B2 HiGS forward + higs_native backward
    # ================================================================
    print("\n--- P3: B2 higs_native ---")
    m = means.detach().clone().requires_grad_(True)
    q = quats.detach().clone().requires_grad_(True)
    s = scales.detach().clone().requires_grad_(True)
    o = opacities.detach().clone().requires_grad_(True)
    c = sh.detach().clone().requires_grad_(True)
    _HIGS_FROZEN_TRACKER.reset()
    h = create_higs_renderer(m, q, s, o, c, sh_degree=SH_DEGREE)
    res = rasterize_gaussian_higs_frozen(
        m, q, s, o, c,
        backward_mode="higs_native", scene=h, freeze_topology=True,
        viewmats=vm, Ks=K, width=width, height=height,
        sh_degree=SH_DEGREE, use_higs_culling=True, radius_clip=0.0,
        tile_sampling_ratio=1.0,
    )
    r_out = res["frame"]
    a_out = res["alpha"]
    img = r_out[0]
    L1 = F.l1_loss(img, gt)
    dssim = ssim_fn(img, gt)
    loss = (1.0 - LAMBDA_DSSIM) * L1 + LAMBDA_DSSIM * dssim
    print("  L1=%.6f, dssim=%.6f, loss=%.6f" % (L1.item(), dssim.item(), loss.item()))
    loss.backward()
    p3_grads = {k: g.detach().cpu().clone() for k, g in
                [("means", m.grad), ("quats", q.grad), ("scales", s.grad),
                 ("opacities", o.grad), ("sh", c.grad)]}
    p3_render = img.detach().cpu().clone()
    h.release()
    del m, q, s, o, c, h, res, r_out, a_out, img, L1, dssim, loss
    cleanup()

    # ================================================================
    # Forward correctness
    # ================================================================
    print("\n--- Forward correctness ---")
    for a, b, label in [(p1_render, p2_render, "P1 vs P2"), (p1_render, p3_render, "P1 vs P3"),
                         (p2_render, p3_render, "P2 vs P3")]:
        diff = (a - b).abs()
        psnr_val = 60.0 if diff.max().item() < 1e-6 else float(-10*np.log10(max((a-b).pow(2).mean().item(), 1e-12)))
        print("  %s: max_abs=%.6e, psnr=%.2f" % (label, diff.max().item(), psnr_val))

    # ================================================================
    # Gradient support
    # ================================================================
    print("\n--- Gradient support ---")
    support = {}
    for path_name, grads, path_n_vis in [("P1", p1_grads, N), ("P2", p2_grads, n_vis), ("P3", p3_grads, n_vis)]:
        print("  %s (N_visible=%d):" % (path_name, path_n_vis))
        path_support = {}
        for pname in param_names:
            s = grad_support(grads[pname], pname, N, path_n_vis)
            path_support[pname] = s
            print("    %s: nz_gaussians=%d/%d (%.1f%% of total, %.1f%% of visible), grad_norm=%.4f" % (
                pname, s["N_nonzero_gaussians"], N,
                100*s["nonzero_over_total"], 100*s["nonzero_over_visible"],
                s["grad_norm"]))
        support[path_name] = path_support

    # ================================================================
    # Gradient equivalence
    # ================================================================
    print("\n--- Gradient equivalence ---")
    equiv = {}
    for comp_name, (ref_g, cand_g, desc) in [
        ("P1_vs_P2", (p1_grads, p2_grads, "B1 vs B2 recompute")),
        ("P2_vs_P3", (p2_grads, p3_grads, "higs_native vs recompute")),
        ("P1_vs_P3", (p1_grads, p3_grads, "B1 vs B2 native")),
    ]:
        print("  %s (%s):" % (comp_name, desc))
        comp = {}
        for pname in param_names:
            m = grad_metrics(ref_g[pname], cand_g[pname], pname)
            comp[pname] = m
            if m["status"] == "OK":
                print("    %s: cos=%.8f, rel_l2=%.6e, max_abs=%.6e, disagree=%d" % (
                    pname, m["cosine"], m["relative_l2"], m["max_abs"],
                    m["zero_nonzero_disagreement"]))
        equiv[comp_name] = comp

    # Classify
    p1_p2_pass = all(equiv["P1_vs_P2"][n]["status"] == "OK" and equiv["P1_vs_P2"][n]["cosine"] > 0.999
                    and equiv["P1_vs_P2"][n]["relative_l2"] < 1e-3 for n in param_names)
    p2_p3_pass = all(equiv["P2_vs_P3"][n]["status"] == "OK" and equiv["P2_vs_P3"][n]["cosine"] > 0.999
                    and equiv["P2_vs_P3"][n]["relative_l2"] < 1e-3 for n in param_names)
    p1_p3_pass = all(equiv["P1_vs_P3"][n]["status"] == "OK" and equiv["P1_vs_P3"][n]["cosine"] > 0.999
                    and equiv["P1_vs_P3"][n]["relative_l2"] < 1e-3 for n in param_names)
    if p1_p3_pass:
        classification = "BACKWARD_EQUIVALENT"
    elif not p2_p3_pass:
        classification = "BACKWARD_MISMATCH_REAL"
    else:
        classification = "INCONCLUSIVE"
    print("\n  CLASSIFICATION: %s (P1v2=%s, P2v3=%s, P1v3=%s)" % (classification, p1_p2_pass, p2_p3_pass, p1_p3_pass))

    cam_results = {
        "camera_idx": cam_idx,
        "camera_name": img_name,
        "N_total": N,
        "N_visible": n_vis,
        "width": width, "height": height,
        "loss_type": "0.8 * L1 + 0.2 * (1 - SSIM)",
        "lambda_dssim": LAMBDA_DSSIM,
        "nonzero_threshold": NZ_THRESH,
        "gradient_support": support,
        "gradient_equivalence": equiv,
        "classification": classification,
        "p1_p2_pass": p1_p2_pass,
        "p2_p3_pass": p2_p3_pass,
        "p1_p3_pass": p1_p3_pass,
        "forward_correctness": {
            "P1_vs_P2_max_abs": float((p1_render - p2_render).abs().max().item()),
            "P1_vs_P3_max_abs": float((p1_render - p3_render).abs().max().item()),
            "P2_vs_P3_max_abs": float((p2_render - p3_render).abs().max().item()),
        },
    }
    all_results[cam_idx] = cam_results
    del gt, p1_grads, p2_grads, p3_grads, p1_render, p2_render, p3_render
    cleanup()

# ================================================================
# Diagnose previous ~6 Gaussians
# ================================================================
print("\n%s" % "="*60)
print("DIAGNOSIS: Previous ~6 nonzero-gradient Gaussians")
print("="*60)
# The previous test used random upstream gradient v_render ~ N(0,1)
# which has ZERO MEAN. For each Gaussian, the gradient is a sum of
# random zero-mean terms weighted by per-pixel contributions.
# By CLT, this sum ~ N(0, sigma) where sigma depends on the number
# of contributing pixels and the gradient weights.
# For most Gaussians, sigma is small enough that |gradient| < 1e-10.
# Only the 6 most dominant Gaussians (largest per-pixel contributions)
# have |gradient| > 1e-10.
#
# With a REAL training loss (L1 + DSSIM), the upstream gradient
# (render - GT) has NONZERO MEAN, so the gradient sum for each
# Gaussian has a systematic component that does NOT cancel.
# This should produce nonzero gradients for many more Gaussians.

p1_means_support_cam0 = all_results[cam_indices[0]]["gradient_support"]["P1"]["means"]
n_nz_real = p1_means_support_cam0["N_nonzero_gaussians"]
n_nz_prev = 6  # from previous random-gradient test

if n_nz_real > 100:
    diagnosis = "EXPECTED_FROM_TEST_CONSTRUCTION"
    explanation = ("Previous test used random upstream gradient v~N(0,1) with zero mean. "
                   "Gradient for each Gaussian is a sum of zero-mean random terms that "
                   "cancel by CLT, leaving only ~%d dominant Gaussians above threshold. "
                   "Real training loss (L1+DSSIM) has nonzero-mean upstream gradient "
                   "(render != GT), producing %d nonzero-gradient Gaussians. "
                   "The ~6 was NOT a harness bug — it was a consequence of the random "
                   "test construction." % (n_nz_prev, n_nz_real))
elif n_nz_real <= 10:
    diagnosis = "REAL_SPARSE_GRADIENT_SUPPORT"
    explanation = ("Even with real training loss, only %d Gaussians have nonzero gradients. "
                   "This may indicate real sparse gradient support or a deeper issue." % n_nz_real)
else:
    diagnosis = "INCONCLUSIVE"
    explanation = "Real loss produced %d nonzero-gradient Gaussians — between 6 and 100." % n_nz_real

print("  Previous (random grad): %d nonzero Gaussians" % n_nz_prev)
print("  Current (real loss):    %d nonzero Gaussians" % n_nz_real)
print("  Diagnosis: %s" % diagnosis)
print("  %s" % explanation)

# ================================================================
# Overall verdict
# ================================================================
all_pass = all(all_results[ci]["classification"] == "BACKWARD_EQUIVALENT" for ci in cam_indices)
coverage_pass = all(
    all_results[ci]["gradient_support"]["P1"]["means"]["N_nonzero_gaussians"] > 100
    for ci in cam_indices
)
overall = "PASS" if (all_pass and coverage_pass) else "FAIL"
print("\n=== OVERALL: %s ===" % overall)
print("  Backward equivalence: %s" % all_pass)
print("  Gradient coverage: %s (%d nonzero Gaussians with real loss)" % (coverage_pass, n_nz_real))

# ================================================================
# Save
# ================================================================
result = {
    "scene": "room",
    "N_total": N,
    "width": width, "height": height,
    "loss_type": "0.8 * L1 + 0.2 * (1 - SSIM)",
    "lambda_dssim": LAMBDA_DSSIM,
    "nonzero_threshold": NZ_THRESH,
    "cameras": {str(ci): all_results[ci] for ci in cam_indices},
    "diagnosis_previous_6_gaussians": {
        "classification": diagnosis,
        "explanation": explanation,
        "previous_nz_gaussians": n_nz_prev,
        "current_nz_gaussians": n_nz_real,
    },
    "overall_verdict": overall,
    "backward_correctness_closed": all_pass,
}

out_path = os.path.join(OUT_DIR, "backward_coverage_sanity.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
print("\nSaved JSON to %s" % out_path)

# CSV
csv_path = os.path.join(OUT_DIR, "backward_coverage_sanity.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "camera_idx", "path", "parameter", "N_total", "N_visible",
                "N_nonzero_gaussians", "nonzero_over_total", "nonzero_over_visible",
                "grad_norm", "grad_max_abs", "grad_mean_abs", "loss_type"])
    for ci in cam_indices:
        r = all_results[ci]
        for path_name in ["P1", "P2", "P3"]:
            for pname in param_names:
                s = r["gradient_support"][path_name][pname]
                w.writerow(["room", ci, path_name, pname, s["N_total"], s["N_visible"],
                           s["N_nonzero_gaussians"], s["nonzero_over_total"],
                           s["nonzero_over_visible"], s["grad_norm"],
                           s["grad_max_abs"], s["grad_mean_abs"],
                           "0.8*L1+0.2*(1-SSIM)"])
    # Equivalence rows
    w.writerow([])
    w.writerow(["scene", "camera_idx", "comparison", "parameter", "cosine", "relative_l2",
                "max_abs", "mean_abs", "zero_nonzero_disagreement", "NaN", "Inf", "classification"])
    for ci in cam_indices:
        r = all_results[ci]
        for comp_name in ["P1_vs_P2", "P2_vs_P3", "P1_vs_P3"]:
            for pname in param_names:
                d = r["gradient_equivalence"][comp_name][pname]
                if d["status"] == "OK":
                    w.writerow(["room", ci, comp_name, pname, d["cosine"], d["relative_l2"],
                               d["max_abs"], d["mean_abs"], d["zero_nonzero_disagreement"],
                               d["NaN"], d["Inf"], r["classification"]])
print("Saved CSV to %s" % csv_path)