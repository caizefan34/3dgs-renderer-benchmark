#!/usr/bin/env python3
"""H1-R Part A: Backward correctness repair — P1/P2/P3 gradient comparison.

Fixes:
1. Memory: run each path, save grads to CPU, delete tensors, empty cache between paths.
2. P1 uses packed=False to match B2's gsplat_recompute internal path (fair comparison).
   Also runs packed=True for reference.
3. Non-zero threshold: > 1e-10 (not > 0) to catch tiny but non-zero gradients.
"""
import os, sys, json, math, csv, gc
import torch
import torch.nn.functional as F
import numpy as np

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
SCENE = "room"
CAM_IDX = 0
SH_DEGREE = 3
NZ_THRESH = 1e-10  # non-zero threshold

SCENES = {
    "room": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
    },
}

def load_ply(path, device):
    ply = PlyData.read(path)
    v = ply["vertex"]
    N = len(v)
    def arr(name):
        return torch.tensor(v[name], dtype=torch.float32, device=device)
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

def load_camera(cams_path, cam_idx, device):
    cams = json.load(open(cams_path))
    cam = cams[cam_idx]
    R = np.asarray(cam["rotation"], dtype=np.float64)
    p = np.asarray(cam["position"], dtype=np.float64)
    Rw2c = R.T
    vm = np.eye(4); vm[:3,:3] = Rw2c; vm[:3,3] = -Rw2c @ p
    width = 2048; height = 1365
    scale = width / float(cam["width"])
    K = np.array([[float(cam["fx"])*scale, 0, (width-1)/2],
                  [0, float(cam["fy"])*scale, (height-1)/2],
                  [0, 0, 1]], dtype=np.float64)
    vm = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    K = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    return vm, K, width, height, cam["id"]

device = torch.device("cuda:0")
print("Loading scene %s cam %d..." % (SCENE, CAM_IDX))
means, quats, scales, opacities, sh = load_ply(SCENES[SCENE]["ply"], device)
vm, K, width, height, cam_id = load_camera(SCENES[SCENE]["cams"], CAM_IDX, device)
N = means.shape[0]
print("  N=%d, res=%dx%d, cam_id=%s" % (N, width, height, cam_id))

# Fixed random upstream gradient (seed=42)
torch.manual_seed(42)
n_pixels = width * height
v_render_flat = torch.randn(n_pixels * 3, device=device, dtype=torch.float32)
v_alpha_flat = torch.randn(n_pixels, device=device, dtype=torch.float32)

def make_upstream(shape_render, shape_alpha):
    return v_render_flat.reshape(shape_render), v_alpha_flat.reshape(shape_alpha)

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
        "NaN": int((ref_f.isnan() | cand_f.isnan()).sum().item()),
        "Inf": int((ref_f.isinf() | cand_f.isinf()).sum().item()),
        "n_elements": int(ref_f.numel()),
    }

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# ================================================================
# P1a: B1 clean gsplat backward (rasterization packed=True)
# ================================================================
print("\n=== P1a: B1 packed=True ===")
m = means.detach().clone().requires_grad_(True)
q = quats.detach().clone().requires_grad_(True)
s = scales.detach().clone().requires_grad_(True)
o = opacities.detach().clone().requires_grad_(True)
c = sh.detach().clone().requires_grad_(True)
out = rasterization(
    means=m.unsqueeze(0), quats=q.unsqueeze(0), scales=s.unsqueeze(0),
    opacities=o.unsqueeze(0), colors=c,
    viewmats=vm, Ks=K, width=width, height=height,
    sh_degree=SH_DEGREE, packed=True, radius_clip=0.0,
)
r_out, a_out = out[0], out[1]
vr, va = make_upstream(r_out.shape, a_out.shape)
loss = (r_out.float() * vr).sum() + (a_out.float() * va).sum()
loss.backward()
p1a_grads = {k: g.detach().cpu().clone() for k, g in
             [("means", m.grad), ("quats", q.grad), ("scales", s.grad),
              ("opacities", o.grad), ("sh", c.grad)]}
p1a_render = r_out.detach().reshape(-1).float().cpu().clone()
print("  loss=%.2f, means_grad norm=%.4f nz=%d/%d" % (
    loss.item(), p1a_grads["means"].norm().item(),
    (p1a_grads["means"].abs() > NZ_THRESH).sum().item(), p1a_grads["means"].numel()))
del m, q, s, o, c, out, r_out, a_out, vr, va, loss
cleanup()

# ================================================================
# P1b: B1 clean gsplat backward (rasterization packed=False)
# ================================================================
print("\n=== P1b: B1 packed=False ===")
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
r_out, a_out = out[0], out[1]
vr, va = make_upstream(r_out.shape, a_out.shape)
loss = (r_out.float() * vr).sum() + (a_out.float() * va).sum()
loss.backward()
p1b_grads = {k: g.detach().cpu().clone() for k, g in
             [("means", m.grad), ("quats", q.grad), ("scales", s.grad),
              ("opacities", o.grad), ("sh", c.grad)]}
p1b_render = r_out.detach().reshape(-1).float().cpu().clone()
print("  loss=%.2f, means_grad norm=%.4f nz=%d/%d" % (
    loss.item(), p1b_grads["means"].norm().item(),
    (p1b_grads["means"].abs() > NZ_THRESH).sum().item(), p1b_grads["means"].numel()))
del m, q, s, o, c, out, r_out, a_out, vr, va, loss
cleanup()

# ================================================================
# P2: B2 HiGS forward + gsplat_recompute backward
# ================================================================
print("\n=== P2: B2 gsplat_recompute ===")
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
r_out, a_out = res["frame"], res["alpha"]
vr, va = make_upstream(r_out.shape, a_out.shape)
loss = (r_out.float() * vr).sum() + (a_out.float() * va).sum()
loss.backward()
p2_grads = {k: g.detach().cpu().clone() for k, g in
            [("means", m.grad), ("quats", q.grad), ("scales", s.grad),
             ("opacities", o.grad), ("sh", c.grad)]}
p2_render = r_out.detach().reshape(-1).float().cpu().clone()
meta_p2 = dict(_HigsAutogradFunction.last_forward_metadata)
p2_vis_ids = meta_p2.get("visible_gaussian_ids")
if p2_vis_ids is not None:
    p2_vis_ids = p2_vis_ids.cpu().clone()
print("  loss=%.2f, means_grad norm=%.4f nz=%d/%d" % (
    loss.item(), p2_grads["means"].norm().item(),
    (p2_grads["means"].abs() > NZ_THRESH).sum().item(), p2_grads["means"].numel()))
if p2_vis_ids is not None:
    print("  B2 visible: %d / %d (culling_ratio=%.3f)" % (
        p2_vis_ids.numel(), N, meta_p2.get("culling_ratio", -1)))
h.release()
del m, q, s, o, c, h, res, r_out, a_out, vr, va, loss
cleanup()

# ================================================================
# P3: B2 HiGS forward + higs_native backward
# ================================================================
print("\n=== P3: B2 higs_native ===")
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
r_out, a_out = res["frame"], res["alpha"]
vr, va = make_upstream(r_out.shape, a_out.shape)
loss = (r_out.float() * vr).sum() + (a_out.float() * va).sum()
loss.backward()
p3_grads = {k: g.detach().cpu().clone() for k, g in
            [("means", m.grad), ("quats", q.grad), ("scales", s.grad),
             ("opacities", o.grad), ("sh", c.grad)]}
p3_render = r_out.detach().reshape(-1).float().cpu().clone()
meta_p3 = dict(_HigsAutogradFunction.last_forward_metadata)
p3_vis_ids = meta_p3.get("visible_gaussian_ids")
if p3_vis_ids is not None:
    p3_vis_ids = p3_vis_ids.cpu().clone()
print("  loss=%.2f, means_grad norm=%.4f nz=%d/%d" % (
    loss.item(), p3_grads["means"].norm().item(),
    (p3_grads["means"].abs() > NZ_THRESH).sum().item(), p3_grads["means"].numel()))
if p3_vis_ids is not None:
    print("  B2 visible: %d / %d (culling_ratio=%.3f)" % (
        p3_vis_ids.numel(), N, meta_p3.get("culling_ratio", -1)))
h.release()
del m, q, s, o, c, h, res, r_out, a_out, vr, va, loss
cleanup()

# ================================================================
# Forward correctness
# ================================================================
print("\n=== Forward correctness ===")
for a, b, label in [(p1a_render, p2_render, "P1a vs P2"), (p1b_render, p2_render, "P1b vs P2"),
                     (p1a_render, p3_render, "P1a vs P3"), (p1b_render, p3_render, "P1b vs P3"),
                     (p2_render, p3_render, "P2 vs P3")]:
    diff = (a - b).abs()
    psnr_val = 60.0 if diff.max().item() < 1e-6 else float(-10*np.log10(max((a-b).pow(2).mean().item(), 1e-12)))
    print("  %s: max_abs=%.6e, psnr=%.2f" % (label, diff.max().item(), psnr_val))

# ================================================================
# Gradient comparisons
# ================================================================
param_names = ["means", "quats", "scales", "opacities", "sh"]

comparisons = {
    "P2_vs_P3": (p2_grads, p3_grads, "higs_native vs gsplat_recompute"),
    "P1b_vs_P2": (p1b_grads, p2_grads, "B1 packed=False vs B2 gsplat_recompute"),
    "P1b_vs_P3": (p1b_grads, p3_grads, "B1 packed=False vs B2 higs_native"),
    "P1a_vs_P1b": (p1a_grads, p1b_grads, "B1 packed=True vs B1 packed=False"),
    "P1a_vs_P3": (p1a_grads, p3_grads, "B1 packed=True vs B2 higs_native"),
}

all_results = {}
for comp_name, (ref_grads, cand_grads, desc) in comparisons.items():
    print("\n=== %s (%s) ===" % (comp_name, desc))
    comp_results = {}
    for name in param_names:
        m_result = grad_metrics(ref_grads[name], cand_grads[name], name)
        comp_results[name] = m_result
        if m_result["status"] == "OK":
            print("  %s: cos=%.8f, rel_l2=%.6e, max_abs=%.6e, nz_ref=%d, nz_cand=%d, disagree=%d" % (
                name, m_result["cosine"], m_result["relative_l2"], m_result["max_abs"],
                m_result["nonzero_ref"], m_result["nonzero_cand"],
                m_result["zero_nonzero_disagreement"]))
        else:
            print("  %s: %s" % (name, m_result["status"]))
    all_results[comp_name] = comp_results

# ================================================================
# Classification
# ================================================================
def check_pass(comp):
    return all(comp[n]["status"] == "OK" and comp[n]["cosine"] > 0.999 and comp[n]["relative_l2"] < 1e-3
               for n in param_names)

p2_p3_pass = check_pass(all_results["P2_vs_P3"])
p1b_p2_pass = check_pass(all_results["P1b_vs_P2"])
p1b_p3_pass = check_pass(all_results["P1b_vs_P3"])
p1a_p1b_pass = check_pass(all_results["P1a_vs_P1b"])

if p1b_p3_pass:
    classification = "BACKWARD_EQUIVALENT"
elif p2_p3_pass and not p1b_p2_pass:
    classification = "BACKWARD_MISMATCH_REAL"
elif not p2_p3_pass:
    classification = "BACKWARD_MISMATCH_REAL"
else:
    classification = "INCONCLUSIVE"

print("\n=== CLASSIFICATION: %s ===" % classification)
print("  P2 vs P3 pass: %s" % p2_p3_pass)
print("  P1b vs P2 pass: %s" % p1b_p2_pass)
print("  P1b vs P3 pass: %s" % p1b_p3_pass)
print("  P1a vs P1b pass: %s" % p1a_p1b_pass)

# ================================================================
# Mapping audit
# ================================================================
print("\n=== Mapping audit ===")
b1b_nz = int((p1b_grads["means"].abs() > NZ_THRESH).any(dim=-1).sum().item())
b1a_nz = int((p1a_grads["means"].abs() > NZ_THRESH).any(dim=-1).sum().item())
if p2_vis_ids is not None:
    n_vis = p2_vis_ids.numel()
    b1b_vis_mask = (p1b_grads["means"].abs() > NZ_THRESH).any(dim=-1)
    b2_vis_mask = torch.zeros(N, dtype=torch.bool)
    b2_vis_mask[p2_vis_ids] = True
    overlap = int((b1b_vis_mask & b2_vis_mask).sum().item())
    b1b_only = int((b1b_vis_mask & ~b2_vis_mask).sum().item())
    b2_only = int((~b1b_vis_mask & b2_vis_mask).sum().item())
    print("  B1a(packed=True) nz gaussians: %d" % b1a_nz)
    print("  B1b(packed=False) nz gaussians: %d" % b1b_nz)
    print("  B2 visible: %d" % n_vis)
    print("  Overlap B1b&B2: %d, B1b-only: %d, B2-only: %d" % (overlap, b1b_only, b2_only))
else:
    n_vis = -1
    overlap = b1b_only = b2_only = -1

# ================================================================
# Save
# ================================================================
result = {
    "scene": SCENE, "camera_idx": CAM_IDX, "camera_id": cam_id,
    "N_total": N, "width": width, "height": height,
    "root_cause_previous_cosine0": "HARNESS_BUG: target=b1_render.clone(), B1/B2 produce identical renders, loss=(render-target).abs().mean()=0, zero gradients, cosine(zero,zero)=0",
    "upstream_gradient": "fixed random (seed=42), v_render~N(0,1)[H*W*3], v_alpha~N(0,1)[H*W]",
    "nonzero_threshold": NZ_THRESH,
    "forward_correctness": {
        "P1a_vs_P2_max_abs": float((p1a_render - p2_render).abs().max().item()),
        "P1b_vs_P2_max_abs": float((p1b_render - p2_render).abs().max().item()),
        "P1a_vs_P3_max_abs": float((p1a_render - p3_render).abs().max().item()),
        "P1b_vs_P3_max_abs": float((p1b_render - p3_render).abs().max().item()),
        "P2_vs_P3_max_abs": float((p2_render - p3_render).abs().max().item()),
    },
    "gradient_comparisons": all_results,
    "classification": classification,
    "p2_p3_pass": p2_p3_pass,
    "p1b_p2_pass": p1b_p2_pass,
    "p1b_p3_pass": p1b_p3_pass,
    "p1a_p1b_pass": p1a_p1b_pass,
    "mapping_audit": {
        "B1a_packed_True_nz_gaussians": b1a_nz,
        "B1b_packed_False_nz_gaussians": b1b_nz,
        "B2_visible": n_vis,
        "overlap_B1b_B2": overlap,
        "B1b_only": b1b_only,
        "B2_only": b2_only,
    },
}

out_path = os.path.join(OUT_DIR, "backward_correctness_repair.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
print("\nSaved to %s" % out_path)

# CSV
csv_path = os.path.join(OUT_DIR, "backward_correctness_repair.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "camera_idx", "comparison", "parameter", "ref_norm", "cand_norm",
                "max_abs", "mean_abs", "relative_l2", "cosine", "nonzero_ref", "nonzero_cand",
                "zero_nonzero_disagreement", "NaN", "Inf", "classification"])
    for comp_name, comp_data in all_results.items():
        for pname in param_names:
            d = comp_data[pname]
            if d["status"] == "OK":
                w.writerow([SCENE, CAM_IDX, comp_name, pname,
                            d["ref_norm"], d["cand_norm"], d["max_abs"], d["mean_abs"],
                            d["relative_l2"], d["cosine"], d["nonzero_ref"], d["nonzero_cand"],
                            d["zero_nonzero_disagreement"], d["NaN"], d["Inf"], classification])
            else:
                w.writerow([SCENE, CAM_IDX, comp_name, pname, "", "", "", "", "", "", "", "", "", "", d["status"]])
print("Saved CSV to %s" % csv_path)