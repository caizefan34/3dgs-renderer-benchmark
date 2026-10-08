"""Frozen Stage-0 fresh-checkpoint spectral diagnostic for SAD event-1500 captures."""
import hashlib
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from gaussian_renderer import render_structgs
from scene import GaussianModel
from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text
from scene.dataset_readers import readColmapCameras
from utils.camera_utils import cameraList_from_camInfos
from utils.freq_utils import compute_projected_axes_subset, update_freq_stats_online
from utils.loss_utils import get_multiscale_structure_tensor_v1


BASE = Path("/root/sadgs_joint_birth_p0_20260928")
DATA = Path("/root/epic05_p16rb/common_data/mipnerf360")
OUT = Path("/root/sadgs_joint_birth_stage0_20260928")
PIPE = Namespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
BG = torch.zeros(3, dtype=torch.float32, device="cuda")
MULT = 0.25
ETA_PARITY_TOL = 1e-4
DISCARDED_TRACE_GATE = 0.01
PARAMS = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")
TENSORS = ("max_radii2D", "xyz_gradient_accum", "xyz_gradient_accum_abs", "denom", "filter_3D",
           "tmp_radii", "accum_eta", "accum_view_count", "max_eta_3ch", "accum_weights_valid",
           "densify_count", "eta_high_count", "eta_high_sum_3ch", "eta_mid_count",
           "eta_mid_sum_3ch", "eta_low_count")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_cameras_only(scene_name):
    root = DATA / scene_name
    sparse = root / "sparse/0"
    try:
        ext = read_extrinsics_binary(str(sparse / "images.bin"))
        intr = read_intrinsics_binary(str(sparse / "cameras.bin"))
    except Exception:
        sparse = root / "sparse"
        ext = read_extrinsics_text(str(sparse / "images.txt"))
        intr = read_intrinsics_text(str(sparse / "cameras.txt"))
    # Match the pinned LLFF holdout split before opening any held-out image.
    all_ext = sorted(ext.items(), key=lambda kv: os.path.basename(kv[1].name))
    train_ext = {key: value for idx, (key, value) in enumerate(all_ext) if idx % 8 != 0}
    infos = readColmapCameras(train_ext, intr, str(root / "images"))
    infos.sort(key=lambda c: c.image_name)
    args = Namespace(resolution=1, data_device="cuda")
    return cameraList_from_camInfos(infos, 1.0, args)


def load_render_model(entry):
    model = GaussianModel(3, "hybrid")
    tensors = entry["tensors"]
    for name in PARAMS:
        setattr(model, name, nn.Parameter(tensors[name].to("cuda"), requires_grad=False))
    for name in TENSORS:
        value = tensors.get(name)
        if value is not None:
            setattr(model, name, value.to("cuda"))
    model.active_sh_degree = int(entry["active_sh_degree"])
    model.spatial_lr_scale = float(entry["spatial_lr_scale"])
    return model


def subset_model(model, ids):
    sub = GaussianModel(3, "hybrid")
    for name in PARAMS:
        setattr(sub, name, nn.Parameter(getattr(model, name).detach()[ids].clone(), requires_grad=False))
    n = int(ids.numel())
    sub.max_eta_3ch = torch.zeros((n, 3), device="cuda")
    sub.accum_eta = torch.zeros(n, device="cuda")
    sub.accum_view_count = torch.zeros(n, device="cuda")
    sub.accum_weights_valid = torch.zeros(n, device="cuda")
    sub.eta_high_count = torch.zeros(n, device="cuda")
    sub.eta_high_sum_3ch = torch.zeros((n, 3), device="cuda")
    sub.eta_mid_count = torch.zeros(n, device="cuda")
    sub.eta_mid_sum_3ch = torch.zeros((n, 3), device="cuda")
    sub.eta_low_count = torch.zeros(n, device="cuda")
    return sub


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"count": 0}
    return {"count": int(x.size), "mean": float(x.mean()), "p50": float(np.quantile(x, .5)),
            "p90": float(np.quantile(x, .9)), "p95": float(np.quantile(x, .95)), "max": float(x.max())}


def qstats(values, mask):
    if mask is None:
        return stats(values)
    return stats(np.asarray(values)[np.asarray(mask, dtype=bool)])


def one_view(model, camera, family_ids_cpu, st_seed, trans_thresh, opacity_thresh, levels=4):
    with torch.no_grad():
        pkg = render_structgs(camera, model, PIPE, BG, MULT)
    ids = family_ids_cpu.to(device="cuda", dtype=torch.long)
    cov_all = pkg["cov2D"]
    cov = cov_all[ids].detach()
    vis = pkg["radii"][ids] > 0
    original = model
    sub = subset_model(original, ids)
    image = camera.original_image.detach().unsqueeze(0)
    with torch.no_grad():
        st = get_multiscale_structure_tensor_v1(image, levels=levels)
    is_active = vis & (cov[:, 6] > trans_thresh) & (sub.get_opacity[:, 0] > opacity_thresh)

    # Run the pinned online updater on exactly this camera/parent subset; no training
    # gradient exists for the fresh-checkpoint replay, so the optional grad gate is off.
    torch.manual_seed(st_seed)
    torch.cuda.manual_seed_all(st_seed)
    jitter_state = torch.cuda.get_rng_state()
    update_freq_stats_online(camera, sub, cov, vis.squeeze(-1) if vis.ndim > 1 else vis,
                             {camera.image_name: st}, viewspace_point_tensor=None,
                             grad_threshold=None, transmittance_threshold=trans_thresh,
                             opacity_threshold=opacity_thresh, eta_compute_mode="wavelength")
    source_eta = sub.max_eta_3ch.detach().clone()
    torch.cuda.set_rng_state(jitter_state)

    active_idx = torch.nonzero(is_active, as_tuple=False).flatten()
    n = int(ids.numel())
    G = torch.zeros((n, 3, 3), dtype=torch.float32, device="cuda")
    H = torch.zeros_like(G)
    eta_h = torch.zeros((n, 3), dtype=torch.float32, device="cuda")
    if active_idx.numel():
        c = cov[active_idx]
        means = c[:, 3:5]
        depths = c[:, 5]
        scales = sub.get_scaling[active_idx]
        rotations = sub.get_rotation[active_idx]
        w, h = camera.image_width, camera.image_height
        xx, xy, yy = c[:, 0], c[:, 1], c[:, 2]
        eps1 = torch.randn_like(xx)
        eps2 = torch.randn_like(xx)
        l11 = torch.sqrt(torch.clamp(xx, min=1e-6))
        l21 = xy / l11
        l22 = torch.sqrt(torch.clamp(yy - l21.square(), min=1e-6))
        sx = means[:, 0] + l11 * eps1
        sy = means[:, 1] + l21 * eps1 + l22 * eps2
        grid = torch.stack([(sx / (w - 1)) * 2 - 1, (sy / (h - 1)) * 2 - 1], dim=-1).view(1, 1, -1, 2)
        sampled = F.grid_sample(st, grid, align_corners=True).squeeze()
        if sampled.ndim == 1:
            sampled = sampled.unsqueeze(1)
        sampled = sampled.permute(1, 0)
        trace = sampled[:, 0] + sampled[:, 2]
        det = sampled[:, 0] * sampled[:, 2] - sampled[:, 1].square()
        delta = torch.sqrt(torch.clamp((trace / 2).square() - det, min=0.0))
        lambda1 = trace / 2 + delta
        freq = torch.sqrt(lambda1) + 1e-5
        axes = compute_projected_axes_subset(means, depths, scales, rotations, camera)
        A = axes.transpose(1, 2)
        G_active = A.transpose(1, 2) @ A
        H_active = G_active * freq[:, None, None].square()
        eta_h_active = torch.sqrt(torch.diagonal(H_active, dim1=1, dim2=2).clamp_min(0))
        G[active_idx] = G_active
        H[active_idx] = H_active
        eta_h[active_idx] = eta_h_active
    parity = (source_eta[is_active] - eta_h[is_active]).abs()
    max_err = float(parity.max().item()) if parity.numel() else 0.0
    mean_err = float(parity.mean().item()) if parity.numel() else 0.0
    if max_err > ETA_PARITY_TOL:
        raise RuntimeError(f"single-view eta/H parity failed on {camera.image_name}: {max_err}")
    result = {
        "G": G.cpu().numpy(), "H": H.cpu().numpy(), "eta_source": source_eta.cpu().numpy(),
        "eta_H": eta_h.cpu().numpy(), "active": is_active.cpu().numpy(),
        "parity": {"camera": camera.image_name, "active_three_axis_parents": int(is_active.sum().item()),
                   "max_abs": max_err, "mean_abs": mean_err, "tolerance": ETA_PARITY_TOL, "passed": True},
    }
    del pkg, cov_all, cov, sub, st, G, H, eta_h
    torch.cuda.empty_cache()
    return result


def analyze_stack(matrices, active, q, label):
    # matrices: [V,N,3,3], active: [V,N], q: [N,3,3] eigenvectors in columns, largest first.
    out = {}
    for v in range(matrices.shape[0]):
        hv = matrices[v]
        diag = np.einsum("nik,nij,njk->nk", q, hv, q, optimize=True)
        tr = np.trace(hv, axis1=1, axis2=2)
        valid = active[v] & (tr > 0)
        for rank in (1, 2):
            discarded = diag[:, rank:].sum(axis=1) / np.maximum(tr, 1e-30)
            out.setdefault(str(rank), {})[str(v)] = {"valid_parents": int(valid.sum()),
                "discarded_trace_fraction": stats(discarded[valid]),
                "parents_under_1pct": int((discarded[valid] <= DISCARDED_TRACE_GATE).sum())}
    return out


def spectral_summary(G, H, active, a_count):
    fit = active[:a_count].any(axis=0)
    gsum = np.where(active[:a_count, :, None, None], G[:a_count], 0).sum(axis=0, dtype=np.float64)
    hsum = np.where(active[:a_count, :, None, None], H[:a_count], 0).sum(axis=0, dtype=np.float64)
    eg, qg = np.linalg.eigh(gsum[fit])
    eh, qh = np.linalg.eigh(hsum[fit])
    eg = eg[:, ::-1]; qg = qg[:, :, ::-1]
    eh = eh[:, ::-1]; qh = qh[:, :, ::-1]
    gtrace = np.maximum(eg.sum(axis=1), 1e-30)
    htrace = np.maximum(eh.sum(axis=1), 1e-30)
    gcorr = np.stack([gsum[fit, 0, 1] / np.sqrt(np.maximum(gsum[fit, 0, 0] * gsum[fit, 1, 1], 1e-30)),
                      gsum[fit, 0, 2] / np.sqrt(np.maximum(gsum[fit, 0, 0] * gsum[fit, 2, 2], 1e-30)),
                      gsum[fit, 1, 2] / np.sqrt(np.maximum(gsum[fit, 1, 1] * gsum[fit, 2, 2], 1e-30))], axis=1)
    hcorr = np.stack([hsum[fit, 0, 1] / np.sqrt(np.maximum(hsum[fit, 0, 0] * hsum[fit, 1, 1], 1e-30)),
                      hsum[fit, 0, 2] / np.sqrt(np.maximum(hsum[fit, 0, 0] * hsum[fit, 2, 2], 1e-30)),
                      hsum[fit, 1, 2] / np.sqrt(np.maximum(hsum[fit, 1, 1] * hsum[fit, 2, 2], 1e-30))], axis=1)
    gspectrum = eg / gtrace[:, None]
    hspectrum = eh / htrace[:, None]
    return {"fit_active_parent_count": int(fit.sum()), "geometry_eigenvalue_ratio": [stats(gspectrum[:, i]) for i in range(3)],
            "frequency_eigenvalue_ratio": [stats(hspectrum[:, i]) for i in range(3)],
            "geometry_abs_offdiag_correlation": stats(np.abs(gcorr).reshape(-1)),
            "frequency_abs_offdiag_correlation": stats(np.abs(hcorr).reshape(-1)),
            "geometry_discarded_response_in_fit_eigenframe": analyze_stack(G, active, qg, a_count),
            "frequency_discarded_response_in_fit_eigenframe": analyze_stack(H, active, qh, a_count),
            "fit_indices": np.flatnonzero(fit), "Q_H": qh, "Q_G": qg, "fit_mask": fit}


def main(scene_name):
    torch.cuda.set_device(0)
    started = time.time()
    out = OUT / scene_name
    out.mkdir(parents=True, exist_ok=True)
    capdir = BASE / scene_name / "capture"
    entry_path = capdir / "split_entry.pt"
    capture_path = capdir / "capture_meta.json"
    capture_meta = json.loads(capture_path.read_text())
    src = BASE
    entry_blob = torch.load(entry_path, map_location="cpu", mmap=True, weights_only=False)
    entry = entry_blob["model"]
    parents = entry_blob["parents"].to(torch.int64)
    native_counts = entry_blob["child_counts"].to(torch.int64)
    eta_all = entry_blob["eta"]
    if eta_all is None:
        raise RuntimeError("capture lacks eta input")
    k_all = torch.sqrt(torch.clamp(eta_all[parents].float(), min=1.0)).ceil().to(torch.int64).clamp(min=1)
    k_product = k_all.prod(dim=1)
    parity_bad = int((k_product != native_counts).sum())
    if parity_bad:
        raise RuntimeError(f"stored eta to native child-count parity mismatch: {parity_bad}")
    three_axis = (k_all > 1).all(dim=1)
    eligible_ids = parents[three_axis]
    historical_children = native_counts[three_axis]
    entry_rows = int(entry["tensors"]["_xyz"].shape[0])
    expected_children = int(capture_meta["split_child_count"])
    if int(native_counts.sum()) != expected_children:
        raise RuntimeError("capture metadata child total mismatch")
    model = load_render_model(entry)
    cameras = train_cameras_only(scene_name)
    n_train = len(cameras)
    index12 = [round(k * (n_train - 1) / 11) for k in range(12)]
    selected = [cameras[index] for index in index12]
    names = [c.image_name for c in selected]
    if names != capture_meta["A_camera_names"] + [
        sorted(c.image_name for c in cameras)[index12[k]] for k in range(1, 12, 2)
    ]:
        # Capture metadata records only even-indexed fit cameras.
        if [names[k] for k in range(0, 12, 2)] != capture_meta["A_camera_names"]:
            raise RuntimeError("fixed 12-camera identities disagree with capture-time A cameras")
    a_names = [names[k] for k in range(0, 12, 2)]
    b_names = [names[k] for k in range(1, 12, 2)]
    trans = 0.8 if scene_name == "room" else 0.0
    opac = 0.5 if scene_name == "room" else 0.05
    levels = int(entry["training_options"].get("st_levels", 4))
    raw = []
    for i, cam in enumerate(selected):
        print(f"VIEW {i+1}/12 {cam.image_name} rows={entry_rows} eligible={len(eligible_ids)}", flush=True)
        raw.append(one_view(model, cam, eligible_ids, 204800 + i, trans, opac, levels=levels))
        torch.cuda.synchronize()
        print(f"PARITY {json.dumps(raw[-1]['parity'])}", flush=True)
    G = np.stack([r["G"] for r in raw])
    H = np.stack([r["H"] for r in raw])
    active = np.stack([r["active"] for r in raw])
    fresh_eta_source = np.stack([r["eta_source"] for r in raw])
    fresh_eta_h = np.stack([r["eta_H"] for r in raw])
    a_count = 6
    spectral = spectral_summary(G, H, active, a_count)
    qh = spectral["Q_H"]
    qg = spectral["Q_G"]
    fit = spectral["fit_mask"]

    # Use A-only fresh directional envelopes to propose diagnostic counts; B remains a check.
    def envelope_counts(frame, matrix_stack, view_mask):
        proj = np.einsum("nik,vnij,njk->vnk", frame, matrix_stack, frame, optimize=True)
        proj = np.maximum(proj, 0.0)
        proj[~view_mask] = 0.0
        eta = np.sqrt(proj)
        env = eta.max(axis=0)
        k = np.ceil(np.sqrt(np.maximum(env, 1.0))).astype(np.int64)
        return k, np.prod(k, axis=1), eta

    k_unrot_a, c_unrot_a, eta_unrot = envelope_counts(np.broadcast_to(np.eye(3), (len(eligible_ids), 3, 3)), H, active[:a_count])
    k_rot_a, c_rot_a, eta_rot = envelope_counts(qh, H, active[:a_count])
    k_rot_b, c_rot_b, _ = envelope_counts(qh, H[a_count:], active[a_count:])
    historical = historical_children.numpy()
    fit_mask = active[:a_count].any(axis=0)
    check_mask = active[a_count:].any(axis=0)
    counts = {
        "eligible_three_axis_parents": int(len(eligible_ids)),
        "historical_native_children": int(historical.sum()),
        "fresh_A_unrotated_children": int(c_unrot_a[fit_mask].sum()),
        "fresh_A_rotated_H_eigenframe_children": int(c_rot_a[fit_mask].sum()),
        "fresh_A_rotated_vs_unrotated_reduction_fraction": float(1 - c_rot_a[fit_mask].sum() / max(1, c_unrot_a[fit_mask].sum())),
        "fresh_B_unrotated_children": int(envelope_counts(np.broadcast_to(np.eye(3), (len(eligible_ids), 3, 3)), H[a_count:], active[a_count:])[1][check_mask].sum()),
        "fresh_B_rotated_H_eigenframe_children": int(c_rot_b[check_mask].sum()),
        "A_active_eligible_parent_count": int(fit_mask.sum()), "B_active_eligible_parent_count": int(check_mask.sum()),
        "per_parent_historical_native_child_count": stats(historical),
        "per_parent_fresh_A_unrotated_child_count": stats(c_unrot_a[fit_mask]),
        "per_parent_fresh_A_rotated_child_count": stats(c_rot_a[fit_mask]),
    }
    parity_rows = [r["parity"] for r in raw]
    report = {
        "status": "STAGE0_FRESH_CHECKPOINT_SPECTRAL_GATE",
        "scene": scene_name, "source_commit": "029814fb5ed773e4a794cec5115951a64f610ab8",
        "capture_iteration": entry.get("context", {}).get("iteration"),
        "eta_mode": "wavelength", "eta_mapping": "k=ceil(sqrt(max(eta,1))); child_count=product(k); no executable upper clamp",
        "native_eta_child_count_parity": {"parents": int(len(parents)), "children": int(native_counts.sum()),
             "matching_parent_products": int((k_product == native_counts).sum()), "mismatch_count": parity_bad},
        "three_axis_native_families": {"parents": int(three_axis.sum()), "children": int(historical.sum())},
        "camera_protocol": {"train_camera_count": n_train, "sorted_index_rule": "round(k*(Ntrain-1)/11), k=0..11",
             "A_fit_camera_names": a_names, "B_internal_check_camera_names": b_names,
             "heldout_images_loaded": False, "B_are_training_views_seen_during_prefix": True},
        "single_view_eta_formula_parity": {"tolerance": ETA_PARITY_TOL, "view_results": parity_rows,
             "all_pass": all(x["passed"] for x in parity_rows),
             "boundary": "fresh event-1500 geometry and recomputed structure tensors; no historical observation/gradient replay"},
        "matrix_definition": {"G_v": "A_v^T A_v", "H_v": "(sqrt(lambda1)+1e-5)^2 * A_v^T A_v (captured wavelength mode)",
             "A_v_columns": "pinned compute_projected_axes_subset axes in parent coordinates",
             "valid_mask": f"renderer-visible AND cov2D max_transmittance>{trans} AND opacity>{opac}; optional training gradient gate omitted because historical per-view gradients are absent",
             "sample_jitter_seed_base": 204800, "structure_tensor": f"pinned get_multiscale_structure_tensor_v1 levels={levels}",
             "camera_aggregation": "equal-weight sum over active A fit views; B excluded from Q; all response/count checks use fresh fixed views"},
        "spectral_summary": {k: v for k, v in spectral.items() if k not in ("fit_indices", "Q_H", "Q_G", "fit_mask")},
        "fresh_vs_historical_counts": counts,
        "discarded_response_gate": {"fraction": DISCARDED_TRACE_GATE,
             "interpretation": "per-view trace fraction in dropped fit-eigenframe dimensions; sample-only descriptive threshold, not full-training conservative bound"},
        "accepted_work": {"status": "NOT_RUN_YET", "reason": "root requested spectral gate first; garden all-split state has 9.13M rows; do not run counter unless spectral opportunity survives a separate memory preflight"},
        "elapsed_seconds": time.time() - started,
    }
    np.savez_compressed(out / "stage0_raw.npz", parent_entry_rows=eligible_ids.numpy(),
                        historical_native_child_counts=historical, active=active, G=G, H=H,
                        fresh_eta_source=fresh_eta_source, fresh_eta_H=fresh_eta_h,
                        native_k=k_all.numpy(), A_camera_names=np.asarray(a_names), B_camera_names=np.asarray(b_names))
    report["provenance"] = {
        "split_entry_sha256": sha256(entry_path), "capture_meta_sha256": sha256(capture_path),
        "pinned_freq_utils_sha256": sha256(BASE / "utils/freq_utils.py"),
        "pinned_loss_utils_sha256": sha256(BASE / "utils/loss_utils.py"),
        "analysis_script_sha256": sha256(__file__), "raw_npz_sha256": sha256(out / "stage0_raw.npz"),
        "gpu": torch.cuda.get_device_name(0),
    }
    (out / "stage0_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"scene": scene_name, "status": report["status"], "eligible": counts["eligible_three_axis_parents"],
                      "native_children": counts["historical_native_children"], "fresh_A_reduction": counts["fresh_A_rotated_vs_unrotated_reduction_fraction"],
                      "report": str(out / "stage0_report.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
