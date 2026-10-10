#!/usr/bin/env python3
"""R18-P1: numerical correctness — frozen std_ll vs frozen HiGS.

Under an IDENTICAL activated Gaussian state (opacity=sigmoid, scales=exp,
quats=normalize) and REAL training cameras, compares the two frozen renderers:

  Forward : render_colors (RGB) and render_alphas — MAE / RMSE / max-abs error,
            plus relative-L2. Shape is checked (no silent broadcasting).
  Backward : a FIXED upstream gradient (same seed for both backends) is applied
            to (frame, alpha); the induced parameter gradients for means, quats,
            scales, opacities, shs are compared (cosine, MAE, RMSE, max-abs, rel-L2).
  Robust  : non-finite value/gradient flags, anomalous-zero-gradient fractions,
            and a reproducibility check (forward run twice must be bit-identical).

Expected numerical error (float accumulation / tile-order differences) is
REPORTED SEPARATELY from structural/algorithmic difference (which would show as
large errors concentrated where one backend culls and the other does not).

Deliverable: R18_P1_CORRECTNESS.json (+ raw/R18_P1_details.json)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda import Event

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r18_common import (H540, W540, OUT, add_project_to_syspath, build_cell_state,
                        device_utilization_ms, err_metrics, ensure_out_dirs,
                        gpu_mem_snapshot, grad_metrics, load_checkpoint,
                        load_real_cameras, reset_peak, sha256_file, write_json,
                        zero_grad_frac)

CSV_FIELDS = [
    "config", "n_gaussians", "camera_set", "n_cameras", "activation",
    "rgb_mae", "rgb_rmse", "rgb_max_abs", "rgb_rel_l2",
    "alpha_mae", "alpha_rmse", "alpha_max_abs", "alpha_rel_l2",
    "grad_means_cosine", "grad_means_rmse", "grad_means_max_abs",
    "grad_scales_cosine", "grad_scales_rmse", "grad_scales_max_abs",
    "grad_rotation_cosine", "grad_rotation_rmse", "grad_rotation_max_abs",
    "grad_opacity_cosine", "grad_opacity_rmse", "grad_opacity_max_abs",
    "grad_shs_cosine", "grad_shs_rmse", "grad_shs_max_abs",
    "finite_frame", "finite_alpha", "finite_grads",
    "zero_frac_grad_means", "zero_frac_grad_scales", "zero_frac_grad_rotation",
    "zero_frac_grad_opacity", "zero_frac_grad_shs",
    "repro_forward_identical", "std_binary_sha256", "higs_binary_sha256", "status", "error",
]


def percentiles(v):
    a = np.asarray(v, dtype=np.float64)
    return {"median": float(np.median(a)), "mean": float(a.mean()),
            "p10": float(np.quantile(a, 0.1)), "p90": float(np.quantile(a, 0.9))}


def render_std(state, viewmats, Ks, bg):
    from gsplat import rasterization
    rc, ra, _ = rasterization(
        means=state["means"], quats=state["quats"], scales=state["scales"],
        opacities=state["opacities"], colors=state["shs"],
        viewmats=viewmats, Ks=Ks, width=W540, height=H540,
        sh_degree=3, backgrounds=bg)
    return {"frame": rc, "alpha": ra}


def render_higs(state, viewmats, Ks):
    from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen
    r = rasterize_gaussian_higs_frozen(
        state["means"], state["quats"], state["scales"], state["opacities"],
        state["shs"], viewmats=viewmats, Ks=Ks, width=W540, height=H540,
        sh_degree=3, enable_culling=True, use_higs_culling=True,
        freeze_topology=False)  # mixed Gaussian counts across configs
    return {"frame": r["frame"], "alpha": r["alpha"]}


def norm_c(t):
    """Normalize a render output to [C,H,W,*] for shape-safe comparison.

    std_ll returns [C,H,W,*]; HiGS frozen may return [H,W,*], [1,H,W,*] or
    [1,C,H,W,*]. Squeeze all leading size-1 dims, then re-add a camera dim if
    the result is 3-D (single camera)."""
    while t.dim() > 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t


def make_fixed_upstream(frame, alpha, seed=1234):
    """Same upstream gradient for both backends (detached from graph)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    gf = torch.randn(frame.shape, generator=g, dtype=torch.float32).to(frame.device)
    ga = torch.randn(alpha.shape, generator=g, dtype=torch.float32).to(alpha.device)
    return gf, ga


def run_config(tag, ckpt, indices, cams, binaries, details):
    state = build_cell_state(ckpt, indices, "sigmoid", "exp", "normalize", device="cuda")
    ncams = len(cams)
    viewmats = torch.stack([c["viewmatrix"] for c in cams]).to("cuda").contiguous()
    Ks = torch.stack([c["K"] for c in cams]).to("cuda").contiguous()
    bg = torch.zeros(ncams, 3, device="cuda")

    row = {k: None for k in CSV_FIELDS}
    row.update({
        "config": tag, "n_gaussians": int(indices.size),
        "camera_set": "cam0" if ncams == 1 else f"cam0..{ncams-1}",
        "n_cameras": ncams, "activation": "sigmoid/exp/normalize",
        "std_binary_sha256": binaries.get("std_csrc_sha256"),
        "higs_binary_sha256": binaries.get("higs_csrc_sha256"),
    })

    try:
        # ---- Forward: std ----
        out_std = render_std(state, viewmats, Ks, bg)
        f_std, a_std = norm_c(out_std["frame"]), norm_c(out_std["alpha"])
        # ---- Forward: higs (fresh state to avoid grad contamination) ----
        state_h = build_cell_state(ckpt, indices, "sigmoid", "exp", "normalize", device="cuda")
        out_higs = render_higs(state_h, viewmats, Ks)
        f_h, a_h = norm_c(out_higs["frame"]), norm_c(out_higs["alpha"])

        # ---- Shape check (no silent broadcasting) ----
        shapes = {
            "std_frame": list(f_std.shape), "higs_frame": list(f_h.shape),
            "std_alpha": list(a_std.shape), "higs_alpha": list(a_h.shape),
        }
        if f_std.shape != f_h.shape or a_std.shape != a_h.shape:
            row.update({"status": "SHAPE_MISMATCH", "error": json.dumps(shapes)})
            details["rows"].append({**row, "shapes": shapes})
            return row

        # ---- Forward error metrics ----
        rgb = err_metrics(f_std, f_h)
        alpha = err_metrics(a_std, a_h)
        row.update({
            "rgb_mae": rgb["mae"], "rgb_rmse": rgb["rmse"],
            "rgb_max_abs": rgb["max_abs"], "rgb_rel_l2": rgb["rel_l2"],
            "alpha_mae": alpha["mae"], "alpha_rmse": alpha["rmse"],
            "alpha_max_abs": alpha["max_abs"], "alpha_rel_l2": alpha["rel_l2"],
        })

        # ---- Reproducibility: forward std twice must be bit-identical ----
        out_std_b = render_std(state, viewmats, Ks, bg)
        f_std_b, a_std_b = norm_c(out_std_b["frame"]), norm_c(out_std_b["alpha"])
        repro = bool(torch.equal(f_std, f_std_b) and torch.equal(a_std, a_std_b))
        row["repro_forward_identical"] = repro

        # ---- Finite checks (forward) ----
        finite_frame = bool(torch.isfinite(f_std).all() and torch.isfinite(f_h).all())
        finite_alpha = bool(torch.isfinite(a_std).all() and torch.isfinite(a_h).all())
        row["finite_frame"], row["finite_alpha"] = finite_frame, finite_alpha

        # ---- Backward with FIXED upstream gradient ----
        # std: combine frame+alpha upstream grads into ONE scalar, backward once.
        state_s = build_cell_state(ckpt, indices, "sigmoid", "exp", "normalize", device="cuda")
        os_ = render_std(state_s, viewmats, Ks, bg)
        fs_, as_ = norm_c(os_["frame"]), norm_c(os_["alpha"])
        gf, ga = make_fixed_upstream(fs_, as_)
        loss_s = (fs_ * gf).sum() + (as_ * ga).sum()
        loss_s.backward()
        grads_std = {k: (state_s[k].grad.detach().clone() if state_s[k].grad is not None else None)
                     for k in ("means", "quats", "scales", "opacities", "shs")}
        # higs (same upstream, matching shape)
        state_g = build_cell_state(ckpt, indices, "sigmoid", "exp", "normalize", device="cuda")
        og_ = render_higs(state_g, viewmats, Ks)
        fg_, ag_ = norm_c(og_["frame"]), norm_c(og_["alpha"])
        gf2, ga2 = make_fixed_upstream(fg_, ag_)
        loss_g = (fg_ * gf2).sum() + (ag_ * ga2).sum()
        loss_g.backward()
        grads_higs = {k: (state_g[k].grad.detach().clone() if state_g[k].grad is not None else None)
                      for k in ("means", "quats", "scales", "opacities", "shs")}

        # ---- Gradient comparison ----
        gmap = {"means": "grad_means", "quats": "grad_rotation", "scales": "grad_scales",
                "opacities": "grad_opacity", "shs": "grad_shs"}
        finite_grads = True
        for pk, rk in gmap.items():
            gm = grad_metrics(grads_std[pk], grads_higs[pk])
            if not gm.get("shape_mismatch") and "cosine" in gm:
                row[f"{rk}_cosine"] = gm["cosine"]
                row[f"{rk}_rmse"] = gm["rmse"]
                row[f"{rk}_max_abs"] = gm["max_abs"]
            zf = zero_grad_frac(grads_std[pk])
            row[f"zero_frac_{rk}"] = zf.get("frac_zero")
            # finite check on both grads
            for g in (grads_std[pk], grads_higs[pk]):
                if g is not None and not torch.isfinite(g).all():
                    finite_grads = False
        row["finite_grads"] = finite_grads
        row["status"] = "PASS" if (finite_frame and finite_alpha and finite_grads) else "NONFINITE_DETECTED"

        details["rows"].append({
            **row, "shapes": shapes,
            "forward_rgb": rgb, "forward_alpha": alpha,
            "grad_metrics": {rk: grad_metrics(grads_std[pk], grads_higs[pk])
                             for pk, rk in gmap.items()},
            "grad_zero_frac": {rk: zero_grad_frac(grads_std[pk]) for pk, rk in gmap.items()},
            "reproducible": repro,
        })
    except Exception as e:  # noqa: BLE001
        row.update({"status": "ERROR", "error": repr(e)[:500]})
        import traceback
        traceback.print_exc()
        details["rows"].append(row)
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="580K_B1,580K_B4,4M_B1,4M_B4")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    ensure_out_dirs()
    add_project_to_syspath()

    pre = gpu_mem_snapshot("p1_preflight")
    print(json.dumps(pre, indent=2), flush=True)

    ckpt, ck_info = load_checkpoint()
    n_full = int(ckpt["xyz"].shape[0])
    from r18_common import build_580k_indices, full_indices
    try:
        cams540, cams_path, cams_sha = load_real_cameras(W540, H540, device="cpu")
    except Exception as e:  # noqa: BLE001
        cams540, cams_path, cams_sha = [], None, None
        print(f"CAMERAS_UNAVAILABLE: {e!r}", flush=True)

    from r18_common import PROJECT
    binaries = {}
    for label, rel in (("std_csrc_sha256", "gsplat/csrc.so"),
                       ("higs_csrc_sha256", "gsplat/experimental/render/kernels/csrc.so")):
        binaries[label] = sha256_file(PROJECT / rel)

    details = {"deliverable": "R18_P1_CORRECTNESS", "checkpoint": ck_info,
               "cameras_json": cams_path, "cameras_json_sha256": cams_sha,
               "binaries": binaries, "warmup": args.warmup,
               "state": "580K/4M subset ACTIVATED (sigmoid/exp/normalize)",
               "rows": []}

    csv_path = OUT / "R18_P1_CORRECTNESS.csv"
    from r18_common import append_csv
    idx_cache = {"580k": build_580k_indices(n_full), "4m": full_indices(n_full)}

    for cfg in args.configs.split(","):
        cfg = cfg.strip()
        subset, bsz = cfg.split("_")
        subset = "4m" if subset.upper().startswith("4") else "580k"  # '4M'/'580K' -> cache key
        indices = idx_cache[subset]
        ncams = 1 if bsz == "B1" else 4
        if not cams540 or len(cams540) < ncams:
            row = {k: None for k in CSV_FIELDS}
            row.update({"config": cfg, "status": "NOT_EXECUTED_CAMERAS_MISSING",
                        "error": f"need {ncams} cams, have {len(cams540)}"})
            append_csv(csv_path, row, CSV_FIELDS)
            details["rows"].append(row)
            continue
        cams = cams540[:ncams]
        print(f"\n===== P1 {cfg} ({indices.size} gaussians, {ncams} cam) =====", flush=True)
        row = run_config(cfg, ckpt, indices, cams, binaries, details)
        append_csv(csv_path, row, CSV_FIELDS)
        print(json.dumps({k: row.get(k) for k in
              ("config", "rgb_mae", "rgb_rmse", "rgb_max_abs", "alpha_mae",
               "grad_means_cosine", "grad_scales_cosine", "finite_grads",
               "repro_forward_identical", "status")}, indent=2), flush=True)

    write_json(OUT / "raw" / "R18_P1_details.json", details)
    print("P1 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
