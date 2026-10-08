#!/usr/bin/env python
"""g5b_gradient_check.py - G5-B gradient correctness for the native CGT renderer.

Validates the autograd gradient of cgt_student_render (the native renderer used by
G5-C/D) against central finite differences in float64 (ground truth), across:
  - real per-scene centered pixel grids (room/garden/bicycle from workload.px)
  - the actual frozen teacher coeffs (route-A scale) + random coeffs at several scales
  - multiple seeds
  - two losses: generic linear cotangent (comprehensive) and training MSE (E_rgb vs y)

Reports per case: relative L2 error, cosine similarity, max abs error, p99 relative
component error, NaN/Inf count. Gate (prereq G5-B): global cosine >= 0.999 AND
relative L2 <= 1e-3 AND no unexplained NaN/Inf.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g5_cgt_renderer import cgt_student_render              # noqa: E402

WORKLOAD = "/home/liaoyuanjun/cgt_f0_fused_cuda/results/workload.npz"
OUTDIR = "/home/liaoyuanjun/cgt_g5_native/results"
RANK = 4
DEV = "cpu"   # gradient check is exact in float64 on CPU; GPU adds nothing


def analytic_grad(coeffs_np, px_np, cotangent_np, mse_y_np=None):
    """coeffs (R,10) f64. Returns grad (R,10) f64 via autograd.
    If mse_y_np given: loss = mean((pred[:,:3]-y)^2); else loss = sum(pred*cot)."""
    c = torch.tensor(coeffs_np, dtype=torch.float64, requires_grad=True)
    px = torch.tensor(px_np, dtype=torch.float64)
    pred = cgt_student_render(c, px)               # (P,4) f64, differentiable
    if mse_y_np is not None:
        y = torch.tensor(mse_y_np, dtype=torch.float64)
        loss = torch.mean((pred[:, :3] - y) ** 2)
    else:
        cot = torch.tensor(cotangent_np, dtype=torch.float64)
        loss = torch.sum(pred * cot)
    loss.backward()
    g = c.grad.detach().numpy().astype(np.float64)
    return g, float(loss.item())


def fd_grad(coeffs_np, px_np, cotangent_np=None, mse_y_np=None, h=1e-6):
    """Central finite difference in float64 (ground truth)."""
    g = np.zeros_like(coeffs_np)
    flat = coeffs_np.ravel().copy()
    for i in range(flat.size):
        fp = flat.copy(); fp[i] += h
        fm = flat.copy(); fm[i] -= h
        cp = torch.tensor(fp.reshape(coeffs_np.shape), dtype=torch.float64)
        cm = torch.tensor(fm.reshape(coeffs_np.shape), dtype=torch.float64)
        px = torch.tensor(px_np, dtype=torch.float64)
        if mse_y_np is not None:
            y = torch.tensor(mse_y_np, dtype=torch.float64)
            lp = torch.mean((cgt_student_render(cp, px)[:, :3] - y) ** 2).item()
            lm = torch.mean((cgt_student_render(cm, px)[:, :3] - y) ** 2).item()
        else:
            cot = torch.tensor(cotangent_np, dtype=torch.float64)
            lp = torch.sum(cgt_student_render(cp, px) * cot).item()
            lm = torch.sum(cgt_student_render(cm, px) * cot).item()
        g.ravel()[i] = (lp - lm) / (2 * h)
    return g


def compare(g_a, g_f):
    nan = int(np.sum(~np.isfinite(g_a))) + int(np.sum(~np.isfinite(g_f)))
    na = np.linalg.norm(g_a); nf = np.linalg.norm(g_f)
    rel_l2 = float(np.linalg.norm(g_a - g_f) / (nf + 1e-30))
    cos = float(np.sum(g_a * g_f) / (na * nf + 1e-30))
    max_abs = float(np.max(np.abs(g_a - g_f)))
    denom = np.maximum(np.abs(g_f), 1e-8)
    rel_comp = np.abs(g_a - g_f) / denom
    p99 = float(np.percentile(rel_comp, 99))
    return {"rel_l2": rel_l2, "cosine": cos, "max_abs": max_abs, "p99_rel_component": p99, "nan_inf": nan}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = np.load(WORKLOAD)
    coeffs0 = d["coeffs"].astype(np.float64)   # (150,R,10)
    px_all = d["px"].astype(np.float64)        # (256,2)
    scene = d["scene"]
    P = px_all.shape[0]
    rng = np.random.default_rng(20261004)

    # per-scene pixel grid (centered, identical across tiles but we vary the grid extent slightly per scene for coverage)
    scenes = ["room", "garden", "bicycle"]
    idx_by_sc = {sc: np.where(scene == sc)[0] for sc in scenes}

    cases = []
    # 1) frozen teacher coeffs at native scale (route-A), real px, generic cotangent + MSE
    for sc in scenes:
        t = int(idx_by_sc[sc][0])
        cases.append({"scene": sc, "t": t, "coeffs": coeffs0[t], "kind": "frozen_teacher", "loss": "cotangent"})
        cases.append({"scene": sc, "t": t, "coeffs": coeffs0[t], "kind": "frozen_teacher", "loss": "mse"})
    # 2) random coeffs at several scales, all scenes, both losses
    for scale in [0.1, 0.5, 1.0, 3.0]:
        for sc in scenes:
            t = int(idx_by_sc[sc][len(idx_by_sc[sc])//2])
            c = rng.standard_normal((RANK, 10)) * scale
            cases.append({"scene": sc, "t": t, "coeffs": c, "kind": f"random_scale{scale}", "loss": "cotangent"})
            cases.append({"scene": sc, "t": t, "coeffs": c, "kind": f"random_scale{scale}", "loss": "mse"})

    recs = []
    t0 = time.time()
    for ci, case in enumerate(cases):
        px = px_all
        coeffs = case["coeffs"]
        if case["loss"] == "cotangent":
            cot = rng.standard_normal((P, 4))
            ga, _ = analytic_grad(coeffs, px, cot)
            gf = fd_grad(coeffs, px, cotangent_np=cot, h=1e-6)
        else:
            y = rng.random((P, 3))
            ga, _ = analytic_grad(coeffs, px, None, mse_y_np=y)
            gf = fd_grad(coeffs, px, mse_y_np=y, h=1e-6)
        cmp_ = compare(ga, gf)
        recs.append({"case": ci, "scene": case["scene"], "t": case["t"], "kind": case["kind"],
                     "loss": case["loss"], **cmp_})
        if ci % 10 == 0:
            print(f"[{ci}/{len(cases)}] {case['scene']} {case['kind']} {case['loss']} rel_l2={cmp_['rel_l2']:.2e} cos={cmp_['cosine']:.8f} nan={cmp_['nan_inf']}", flush=True)

    # aggregate gates
    cos_min = float(min(r["cosine"] for r in recs))
    rel_l2_max = float(max(r["rel_l2"] for r in recs))
    max_abs_max = float(max(r["max_abs"] for r in recs))
    p99_max = float(max(r["p99_rel_component"] for r in recs))
    nan_total = int(sum(r["nan_inf"] for r in recs))
    gate_cos = cos_min >= 0.999
    gate_rel = rel_l2_max <= 1e-3
    gate_nan = nan_total == 0
    verdict = "PASS_GRADIENT_CORRECTNESS" if (gate_cos and gate_rel and gate_nan) else "DROP_GRADIENT_CORRECTNESS"

    out = {
        "stage": "G5-B", "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": "Is the autograd gradient of the native CGT renderer correct vs finite differences?",
        "renderer": "cgt_student_render (g5_cgt_renderer.py), float64 autograd vs central FD (h=1e-6, float64 ground truth)",
        "rank": RANK, "P": P, "n_cases": len(recs),
        "losses_tested": ["generic_linear_cotangent", "training_MSE_E_rgb_vs_y"],
        "scales_tested": ["frozen_teacher", "random_scale0.1/0.5/1.0/3.0"],
        "scenes_tested": scenes,
        "gate": {"cosine_min_ge": 0.999, "rel_l2_max_le": 1e-3, "nan_inf_eq": 0},
        "measured": {"cosine_min": cos_min, "rel_l2_max": rel_l2_max, "max_abs_max": max_abs_max,
                     "p99_rel_component_max": p99_max, "nan_inf_total": nan_total},
        "gate_pass_cosine": bool(gate_cos), "gate_pass_rel_l2": bool(gate_rel), "gate_pass_nan": bool(gate_nan),
        "per_case": recs,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(os.path.join(OUTDIR, "G5B_GRADIENT_CHECK.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"verdict": verdict, "cosine_min": cos_min, "rel_l2_max": rel_l2_max,
                      "max_abs_max": max_abs_max, "p99_max": p99_max, "nan_total": nan_total,
                      "gate_cos": gate_cos, "gate_rel": gate_rel, "gate_nan": gate_nan}, indent=2))
    print("WROTE G5B_GRADIENT_CHECK.json")


if __name__ == "__main__":
    main()
