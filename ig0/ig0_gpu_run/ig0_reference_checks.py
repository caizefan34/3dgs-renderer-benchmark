"""InfoGeo-GS IG0 - CPU fp64 reference correctness harness (Section 10).

Runs the renderer-independent correctness controls of IG0_PREREG.md and emits
the supporting JSON artifacts. DOES NOT touch the frozen renderer or any GPU.

The GPU scientific probes (Section 9: IG0_PROBES.json / IG0_TRUE_HOLDOUT.json)
and the mechanical verdict are intentionally NOT produced here: the frozen
checkpoints and referenced commits are absent and local GPU execution is out of
scope for this run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import ig0_fisher as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEED = 20260930
N_GAUSS = 1536          # >= 1024 required by control 2
MIN_N_METRIC = 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_gaussians(rng, n):
    """Deterministic anisotropic Gaussians covering iso / near-iso / anisotropic."""
    q = rng.standard_normal((n, 4))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    base = rng.uniform(-4.0, 0.5, size=(n, 1))
    iso = np.repeat(base, 3, axis=1)
    near_iso = np.hstack([base, base, base + rng.uniform(-1e-7, 1e-7, size=(n, 1))])
    aniso = np.hstack([
        base,
        base + rng.uniform(0.5, 2.0, size=(n, 1)),
        base + rng.uniform(1.5, 4.0, size=(n, 1)),
    ])
    wide = np.hstack([
        base + rng.uniform(1.0, 3.0, size=(n, 1)),
        base + rng.uniform(2.0, 4.0, size=(n, 1)),
        base + rng.uniform(3.0, 5.0, size=(n, 1)),
    ])
    regime = rng.integers(0, 4, size=n)
    ls = np.where(regime[:, None] == 0, iso,
          np.where(regime[:, None] == 1, near_iso,
          np.where(regime[:, None] == 2, aniso, wide)))
    mu = rng.uniform(-2.0, 2.0, size=(n, 3))
    return mu, ls, q


def make_gradient_field(mu, ls, q, rng, zero_idx):
    """Deterministic native-style gradient field in tangent coordinates."""
    n = mu.shape[0]
    g_mu = np.stack([
        np.sin(mu[:, 0]) + 0.1 * mu[:, 1],
        np.cos(mu[:, 1]) - 0.1 * mu[:, 2],
        0.5 * mu[:, 2] * mu[:, 0],
    ], axis=1)
    g_logscale = np.stack([
        0.3 * np.cos(ls[:, 0]),
        0.2 * np.sin(ls[:, 1]) + 0.05,
        0.25 * (ls[:, 2] - ls[:, 0]),
    ], axis=1)
    g_q = rng.standard_normal((n, 4))
    g_q = g_q / np.linalg.norm(g_q, axis=1, keepdims=True)
    g_omega = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        g_omega[i] = F.quat_grad_to_tangent(g_q[i], q[i])
    g_mu[zero_idx] = 0.0
    g_logscale[zero_idx] = 0.0
    g_omega[zero_idx] = 0.0
    return g_mu, g_logscale, g_omega


def control_psd(Sigma_inv_list, lam_list):
    """Control 1: Fisher metric positive-semidefinite on sampled Gaussians."""
    min_eig = np.inf
    n_checked = 0
    all_ok = True
    for Sinv, lm in zip(Sigma_inv_list, lam_list):
        G = F.fisher_metric_matrix(Sinv, lm)
        w = np.linalg.eigvalsh(G)
        min_eig = min(min_eig, float(w[0]))
        n_checked += 1
        if w[0] < -1e-9:
            all_ok = False
    return {
        "control": "1_fisher_metric_psd",
        "prereg_section": "10.1",
        "n_checked": n_checked,
        "min_eigenvalue": min_eig,
        "tolerance": -1e-9,
        "pass": bool(all_ok and min_eig >= -1e-9),
    }


def control_metric_parity(q_list, ls_list, mu_list, rng):
    """Control 2: closed-form Fisher metric vs direct differential (>=1024)."""
    worst = 0.0
    n = len(q_list)
    for i in range(n):
        q = q_list[i]; ls = ls_list[i]; mu = mu_list[i]
        Sigma, R, lam = F.sigma_from_quat_logscale(q, ls)
        Sinv = np.linalg.inv(Sigma)
        dmu = rng.standard_normal(3)
        dls = rng.standard_normal(3)
        dom = rng.standard_normal(3)
        a = F.fisher_block_norm2_full(Sinv, lam, dmu, dls, dom)
        b = F.fisher_differential_norm2(q, ls, dmu, dls, dom, R=R, lam=lam)
        rel = abs(a - b) / max(abs(b), 1e-30)
        worst = max(worst, rel)
    return {
        "control": "2_fisher_metric_parity",
        "prereg_section": "10.2",
        "n_compared": n,
        "min_required": MIN_N_METRIC,
        "worst_relative_error": worst,
        "tolerance": 1e-5,
        "pass": bool(n >= MIN_N_METRIC and worst <= 1e-5),
    }


def control_quaternion_jacobian(q_list, rng, n_sample=64):
    """Control 3: quaternion tangent Jacobian parity (autograd + finite diff)."""
    import torch
    dev = torch.device("cpu")
    torch.manual_seed(0)

    def quat_exp_pure(v):
        s = v.norm()
        s_safe = s.clamp_min(1e-12)
        coeff = torch.where(s < 1e-8,
                            1.0 - s * s / 6.0,
                            torch.sin(s_safe) / s_safe)
        w = torch.cos(s_safe)
        return torch.cat([w.reshape(1), coeff * v])

    def q_of_omega(omega, q0):
        half = 0.5 * omega
        dq = quat_exp_pure(half)
        out = torch.stack([
            q0[0] * dq[0] - (q0[1:] * dq[1:]).sum(),
            q0[0] * dq[1] + q0[1] * dq[0] + (q0[2] * dq[3] - q0[3] * dq[2]),
            q0[0] * dq[2] - q0[1] * dq[3] + q0[2] * dq[0] + q0[3] * dq[1],
            q0[0] * dq[3] + q0[1] * dq[2] - q0[2] * dq[1] + q0[3] * dq[0],
        ])
        return out / out.norm()

    worst_autograd = 0.0
    worst_fd = 0.0
    idxs = rng.choice(len(q_list), size=min(n_sample, len(q_list)), replace=False)
    for i in idxs:
        q0 = q_list[i]
        q0t = torch.tensor(q0, dtype=torch.float64, device=dev)
        omega = torch.zeros(3, dtype=torch.float64, device=dev, requires_grad=True)
        q_out = q_of_omega(omega, q0t)
        J_auto = torch.zeros((4, 3), dtype=torch.float64)
        for k in range(4):
            grad = torch.autograd.grad(q_out[k], omega, retain_graph=True)[0]
            J_auto[k, :] = grad
        J_an = torch.tensor(F.quat_tangent_jacobian(q0), dtype=torch.float64)
        worst_autograd = max(worst_autograd, float((J_auto - J_an).abs().max()))
        h = 1e-6
        for k in range(3):
            ep = np.zeros(3); ep[k] = h
            qp = F.normalize_q(F.quat_mul(q0, F.exp_quat(ep)))
            qm = F.normalize_q(F.quat_mul(q0, F.exp_quat(-ep)))
            fd = (qp - qm) / (2 * h)
            worst_fd = max(worst_fd, float(np.max(np.abs(fd - J_an[:, k].numpy()))))
    return {
        "control": "3_quaternion_jacobian_parity",
        "prereg_section": "10.3",
        "n_sample": int(len(idxs)),
        "worst_autograd_abs_error": worst_autograd,
        "worst_finite_difference_abs_error": worst_fd,
        "tolerance": 1e-6,
        "pass": bool(worst_autograd <= 1e-6 and worst_fd <= 1e-5),
        "note": "autograd in float64 CPU; no CUDA used.",
    }


def adam_geometry_displacement(mu, ls, q, g_mu, g_logscale, g_q, lr=1.0):
    """Section 6 reference: one exact native Adam step -> geometry tangent disp.

    Adam runs on native params (xyz, log_scale, quaternion-4). The quaternion
    displacement is converted to the right-trivialized tangent via the exact
    Section-4 relative log-map. lr is a positive stand-in; the FR matching is
    scale-invariant by construction so lr does not affect parity validity.
    """
    n = mu.shape[0]
    adam_geo = []
    for i in range(n):
        _, _, dxyz = F.adam_step_displacement(g_mu[i], np.zeros(3), np.zeros(3), 1, lr)
        _, _, dls = F.adam_step_displacement(g_logscale[i], np.zeros(3), np.zeros(3), 1, lr)
        _, _, dq = F.adam_step_displacement(g_q[i], np.zeros(4), np.zeros(4), 1, lr)
        q1 = F.normalize_q(q[i] + dq)
        domega = F.quat_relative_log(q[i], q1)
        adam_geo.append((dxyz, dls, domega))
    return adam_geo


def natural_geometry(mu, ls, q, g_mu, g_logscale, g_omega):
    """Section 5: per-Gaussian Fisher natural direction in tangent coords."""
    nat = []
    for i in range(mu.shape[0]):
        Sigma, R, lam = F.sigma_from_quat_logscale(q[i], ls[i])
        Sinv = np.linalg.inv(Sigma)
        d_mu, d_ls, d_om = F.natural_direction(g_mu[i], g_logscale[i], g_omega[i], Sinv, lam)
        nat.append((d_mu, d_ls, d_om))
    return nat


def control_radius_global(adam_geo, nat_geo, Sinv_list, lam_list):
    """Control 4: FR-GLOBAL Fisher radius matches native Adam (rel err <= 1e-5)."""
    alpha = F.fr_global_alpha(adam_geo, nat_geo, Sinv_list, lam_list)
    r_a2 = 0.0
    r_m2 = 0.0
    for (dmu, dls, dom), (nmu, nls, nom), Sinv, lm in zip(adam_geo, nat_geo, Sinv_list, lam_list):
        r_a2 += F.fisher_radius_geom(Sinv, lm, dmu, dls, dom)
        r_m2 += F.fisher_radius_geom(Sinv, lm, alpha * nmu, alpha * nls, alpha * nom)
    rel = abs(np.sqrt(r_m2) - np.sqrt(r_a2)) / max(np.sqrt(r_a2), 1e-300)
    return {
        "control": "4_fr_global_radius_parity",
        "prereg_section": "10.4",
        "alpha": alpha,
        "adam_fisher_radius": float(np.sqrt(r_a2)),
        "matched_fisher_radius": float(np.sqrt(r_m2)),
        "relative_error": rel,
        "tolerance": 1e-5,
        "pass": bool(rel <= 1e-5),
    }


def control_radius_local(adam_geo, nat_geo, Sinv_list, lam_list):
    """Control 5: FR-LOCAL per-Gaussian radius parity (median<=1e-5, p99<=1e-4)."""
    alphas = F.fr_local_alpha(adam_geo, nat_geo, Sinv_list, lam_list)
    rels = []
    zero_radius = 0
    for a, (dmu, dls, dom), (nmu, nls, nom), Sinv, lm in zip(
        alphas, adam_geo, nat_geo, Sinv_list, lam_list
    ):
        r_a2 = F.fisher_radius_geom(Sinv, lm, dmu, dls, dom)
        r_n2 = F.fisher_radius_geom(Sinv, lm, nmu, nls, nom)
        if r_n2 <= 0.0:
            zero_radius += 1
            continue
        r_ai = np.sqrt(r_a2)
        if r_ai <= 0.0:
            zero_radius += 1
            continue
        r_mi = np.sqrt(F.fisher_radius_geom(Sinv, lm, a * nmu, a * nls, a * nom))
        rels.append(abs(r_mi - r_ai) / r_ai)
    rels = np.array(rels) if rels else np.array([0.0])
    return {
        "control": "5_fr_local_radius_parity",
        "prereg_section": "10.5",
        "n_compared": int(rels.size),
        "n_zero_radius_excluded": int(zero_radius),
        "median_relative_error": float(np.median(rels)),
        "p99_relative_error": float(np.percentile(rels, 99)),
        "median_tolerance": 1e-5,
        "p99_tolerance": 1e-4,
        "pass": bool(np.median(rels) <= 1e-5 and np.percentile(rels, 99) <= 1e-4),
    }


def control_nan(mu, ls, q, cand_states):
    """Control 6: no invalid scale/quaternion/NaN across base and candidate states."""
    nan_inf = 0
    invalid = 0
    for arr in [mu, ls, q] + [c for st in cand_states for c in st]:
        a = np.asarray(arr, dtype=np.float64)
        nan_inf += int(np.sum(~np.isfinite(a)))
    for qi in q:
        if abs(np.linalg.norm(qi) - 1.0) > 1e-6:
            invalid += 1
    for lsi in ls:
        if np.any(np.exp(2.0 * lsi) <= 0.0) or not np.all(np.isfinite(np.exp(2.0 * lsi))):
            invalid += 1
    return {
        "control": "6_no_nan_inf_or_invalid",
        "prereg_section": "10.6",
        "nan_inf_count": nan_inf,
        "invalid_quaternion_or_scale": invalid,
        "pass": bool(nan_inf == 0 and invalid == 0),
    }


CELLS = ["room@10000", "room@15000", "bicycle@10000", "bicycle@15000"]
SCENE_OF = {"room": "room", "bicycle": "bicycle"}


def midpoint_stratified(pool_idx, k=8):
    """Deterministic midpoint-stratified selection of k indices from an ordered pool."""
    M = len(pool_idx)
    if M == 0:
        return []
    if M <= k:
        return list(pool_idx)
    chosen = []
    for j in range(k):
        lo = int(round(j * M / k))
        hi = int(round((j + 1) * M / k))
        mid = (lo + hi) // 2
        mid = min(max(mid, 0), M - 1)
        chosen.append(pool_idx[mid])
    return chosen


def load_ordered_cameras(scene):
    path = REPO / "data" / "official" / "mipnerf360" / scene / "cameras.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def camera_semantics():
    """Section 2 camera semantics: corrected-SB0 true-holdout split + stratification."""
    out = {
        "prereg_section": "2",
        "true_holdout_rule": "index i with i%8==0 (corrected SB0 true holdout)",
        "source_train_rule": "index i with i%8!=0",
        "selection": "midpoint-stratified, deterministic",
        "n_source_train": 8,
        "n_true_holdout": 8,
        "cells": {},
        "data_root": str(REPO / "data" / "official" / "mipnerf360"),
    }
    for scene in ["room", "bicycle"]:
        cams = load_ordered_cameras(scene)
        if cams is None:
            out["cells"][scene] = {"status": "cameras.json not found"}
            continue
        ids = [c.get("id", i) for i, c in enumerate(cams)]
        holdout_pool = [ids[i] for i in range(len(ids)) if i % 8 == 0]
        train_pool = [ids[i] for i in range(len(ids)) if i % 8 != 0]
        sel_holdout = midpoint_stratified(holdout_pool, 8)
        sel_train = midpoint_stratified(train_pool, 8)
        out["cells"][scene] = {
            "n_cameras_total": len(ids),
            "n_train_pool": len(train_pool),
            "n_holdout_pool": len(holdout_pool),
            "selected_source_train_ids": sel_train,
            "selected_true_holdout_ids": sel_holdout,
            "note": "identical across @10K/@15K checkpoints (camera split is ckpt-independent)",
        }
    return out


def build_candidate_states(mu, ls, q, adam_geo, nat_geo, Sinv_list, lam_list):
    """Apply FR-GLOBAL and FR-LOCAL to base state for the NaN/invalid control."""
    n = mu.shape[0]
    a_g = F.fr_global_alpha(adam_geo, nat_geo, Sinv_list, lam_list)
    a_l = F.fr_local_alpha(adam_geo, nat_geo, Sinv_list, lam_list)
    muG = np.empty_like(mu); lsG = np.empty_like(ls); qG = np.empty_like(q)
    muL = np.empty_like(mu); lsL = np.empty_like(ls); qL = np.empty_like(q)
    for i in range(n):
        nmu, nls, nom = nat_geo[i]
        muG[i], lsG[i], qG[i] = F.apply_geometry(mu[i], ls[i], q[i], a_g * nmu, a_g * nls, a_g * nom)
        muL[i], lsL[i], qL[i] = F.apply_geometry(mu[i], ls[i], q[i], a_l[i] * nmu, a_l[i] * nls, a_l[i] * nom)
    return [(muG, lsG, qG), (muL, lsL, qL)], a_g, a_l


def natural_descent_fraction(g_mu, g_logscale, g_omega, nat_geo):
    """Diagnostic: fraction of Gaussians with negative directional derivative."""
    ok = 0
    n = len(nat_geo)
    for i in range(n):
        dmu, dls, dom = nat_geo[i]
        deriv = g_mu[i] @ dmu + g_logscale[i] @ dls + g_omega[i] @ dom
        if deriv < 0:
            ok += 1
    return ok / max(n, 1)


def run_reference_checks():
    rng = np.random.default_rng(SEED)
    mu, ls, q = make_gaussians(rng, N_GAUSS)
    zero_idx = np.arange(0, 5)
    g_mu, g_logscale, g_omega = make_gradient_field(mu, ls, q, rng, zero_idx)
    g_q = F.normalize_q(rng.standard_normal((N_GAUSS, 4)))

    Sinv_list = []
    lam_list = []
    for i in range(N_GAUSS):
        Sigma, R, lam = F.sigma_from_quat_logscale(q[i], ls[i])
        Sinv_list.append(np.linalg.inv(Sigma))
        lam_list.append(lam)

    c1 = control_psd(Sinv_list, lam_list)
    c2 = control_metric_parity([q[i] for i in range(N_GAUSS)],
                               [ls[i] for i in range(N_GAUSS)],
                               [mu[i] for i in range(N_GAUSS)], rng)
    c3 = control_quaternion_jacobian([q[i] for i in range(N_GAUSS)], rng)
    c6 = control_nan(mu, ls, q, [])  # placeholder, real candidates built below

    adam_geo = adam_geometry_displacement(mu, ls, q, g_mu, g_logscale, g_q)
    nat_geo = natural_geometry(mu, ls, q, g_mu, g_logscale, g_omega)
    c4 = control_radius_global(adam_geo, nat_geo, Sinv_list, lam_list)
    c5 = control_radius_local(adam_geo, nat_geo, Sinv_list, lam_list)

    cand_states, a_g, a_l = build_candidate_states(mu, ls, q, adam_geo, nat_geo, Sinv_list, lam_list)
    c6 = control_nan(mu, ls, q, cand_states)

    descent_frac = natural_descent_fraction(g_mu, g_logscale, g_omega, nat_geo)

    controls = [c1, c2, c3, c4, c5, c6]
    correctness_pass = all(c["pass"] for c in controls)

    fisher_ref = {
        "artifact": "IG0_FISHER_REFERENCE.json",
        "prereg_section": "10.1,10.2,10.6",
        "n_gaussians": N_GAUSS,
        "controls": [c1, c2, c6],
        "correctness_pass": bool(c1["pass"] and c2["pass"] and c6["pass"]),
        "note": "fp64 CPU; closed-form Fisher metric vs direct differential; PSD; NaN/invalid.",
    }
    quat_check = {
        "artifact": "IG0_QUATERNION_CHECK.json",
        "prereg_section": "10.3",
        "controls": [c3],
        "correctness_pass": bool(c3["pass"]),
        "note": "Jacobian vs float64 torch autograd and central finite differences.",
    }
    radius_parity = {
        "artifact": "IG0_RADIUS_PARITY.json",
        "prereg_section": "10.4,10.5",
        "fr_global_alpha": a_g,
        "fr_local_alpha_median": float(np.median(F.fr_local_alpha(adam_geo, nat_geo, Sinv_list, lam_list))),
        "controls": [c4, c5],
        "correctness_pass": bool(c4["pass"] and c5["pass"]),
        "gradient_source": "deterministic_analytic_field_NOT_frozen_renderer",
        "note": "Validates the magnitude-matching algebra (Sections 5-7). The frozen-renderer Adam displacement is GPU-gated; the matching arithmetic itself is exact by construction.",
    }
    diagnostics = {
        "natural_direction_descent_fraction": descent_frac,
        "note": "fraction of Gaussians where g.dNat<0 (should be ~1.0 for PD metric).",
    }
    return {
        "controls": controls,
        "correctness_pass": bool(correctness_pass),
        "fisher_ref": fisher_ref,
        "quat_check": quat_check,
        "radius_parity": radius_parity,
        "diagnostics": diagnostics,
    }


def source_identity():
    """Section 1: frozen C0 identity. Fail-closed: commits/checkpoints absent here."""
    import subprocess
    def git_present(sha):
        try:
            r = subprocess.run(
                ["git", "cat-file", "-t", sha],
                cwd=str(REPO), capture_output=True, text=True, timeout=20,
            )
            return r.returncode == 0 and r.stdout.strip() == "commit"
        except Exception:
            return False

    ext = "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11"
    core = "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"
    ext_ok = git_present(ext)
    core_ok = git_present(core)
    ckpt_dir = REPO / "data" / "official" / "mipnerf360"
    return {
        "artifact": "IG0_SOURCE_IDENTITY.json",
        "prereg_section": "1",
        "c0_extension_commit": ext,
        "c0_core_commit": core,
        "extension_commit_present_in_repo": bool(ext_ok),
        "core_commit_present_in_repo": bool(core_ok),
        "checkpoints_present": bool(ckpt_dir.exists()),
        "identity_verified": bool(ext_ok and core_ok),
        "fail_closed": True,
        "verdict": "IDENTITY_UNVERIFIED_GPU_PATH_BLOCKED",
        "note": ("Frozen extension/core commits are not present in this repo and no "
                 "room/bicycle 10K/15K checkpoints are available, so the Section 1 "
                 "identity check fails closed. The GPU scientific probes cannot be "
                 "identity-verified or executed in this CPU-only run."),
    }


def environment():
    import numpy as _np
    info = {
        "artifact": "IG0_ENVIRONMENT.json",
        "python": sys.version.split()[0],
        "numpy": _np.__version__,
        "platform": platform.platform(),
        "gpu_execution": False,
        "cuda_used": False,
        "torch": None,
        "torch_cuda_available": None,
        "execution_mode": "cpu_only_fp64",
        "constraint": "local GPU execution is out of scope for this run (goal: not local GPU).",
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover
        info["torch"] = "unavailable: %s" % exc
    return info


def write_json(name, obj):
    path = HERE / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
    return path


def write_report(ctx, paths):
    rp = ctx["radius_parity"]
    fr = ctx["fisher_ref"]
    qc = ctx["quat_check"]
    diag = ctx["diagnostics"]
    ident = ctx["identity"]
    env = ctx["environment"]
    cams = ctx["camera_semantics"]
    lines = []
    lines.append("# InfoGeo-GS IG0 Report (CPU reference run)")
    lines.append("")
    lines.append("Date: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    lines.append("")
    lines.append("## Scope")
    lines.append("- Executed: Sections 3-8 math + Section 10 fp64 CPU correctness controls.")
    lines.append("- NOT executed: Section 9 GPU scientific probes (frozen renderer + checkpoints).")
    lines.append("- Constraint honored: no local GPU / no production CUDA.")
    lines.append("")
    lines.append("## Correctness controls (Section 10)")
    for c in ctx["controls"]:
        lines.append("- %s (sec %s): %s" % (c["control"], c["prereg_section"], "PASS" if c["pass"] else "FAIL"))
    lines.append("")
    lines.append("Fisher metric worst relative error (closed-form vs differential): %.3e" % fr["controls"][1]["worst_relative_error"])
    lines.append("Quaternion Jacobian worst abs error (autograd): %.3e" % qc["controls"][0]["worst_autograd_abs_error"])
    lines.append("FR-GLOBAL radius parity relative error: %.3e" % rp["controls"][0]["relative_error"])
    lines.append("FR-LOCAL radius parity median / p99: %.3e / %.3e" % (rp["controls"][1]["median_relative_error"], rp["controls"][1]["p99_relative_error"]))
    lines.append("Natural-direction descent fraction: %.4f" % diag["natural_direction_descent_fraction"])
    lines.append("")
    lines.append("## Overall correctness")
    lines.append("- correctness_pass: %s" % ("true" if ctx["correctness_pass"] else "false"))
    lines.append("")
    lines.append("## Blocked (GPU) inputs")
    lines.append("- identity_verified: %s (%s)" % (ident["identity_verified"], ident["verdict"]))
    lines.append("- checkpoints_present: %s" % ident["checkpoints_present"])
    lines.append("- The mechanical verdict (IG0_STRONG_PROMOTE / IG0_PROMOTE / IG0_DROP)")
    lines.append("  requires IG0_PROBES.json / IG0_TRUE_HOLDOUT.json, which need the frozen")
    lines.append("  renderer on GPU. They are NOT produced here; no summary or verdict is fabricated.")
    lines.append("")
    lines.append("## Camera semantics (Section 2)")
    for scene, d in cams.get("cells", {}).items():
        if "selected_source_train_ids" in d:
            lines.append("- %s: train=%s holdout=%s (of %d cams)" % (
                scene, d["selected_source_train_ids"], d["selected_true_holdout_ids"], d["n_cameras_total"]))
    lines.append("")
    lines.append("## Artifacts")
    for pth in paths:
        lines.append("- %s" % Path(pth).name)
    lines.append("")
    (HERE / "IG0_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return HERE / "IG0_REPORT.md"


OUTPUT_JSONS = [
    "IG0_FISHER_REFERENCE.json",
    "IG0_QUATERNION_CHECK.json",
    "IG0_RADIUS_PARITY.json",
    "IG0_SOURCE_IDENTITY.json",
    "IG0_ENVIRONMENT.json",
    "IG0_CAMERA_SEMANTICS.json",
]


def build_manifest():
    packet = Path(r"C:\Users\36570\Downloads\IG0")
    man = {"generated_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    if packet.exists():
        for f in sorted(packet.glob("*")):
            if f.is_file():
                man["files"][f.name] = {"bytes": f.stat().st_size, "sha256": sha256_file(f)}
    for name in OUTPUT_JSONS + ["IG0_REPORT.md", "README.md", "IG0_REFERENCE_LOG.txt"]:
        f = HERE / name
        if f.exists():
            man["files"][name] = {"bytes": f.stat().st_size, "sha256": sha256_file(f)}
    for name in ["ig0_fisher.py", "ig0_reference_checks.py", "ig0_analyze.py"]:
        f = HERE / name
        if f.exists():
            man["files"][name] = {"bytes": f.stat().st_size, "sha256": sha256_file(f)}
    return man


def main():
    ctx = run_reference_checks()
    ctx["identity"] = source_identity()
    ctx["environment"] = environment()
    ctx["camera_semantics"] = camera_semantics()

    paths = []
    paths.append(write_json("IG0_FISHER_REFERENCE.json", ctx["fisher_ref"]))
    paths.append(write_json("IG0_QUATERNION_CHECK.json", ctx["quat_check"]))
    paths.append(write_json("IG0_RADIUS_PARITY.json", ctx["radius_parity"]))
    paths.append(write_json("IG0_SOURCE_IDENTITY.json", ctx["identity"]))
    paths.append(write_json("IG0_ENVIRONMENT.json", ctx["environment"]))
    paths.append(write_json("IG0_CAMERA_SEMANTICS.json", ctx["camera_semantics"]))

    rep = write_report(ctx, paths)
    paths.append(rep)

    man = build_manifest()
    write_json("IG0_MANIFEST.json", man)

    print("=== IG0 CPU reference correctness ===")
    for c in ctx["controls"]:
        print("  %-32s %s" % (c["control"], "PASS" if c["pass"] else "FAIL"))
    print("correctness_pass:", ctx["correctness_pass"])
    print("identity_verified:", ctx["identity"]["identity_verified"])
    print("wrote %d artifacts + manifest" % len(paths))
    return 0 if ctx["correctness_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
