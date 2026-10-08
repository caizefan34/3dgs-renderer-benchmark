#!/usr/bin/env python3
"""OPTLAWS0 / MG0-S Part A — gradient-variance concentration (sidecar to MG0).

Usage: mg0s_variance.py <room|bicycle> <20000|25000>

Frozen reuse
------------
MG0 (mg0_oneview_hotcv/scripts/mg0_*.py) is treated as the frozen definition.
This script imports mg0_core and therefore reuses VERBATIM:
  - the frozen 64-camera set            (MG0_CAMERA_SET.json, MG0 Stage 1)
  - the frozen N1-snapshot checkpoints  (ckpt_15000/20000/25000.pt)
  - the c0 render branch, P12 fused loss, Adam preconditioner P_t
  - the 59-dim row layout and MG0 parameter groups
No MG0 script or gate is modified.

Quantity (MG0S_PREREG Part A; identical to MG0's own `score`):

    V_j = mean_v || P(g_v,j) - mean_u P(g_u,j) ||^2 ,  P_t(x) = x/(sqrt(v_t)+eps)

Zeros from cameras where a row receives zero gradient stay zero. Per-dimension
(fp64) accumulation is two-pass, exactly as MG0 does, so the summed V_j is the
same frozen quantity MG0 stores in MG0_SCORE_*.npy (cross-checked here).

Emits
-----
  MG0S_VARIANCE_{scene}_{ckpt}.npz      full per-row V_j + per-group V_j
  MG0S_CONCENTRATION_{scene}_{ckpt}.json
and freezes the deterministic 4096-row low-rank sample IDs consumed by
mg0s_lowrank.py (row IDs are frozen here, before any SVD).

Telemetry only: no training, no densification, no formal test views,
no renderer / optimizer / loss / resolution change.
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

# ---- locate the frozen MG0 scripts (import mg0_core as the frozen core) -----
MG0_SCRIPTS = os.environ.get(
    "MG0_SCRIPTS_DIR",
    "/root/epic05_mg0/scripts" if os.path.isdir("/root/epic05_mg0/scripts")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, MG0_SCRIPTS)
import mg0_core as C  # noqa: E402

# ---- frozen MG0-S constants ------------------------------------------------
N_HOT = 2048                 # highest-V portion of the analysis set
N_UNIFORM = 2048             # uniform nonzero-V portion
SAMPLE_SEED = 20260929       # frozen before any singular-value analysis
TOP_FRACTIONS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
ENERGY_TARGETS = (0.50, 0.80, 0.90, 0.95)
GROUP_COLS = {name: (a, b) for name, a, b in C.GROUPS}


# ---- concentration statistics (pure numpy, fp64) ---------------------------
def _gini(v):
    """Gini coefficient of a nonnegative vector (0 => perfectly uniform)."""
    if v.size == 0 or v.sum() <= 0:
        return 0.0
    s = np.sort(v)
    n = s.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * float(np.sum(idx * s))) / (n * float(s.sum())) - (n + 1.0) / n)


def _effective_support(v):
    """(sum V)^2 / sum(V^2) — number of rows carrying most of the energy."""
    d = float(np.sum(v.astype(np.float64) ** 2))
    if d <= 0:
        return 0.0
    s = float(v.sum(dtype=np.float64))
    return float(s * s / d)


def _cumulative_top(v_desc, fractions):
    """Energy fraction captured by the top p-fraction of rows (v_desc sorted)."""
    n = v_desc.size
    total = float(v_desc.sum(dtype=np.float64))
    cum = np.cumsum(v_desc, dtype=np.float64)
    out = {}
    for p in fractions:
        m = max(1, int(math.ceil(p * n)))
        frac = float(cum[m - 1] / total) if total > 0 else 0.0
        out[f"{p * 100:g}%"] = {"rows": m, "energy_fraction": frac}
    return out


def _rows_for_energy(v_desc, targets):
    """Smallest row count whose top-k energy reaches each target."""
    n = v_desc.size
    total = float(v_desc.sum(dtype=np.float64))
    cum = np.cumsum(v_desc, dtype=np.float64)
    out = {}
    for e in targets:
        if total <= 0:
            out[f"{e * 100:g}%"] = None
            continue
        k = int(np.searchsorted(cum, e * total - 1e-18) + 1)
        out[f"{e * 100:g}%"] = min(k, n)
    return out


def _concentration(v):
    """Full per-vector concentration summary in the prereg's language."""
    v = np.asarray(v, dtype=np.float64)
    vd = np.sort(v)[::-1]
    total = float(v.sum())
    out = {
        "total_variance": total,
        "gini": _gini(v),
        "effective_support": _effective_support(v),
        "cumulative_top_fraction": _cumulative_top(vd, TOP_FRACTIONS),
        "rows_for_energy": _rows_for_energy(vd, ENERGY_TARGETS),
        "rows_for_energy_fraction": {
            k: (None if k_v is None else k_v / max(1, v.size))
            for k, k_v in _rows_for_energy(vd, ENERGY_TARGETS).items()},
    }
    return out


def _sha256_arr(path):
    return C.sha256_file(path)


def main():
    scene = sys.argv[1]
    stage = int(sys.argv[2])
    assert scene in C.SCENES, scene
    assert stage in (20000, 25000), "MG0-S cells are 20K and 25K only"
    t_start = time.time()

    cams = C.load_camera_set(scene)
    views = [int(v) for v in cams["camera_ids"]]
    assert len(views) == C.N_VIEWS, f"camera set is not the frozen 64 ({len(views)})"

    C.bootstrap_c0()
    fused_loss = C.make_fused_loss()
    dataset = C.load_scene_dataset(scene)
    ck = C.load_checkpoint(scene, stage)
    model = C.build_model(ck)
    C.scene_reset()
    del ck
    torch.cuda.empty_cache()
    N = int(model._xyz.shape[0])
    denom = C.flat_denom(model)                       # [N,59] sqrt(v_t)+eps
    print(f"[mg0s_variance] scene={scene} ckpt={stage} N={N} views={len(views)} "
          f"mem={C.gpu_mem_gb():.1f}GB", flush=True)

    # ---- Pass 1: fp64 view-mean of the preconditioned rows m_P = E_v P(g_v) --
    t0 = time.time()
    S_P = torch.zeros((N, C.ROW_DIM), dtype=torch.float64, device="cuda")
    view_losses = []
    for v in views:
        cam, gt = dataset.get_item(v)
        loss, g = C.view_grad_and_loss(model, cam, gt, fused_loss)
        view_losses.append(float(loss))
        S_P += (g / denom).double()
        del g
    m_P = S_P / len(views)
    del S_P
    t_a1 = time.time() - t0
    print(f"[mg0s_variance] Pass 1 (m_P) done {t_a1:.0f}s mem={C.gpu_mem_gb():.1f}GB",
          flush=True)

    # ---- Pass 2: fp64 per-dimension V = E_v (P(g_v) - m_P)^2 ----------------
    t0 = time.time()
    Q = torch.zeros((N, C.ROW_DIM), dtype=torch.float64, device="cuda")
    for v in views:
        cam, gt = dataset.get_item(v)
        loss, g = C.view_grad_and_loss(model, cam, gt, fused_loss)
        Q += (g / denom - m_P) ** 2
        del g
    t_a2 = time.time() - t0
    del m_P
    # identical expression to MG0's mg0_variance.py: score = Q.sum(dim=1)/n_views
    V_j = Q.sum(dim=1) / len(views)                    # [N] per-row V_j (fp64)
    Q /= len(views)                                    # per-dim V, fp64 (per-group views)
    print(f"[mg0s_variance] Pass 2 (V) done {t_a2:.0f}s", flush=True)

    # ---- per-group V vectors ------------------------------------------------
    per_group = {}
    for name, a, b in C.GROUPS:
        per_group[name] = Q[:, a:b].sum(dim=1)

    # ---- MG0 frozen-quantity cross-check ------------------------------------
    cross = {"mg0_score_file": f"MG0_SCORE_{scene}_{stage}.npy", "present": False}
    score_path = os.path.join(C.ARTIFACTS, cross["mg0_score_file"])
    if os.path.exists(score_path):
        mg0_score = np.load(score_path).astype(np.float64)
        v_np = V_j.cpu().numpy()
        assert mg0_score.shape == v_np.shape, (mg0_score.shape, v_np.shape)
        den = max(float(np.abs(mg0_score).max()), 1e-30)
        cross.update({
            "present": True,
            "sha256": _sha256_arr(score_path),
            "max_abs_diff": float(np.abs(mg0_score - v_np).max()),
            "max_rel_diff_vs_absmax": float(np.abs(mg0_score - v_np).max() / den),
            "note": "V_j is the same frozen quantity as MG0's score; this is a "
                    "numerical-parity cross-check, not a re-derivation.",
        })
        print(f"[mg0s_variance] MG0_SCORE cross-check max_abs_diff="
              f"{cross['max_abs_diff']:.3e}", flush=True)

    # ---- freeze the 4096-row low-rank sample (before any SVD) ---------------
    v_cpu = V_j.cpu()
    order = torch.argsort(v_cpu, descending=True, stable=True).to(torch.int64)
    hot_ids = order[:N_HOT]
    n_nonzero = int((v_cpu > 0).sum())
    hot_set = set(hot_ids.tolist())
    pool = torch.tensor([int(i) for i in torch.nonzero(v_cpu > 0).flatten().tolist()
                         if int(i) not in hot_set], dtype=torch.int64)
    assert pool.numel() >= N_UNIFORM, f"nonzero pool too small: {pool.numel()}"
    rng = np.random.default_rng(SAMPLE_SEED)
    pick = rng.choice(pool.numpy(), size=N_UNIFORM, replace=False)
    unif_ids = torch.tensor(np.sort(pick), dtype=torch.int64)
    sample_ids = torch.cat([hot_ids, unif_ids])
    partition = torch.cat([torch.zeros(N_HOT, dtype=torch.int8),
                           torch.ones(N_UNIFORM, dtype=torch.int8)])
    assert sample_ids.numel() == N_HOT + N_UNIFORM
    assert len(set(sample_ids.tolist())) == sample_ids.numel(), "sample not distinct"

    npz_path = os.path.join(C.ARTIFACTS, f"MG0S_VARIANCE_{scene}_{stage}.npz")
    np.savez_compressed(
        npz_path,
        V_j=V_j.cpu().numpy().astype(np.float64),
        V_j_per_group=np.stack([per_group[n].cpu().numpy().astype(np.float64)
                                for n, _, _ in C.GROUPS], axis=1),
        group_names=np.array([n for n, _, _ in C.GROUPS]),
        group_bounds=np.array([[a, b] for _, a, b in C.GROUPS], dtype=np.int64),
        sample_ids=sample_ids.numpy().astype(np.int64),
        sample_partition=partition.numpy().astype(np.int8),
        row_dim=np.array(C.ROW_DIM),
    )
    print(f"[mg0s_variance] wrote {npz_path}", flush=True)

    # ---- concentration summary ---------------------------------------------
    conc = {
        "campaign": "OPTLAWS0", "stage": "MG0S_partA_concentration",
        "laws": "L1 Gradient-Variance Concentration",
        "scene": scene, "checkpoint": stage, "n_views": len(views),
        "N_rows": N, "n_zero_variance_rows": N - n_nonzero,
        "n_nonzero_variance_rows": n_nonzero,
        "definition": "V_j = mean_v ||P(g_v,j) - mean_u P(g_u,j)||^2 ; "
                      "P_t(x) = x/(sqrt(v_t)+eps) with the checkpoint's "
                      "per-group eps; zeros stay zeros",
        "reuse": {"mg0_scripts_dir": MG0_SCRIPTS,
                  "camera_set": "MG0_CAMERA_SET.json (MG0 Stage 1, frozen)",
                  "checkpoint": f"N1 snapshot ckpt_{stage}.pt",
                  "mg0_core_sha256": C.sha256_file(os.path.join(MG0_SCRIPTS, "mg0_core.py"))},
        "view_fused_losses": view_losses,
        "loss_mean": float(np.mean(view_losses)),
        "mg0_score_crosscheck": cross,
        "overall": _concentration(v_cpu.numpy()),
        "per_group": {n: _concentration(per_group[n].cpu().numpy())
                      for n, _, _ in C.GROUPS},
        "sample_freeze": {
            "n_hot": N_HOT, "n_uniform": N_UNIFORM, "seed": SAMPLE_SEED,
            "rule": "hot = top-2048 rows of the frozen V_j ranking (stable argsort, "
                    "ties by row index); uniform = 2048 rows drawn without "
                    "replacement from the nonzero-V rows not already hot, "
                    "seed 20260929, stored sorted; union is distinct and exactly 4096",
            "ids_sha256": hashlib.sha256(
                sample_ids.numpy().astype(np.int64).tobytes()).hexdigest(),
            "variance_npz": os.path.basename(npz_path),
            "variance_npz_sha256": _sha256_arr(npz_path),
        },
        "timing": {"pass1_mP_s": t_a1, "pass2_V_s": t_a2,
                   "total_s": time.time() - t_start},
        "memory_gb_max": C.gpu_mem_gb(),
        "gpu_uuid": os.environ.get("MG0_GPU_UUID", "unknown"),
        "script_sha256": {"mg0s_variance.py": C.sha256_file(os.path.abspath(__file__))},
    }
    C.json_dump(conc, os.path.join(C.ARTIFACTS,
                                   f"MG0S_CONCENTRATION_{scene}_{stage}.json"))
    print(f"[mg0s_variance] {scene}@{stage} COMPLETE {time.time() - t_start:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()