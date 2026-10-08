#!/usr/bin/env python3
"""OPTLAWS0 / MG0-S Part B — view-gradient low-rank structure (sidecar to MG0).

Usage: mg0s_lowrank.py <room|bicycle> <20000|25000>

Consumes the MG0-S variance artifact
  MG0S_VARIANCE_{scene}_{ckpt}.npz
whose sample IDs were frozen (before any singular-value analysis) by
mg0s_variance.py. Uses the frozen MG0 machinery via mg0_core (same 64 cameras,
same checkpoint, same Adam preconditioner P_t, same 59-dim row layout).

For every selected row j, the 64 x D matrix of Adam-preconditioned rows
X_j[v,:] = P(g_v,j) is built, centered across views, and decomposed.

Two modes:
  all_views    : zeros from invisible / no-gradient cameras stay in X.
  visible_only : only views with nonzero row-gradient norm; requires >= 8,
                 otherwise the row is labelled insufficient.

Reported per mode and per sample partition (hot / uniform):
  rank for 80/90/95% centered energy; rank-1/2/4/8/16 explained energy;
  effective rank r_eff = (sum s^2)^2 / sum s^4; plus per-parameter-group
  matrices (geometry = xyz+scale+quat, opacity, appearance = sh_dc+sh_rest).

Emits MG0S_LOWRANK_SAMPLE_{scene}_{ckpt}.npz (frozen IDs + the preconditioned
gradient rows, enough to independently re-verify every rank metric) and
MG0S_LOWRANK_{scene}_{ckpt}.json.

Telemetry only. No training, no densification, no formal test views,
no renderer / optimizer / loss / resolution change, no MG0 gate touched.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

MG0_SCRIPTS = os.environ.get(
    "MG0_SCRIPTS_DIR",
    "/root/epic05_mg0/scripts" if os.path.isdir("/root/epic05_mg0/scripts")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, MG0_SCRIPTS)
import mg0_core as C  # noqa: E402

ENERGY_TARGETS = (0.80, 0.90, 0.95)
RANK_LIST = (1, 2, 4, 8, 16)
MIN_VISIBLE = 8

# parameter-group matrices required by MG0S_PREREG Part B
GROUP_MATRICES = {
    "geometry": (0, 10),      # xyz + scale + quat
    "opacity": (10, 11),
    "appearance": (11, 59),   # sh_dc + sh_rest
}


# ---------------------------------------------------------------- rank metrics
def _metrics_from_singular(s2):
    """Rank metrics from a descending squared-singular-value vector (fp64)."""
    s2 = np.asarray(s2, dtype=np.float64)
    total = float(s2.sum())
    if total <= 0 or s2.size == 0:
        return {"total_energy": 0.0, "insufficient": True,
                "rank_for_energy": {f"{e * 100:g}%": None for e in ENERGY_TARGETS},
                "rank_k_explained": {str(k): None for k in RANK_LIST},
                "r_eff": 0.0}
    cum = np.cumsum(s2)
    rank_for = {}
    for e in ENERGY_TARGETS:
        k = int(np.searchsorted(cum, e * total - 1e-18) + 1)
        rank_for[f"{e * 100:g}%"] = min(k, int(s2.size))
    rk = {}
    for k in RANK_LIST:
        rk[str(k)] = float(cum[k - 1] / total) if k <= s2.size else float(cum[-1] / total)
    s4 = float((s2 ** 2).sum())
    r_eff = float(total * total / s4) if s4 > 0 else 0.0
    return {"total_energy": total, "insufficient": False,
            "rank_for_energy": rank_for, "rank_k_explained": rk, "r_eff": r_eff}


def _agg(recs):
    """Median / decile aggregate of per-row metric records."""
    if not recs:
        return {"n": 0}
    ok = [r for r in recs if not r["insufficient"]]
    out = {"n": len(recs), "n_insufficient": len(recs) - len(ok)}
    if not ok:
        return out
    r8 = np.array([r["rank_k_explained"]["8"] for r in ok], dtype=np.float64)
    r16 = np.array([r["rank_k_explained"]["16"] for r in ok], dtype=np.float64)
    reff = np.array([r["r_eff"] for r in ok], dtype=np.float64)
    r90 = np.array([r["rank_for_energy"]["90%"] for r in ok], dtype=np.float64)
    out.update({
        "median_rank8_explained": float(np.median(r8)),
        "p10_rank8_explained": float(np.percentile(r8, 10)),
        "p90_rank8_explained": float(np.percentile(r8, 90)),
        "median_rank16_explained": float(np.median(r16)),
        "median_r_eff": float(np.median(reff)),
        "p10_r_eff": float(np.percentile(reff, 10)),
        "p90_r_eff": float(np.percentile(reff, 90)),
        "median_rank_for_90pct": float(np.median(r90)),
    })
    return out


def _row_metrics(X, mask):
    """Per-row SVD metrics. X [n,nv,d] fp64; mask [n,nv] bool (visible-only)."""
    n, nv, d = X.shape
    out = []
    for j in range(n):
        if mask is None:
            M = X[j]
        else:
            vis = mask[j]
            if int(vis.sum()) < MIN_VISIBLE:
                out.append({"insufficient": True, "total_energy": 0.0,
                            "rank_for_energy": {}, "rank_k_explained": {}, "r_eff": 0.0})
                continue
            M = X[j][vis]
        M = M - M.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(M)
        out.append(_metrics_from_singular((s.double() ** 2).cpu().numpy()))
    return out


def main():
    scene = sys.argv[1]
    stage = int(sys.argv[2])
    assert scene in C.SCENES and stage in (20000, 25000)
    t_start = time.time()

    npz_in = os.path.join(C.ARTIFACTS, f"MG0S_VARIANCE_{scene}_{stage}.npz")
    z = np.load(npz_in)
    sample_ids = torch.tensor(z["sample_ids"], dtype=torch.int64)
    partition = z["sample_partition"]
    n_sample = int(sample_ids.numel())
    print(f"[mg0s_lowrank] scene={scene} ckpt={stage} n_sample={n_sample}", flush=True)

    cams = C.load_camera_set(scene)
    views = [int(v) for v in cams["camera_ids"]]
    assert len(views) == C.N_VIEWS

    C.bootstrap_c0()
    fused_loss = C.make_fused_loss()
    dataset = C.load_scene_dataset(scene)
    ck = C.load_checkpoint(scene, stage)
    model = C.build_model(ck)
    C.scene_reset()
    del ck
    torch.cuda.empty_cache()
    denom = C.flat_denom(model)

    # ---- Pass 3: preconditioned rows for the frozen sample -----------------
    # Stored in float32, not float16.  The frozen preconditioner
    # P_t(x) = x/(sqrt(v_t)+eps) maps a coordinate whose Adam second moment is
    # still exactly zero (a row materialised by the last densification event,
    # whose optimizer state is padded with zeros) to g/eps, which for this
    # checkpoint reaches |P| ~ 2.3e8 -- far beyond float16's 65504.  float16
    # storage silently turned those entries into inf and every SVD touching
    # them into NaN.  float32 stores the frozen quantity exactly as computed.
    t0 = time.time()
    idx = sample_ids.to("cuda")
    X = torch.empty((len(views), n_sample, C.ROW_DIM), dtype=torch.float32)
    denom_rows = denom.index_select(0, idx).to("cpu", torch.float32)
    visible_norm = torch.empty((n_sample, len(views)), dtype=torch.float32)
    for i, v in enumerate(views):
        cam, gt = dataset.get_item(v)
        loss, g = C.view_grad_and_loss(model, cam, gt, fused_loss)
        Pg = g.index_select(0, idx) / denom.index_select(0, idx)
        X[i] = Pg.to("cpu", torch.float32)
        visible_norm[:, i] = Pg.norm(dim=1).to("cpu", torch.float32)
        del g, Pg
    t_pass3 = time.time() - t0
    print(f"[mg0s_lowrank] Pass 3 done {t_pass3:.0f}s", flush=True)

    Xnpz = os.path.join(C.ARTIFACTS, f"MG0S_LOWRANK_SAMPLE_{scene}_{stage}.npz")
    x_absmax = float(X.abs().max().item())
    n_nonfinite = int((~torch.isfinite(X)).sum().item())
    assert n_nonfinite == 0, f"preconditioned rows not finite ({n_nonfinite})"
    np.savez_compressed(
        Xnpz,
        sample_ids=sample_ids.numpy().astype(np.int64),
        sample_partition=partition.astype(np.int8),
        X_preconditioned_fp32=X.numpy().astype(np.float32),
        denom_rows_fp32=denom_rows.numpy().astype(np.float32),
        visible_row_norm=visible_norm.numpy().astype(np.float32),
        view_ids=np.array(views, dtype=np.int64),
        row_dim=np.array(C.ROW_DIM),
        group_bounds=np.array([[a, b] for _, a, b in C.GROUPS], dtype=np.int64),
        group_names=np.array([n for n, _, _ in C.GROUPS]),
    )
    print(f"[mg0s_lowrank] wrote {Xnpz} absmax={x_absmax:.4e}", flush=True)

    # ---- rank analysis -----------------------------------------------------
    Xg = X.to("cuda", torch.float64)                      # [nv,n,59]
    Xg = Xg.permute(1, 0, 2).contiguous()                 # [n,nv,59]
    mask = (visible_norm > 0).numpy()                     # [n,nv]

    hot = partition == 0
    unif = partition == 1
    modes = {}
    for mode, m in (("all_views", None), ("visible_only", mask)):
        recs = _row_metrics(Xg, m)
        modes[mode] = {
            "hot": _agg([recs[j] for j in np.nonzero(hot)[0]]),
            "uniform": _agg([recs[j] for j in np.nonzero(unif)[0]]),
            "visible_count": {
                "n_rows_with_ge8_visible": int((mask.sum(axis=1) >= MIN_VISIBLE).sum()),
                "median_visible_views": float(np.median(mask.sum(axis=1))),
                "p10_visible_views": float(np.percentile(mask.sum(axis=1), 10)),
            } if mode == "visible_only" else None,
        }
        print(f"[mg0s_lowrank] mode={mode} hot.median_rank8="
              f"{modes[mode]['hot'].get('median_rank8_explained')} "
              f"uniform.median_rank8="
              f"{modes[mode]['uniform'].get('median_rank8_explained')}", flush=True)

    # ---- parameter-group matrices -----------------------------------------
    per_group = {}
    for gname, (a, b) in GROUP_MATRICES.items():
        sub = Xg[:, :, a:b].contiguous()
        per_group[gname] = {}
        for mode, m in (("all_views", None), ("visible_only", mask)):
            recs = _row_metrics(sub, m)
            per_group[gname][mode] = {
                "hot": _agg([recs[j] for j in np.nonzero(hot)[0]]),
                "uniform": _agg([recs[j] for j in np.nonzero(unif)[0]]),
            }

    out = {
        "campaign": "OPTLAWS0", "stage": "MG0S_partB_lowrank",
        "laws": "L2 View-Gradient Low-Rank Structure",
        "scene": scene, "checkpoint": stage, "n_views": len(views),
        "views": views, "n_sample": n_sample,
        "row_layout": ["xyz3", "scale3", "quat4", "opacity1", "sh_dc3", "sh_rest45"],
        "sample": {
            "n_hot": int(hot.sum()), "n_uniform": int(unif.sum()),
            "seed": 20260929,
            "frozen_ids_source": os.path.basename(npz_in),
            "frozen_ids_sha256": hashlib.sha256(
                sample_ids.numpy().astype(np.int64).tobytes()).hexdigest(),
            "sample_npz": os.path.basename(Xnpz),
            "sample_npz_sha256": C.sha256_file(Xnpz),
            "X_storage": "float32",
            "X_absmax": x_absmax,
            "X_nonfinite": n_nonfinite,
        },
        "definitions": {
            "X_j": "X_j[v,:] = P(g_v,j), P_t(x)=x/(sqrt(v_t)+eps)",
            "centering": "Xc_j = X_j - mean_v X_j (mean over the mode's views)",
            "r_eff": "(sum s^2)^2 / sum s^4",
            "rank_for_energy": "smallest k with cumulative s^2 / total >= target",
            "rank_k_explained": "cumulative s^2 of the top k / total",
            "visible_only_min_views": MIN_VISIBLE,
        },
        "modes": modes,
        "per_group_matrices": per_group,
        "notes": [
            "Scalar opacity rank must NOT be interpreted as evidence for the "
            "global law (MG0S_PREREG Part B).",
            "Sampling used the MG0-S frozen V_j ranking; row IDs were frozen in "
            "MG0S_VARIANCE before this SVD pass.",
            "Descriptive discovery labels are computed by mg0s_gate.py; no "
            "scientific verdict is issued here.",
            "Storage dtype note (post-hoc fix, 2026-09-29): the first run stored "
            "X in float16, which overflowed to inf for 4 hot rows whose Adam "
            "second moment is still exactly zero (|P| up to 2.3e8 >> 65504) and "
            "turned their SVD metrics into NaN. X is now stored in float32; no "
            "definition, sample id, view set or metric was changed.",
        ],
        "timing": {"pass3_sample_grads_s": t_pass3,
                   "total_s": time.time() - t_start},
        "memory_gb_max": C.gpu_mem_gb(),
        "gpu_uuid": os.environ.get("MG0_GPU_UUID", "unknown"),
        "source_variance_npz_sha256": C.sha256_file(npz_in),
        "script_sha256": {"mg0s_lowrank.py": C.sha256_file(os.path.abspath(__file__))},
    }
    C.json_dump(out, os.path.join(C.ARTIFACTS, f"MG0S_LOWRANK_{scene}_{stage}.json"))
    print(f"[mg0s_lowrank] {scene}@{stage} COMPLETE {time.time() - t_start:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()