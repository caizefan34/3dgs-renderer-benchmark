#!/usr/bin/env python3
"""OPTLAWS0 / OC0 step 5 — frozen univariate + multivariate analysis.

Usage: oc0_analysis.py

Consumes ONLY the frozen artifacts:
  OC0_SELECTION.json, OC0_RENDERER_FEATURES_{scene}_{ckpt}.parquet,
  OC0_GRADIENT_ROWS_{scene}_{ckpt}.npz, OC0_INSTRUMENTATION_PARITY.json.

Emits
-----
  OC0_ANALYSIS.json
      * A. per-cell and pooled univariate Spearman(feature, Y_ab) and
        AUC(feature -> conflict_ab), with the AUC sign orientation taken from
        the room-20K cell only and frozen for every other cell;
      * B. the frozen multivariate predictor (exactly the six primary pairwise
        renderer features, L2 logistic regression, C=1.0, class_weight=None)
        under every cross-scene and temporal train/test split required by the
        prereg, with ROC AUC / average precision / Brier score / top-decile
        conflict enrichment;
      * the raw numeric inputs of the frozen discovery-label criteria, as
        booleans over the frozen thresholds.

Preprocessing note (declared, not tuned): the six features carry different
units (log ratios, counts, a 0/1 indicator), so each feature is standardised
with the TRAIN-cell mean/std and the same transform is applied to the test
cell.  Standardisation is not a hyperparameter; C=1.0 and class_weight=None
are used verbatim and nothing else is selected.  All rank-based numbers
(Spearman, AUC, AP, enrichment) are scale-invariant.

No discovery label / scientific verdict is emitted here: the DSH task forbids
issuing one, so the raw criterion values are returned for independent analysis.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "MG0_SCRIPTS_DIR",
    "/root/epic05_mg0/scripts" if os.path.isdir("/root/epic05_mg0/scripts")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")))
import mg0_core as C  # noqa: E402
import oc0_common as K  # noqa: E402

FEATURES = K.PAIRWISE_FEATURES
TRAIN_CELL = ("room", 20000)
TEST_SPLITS = (
    (("room", 20000), ("bicycle", 20000), "cross_scene_room20k_to_bicycle20k"),
    (("room", 20000), ("bicycle", 25000), "cross_scene_room20k_to_bicycle25k"),
    (("bicycle", 20000), ("room", 20000), "cross_scene_bicycle20k_to_room20k"),
    (("bicycle", 20000), ("room", 25000), "cross_scene_bicycle20k_to_room25k"),
    (("room", 20000), ("room", 25000), "temporal_room20k_to_room25k"),
    (("bicycle", 20000), ("bicycle", 25000), "temporal_bicycle20k_to_bicycle25k"),
)

COLUMNS = ("mean_alpha", "mean_T_before", "p10_T_before", "p50_T_before",
           "mean_depth_percentile", "mean_prior_contrib_count",
           "mean_remaining_contrib_count", "radius_x", "radius_y",
           "radius_max", "mean_radius")


# --------------------------------------------------------------------- loading
def load_cell(scene, stage):
    """Return the frozen per-camera matrices and the frozen gradient labels."""
    import pyarrow.parquet as pq

    tag = K.cell_tag(scene, stage)
    sel = json.load(open(K.artifact("OC0_SELECTION.json")))
    cell = sel["cells"][tag]
    cam_ids = np.asarray(cell["oc0_camera_ids"], dtype=np.int64)
    sel_ids = np.asarray(cell["selection_ids"], dtype=np.int64)
    hot_ids = set(int(x) for x in cell["hot_ids"])
    ng, nc = len(sel_ids), len(cam_ids)

    tab = pq.read_table(K.parquet_path(scene, stage))
    n_rows = tab.num_rows
    assert n_rows == ng * nc, (n_rows, ng, nc, "row order mismatch")
    got_cam = np.asarray(tab.column("camera_slot").to_numpy(), dtype=np.int64)
    assert np.array_equal(got_cam.reshape(ng, nc)[0], np.arange(nc))
    got_row = np.asarray(tab.column("gaussian_row").to_numpy(), dtype=np.int64)
    assert np.array_equal(got_row.reshape(ng, nc)[:, 0], sel_ids)

    def mat(name):
        return np.asarray(tab.column(name).to_numpy(zero_copy_only=False),
                          dtype=np.float64).reshape(ng, nc)

    F = {c: mat(c) for c in COLUMNS}
    F["accepted_hits"] = mat("accepted_hits")
    F["visible"] = mat("visible").astype(bool)
    part = np.asarray([0 if int(r) in hot_ids else 1 for r in sel_ids],
                      dtype=np.int8)

    gz = np.load(K.artifact(f"OC0_GRADIENT_ROWS_{scene}_{stage}.npz"))
    assert np.array_equal(gz["selection_ids"], sel_ids)
    assert np.array_equal(gz["oc0_camera_ids"], cam_ids)
    cos = np.asarray(gz["cos_fp32"], dtype=np.float64)          # [ng,n_pair]
    pv = np.asarray(gz["pair_valid"], dtype=bool)
    pair_a = np.asarray(gz["pair_a_slot"], dtype=np.int64)
    pair_b = np.asarray(gz["pair_b_slot"], dtype=np.int64)
    negfrac = np.asarray(gz["negative_pair_fraction"], dtype=np.float64)
    sevfrac = np.asarray(gz["severe_pair_fraction"], dtype=np.float64)
    disp = np.asarray(gz["directional_dispersion"], dtype=np.float64)

    return {
        "tag": tag, "scene": scene, "checkpoint": stage,
        "ng": ng, "nc": nc, "cam_ids": cam_ids, "sel_ids": sel_ids,
        "part": part, "F": F, "cos": cos, "pair_valid": pv,
        "pair_a": pair_a, "pair_b": pair_b,
        "negative_pair_fraction": negfrac,
        "severe_pair_fraction": sevfrac,
        "directional_dispersion": disp,
        "source_parquet_sha256": C.sha256_file(K.parquet_path(scene, stage)),
        "source_gradient_npz_sha256": C.sha256_file(
            K.artifact(f"OC0_GRADIENT_ROWS_{scene}_{stage}.npz")),
    }


def pair_tables(cell):
    """[ng, n_pair] feature matrices + labels from the frozen definitions."""
    F = cell["F"]
    a, b = cell["pair_a"], cell["pair_b"]

    def dif(name):
        v = F[name]
        return np.abs(v[:, a] - v[:, b])

    T = F["mean_T_before"]
    lt = np.abs(np.log(T[:, a] + K.T_EPS) - np.log(T[:, b] + K.T_EPS))
    lh = np.abs(np.log(F["accepted_hits"][:, a] + 1.0)
                - np.log(F["accepted_hits"][:, b] + 1.0))
    va = F["visible"][:, a]
    vb = F["visible"][:, b]
    ta = T[:, a]
    tb = T[:, b]
    role = ((va != vb) | ((ta < K.ROLE_T_MIDPOINT) != (tb < K.ROLE_T_MIDPOINT)))
    feats = {
        "abs_logT_diff": lt,
        "abs_depth_pct_diff": dif("mean_depth_percentile"),
        "abs_loghits_diff": lh,
        "abs_mean_alpha_diff": dif("mean_alpha"),
        "role_mismatch": role.astype(np.float64),
        "abs_prior_count_diff": dif("mean_prior_contrib_count"),
    }
    valid = cell["pair_valid"]
    cos = cell["cos"]
    return feats, valid, cos


# ------------------------------------------------------------------- metrics
def _rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if d == 0:
        return float("nan")
    return float((rx * ry).sum() / d)


def auc(score, label):
    """Rank-based ROC AUC (ties handled by average ranks)."""
    lab = np.asarray(label).astype(bool)      # accept bool OR 0/1 integer labels
    n_pos = int(lab.sum())
    n_neg = int(lab.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rank(score.astype(np.float64))
    s = r[lab].sum()
    return float((s - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(score, label):
    order = np.argsort(-score, kind="mergesort")
    y = label[order].astype(np.float64)
    cum = np.cumsum(y)
    prec = cum / (np.arange(y.size) + 1.0)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    return float((prec * y).sum() / n_pos)


def brier(prob, label):
    return float(np.mean((prob - label.astype(np.float64)) ** 2))


def top_decile_enrichment(score, label, frac=0.1):
    n = label.size
    k = max(int(np.floor(n * frac)), 1)
    order = np.argsort(-score, kind="mergesort")[:k]
    base = float(label.mean())
    if base == 0.0:
        return float("nan"), float(base), float(label[order].mean())
    return (float(label[order].mean() / base), float(base),
            float(label[order].mean()))


def fit_logreg(X, y, C_reg=1.0, max_iter=400):
    """L2 logistic regression: 0.5*||w||^2 + C*sum logloss, C=1.0 verbatim."""
    Xt = torch.as_tensor(X, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(Xt.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, tolerance_grad=1e-12,
                            tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ w + b
        obj = (torch.nn.functional.binary_cross_entropy_with_logits(
            z, yt, reduction="sum") * C_reg + 0.5 * (w * w).sum())
        obj.backward()
        return obj

    opt.step(closure)
    return w.detach().numpy(), float(b.detach().numpy()[0])


def _predict(w, b, X):
    z = X @ w + b
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------- main
def main():
    t0 = time.time()
    cells = {}
    for scene, stage in K.CELLS:
        tag = K.cell_tag(scene, stage)
        c = load_cell(scene, stage)
        feats, valid, cos = pair_tables(c)
        c["feats"] = feats
        c["valid"] = valid
        c["conflict"] = cos < K.CONFLICT_COSINE
        c["Y"] = 1.0 - cos
        cells[tag] = c
        print(f"[oc0_analysis] loaded {tag} ng={c['ng']} pairs={valid.size} "
              f"valid={valid.mean():.4f}", flush=True)

    # ---------- finite mask shared by every feature and the label -----------
    for tag, c in cells.items():
        m = c["valid"].copy()
        for f in FEATURES:
            m &= np.isfinite(c["feats"][f])
        c["mask"] = m
        print(f"[oc0_analysis] {tag} usable pairs={int(m.sum())} "
              f"({m.mean() * 100:.2f}% of all pairs)", flush=True)

    def flat(tag, key):
        return cells[tag][key][cells[tag]["mask"]]

    def sel(tag, arr):
        return arr[cells[tag]["mask"]]

    # ---------- A. univariate -----------------------------------------------
    uni = {"per_cell": {}, "pooled": {}, "orientation": {
        "rule": ("AUC sign orientation is taken from the room-20K training cell "
                 "only and frozen for every other cell"),
        "cell": K.cell_tag(*TRAIN_CELL)}}
    tr = K.cell_tag(*TRAIN_CELL)
    for tag, c in cells.items():
        m = c["mask"]
        y = c["Y"][m]
        conf = c["conflict"][m].astype(np.int8)
        rec = {}
        for f in FEATURES:
            x = c["feats"][f][m]
            a = auc(x, conf)
            orient = 1.0
            if tag == tr:
                orient = 1.0 if (not np.isnan(a) and a >= 0.5) else -1.0
            rec[f] = {
                "spearman_feature_vs_Y_ab": spearman(x, y),
                "auc_raw": a,
                "n_pairs": int(m.sum()),
            }
        # frozen orientation re-applied after room-20K is available
        uni["per_cell"][tag] = {"orient_placeholder": rec}
        uni["per_cell"][tag]["base_conflict_rate"] = float(conf.mean())
    orient = {}
    for f in FEATURES:
        a = uni["per_cell"][tr]["orient_placeholder"][f]["auc_raw"]
        orient[f] = 1.0 if (not np.isnan(a) and a >= 0.5) else -1.0
    uni["orientation"]["sign"] = orient
    for tag, c in cells.items():
        m = c["mask"]
        conf = c["conflict"][m].astype(np.int8)
        rec = uni["per_cell"][tag].pop("orient_placeholder")
        for f in FEATURES:
            raw = c["feats"][f][m]
            rec[f]["auc_oriented"] = auc(orient[f] * raw, conf)
        uni["per_cell"][tag]["features"] = rec

    pooled_x = {f: np.concatenate([sel(t, cells[t]["feats"][f]) for t in cells])
                for f in FEATURES}
    pooled_y = np.concatenate([flat(t, "Y") for t in cells])
    pooled_c = np.concatenate([flat(t, "conflict") for t in cells]).astype(np.int8)
    for f in FEATURES:
        uni["pooled"][f] = {
            "spearman_feature_vs_Y_ab": spearman(pooled_x[f], pooled_y),
            "auc_oriented": auc(orient[f] * pooled_x[f], pooled_c),
            "n_pairs": int(pooled_y.size),
        }
    uni["pooled"]["base_conflict_rate"] = float(pooled_c.mean())
    uni["pooled"]["n_pairs"] = int(pooled_y.size)

    # ---------- B. frozen multivariate predictor ----------------------------
    X = {t: np.column_stack([cells[t]["feats"][f][cells[t]["mask"]]
                             for f in FEATURES]) for t in cells}
    Yc = {t: cells[t]["conflict"][cells[t]["mask"]].astype(np.float64)
          for t in cells}
    pred = {"features": list(FEATURES), "model":
            "L2 logistic regression, C=1.0, class_weight=None (verbatim)",
            "preprocessing": ("train-cell mean/std standardisation; the same "
                              "transform is applied to the test cell"),
            "splits": {}}
    for (tr_cell, te_cell, name) in TEST_SPLITS:
        tt, et = K.cell_tag(*tr_cell), K.cell_tag(*te_cell)
        Xtr, ytr = X[tt], Yc[tt]
        Xte, yte = X[et], Yc[et]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        w, b = fit_logreg((Xtr - mu) / sd, ytr)
        p_tr = _predict(w, b, (Xtr - mu) / sd)
        p_te = _predict(w, b, (Xte - mu) / sd)
        enr, base, top = top_decile_enrichment(p_te, yte)
        pred["splits"][name] = {
            "train_cell": tt, "test_cell": et,
            "n_train": int(ytr.size), "n_test": int(yte.size),
            "train_base_conflict_rate": float(ytr.mean()),
            "test_base_conflict_rate": float(yte.mean()),
            "standardizer": {"mean": mu.tolist(), "std": sd.tolist()},
            "weights": dict(zip(FEATURES, w.tolist())),
            "intercept": b,
            "train": {"auc": auc(p_tr, ytr.astype(np.int8)),
                      "average_precision": average_precision(p_tr, ytr.astype(np.int8)),
                      "brier": brier(p_tr, ytr)},
            "test": {"roc_auc": auc(p_te, yte.astype(np.int8)),
                     "average_precision": average_precision(p_te, yte.astype(np.int8)),
                     "brier": brier(p_te, yte),
                     "top_decile_enrichment": enr,
                     "top_decile_conflict_rate": top,
                     "test_base_conflict_rate": base},
        }
        print(f"[oc0_analysis] {name}: auc={pred['splits'][name]['test']['roc_auc']:.4f} "
              f"ap={pred['splits'][name]['test']['average_precision']:.4f} "
              f"enr={enr:.3f}", flush=True)

    # ---------- frozen criterion inputs (no label emitted) ------------------
    med_hot = {t: float(np.median(cells[t]["negative_pair_fraction"]
                                  [cells[t]["part"] == 0])) for t in cells}
    c1 = {t: bool(med_hot[t] >= 0.10) for t in cells}
    tdr = {}
    for t in cells:
        m = cells[t]["mask"]
        best_f, best_v = None, 0.0
        for f in K.T_DEPTH_ROLE_FEATURES:
            v = abs(spearman(cells[t]["feats"][f][m], cells[t]["Y"][m]))
            tdr.setdefault(t, {})[f] = v
            if v > best_v:
                best_f, best_v = f, v
        tdr[t] = {"per_feature_abs_spearman": tdr[t],
                  "max_abs_spearman": best_v, "argmax_feature": best_f,
                  "satisfied_ge_0.35": bool(best_v >= 0.35)}
    c2 = {t: bool(tdr[t]["satisfied_ge_0.35"]) for t in cells}
    test_auc = {n: pred["splits"][n]["test"]["roc_auc"]
                for (_, _, n) in TEST_SPLITS}
    test_enr = {n: pred["splits"][n]["test"]["top_decile_enrichment"]
                for (_, _, n) in TEST_SPLITS}
    crit = {
        "criterion_1_nontrivial_conflict": {
            "rule": "median negative-pair fraction among the high-V sample "
                    ">= 0.10 in all cells",
            "per_cell_median_negative_pair_fraction_hot": med_hot,
            "per_cell_satisfied": c1,
            "all_cells_satisfied": bool(all(c1.values())),
        },
        "criterion_2_t_depth_role_spearman": {
            "rule": "at least one T/depth-role feature has |Spearman| >= 0.35 "
                    "in all four cells",
            "per_cell": tdr,
            "all_cells_satisfied": bool(all(c2.values())),
        },
        "criterion_3_auc": {
            "rule_strong": "every cross-scene/temporal test has ROC AUC >= 0.75",
            "rule_moderate": "every cross-scene/temporal test has ROC AUC >= 0.65",
            "per_test": test_auc,
            "strong_satisfied": bool(all(v >= 0.75 for v in test_auc.values())),
            "moderate_satisfied": bool(all(v >= 0.65 for v in test_auc.values())),
        },
        "criterion_4_enrichment": {
            "rule_strong": "top-decile predicted-risk pairs have >= 2.0x the base "
                           "conflict rate in every test cell",
            "rule_moderate": ">= 1.5x",
            "per_test": test_enr,
            "strong_satisfied": bool(all(v >= 2.0 for v in test_enr.values())),
            "moderate_satisfied": bool(all(v >= 1.5 for v in test_enr.values())),
        },
    }
    crit["discovery_label"] = None
    crit["note"] = ("the frozen label follows mechanically from the criterion "
                    "booleans above, but DSH_TASK_OPTLAWS0.md forbids issuing a "
                    "scientific verdict, so no label is emitted here")

    out = {
        "campaign": "OPTLAWS0", "stage": "OC0_analysis",
        "prereg": "OC0_PREREG.md", "schema": "SCHEMA.json",
        "creation_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thresholds": {"CONFLICT_COSINE": K.CONFLICT_COSINE,
                       "SEVERE_COSINE": K.SEVERE_COSINE,
                       "T_EPS": K.T_EPS,
                       "ROLE_T_MIDPOINT": K.ROLE_T_MIDPOINT},
        "cells": {t: {
            "scene": cells[t]["scene"], "checkpoint": cells[t]["checkpoint"],
            "n_gaussians": cells[t]["ng"], "n_cameras": cells[t]["nc"],
            "n_pairs_total": int(cells[t]["valid"].size),
            "n_pairs_gradient_valid": int(cells[t]["valid"].sum()),
            "n_pairs_usable": int(cells[t]["mask"].sum()),
            "base_conflict_rate": uni["per_cell"][t]["base_conflict_rate"],
            "median_negative_pair_fraction_hot": med_hot[t],
            "median_negative_pair_fraction_uniform": float(np.median(
                cells[t]["negative_pair_fraction"][cells[t]["part"] == 1])),
            "median_severe_pair_fraction_hot": float(np.median(
                cells[t]["severe_pair_fraction"][cells[t]["part"] == 0])),
            "mean_directional_dispersion_hot": float(np.nanmean(
                cells[t]["directional_dispersion"][cells[t]["part"] == 0])),
            "source_parquet_sha256": cells[t]["source_parquet_sha256"],
            "source_gradient_npz_sha256": cells[t]["source_gradient_npz_sha256"],
        } for t in cells},
        "A_univariate": uni,
        "B_predictor": pred,
        "prereg_criteria_check": crit,
        "timing_s": time.time() - t0,
        "script_sha256": {os.path.basename(__file__):
                          C.sha256_file(os.path.abspath(__file__))},
    }
    if os.path.exists(K.artifact("OC0_INSTRUMENTATION_PARITY.json")):
        pf = json.load(open(K.artifact("OC0_INSTRUMENTATION_PARITY.json")))
        out["parity_ok"] = pf.get("parity_ok")
    C.json_dump(out, K.artifact("OC0_ANALYSIS.json"))
    print(f"[oc0_analysis] COMPLETE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()