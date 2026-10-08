#!/usr/bin/env python3
"""D0 Stage 4: cross-scene predictability diagnostic (frozen protocol).

Prereg-driven, no hyperparameter search, no test-scene tuning:
  - label: RETROSPECTIVELY_DEAD (post-init birth roots)
  - features: the 30 preregistered pre-birth scalars (birth-table columns)
  - model 1: standardized L2 logistic regression, C=1.0, lbfgs, max_iter=5000
  - model 2: best single pre-birth scalar by training-scene AUROC
    (sign chosen on the training scene; ties -> lowest feature index)
  - cross-scene only: room -> bicycle and bicycle -> room
  - threshold: highest recall subject to training-scene survivor-lineage
    FPR <= 1% (ties -> higher threshold); frozen and applied to the test scene

STRONG (both directions, model 1): test AUROC >= 0.85,
test survivor FPR <= 2%, test cost-weighted dead recall >= 50%.
Cost weight per lineage l: f_FB * 20 * sum_s tiles_sampled(l)
+ f_OPT * lineage_population_integral(l).

Writes /root/epic05_d0/artifacts/D0_PREDICTABILITY.json.  Exit 0 always when
the analysis ran (the verdict inside the file is authoritative); exit 2 if the
headroom prerequisite is not met and --force is not given.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ART = "/root/epic05_d0/artifacts"
F_FB = 13.915 / 19.369
F_OPT = 4.140 / 19.369
SAMPLE_STRIDE = 20.0
FPR_TRAIN_MAX = 0.01
STRONG = {"auroc_min": 0.85, "fpr_max": 0.02, "cost_recall_min": 0.50}

FEATURE_NAMES = [
    "parent_age", "parent_opacity", "parent_xyz_norm", "depth_last_visible",
    "scale_x", "scale_y", "scale_z", "scale_volume", "radius_last_px",
    "sh_degree_active",
    "grad_accum_row", "grad_denom_row", "grad_mean_norm", "threshold_margin",
    "max_radii2D_row", "last_absgrad_norm_px", "branch",
    "vis_count_32", "vis_count_128", "vis_frac_32", "vis_frac_128",
    "n_distinct_cameras", "view_dir_spread_32",
    "res_mean_current_tile", "res_p90_current_tile", "res_roll_mean_32",
    "res_roll_var_32", "grad_dir_stability_32", "footprint_tiles_last_obs",
    "footprint_tiles_last32_mean",
]


def load_scene(scene):
    lin = pd.read_parquet(os.path.join(ART, "D0_LINEAGE_OUTCOMES_%s.parquet" % scene))
    bt = pd.read_parquet(os.path.join(ART, "D0_BIRTH_TABLE_%s.parquet" % scene))
    feat_by_uid = bt.set_index("uid")[FEATURE_NAMES]
    roots = lin["root_uid"].to_numpy(dtype=np.int64)
    missing = [int(r) for r in roots if r not in feat_by_uid.index]
    if missing:
        raise SystemExit("root uid(s) missing from birth table: %s" % missing[:5])
    X = feat_by_uid.loc[roots, FEATURE_NAMES].to_numpy(dtype=np.float64)
    dead = lin["retrospectively_dead"].to_numpy(dtype=bool)
    alive = lin["has_final_descendant"].to_numpy(dtype=bool)
    if not np.array_equal(dead, ~alive):
        raise SystemExit("dead/alive labels inconsistent on %s" % scene)
    w = (F_FB * SAMPLE_STRIDE * lin["tiles_sampled_full"].to_numpy(dtype=np.float64)
         + F_OPT * lin["lineage_population_integral"].to_numpy(dtype=np.float64))
    return {"X": X, "y": dead, "alive": alive, "w": w,
            "n": int(roots.size), "n_dead": int(dead.sum()),
            "dead_frac": float(dead.mean()),
            "cost_total": float(w.sum()),
            "cost_dead_total": float(w[dead].sum())}


def fill_and_standardize(Xtr, Xte):
    """NaN/inf -> train column mean; drop constant train columns; standardize
    with train statistics only."""
    Xtr = np.where(np.isfinite(Xtr), Xtr, np.nan)
    Xte = np.where(np.isfinite(Xte), Xte, np.nan)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Xtr_f = np.where(np.isnan(Xtr), mu[None, :], Xtr)
    Xte_f = np.where(np.isnan(Xte), mu[None, :], Xte)
    sd = Xtr_f.std(axis=0)
    keep = sd > 0
    Ztr = (Xtr_f[:, keep] - Xtr_f[:, keep].mean(0)) / Xtr_f[:, keep].std(0)
    Zte = (Xte_f[:, keep] - Xtr_f[:, keep].mean(0)) / Xtr_f[:, keep].std(0)
    return Ztr, Zte, Xtr_f, Xte_f, keep


def pick_threshold(scores, alive, fpr_max=FPR_TRAIN_MAX):
    """Highest-recall cutpoint on the training scene with survivor-lineage
    FPR <= fpr_max.  Ties -> higher threshold.  Cutpoints are score values;
    selecting `scores >= thr` takes a whole tie block, so block statistics
    are evaluated at the block's last index (integer-valued features make
    large tie blocks common in model 2)."""
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    surv = alive[order]
    n_surv = int(surv.sum())
    n_dead = int((~surv).sum())
    if n_surv == 0 or n_dead == 0:
        return None
    cum_surv = np.cumsum(surv)
    cum_dead = np.cumsum(~surv)
    fpr = cum_surv / n_surv
    rec = cum_dead / n_dead
    is_boundary = np.empty(s.size, dtype=bool)
    is_boundary[-1] = True
    is_boundary[:-1] = s[1:] != s[:-1]
    cand = np.nonzero(is_boundary)[0]
    ok = cand[fpr[cand] <= fpr_max + 1e-12]
    if ok.size == 0:
        return None
    rec_c = rec[ok]
    best_rec = rec_c.max()
    chosen = ok[rec_c == best_rec][0]  # ascending index = descending score
    return float(s[chosen])


def at_threshold(scores, y_dead, alive, w, thr):
    sel = scores >= thr
    n_surv = max(int(alive.sum()), 1)
    n_dead = max(int((~alive).sum()), 1)
    w_dead = float(w[~alive].sum())
    return {
        "auroc": float(roc_auc_score(y_dead, scores)),
        "auprc": float(average_precision_score(y_dead, scores)),
        "survivor_fpr": float((sel & alive).sum()) / n_surv,
        "birth_recall": float((sel & ~alive).sum()) / n_dead,
        "cost_weighted_dead_recall": (float(w[sel & ~alive].sum()) / w_dead
                                      if w_dead > 0 else None),
        "n_selected": int(sel.sum()),
    }


def run_direction(train_scene, test_scene, data):
    tr, te = data[train_scene], data[test_scene]
    Ztr, Zte, Xtr_f, Xte_f, keep = fill_and_standardize(tr["X"], te["X"])
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000,
                             class_weight=None)
    clf.fit(Ztr, tr["y"].astype(np.int64))
    s_tr = clf.predict_proba(Ztr)[:, 1]
    s_te = clf.predict_proba(Zte)[:, 1]
    thr = pick_threshold(s_tr, tr["alive"])
    m1 = {
        "dropped_constant_columns": [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES))
                                     if not keep[i]],
        "n_features_used": int(keep.sum()),
        "threshold": thr,
        "train": at_threshold(s_tr, tr["y"], tr["alive"], tr["w"], thr)
        if thr is not None else None,
        "test": at_threshold(s_te, te["y"], te["alive"], te["w"], thr)
        if thr is not None else None,
    }

    # model 2: best single pre-birth scalar by training-scene AUROC
    au, sign = np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES))
    for i in range(len(FEATURE_NAMES)):
        au_raw = roc_auc_score(tr["y"], Xtr_f[:, i])
        au[i] = max(au_raw, 1.0 - au_raw)
        sign[i] = 1.0 if au_raw >= 0.5 else -1.0
    best = int(np.argmax(au))  # ties -> lowest index
    s2_tr = sign[best] * Xtr_f[:, best]
    s2_te = sign[best] * Xte_f[:, best]
    thr2 = pick_threshold(s2_tr, tr["alive"])
    m2 = {
        "feature": FEATURE_NAMES[best],
        "feature_index": best,
        "sign": sign[best],
        "train_auroc": float(au[best]),
        "threshold": thr2,
        "train": at_threshold(s2_tr, tr["y"], tr["alive"], tr["w"], thr2)
        if thr2 is not None else None,
        "test": at_threshold(s2_te, te["y"], te["alive"], te["w"], thr2)
        if thr2 is not None else None,
    }
    return {"train_scene": train_scene, "test_scene": test_scene,
            "model1_logreg": m1, "model2_single_scalar": m2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    head_path = os.path.join(ART, "D0_HEADROOM.json")
    head = None
    if os.path.exists(head_path):
        with open(head_path) as f:
            head = json.load(f)
    if head is None or head.get("verdict") != "BOTH_SCENES_HEADROOM":
        msg = ("headroom prerequisite not met (D0_HEADROOM.json verdict=%s)"
               % (head or {}).get("verdict"))
        if not args.force:
            print("STAGE4 BLOCKED:", msg)
            return 2
        print("STAGE4 WARNING (--force):", msg)

    data = {s: load_scene(s) for s in ("room", "bicycle")}
    directions = {
        "room_to_bicycle": run_direction("room", "bicycle", data),
        "bicycle_to_room": run_direction("bicycle", "room", data),
    }

    reasons = []
    for name, d in directions.items():
        t = d["model1_logreg"]["test"] or {}
        if t.get("auroc", 0) < STRONG["auroc_min"]:
            reasons.append("%s: model1 test AUROC %.3f < %.2f"
                           % (name, t.get("auroc", 0), STRONG["auroc_min"]))
        if t.get("survivor_fpr", 1) > STRONG["fpr_max"]:
            reasons.append("%s: model1 test survivor FPR %.4f > %.2f"
                           % (name, t.get("survivor_fpr", 1), STRONG["fpr_max"]))
        if (t.get("cost_weighted_dead_recall") or 0) < STRONG["cost_recall_min"]:
            reasons.append("%s: model1 test cost-weighted dead recall %.3f < %.2f"
                           % (name, t.get("cost_weighted_dead_recall") or 0,
                              STRONG["cost_recall_min"]))
    verdict = "PREDICTABILITY_STRONG" if not reasons else "PREDICTABILITY_NOT_STRONG"

    out = {
        "artifact": "D0_PREDICTABILITY.json",
        "protocol": {
            "label": "RETROSPECTIVELY_DEAD (post-init birth roots)",
            "features": "30 preregistered pre-birth scalars",
            "nan_rule": "NaN/inf -> training-scene column mean",
            "constant_columns": "dropped using training scene only",
            "model1": "LogisticRegression(C=1.0, solver=lbfgs, max_iter=5000, "
                      "class_weight=None) on standardized features",
            "model2": "best single pre-birth scalar by training-scene AUROC; "
                      "sign chosen on training scene; ties -> lowest index",
            "threshold_rule": "highest recall with training-scene survivor-lineage "
                              "FPR <= 1%% (ties -> higher threshold); frozen and "
                              "applied to the test scene; no test-scene tuning",
            "cross_scene_only": ["room->bicycle", "bicycle->room"],
            "cost_weight": "f_FB*20*sum_s tiles_sampled(l) + "
                           "f_OPT*lineage_population_integral(l)",
            "f_fb": F_FB, "f_opt": F_OPT, "sample_stride": SAMPLE_STRIDE,
        },
        "strong_definition": {"applies_to": "model1_logreg",
                              "both_directions": True, **STRONG},
        "headroom_prerequisite": (head or {}).get("verdict"),
        "scenes": {s: {k: v for k, v in data[s].items() if k != "X"}
                   for s in data},
        "directions": directions,
        "verdict": verdict,
        "reasons": reasons,
    }
    with open(os.path.join(ART, "D0_PREDICTABILITY.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    # flat rows in the bundle verdict tool's contract (model "logreg" rows are
    # the frozen model 1; missing threshold -> fail-safe defaults, no rescue)
    rows = []
    for name, d in directions.items():
        t = d["model1_logreg"]["test"] or {}
        rows.append({
            "model": "logreg",
            "train_scene": d["train_scene"],
            "test_scene": d["test_scene"],
            "threshold": d["model1_logreg"]["threshold"],
            "n_features_used": d["model1_logreg"]["n_features_used"],
            "auroc": float(t.get("auroc") or 0.0),
            "auprc": float(t.get("auprc") or 0.0),
            "test_survivor_fpr": float(t.get("survivor_fpr")
                                       if t.get("survivor_fpr") is not None
                                       else 1.0),
            "birth_recall": float(t.get("birth_recall") or 0.0),
            "test_dead_cost_weighted_recall": float(
                t.get("cost_weighted_dead_recall") or 0.0),
        })
    with open(os.path.join(ART, "D0_PREDICTABILITY_ROWS.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(json.dumps({"verdict": verdict, "reasons": reasons,
                      "directions": {k: {"m1_test": v["model1_logreg"]["test"],
                                         "m2_feature": v["model2_single_scalar"]["feature"],
                                         "m2_test": v["model2_single_scalar"]["test"]}
                                     for k, v in directions.items()}},
                     indent=2, default=str))
    print("STAGE4", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
