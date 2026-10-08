#!/usr/bin/env python3
"""OPTLAWS0 / OC0 step 6 — contamination audit of the L3 association.

Usage: oc0_contamination.py

Emits
-----
  OC0_CONTAMINATION.json

Purpose (diagnostic only, declared a priori): the OC0 association is measured
between renderer-state features and cross-view gradient conflict, and several
non-scientific explanations could produce the same correlation.  This script
quantifies them so the independent analyst can discount the result:

  R1 label-support selection
     The conflict label only exists where BOTH row gradients have nonzero norm,
     and a nonzero norm is itself a visibility-mediated event.  Measured by the
     AUC with which "both cameras have an accepted hit" predicts pair validity,
     plus the structural checks that (a) a pair cannot be gradient-valid when a
     camera is invisible and (b) the counts involved.
  R2 shared measurement origin
     Features and labels come from the SAME forward pass at the SAME checkpoint.
     Declared; not removable.  Quantified by the Spearman between a Gaussian's
     mean row-gradient norm and its mean accepted-hit count.
  R3 sample-partition imbalance
     The hot half of the selection is defined by high gradient variance, which
     may be systematically more visible than the uniform half.  Measured by the
     standardised mean difference of every feature and of accept/visibility
     counts between the two partitions.
  R4 degenerate features / missingness
     Near-constant features and dropped pairs.

Frozen flag thresholds are declared in `flag_rules` and are NOT tuned.
No discovery label / verdict is emitted.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get(
    "MG0_SCRIPTS_DIR",
    "/root/epic05_mg0/scripts" if os.path.isdir("/root/epic05_mg0/scripts")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")))
import mg0_core as C  # noqa: E402
import oc0_common as K  # noqa: E402
import oc0_analysis as A  # noqa: E402

FLAG_RULES = {
    "label_support_visibility_auc_ge": 0.90,
    "gradnorm_visibility_abs_spearman_ge": 0.50,
    "partition_abs_smd_ge": 0.80,
    "feature_nunique_lt": 3,
    "usable_pair_fraction_lt": 0.50,
    "structural_inconsistency_gt": 0,
}


def _smd(a, b):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    v = 0.5 * (a.var(ddof=1) + b.var(ddof=1))
    if v <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(v))


def main():
    t0 = time.time()
    out = {
        "campaign": "OPTLAWS0", "stage": "OC0_contamination",
        "prereg": "OC0_PREREG.md",
        "creation_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "declared_risks": {
            "R1_label_support_selection": "the binary conflict label is only "
                "defined where both row gradients have nonzero norm; a nonzero "
                "row-gradient norm is visibility-mediated, so any visibility "
                "feature can appear predictive through label support alone",
            "R2_shared_measurement_origin": "features and labels are produced by "
                "the SAME forward pass of the SAME checkpoint with the SAME "
                "cameras; this is by design (the question is whether conflict is "
                "predictable from renderer state) and is declared, not removed",
            "R3_partition_imbalance": "the hot selection half is defined by high "
                "gradient variance and may be systematically more visible",
            "R4_degeneracy_and_missingness": "near-constant features and pairs "
                "dropped for missing native quantities",
        },
        "flag_rules": FLAG_RULES,
        "per_cell": {}, "flags": [],
        "script_sha256": {os.path.basename(__file__):
                          C.sha256_file(os.path.abspath(__file__))},
    }

    for scene, stage in K.CELLS:
        tag = K.cell_tag(scene, stage)
        c = A.load_cell(scene, stage)
        feats, valid, cos = A.pair_tables(c)
        F = c["F"]
        ng, nc = c["ng"], c["nc"]
        part = c["part"]
        hot = part == 0
        both_visible = F["visible"][:, c["pair_a"]] & F["visible"][:, c["pair_b"]]

        m = valid.copy()
        for f in K.PAIRWISE_FEATURES:
            m &= np.isfinite(feats[f])
        n_total = int(valid.size)
        n_valid = int(valid.sum())
        n_usable = int(m.sum())

        inv_a = ~F["visible"][:, c["pair_a"]]
        inv_b = ~F["visible"][:, c["pair_b"]]
        rec = {
            "scene": scene, "checkpoint": stage,
            "n_pairs_total": n_total,
            "n_pairs_gradient_valid": n_valid,
            "gradient_valid_fraction": n_valid / n_total,
            "n_pairs_usable": n_usable,
            "usable_pair_fraction": n_usable / n_total,
            "n_pairs_dropped_missing_feature": n_valid - n_usable,
            "R1_visibility_support": {
                "n_pairs_both_visible": int(both_visible.sum()),
                "n_pairs_both_visible_and_gradient_valid": int(
                    (both_visible & valid).sum()),
                "n_pairs_gradient_valid_but_a_invisible": int((valid & inv_a).sum()),
                "n_pairs_gradient_valid_but_b_invisible": int((valid & inv_b).sum()),
                "auc_both_visible_predicts_pair_valid": A.auc(
                    both_visible.ravel().astype(np.float64), valid.ravel()),
                "base_rate_pair_valid": float(valid.mean()),
            },
            "R2_shared_measurement_origin": {
                "declared_shared_forward_pass": True,
            },
            "R3_partition": {
                "n_hot_rows": int(hot.sum()),
                "n_uniform_rows": int((~hot).sum()),
                "feature_abs_smd_hot_vs_uniform": {},
                "accepted_hits_mean_hot": float(np.nanmean(
                    F["accepted_hits"][hot])),
                "accepted_hits_mean_uniform": float(np.nanmean(
                    F["accepted_hits"][~hot])),
                "hm_rows_ever_in_frustum_fraction_hot": float(
                    F["visible"][hot].any(axis=1).mean()),
                "hm_rows_ever_in_frustum_fraction_uniform": float(
                    F["visible"][~hot].any(axis=1).mean()),
            },
            "R4_degeneracy": {
                "nunique_per_feature": {}, "zero_fraction_per_feature": {},
            },
        }
        for f in K.PAIRWISE_FEATURES:
            rec["R3_partition"]["feature_abs_smd_hot_vs_uniform"][f] = abs(_smd(
                feats[f][hot][m[hot]], feats[f][~hot][m[~hot]]))
            v = feats[f][m]
            rec["R4_degeneracy"]["nunique_per_feature"][f] = int(
                np.unique(v).size)
            rec["R4_degeneracy"]["zero_fraction_per_feature"][f] = float(
                (v == 0).mean())

        # R2 quantification: per-Gaussian mean row-gradient norm vs visibility
        gz = np.load(K.artifact(f"OC0_GRADIENT_ROWS_{scene}_{stage}.npz"))
        row_norm = np.asarray(gz["row_norm_fp32"], dtype=np.float64)   # [nc,ng]
        gn = np.nanmean(np.where(row_norm > 0, row_norm, np.nan), axis=0)
        hits_mean = np.nanmean(F["accepted_hits"], axis=1)
        vis_ever = F["visible"].any(axis=1).astype(np.float64)
        rec["R2_shared_measurement_origin"].update({
            "spearman_rowgradnorm_vs_mean_accepted_hits": A.spearman(
                gn, hits_mean),
            "spearman_rowgradnorm_vs_ever_visible_indicator": A.spearman(
                gn, vis_ever),
            "mean_rowgradnorm_hot": float(np.nanmean(gn[hot])),
            "mean_rowgradnorm_uniform": float(np.nanmean(gn[~hot])),
        })

        # ------------------------------------------------------------------
        r1 = rec["R1_visibility_support"]
        r2 = rec["R2_shared_measurement_origin"]
        r3 = rec["R3_partition"]
        r4 = rec["R4_degeneracy"]
        if r1["auc_both_visible_predicts_pair_valid"] >= \
                FLAG_RULES["label_support_visibility_auc_ge"]:
            out["flags"].append(f"{tag}:label_support_visibility_auc")
        if r1["n_pairs_gradient_valid_but_a_invisible"] > 0 or \
                r1["n_pairs_gradient_valid_but_b_invisible"] > 0:
            out["flags"].append(f"{tag}:structural_inconsistency")
        for k in ("spearman_rowgradnorm_vs_mean_accepted_hits",
                  "spearman_rowgradnorm_vs_ever_visible_indicator"):
            if abs(r2[k]) >= FLAG_RULES["gradnorm_visibility_abs_spearman_ge"]:
                out["flags"].append(f"{tag}:gradnorm_visibility_leak_{k}")
        for f, v in r3["feature_abs_smd_hot_vs_uniform"].items():
            if v == v and v >= FLAG_RULES["partition_abs_smd_ge"]:
                out["flags"].append(f"{tag}:partition_imbalance_{f}")
        for f, n in r4["nunique_per_feature"].items():
            if n < FLAG_RULES["feature_nunique_lt"]:
                out["flags"].append(f"{tag}:feature_near_constant_{f}")
        if rec["usable_pair_fraction"] < FLAG_RULES["usable_pair_fraction_lt"]:
            out["flags"].append(f"{tag}:low_usable_pair_fraction")

        out["per_cell"][tag] = rec
        print(f"[oc0_contamination] {tag} valid={rec['gradient_valid_fraction']:.3f} "
              f"usable={rec['usable_pair_fraction']:.3f} "
              f"auc_vis->valid={r1['auc_both_visible_predicts_pair_valid']:.3f} "
              f"gn_leak={r2['spearman_rowgradnorm_vs_mean_accepted_hits']:.3f}",
              flush=True)

    out["n_flags"] = len(out["flags"])
    out["interpretation_boundary"] = (
        "flags describe non-scientific explanations for an observed association; "
        "their absence does not establish a causal or mechanism claim, and the "
        "frozen discovery-label rule is left to the independent analyst")
    C.json_dump(out, K.artifact("OC0_CONTAMINATION.json"))
    print(f"[oc0_contamination] COMPLETE {time.time() - t0:.0f}s "
          f"flags={out['n_flags']}", flush=True)


if __name__ == "__main__":
    main()