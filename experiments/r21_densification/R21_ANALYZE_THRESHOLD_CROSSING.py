#!/usr/bin/env python3
"""Explain candidate changes by score, threshold, scale and visibility history.
Status: IMPLEMENTED; MX input evidence NOT_EXECUTED locally.
"""
from __future__ import annotations

import argparse
import torch
from R21_STATE import align_ids, load_state, write_json


def analyze(a,b):
    if a["phase"] != "pre_topology" or b["phase"] != "pre_topology" or a["iteration"] != b["iteration"]:
        raise ValueError("Use paired pre_topology snapshots at the same iteration")
    ai,bi=align_ids(a["stable_ids"],b["stable_ids"])
    ca,cb=a["candidates"],b["candidates"]
    if ca["threshold"] != cb["threshold"] or ca["scale_threshold"] != cb["scale_threshold"]:
        raise ValueError("Threshold/extent contract differs")
    mismatch=(ca["clone"][ai]!=cb["clone"][bi]) | (ca["split"][ai]!=cb["split"][bi])
    rows=[]
    for i,j in zip(ai[mismatch].tolist(),bi[mismatch].tolist()):
        score_a,score_b=float(ca["grow_score"][i]),float(cb["grow_score"][j])
        scale_a,scale_b=float(ca["scale_max"][i]),float(cb["scale_max"][j])
        rows.append({"stable_id":int(a["stable_ids"][i]), "score_left":score_a,"score_right":score_b,
            "score_margin_left":score_a-ca["threshold"],"score_margin_right":score_b-ca["threshold"],
            "score_delta":score_b-score_a,"scale_left":scale_a,"scale_right":scale_b,
            "scale_margin_left":scale_a-ca["scale_threshold"],"scale_margin_right":scale_b-ca["scale_threshold"],
            "accum_left":float(a["stats"]["xyz_gradient_accum"][i]),
            "accum_right":float(b["stats"]["xyz_gradient_accum"][j]),
            "denom_left":float(a["stats"]["denom"][i]),"denom_right":float(b["stats"]["denom"][j]),
            "visible_left":bool(a["visibility"][i]),"visible_right":bool(b["visibility"][j]),
            "clone_left":bool(ca["clone"][i]),"clone_right":bool(cb["clone"][j]),
            "split_left":bool(ca["split"][i]),"split_right":bool(cb["split"][j]),
            "grow_threshold_crossed":(score_a>=ca["threshold"])!=(score_b>=ca["threshold"]),
            "scale_partition_crossed":(scale_a<=ca["scale_threshold"])!=(scale_b<=ca["scale_threshold"])})
    # Decompose g=A/D. Hold one term fixed; this is algebra, not a causal experiment.
    da=a["stats"]["denom"][ai].double(); db=b["stats"]["denom"][bi].double()
    aa=a["stats"]["xyz_gradient_accum"][ai].double(); ab=b["stats"]["xyz_gradient_accum"][bi].double()
    valid=(da>0)&(db>0)
    accum_term=torch.where(valid,(ab-aa)/da.clamp_min(1),0)
    denom_term=torch.where(valid,ab*(1/db.clamp_min(1)-1/da.clamp_min(1)),0)
    return {"schema":"r21.threshold-analysis.v1","status":"ANALYSIS_COMPLETE","iteration":a["iteration"],
            "threshold":ca["threshold"],"scale_threshold":ca["scale_threshold"],"mismatch_count":len(rows),"rows":rows,
            "denom_mismatch_count":int((da!=db).sum()),"valid_decomposition_rows":int(valid.sum()),
            "mean_abs_accum_term":float(accum_term.abs().mean()) if valid.numel() else 0.,
            "mean_abs_denom_term":float(denom_term.abs().mean()) if valid.numel() else 0.,
            "root_cause":"NOT_ESTABLISHED","interpretation":"Crossing is an observation. Repeatability and intervention evidence are needed for H4."}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left",required=True);ap.add_argument("--right",required=True);ap.add_argument("--output",required=True)
    args=ap.parse_args();write_json(args.output,analyze(load_state(args.left),load_state(args.right)))


if __name__=="__main__":
    main()
