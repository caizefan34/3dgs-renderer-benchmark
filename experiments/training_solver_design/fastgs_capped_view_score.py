"""Test one fixed view-balanced FastGS densification rule.

Run from the pinned FastGS checkout with normal train.py arguments and a
common --start_checkpoint. This is a research branch, not production code.
It retains the author's 10 cameras, renders, loss, pruning score, and exact
numbers of clone/split parents per event. Only their ranking changes.
"""

import json
import runpy
import sys

import torch

from utils import fast_utils


PER_VIEW_CAP = 32
original_render = fast_utils.render_fastgs
original_score = fast_utils.compute_gaussian_score_fastgs
counts_by_view = []


def capturing_render(*args, **kwargs):
    result = original_render(*args, **kwargs)
    if kwargs.get("get_flag"):
        counts_by_view.append(result["accum_metric_counts"].clone())
    return result


def candidate_score(camlist, gaussians, pipe, bg, args, DENSIFY=False):
    counts_by_view.clear()
    importance, pruning = original_score(camlist, gaussians, pipe, bg, args, DENSIFY=DENSIFY)
    if not DENSIFY:
        return importance, pruning
    if len(counts_by_view) != len(camlist):
        raise AssertionError("one metric-count render per scoring camera expected")

    with torch.no_grad():
        counts = torch.stack(counts_by_view)
        total = counts.to(torch.int64).sum(0)
        baseline_mask = importance > 5
        if not torch.equal(baseline_mask, total >= 6 * len(camlist)):
            raise AssertionError("author metric gate mismatch")

        grad = gaussians.xyz_gradient_accum / gaussians.denom
        grad[grad.isnan()] = 0.0
        abs_grad = gaussians.xyz_gradient_accum_abs / gaussians.denom
        abs_grad[abs_grad.isnan()] = 0.0
        clone_size = gaussians.get_scaling.max(dim=1).values <= args.dense * gaussians.spatial_lr_scale
        clone_eligible = clone_size & (grad.norm(dim=-1) >= args.grad_thresh)
        split_eligible = (~clone_size) & (abs_grad.norm(dim=-1) >= args.grad_abs_thresh)

        capped_score = counts.clamp(max=PER_VIEW_CAP).to(torch.int64).sum(0)
        selected = torch.zeros_like(baseline_mask)
        baseline_clone = clone_eligible & baseline_mask
        baseline_split = split_eligible & baseline_mask
        for eligible, baseline in ((clone_eligible, baseline_clone), (split_eligible, baseline_split)):
            target = int(baseline.sum())
            if target == 0:
                continue
            eligible_indices = eligible.nonzero(as_tuple=True)[0]
            top_local = torch.topk(capped_score[eligible_indices], target, sorted=False).indices
            selected[eligible_indices[top_local]] = True

        new_importance = torch.zeros_like(importance)
        new_importance[selected] = 6
        record = {
            "per_view_cap": PER_VIEW_CAP,
            "gaussians_pre_event": int(total.numel()),
            "baseline_clone": int(baseline_clone.sum()),
            "baseline_split": int(baseline_split.sum()),
            "candidate_clone": int((selected & clone_eligible).sum()),
            "candidate_split": int((selected & split_eligible).sum()),
            "overlap_clone": int((selected & baseline_clone).sum()),
            "overlap_split": int((selected & baseline_split).sum()),
        }
        if record["candidate_clone"] != record["baseline_clone"] or record["candidate_split"] != record["baseline_split"]:
            raise AssertionError("matched clone/split counts failed")
    print("FASTGS_CAPPED_VIEW_SCORE " + json.dumps(record, sort_keys=True), flush=True)
    return new_importance, pruning


fast_utils.render_fastgs = capturing_render
fast_utils.compute_gaussian_score_fastgs = candidate_score
sys.argv[0] = "train.py"
runpy.run_path("train.py", run_name="__main__")
