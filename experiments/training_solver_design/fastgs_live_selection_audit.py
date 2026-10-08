"""Observe exact FastGS pre-densification candidate sets without changing them.

Run from the pinned FastGS checkout with normal train.py arguments. This
diagnostic wraps the author's score renderer, then reproduces only the Boolean
clone/split selection masks before the author method performs its edit.
"""

import json
import runpy
import sys

import torch

from utils import fast_utils


original_render = fast_utils.render_fastgs
original_score = fast_utils.compute_gaussian_score_fastgs
captured_counts = []
captured_projected = []
event_index = 0


def capturing_render(*args, **kwargs):
    result = original_render(*args, **kwargs)
    if kwargs.get("get_flag"):
        captured_counts.append(result["accum_metric_counts"].clone())
        captured_projected.append((result["radii"] > 0).clone())
    return result


def audited_score(camlist, gaussians, pipe, bg, args, DENSIFY=False):
    global event_index
    captured_counts.clear()
    captured_projected.clear()
    result = original_score(camlist, gaussians, pipe, bg, args, DENSIFY=DENSIFY)
    if not DENSIFY:
        return result
    event_index += 1
    if len(captured_counts) != len(camlist):
        raise AssertionError("one metric-count render per scoring camera expected")

    with torch.no_grad():
        counts = torch.stack(captured_counts).to(torch.int64)
        totals = counts.sum(0)
        metric_gate = result[0] > 5
        if not torch.equal(metric_gate, totals >= 6 * len(camlist)):
            raise AssertionError("recomputed metric gate differs from author score")

        avg_grad = gaussians.xyz_gradient_accum / gaussians.denom
        avg_grad[avg_grad.isnan()] = 0.0
        avg_abs_grad = gaussians.xyz_gradient_accum_abs / gaussians.denom
        avg_abs_grad[avg_abs_grad.isnan()] = 0.0
        clone_size = gaussians.get_scaling.max(dim=1).values <= args.dense * gaussians.spatial_lr_scale
        clone_gate = clone_size & (avg_grad.norm(dim=-1) >= args.grad_thresh)
        split_gate = (~clone_size) & (avg_abs_grad.norm(dim=-1) >= args.grad_abs_thresh)
        actual_gate = metric_gate & (clone_gate | split_gate)

        selected_counts = counts[:, actual_gate]
        selected_totals = totals[actual_gate]
        projected = torch.stack(captured_projected).sum(0)[actual_gate]
        nonzero_views = (selected_counts > 0).sum(0)
        top_fraction = selected_counts.max(0).values / selected_totals.clamp_min(1)
        effective_views = selected_totals.square() / selected_counts.square().sum(0).clamp_min(1)
        concentrated = (projected >= 3) & (effective_views < 1.5)
        record = {
            "event_index": event_index,
            "cameras": len(camlist),
            "gaussians_pre_event": int(totals.numel()),
            "metric_gate": int(metric_gate.sum()),
            "gradient_clone_gate": int(clone_gate.sum()),
            "gradient_split_gate": int(split_gate.sum()),
            "actual_clone": int((metric_gate & clone_gate).sum()),
            "actual_split": int((metric_gate & split_gate).sum()),
            "actual_total": int(actual_gate.sum()),
            "actual_projected_ge_3": int((projected >= 3).sum()),
            "actual_projected_ge_3_effective_views_lt_1_5": int(concentrated.sum()),
            "actual_nonzero_views_le_1": int((nonzero_views <= 1).sum()),
            "actual_top_view_share_ge_0_75": int((top_fraction >= 0.75).sum()),
            "actual_effective_views_median": float(effective_views.median()) if effective_views.numel() else None,
        }
    print("FASTGS_LIVE_SELECTION " + json.dumps(record, sort_keys=True), flush=True)
    return result


fast_utils.render_fastgs = capturing_render
fast_utils.compute_gaussian_score_fastgs = audited_score
sys.argv[0] = "train.py"
runpy.run_path("train.py", run_name="__main__")
