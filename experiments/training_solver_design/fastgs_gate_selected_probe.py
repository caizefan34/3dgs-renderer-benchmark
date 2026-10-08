"""Read the native split-parent opacity at one FastGS densification event.

Runs the unmodified trainer from a saved checkpoint; the wrapper observes the
author's exact selection mask before the split and suppresses only final save.
"""

import json
import math
import os
import runpy
import sys
from pathlib import Path

import torch


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from scene import Scene  # noqa: E402
from utils import fast_utils  # noqa: E402


original_score = fast_utils.compute_gaussian_score_fastgs
cutoff = 1.0 / 255.0
records = []


def summarize(opacity: torch.Tensor) -> dict:
    values = opacity.detach().reshape(-1).to("cpu", torch.float64)
    center_alpha = values.clamp(max=0.99)
    viable = center_alpha >= cutoff
    threshold = torch.full_like(values, float("inf"))
    threshold[viable] = math.log1p(-cutoff) / torch.log1p(-center_alpha[viable])
    finite = threshold[viable]
    return {
        "count": int(values.numel()),
        "never_reaches_center_cutoff": int((~viable).sum()),
        "center_gate_threshold_quantiles": {
            str(p): float(torch.quantile(finite, p))
            for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        } if finite.numel() else {},
        "center_gate_threshold_above": {
            str(g): int((threshold > g).sum())
            for g in (0.01, 0.05, 0.1, 0.25, 0.5)
        },
    }


def observed_score(camlist, gaussians, pipe, bg, args, DENSIFY=False):
    result = original_score(camlist, gaussians, pipe, bg, args, DENSIFY=DENSIFY)
    if DENSIFY:
        with torch.no_grad():
            grads_abs = gaussians.xyz_gradient_accum_abs / gaussians.denom
            grads_abs[grads_abs.isnan()] = 0.0
            split_qualifiers = gaussians.get_scaling.max(dim=1).values > args.dense * gaussians.spatial_lr_scale
            split_gate = grads_abs.norm(dim=-1) >= args.grad_abs_thresh
            selected = (result[0] > 5) & split_qualifiers & split_gate
            mask_path = os.environ.get("FASTGS_GATE_MASK_PATH")
            if mask_path:
                torch.save({"selected_splits": selected.cpu(), "pre_event_count": int(selected.numel())}, mask_path)
            record = {
                "event": len(records) + 1,
                "gaussians_pre_event": int(selected.numel()),
                "selected_splits": summarize(gaussians.get_opacity[selected]),
                "all_gaussians": summarize(gaussians.get_opacity),
                "native_split_copies_parent_opacity": True,
                "native_pixel_alpha_cap": 0.99,
                "scope": "center-pixel threshold bound; no renderer VJP or training benefit",
            }
            records.append(record)
            print("FASTGS_GATE_SELECTED " + json.dumps(record, sort_keys=True), flush=True)
    return result


fast_utils.compute_gaussian_score_fastgs = observed_score
Scene.save = lambda self, iteration: None
sys.argv = [
    "train.py", "-s", "/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
    "-i", "images", "--eval", "--iterations", "4100",
    "--densification_interval", "100", "--optimizer_type", "default",
    "--highfeature_lr", "0.02", "--grad_abs_thresh", "0.0004",
    "--test_iterations", "999999", "--save_iterations", "999999",
    "--start_checkpoint", str(REPO / "checkpoints/room_big_4099.pth"),
    "--model_path", str(REPO / "output/room_big_gate_selected_4100"),
]
runpy.run_path(str(REPO / "train.py"), run_name="__main__")
if len(records) != 1:
    raise AssertionError(f"expected one densification event, got {len(records)}")
