"""Paired FastGS continuation: one native event with or without a gate-score veto.

Only the 4100 split is changed. Both arms use the author's optimizer, loss,
full-resolution images and the same pre-sampled native-rule child perturbations.
Run with `control` or `veto` as the first argument, followed by train.py args.
"""

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import torch

REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from scene.gaussian_model import GaussianModel
from utils import fast_utils
from utils.general_utils import build_rotation


if len(sys.argv) < 2 or sys.argv[1] not in ("control", "veto"):
    raise SystemExit("usage: python fastgs_gate_score_4100_branch.py control|veto [train.py args]")
MODE = sys.argv.pop(1)
CHILD_SEED = int(os.environ.get("FASTGS_CHILD_SEED", "20260927"))
POLICY = torch.load(f"/root/fastgs_event_gate_stable_positive_mask_{CHILD_SEED}.pt",
                    map_location="cpu", weights_only=True)
if POLICY["mask_sha256"] != "d2a393e0513f71bbd19996a9c5044da9dde7faf11740939f845b35c9b7c8b0a7":
    raise AssertionError("wrong split-selection evidence")
PARENTS = POLICY["family_parent_ids"].long()
REJECT = POLICY["stable_positive"].bool()
if PARENTS.numel() != 9345 or POLICY["child_seed"] != CHILD_SEED:
    raise AssertionError("policy dimensions changed")

original_score = fast_utils.compute_gaussian_score_fastgs
original_split = GaussianModel.densify_and_split_fastgs
full_split_mask = None
seen_event = False


def score_with_fixed_veto(camlist, gaussians, pipe, bg, args, DENSIFY=False):
    global full_split_mask, seen_event
    importance, pruning = original_score(camlist, gaussians, pipe, bg, args, DENSIFY=DENSIFY)
    if not DENSIFY:
        return importance, pruning
    if seen_event:
        raise AssertionError("pilot permits one densification event only")
    seen_event = True
    grad_abs = gaussians.xyz_gradient_accum_abs / gaussians.denom
    grad_abs[grad_abs.isnan()] = 0.0
    split_eligible = (gaussians.get_scaling.max(dim=1).values > args.dense * gaussians.spatial_lr_scale)
    split_eligible &= grad_abs.norm(dim=-1) >= args.grad_abs_thresh
    full_split_mask = (importance > 5) & split_eligible
    actual = full_split_mask.nonzero().flatten().cpu()
    if not torch.equal(actual, PARENTS):
        raise AssertionError({"mask_count": actual.numel(),
                              "expected_count": PARENTS.numel(),
                              "matching_prefix": int((actual[:min(actual.numel(), PARENTS.numel())] ==
                                                       PARENTS[:min(actual.numel(), PARENTS.numel())]).sum())})
    output = importance.clone()
    if MODE == "veto":
        output[PARENTS[REJECT].to(output.device)] = 0
    print("FASTGS_GATE_BRANCH_SELECTION " + json.dumps({
        "mode": MODE,
        "original_split_parents": int(full_split_mask.sum()),
        "vetoed_split_parents": int(REJECT.sum()) if MODE == "veto" else 0,
        "kept_split_parents": int(full_split_mask.sum()) - (int(REJECT.sum()) if MODE == "veto" else 0),
        "score_camera_names": POLICY["training_camera_names"],
        "child_seed": CHILD_SEED,
    }, sort_keys=True), flush=True)
    return output, pruning


def matched_child_split(self, metric_mask, filter, N=2):
    global full_split_mask
    if full_split_mask is None:
        return original_split(self, metric_mask, filter, N)
    n_init = self.get_xyz.shape[0]
    old_n = full_split_mask.numel()
    if filter.numel() != old_n or metric_mask.numel() != old_n:
        raise AssertionError("split mask size changed after clone")
    full = torch.zeros(n_init, dtype=torch.bool, device="cuda")
    selected = torch.zeros_like(full)
    full[:old_n] = full_split_mask & filter
    selected[:old_n] = metric_mask & filter
    if bool((selected & ~full).any()):
        raise AssertionError("policy invented a split")
    full_ids = full.nonzero().flatten()
    selected_ids = selected.nonzero().flatten()
    full_stds = self.get_scaling[full_ids].repeat(N, 1)
    torch.manual_seed(CHILD_SEED)
    full_samples = torch.normal(mean=torch.zeros_like(full_stds), std=full_stds)
    child_keep = selected[full_ids].repeat(N)
    samples = full_samples[child_keep]
    stds = self.get_scaling[selected_ids].repeat(N, 1)
    rots = build_rotation(self._rotation[selected_ids]).repeat(N, 1, 1)
    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_ids].repeat(N, 1)
    new_scaling = self.scaling_inverse_activation(stds / (0.8 * N))
    new_rotation = self._rotation[selected_ids].repeat(N, 1)
    new_features_dc = self._features_dc[selected_ids].repeat(N, 1, 1)
    new_features_rest = self._features_rest[selected_ids].repeat(N, 1, 1)
    new_opacity = self._opacity[selected_ids].repeat(N, 1)
    new_tmp_radii = self.tmp_radii[selected_ids].repeat(N)
    sample_sha256 = hashlib.sha256(full_samples.detach().cpu().numpy().tobytes()).hexdigest()
    self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                               new_opacity, new_scaling, new_rotation, new_tmp_radii)
    prune_filter = torch.cat((selected, torch.zeros(N * selected_ids.numel(), device="cuda", dtype=torch.bool)))
    self.prune_points(prune_filter)
    print("FASTGS_GATE_BRANCH_CHILDREN " + json.dumps({
        "mode": MODE, "all_child_sample_sha256": sample_sha256,
        "child_seed": CHILD_SEED,
        "full_parent_count": int(full_ids.numel()),
        "retained_parent_count": int(selected_ids.numel()),
        "gaussians_after_split": int(self.get_xyz.shape[0]),
    }, sort_keys=True), flush=True)
    full_split_mask = None


fast_utils.compute_gaussian_score_fastgs = score_with_fixed_veto
GaussianModel.densify_and_split_fastgs = matched_child_split
sys.argv[0] = "train.py"
runpy.run_path("train.py", run_name="__main__")
