"""Compare one full-resolution FastGS split with frozen child placement choices."""

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

from scene.gaussian_model import GaussianModel  # noqa: E402
from utils import fast_utils  # noqa: E402
from utils.general_utils import build_rotation  # noqa: E402


if len(sys.argv) < 2 or sys.argv[1] not in ("control", "choice"):
    raise SystemExit("usage: fastgs_child_proposal_branch.py control|choice [train.py args]")
MODE = sys.argv.pop(1)
POLICY = torch.load("/root/fastgs_child_proposal_choice_20260927_20260928.pt",
                    map_location="cpu", weights_only=True)
PARENTS = POLICY["family_parent_ids"].long()
CHOOSE28 = POLICY["choose_seed20260928"].bool()
if PARENTS.numel() != 9345 or CHOOSE28.numel() != PARENTS.numel():
    raise AssertionError("policy dimensions changed")
if POLICY["mask_sha256"] != "d2a393e0513f71bbd19996a9c5044da9dde7faf11740939f845b35c9b7c8b0a7":
    raise AssertionError("wrong split-selection evidence")

original_score = fast_utils.compute_gaussian_score_fastgs
original_split = GaussianModel.densify_and_split_fastgs
full_split_mask = None
seen_event = False


def checked_score(camlist, gaussians, pipe, bg, args, DENSIFY=False):
    global full_split_mask, seen_event
    importance, pruning = original_score(camlist, gaussians, pipe, bg, args, DENSIFY=DENSIFY)
    if not DENSIFY:
        return importance, pruning
    if seen_event:
        raise AssertionError("pilot permits one densification event only")
    seen_event = True
    grad_abs = gaussians.xyz_gradient_accum_abs / gaussians.denom
    grad_abs[grad_abs.isnan()] = 0.0
    split_eligible = gaussians.get_scaling.max(dim=1).values > args.dense * gaussians.spatial_lr_scale
    split_eligible &= grad_abs.norm(dim=-1) >= args.grad_abs_thresh
    full_split_mask = (importance > 5) & split_eligible
    actual = full_split_mask.nonzero().flatten().cpu()
    if not torch.equal(actual, PARENTS):
        common = min(actual.numel(), PARENTS.numel())
        raise AssertionError({"actual": actual.numel(), "frozen": PARENTS.numel(),
                              "matching_prefix": int((actual[:common] == PARENTS[:common]).sum()),
                              "actual_head": actual[:10].tolist(), "frozen_head": PARENTS[:10].tolist()})
    print("FASTGS_CHILD_CHOICE_SELECTION " + json.dumps({
        "mode": MODE, "split_parents": int(actual.numel()),
        "choose_seed20260928": int(CHOOSE28.sum()),
    }, sort_keys=True), flush=True)
    return importance, pruning


def matched_split(self, metric_mask, filter, N=2):
    global full_split_mask
    if full_split_mask is None:
        return original_split(self, metric_mask, filter, N)
    if N != 2:
        raise AssertionError("only native two-child split is frozen")
    n_init = self.get_xyz.shape[0]
    old_n = full_split_mask.numel()
    if filter.numel() != old_n or metric_mask.numel() != old_n:
        raise AssertionError("split mask size changed after clone")
    selected = torch.zeros(n_init, dtype=torch.bool, device="cuda")
    selected[:old_n] = metric_mask & filter
    ids = selected.nonzero().flatten()
    if not torch.equal(ids.cpu(), PARENTS):
        raise AssertionError("split call parent order differs")
    stds = self.get_scaling[ids].repeat(N, 1)
    torch.manual_seed(20260927)
    samples27 = torch.normal(mean=torch.zeros_like(stds), std=stds)
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state()
    torch.manual_seed(20260928)
    samples28 = torch.normal(mean=torch.zeros_like(stds), std=stds)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(gpu_rng)
    sample_hashes = [hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
                     for x in (samples27, samples28)]
    if sample_hashes != [
        "185bd364a5eae3638bbf45099bfc7a69b62d70d20d743364ff8b587be88b2b40",
        "be02252c9a83e6f36f9cd67b47862e8662379190bb42e0296f4252d217327046",
    ]:
        raise AssertionError("child proposal bytes changed")
    samples = samples27 if MODE == "control" else torch.where(
        CHOOSE28.to(stds.device).repeat(N).unsqueeze(1), samples28, samples27)
    rots = build_rotation(self._rotation[ids]).repeat(N, 1, 1)
    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[ids].repeat(N, 1)
    new_scaling = self.scaling_inverse_activation(stds / (0.8 * N))
    self.densification_postfix(
        new_xyz, self._features_dc[ids].repeat(N, 1, 1),
        self._features_rest[ids].repeat(N, 1, 1), self._opacity[ids].repeat(N, 1),
        new_scaling, self._rotation[ids].repeat(N, 1), self.tmp_radii[ids].repeat(N))
    self.prune_points(torch.cat((selected, torch.zeros(N * ids.numel(), device="cuda", dtype=torch.bool))))
    print("FASTGS_CHILD_CHOICE_CHILDREN " + json.dumps({
        "mode": MODE, "sample_hashes": sample_hashes,
        "chosen_sample_sha256": hashlib.sha256(samples.detach().cpu().numpy().tobytes()).hexdigest(),
        "gaussians_after_split": int(self.get_xyz.shape[0]),
    }, sort_keys=True), flush=True)
    full_split_mask = None


fast_utils.compute_gaussian_score_fastgs = checked_score
GaussianModel.densify_and_split_fastgs = matched_split
sys.argv[0] = "train.py"
runpy.run_path("train.py", run_name="__main__")
