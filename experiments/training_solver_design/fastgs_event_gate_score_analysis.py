"""Analyze one FastGS event's per-view loss VJPs without touching test views."""

import json
import os
from pathlib import Path

import torch


HERE = Path(__file__).parent
CHILD_SEED = int(os.environ.get("FASTGS_CHILD_SEED", "20260927"))
SOURCE = HERE / f"fastgs_event_gate_multiview_scores_{CHILD_SEED}.pt"
OUT = HERE / f"fastgs_event_gate_score_analysis_{CHILD_SEED}.json"
POLICY_OUT = HERE / f"fastgs_event_gate_stable_positive_mask_{CHILD_SEED}.pt"
EPS = 1e-10


def main():
    bundle = torch.load(SOURCE, map_location="cpu", weights_only=True)
    matrix = bundle["camera_family_loss_vjp"].double()
    views, families = matrix.shape
    if (views, families) != (10, 9345):
        raise AssertionError(matrix.shape)
    mean = matrix.mean(dim=0)
    mean_sign = torch.sign(mean)
    positive = matrix > EPS
    negative = matrix < -EPS
    supported = (positive | negative).sum(dim=0)
    both_signs = positive.any(dim=0) & negative.any(dim=0)
    loo = (matrix.sum(dim=0).unsqueeze(0) - matrix) / (views - 1)
    meaningful_mean = mean.abs() > EPS
    loo_flip = ((loo * mean.unsqueeze(0) < 0) & meaningful_mean).any(dim=0)
    median = matrix.median(dim=0).values
    median_disagrees = (median * mean < 0) & meaningful_mean & (median.abs() > EPS)

    generator = torch.Generator().manual_seed(20260927)
    draws = torch.randint(views, (1000, views), generator=generator)
    negative_counts = torch.zeros(families, dtype=torch.float64)
    for batch in draws.split(50):
        negative_counts += (matrix[batch].mean(dim=1) < -EPS).double().sum(dim=0)
    negative_probability = negative_counts / draws.size(0)
    stable_negative = (mean < -EPS) & (negative_probability >= .95)
    stable_positive = (mean > EPS) & (negative_probability <= .05)
    l1_by_view = matrix.abs().sum(dim=1)
    torch.save({
        "ply_sha256": bundle["ply_sha256"],
        "mask_sha256": bundle["mask_sha256"],
        "family_parent_ids": bundle["family_parent_ids"],
        "stable_positive": stable_positive,
        "mean_vjp": mean.float(),
        "bootstrap_negative_probability": negative_probability.float(),
        "training_camera_names": bundle["camera_names"],
        "threshold": EPS,
        "bootstrap_draws": 1000,
        "seed": 20260927,
        "child_seed": CHILD_SEED,
    }, POLICY_OUT)
    sensitivity = []
    for threshold in (0., 1e-11, 1e-10, 1e-9, 1e-8):
        pos = matrix > threshold
        neg = matrix < -threshold
        support = (pos | neg).sum(dim=0)
        significant_mean = mean.abs() > threshold
        sensitivity.append({
            "threshold": threshold,
            "supported_two": int((support >= 2).sum()),
            "supported_two_opposite_signs": int(((support >= 2) & pos.any(dim=0) & neg.any(dim=0)).sum()),
            "mean_significant": int(significant_mean.sum()),
            "mean_leave_one_out_sign_flip": int(((loo * mean.unsqueeze(0) < 0) & significant_mean).any(dim=0).sum()),
        })
    result = {
        "source": SOURCE.name,
        "child_seed": CHILD_SEED,
        "exploratory_policy_mask": POLICY_OUT.name,
        "ply_sha256": bundle["ply_sha256"],
        "mask_sha256": bundle["mask_sha256"],
        "camera_names": bundle["camera_names"],
        "families": families,
        "significance_threshold_abs_loss_vjp": EPS,
        "supported_on_two_views": int((supported >= 2).sum()),
        "supported_two_with_opposite_signs": int(((supported >= 2) & both_signs).sum()),
        "mean_negative": int((mean < -EPS).sum()),
        "mean_positive": int((mean > EPS).sum()),
        "mean_near_zero": int((mean.abs() <= EPS).sum()),
        "leave_one_out_any_sign_flip": int(loo_flip.sum()),
        "median_mean_opposite_sign": int(median_disagrees.sum()),
        "bootstrap_1000_stable_negative_p_ge_0_95": int(stable_negative.sum()),
        "bootstrap_1000_stable_positive_p_le_0_05": int(stable_positive.sum()),
        "bootstrap_1000_unstable_meaningful": int((meaningful_mean & ~(stable_negative | stable_positive)).sum()),
        "per_view_l1_vjp": l1_by_view.tolist(),
        "per_view_meaningful_negative": negative.sum(dim=1).tolist(),
        "per_view_meaningful_positive": positive.sum(dim=1).tolist(),
        "threshold_sensitivity": sensitivity,
        "scope": "One room event, one sampled child geometry and ten fixed training views. Resampling cameras estimates score instability only; it does not estimate unseen-view reconstruction or long-horizon split utility.",
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
