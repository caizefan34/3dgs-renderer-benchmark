"""Audit how many views support FastGS's pixel-count densification gate.

Run from a pinned FastGS checkout. This reads a saved model and training cameras;
it does not update the model or evaluate official test views.
"""

import argparse
import json
import random
from types import SimpleNamespace

import torch

from gaussian_renderer import render_fastgs
from scene import GaussianModel, Scene
from utils.fast_utils import get_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--loss-thresh", type=float, default=0.1)
    parser.add_argument("--mult", type=float, default=0.5)
    args = parser.parse_args()

    model_args = SimpleNamespace(
        source_path=args.source,
        model_path=args.model,
        images="images",
        eval=True,
        resolution=-1,
        white_background=False,
        data_device="cuda",
        sh_degree=3,
    )
    gaussians = GaussianModel(3, "default")
    scene = Scene(model_args, gaussians, load_iteration=args.iteration, shuffle=False)
    pipe = SimpleNamespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
    rng = random.Random(args.seed)
    cameras = rng.sample(scene.getTrainCameras(), 10)
    background = torch.zeros(3, device="cuda")
    per_view = []
    projected_views = []

    with torch.no_grad():
        for camera in cameras:
            first = render_fastgs(camera, gaussians, pipe, background, args.mult)
            image = first["render"]
            projected_views.append(first["radii"] > 0)
            truth = camera.original_image.cuda()
            metric_map = (get_loss(image, truth) > args.loss_thresh).int()
            counts = render_fastgs(
                camera,
                gaussians,
                pipe,
                background,
                args.mult,
                get_flag=True,
                metric_map=metric_map,
            )["accum_metric_counts"]
            per_view.append(counts)

    counts = torch.stack(per_view).to(torch.int64)
    totals = counts.sum(0)
    selected = totals >= 60  # floor(total / 10) > 5 in the pinned implementation.
    selected_counts = counts[:, selected]
    nonzero_views = (selected_counts > 0).sum(0)
    projected = torch.stack(projected_views).sum(0)[selected]
    top_fraction = selected_counts.max(0).values / totals[selected].clamp_min(1)
    effective_views = totals[selected].square() / selected_counts.square().sum(0).clamp_min(1)
    result = {
        "selection_scope": "metric_gate_only; gradient qualifiers unavailable in saved PLY",
        "source": args.source,
        "model": args.model,
        "iteration": args.iteration,
        "seed": args.seed,
        "loss_thresh": args.loss_thresh,
        "mult": args.mult,
        "cameras": [camera.image_name for camera in cameras],
        "gaussians": int(totals.numel()),
        "selected": int(selected.sum()),
        "selected_nonzero_views_le_1": int((nonzero_views <= 1).sum()),
        "selected_nonzero_views_le_2": int((nonzero_views <= 2).sum()),
        "selected_projected_ge_3": int((projected >= 3).sum()),
        "selected_projected_ge_3_nonzero_le_1": int(((projected >= 3) & (nonzero_views <= 1)).sum()),
        "selected_projected_ge_3_nonzero_le_2": int(((projected >= 3) & (nonzero_views <= 2)).sum()),
        "selected_projected_ge_3_effective_views_lt_1_5": int(((projected >= 3) & (effective_views < 1.5)).sum()),
        "selected_top_view_share_ge_0_75": int((top_fraction >= 0.75).sum()),
        "selected_top_view_share_ge_0_9": int((top_fraction >= 0.9).sum()),
        "nonzero_views_histogram": torch.bincount(nonzero_views, minlength=11).cpu().tolist(),
        "top_view_share_median": float(top_fraction.median()) if top_fraction.numel() else None,
        "effective_views_median": float(effective_views.median()) if effective_views.numel() else None,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
