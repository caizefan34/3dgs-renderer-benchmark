"""Measure local multi-view position-gradient agreement on a frozen FastGS state."""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def voxel_summary(xyz, per_view, divisions):
    gradient = np.mean(per_view, axis=0)
    magnitudes = np.linalg.norm(gradient, axis=1)
    low, high = np.percentile(xyz, [1, 99], axis=0)
    inside = np.all((xyz >= low) & (xyz < high), axis=1) & (magnitudes > 0)
    indices = np.floor((xyz[inside] - low) / (high - low) * divisions).astype(np.int64)
    keys = indices[:, 0] + divisions * (indices[:, 1] + divisions * indices[:, 2])
    count = divisions ** 3
    population = np.bincount(keys, minlength=count)
    norms = np.bincount(keys, weights=magnitudes[inside], minlength=count)
    group_gradient = np.stack([np.bincount(keys, weights=gradient[inside, j], minlength=count)
                               for j in range(3)], axis=1)
    ratio = np.linalg.norm(group_gradient, axis=1) / np.maximum(norms, 1e-30)
    view_groups = np.stack([
        np.stack([np.bincount(keys, weights=g[inside, j], minlength=count) for j in range(3)], axis=1)
        for g in per_view
    ])
    view_norms = np.linalg.norm(view_groups, axis=2)
    valid_view = view_norms > 1e-12
    view_count = valid_view.sum(axis=0)
    cross_cosine = np.zeros(count, dtype=np.float64)
    pair_count = np.zeros(count, dtype=np.int64)
    for a in range(len(per_view)):
        for b in range(a + 1, len(per_view)):
            both = valid_view[a] & valid_view[b]
            cosine = np.sum(view_groups[a] * view_groups[b], axis=1) / np.maximum(view_norms[a] * view_norms[b], 1e-30)
            cross_cosine[both] += cosine[both]
            pair_count[both] += 1
    cross_cosine /= np.maximum(pair_count, 1)
    eligible = (population >= 8) & (view_count >= 3)
    denominator = float(norms[eligible].sum())
    result = {"divisions_per_axis": divisions, "eligible_voxels": int(eligible.sum()),
              "eligible_gaussians": int(population[eligible].sum()),
              "visible_inside_gaussians": int(inside.sum()),
              "median_eligible_population": float(np.median(population[eligible])) if eligible.any() else None,
              "gradient_norm_coherence_p50": float(np.median(ratio[eligible])) if eligible.any() else None,
              "gradient_norm_coherence_p90": float(np.percentile(ratio[eligible], 90)) if eligible.any() else None,
              "cross_view_group_cosine_p50": float(np.median(cross_cosine[eligible])) if eligible.any() else None}
    for threshold in (0.5, 0.7, 0.9):
        mask = eligible & (ratio >= threshold) & (cross_cosine > 0)
        result[f"positive_cross_view_energy_share_at_coherence_{threshold}"] = (
            float(norms[mask].sum() / denominator) if denominator else None)
        result[f"positive_cross_view_voxel_count_at_coherence_{threshold}"] = int(mask.sum())
    return result


def main():
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output/room_big_capture4100"),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=True,
    )
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=4100, shuffle=False)
    xyz = model.get_xyz.detach().cpu().numpy().astype(np.float64)
    gradients = []
    cameras = scene.getTrainCameras()
    for index in (0, 17, 54, 121):
        for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
            getattr(model, name).grad = None
        camera = cameras[index]
        image = render_fastgs(camera, model, pipe, torch.zeros(3, device="cuda"), 0.5)["render"]
        gt = camera.original_image
        loss = 0.8 * l1_loss(image, gt) + 0.2 * (1 - fast_ssim(image.unsqueeze(0), gt.unsqueeze(0)))
        loss.backward()
        gradients.append(model._xyz.grad.detach().cpu().numpy().astype(np.float64))
        print("GROUP_MOTION_VIEW " + json.dumps({"camera_index": index, "camera": camera.image_name,
                                                  "original_loss": float(loss.detach()),
                                                  "nonzero_xyz_grad": int(np.count_nonzero(np.linalg.norm(gradients[-1], axis=1)))},
                                                 sort_keys=True), flush=True)
    per_view = np.stack(gradients)
    result = {"source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
              "model_ply_sha256": sha256(REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"),
              "camera_indices": [0, 17, 54, 121],
              "description": "Full-resolution original-loss xyz gradients at one frozen room state; 1-99 percentile voxelization; no parameter edits",
              "summaries": [voxel_summary(xyz, per_view, d) for d in (16, 32, 64)]}
    print("GROUP_MOTION_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
