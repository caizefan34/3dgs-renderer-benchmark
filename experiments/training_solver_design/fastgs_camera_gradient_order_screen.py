"""Frozen-state full-resolution FastGS camera-gradient order opportunity screen."""

import argparse
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


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantiles(values):
    return {"min": float(np.min(values)), "p10": float(np.percentile(values, 10)),
            "p50": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values))}


def max_prefix_norm(centered, order):
    dot_with_prefix = np.zeros(len(order), dtype=np.float64)
    norm_sq = 0.0
    maximum = 0.0
    for index in order:
        norm_sq += 2 * dot_with_prefix[index] + centered[index, index]
        dot_with_prefix += centered[:, index]
        maximum = max(maximum, np.sqrt(max(0.0, norm_sq)))
    return float(maximum)


def greedy_balanced_order(centered):
    n = len(centered)
    available = np.ones(n, dtype=bool)
    dot_with_prefix = np.zeros(n, dtype=np.float64)
    order = []
    diagonal = np.diag(centered)
    for _ in range(n):
        score = 2 * dot_with_prefix + diagonal
        index = int(np.argmin(np.where(available, score, np.inf)))
        available[index] = False
        order.append(index)
        dot_with_prefix += centered[:, index]
    return order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="room_big_capture4100")
    parser.add_argument("--iteration", type=int, default=4100)
    parser.add_argument("--expected-hash", default="64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc")
    parser.add_argument("--output-prefix", default="/root/fastgs_camera")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    source = REPO / "output" / args.model_name / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    expected_hash = args.expected_hash
    assert digest(source) == expected_hash
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output" / args.model_name), images="images",
        resolution=-1, white_background=False, data_device="cuda", eval=True)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False,
                           convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=args.iteration, shuffle=False)
    cameras = scene.getTrainCameras()
    assert len(cameras) == 272
    names = []
    losses = []
    xyz_gradients = []
    for index, camera in enumerate(cameras):
        for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
            getattr(model, name).grad = None
        image = render_fastgs(camera, model, pipe, torch.zeros(3, device="cuda"), .5)["render"]
        gt = camera.original_image
        loss = .8 * l1_loss(image, gt) + .2 * (1 - fast_ssim(image.unsqueeze(0), gt.unsqueeze(0)))
        loss.backward()
        xyz_gradients.append(model._xyz.grad.detach().flatten().clone())
        names.append(camera.image_name)
        losses.append(float(loss.detach()))
        if (index + 1) % 32 == 0:
            print(f"CAMERA_GRADIENT_PROGRESS {index + 1}/{len(cameras)}", flush=True)
    gradients = torch.stack(xyz_gradients)
    gram_gpu = gradients @ gradients.T
    gram = gram_gpu.double().cpu().numpy()
    diag_check = torch.sum(gradients.double().square(), dim=1).cpu().numpy()
    relative_diagonal_error = float(np.max(np.abs(np.diag(gram) - diag_check) /
                                           np.maximum(diag_check, 1e-30)))
    if relative_diagonal_error > 1e-4:
        raise AssertionError(relative_diagonal_error)
    gram_path = f"{args.output_prefix}_xyz_gram.npz"
    np.savez_compressed(gram_path, gram=gram,
                        camera_names=np.asarray(names), losses=np.asarray(losses))
    n = len(names)
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    mean_gradient_sq = float(gram.sum() / n**2)
    uniform_second_moment = float(np.mean(norms**2))
    uniform_variance = uniform_second_moment - mean_gradient_sq

    def importance(policy):
        weights = np.asarray(policy, dtype=np.float64)
        weights = np.maximum(weights, 1e-30)
        weights /= weights.sum()
        probabilities = .5 / n + .5 * weights
        variance = float(np.sum(norms**2 / probabilities) / n**2 - mean_gradient_sq)
        return {"variance_ratio_to_uniform": variance / uniform_variance,
                "probability_range_ratio": float(probabilities.max() / probabilities.min())}

    centered = gram - gram.mean(axis=1)[:, None] - gram.mean(axis=0)[None, :] + gram.mean()
    rng = np.random.default_rng(20260927)
    random_maxima = [max_prefix_norm(centered, rng.permutation(n)) for _ in range(100)]
    greedy_order = greedy_balanced_order(centered)
    greedy_maximum = max_prefix_norm(centered, greedy_order)
    cosine = gram / np.maximum(norms[:, None] * norms[None, :], 1e-30)
    result = {
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "model_ply_sha256": expected_hash,
        "gram_sha256": digest(gram_path),
        "camera_count": n,
        "model_name": args.model_name,
        "model_iteration": args.iteration,
        "gaussian_count": len(model.get_xyz),
        "resolution": [cameras[0].image_width, cameras[0].image_height],
        "gradient_coordinates": int(gradients.shape[1]),
        "gradient_norm": quantiles(norms),
        "loss": quantiles(losses),
        "pairwise_cosine": quantiles(cosine[np.triu_indices(n, 1)]),
        "mean_gradient_norm": float(np.sqrt(max(mean_gradient_sq, 0))),
        "one_draw_iid_variance": {
            "uniform": uniform_variance,
            "optimal_norm_probability_variance_ratio": float(
                ((norms.mean())**2 - mean_gradient_sq) / uniform_variance),
            "half_mixture_loss": importance(losses),
            "half_mixture_xyz_gradient_norm": importance(norms),
        },
        "frozen_gradient_epoch_order": {
            "random_max_prefix_norm": quantiles(random_maxima),
            "greedy_balanced_max_prefix_norm": greedy_maximum,
            "ratio_greedy_to_random_median": greedy_maximum / np.median(random_maxima),
        },
        "gram_diagonal_max_relative_error": relative_diagonal_error,
        "scope": "Frozen full-resolution original-loss xyz gradients; no camera order trained, no optimizer update, no unseen-view quality or end-to-end speed result; iid formula differs from author without-replacement epoch sampling",
    }
    Path(f"{args.output_prefix}_gradient_order_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("CAMERA_GRADIENT_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
