"""Evaluate two FastGS room checkpoints on the fixed 39-view development set."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    from fused_ssim import fused_ssim  # noqa: E402
    from gaussian_renderer import render_fastgs  # noqa: E402
    from lpipsPyTorch.modules.lpips import LPIPS  # noqa: E402
    from scene import GaussianModel, Scene  # noqa: E402
    from utils.image_utils import psnr  # noqa: E402

    dataset = SimpleNamespace(
        source_path=str(args.dataset), model_path=str(args.control), images="images",
        resolution=-1, white_background=False, data_device="cuda", eval=True)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=args.iteration, shuffle=False)
    train_cameras = scene.getTrainCameras()
    cameras = scene.getTestCameras()
    if len(train_cameras) != 272:
        raise RuntimeError(f"Expected 272 room training cameras, found {len(train_cameras)}")
    if len(cameras) != 39:
        raise RuntimeError(f"Expected 39 room development cameras, found {len(cameras)}")

    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    vgg = LPIPS("vgg", "0.1").cuda().eval()
    result = {
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "iteration": args.iteration,
        "training_camera_count": len(train_cameras),
        "development_camera_count": len(cameras),
        "camera_names": [camera.image_name for camera in cameras],
        "resolution": [cameras[0].image_width, cameras[0].image_height],
        "arms": {},
        "scope": "39 room development-retain views; no formal test views or tuning.",
    }
    with torch.no_grad():
        for name, model_path in (("p1_control", args.control), ("p75_sparse_adjoint", args.sparse)):
            ply = model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
            model.load_ply(str(ply))
            rows = []
            for camera in cameras:
                rendered = render_fastgs(camera, model, pipe, background, 0.5)["render"].clamp(0, 1)
                target = camera.original_image.cuda().clamp(0, 1)
                rows.append({
                    "camera": camera.image_name,
                    "psnr": float(psnr(rendered.unsqueeze(0), target.unsqueeze(0)).mean()),
                    "ssim": float(fused_ssim(rendered.unsqueeze(0), target.unsqueeze(0))),
                    "lpips": float(vgg(rendered.unsqueeze(0), target.unsqueeze(0))),
                })
            result["arms"][name] = {
                "model_path": str(model_path),
                "ply_sha256": sha256(ply),
                "gaussians": int(model.get_xyz.shape[0]),
                "mean": {
                    metric: sum(row[metric] for row in rows) / len(rows)
                    for metric in ("psnr", "ssim", "lpips")
                },
                "per_view": rows,
            }
            print("FASTGS_SPARSE_ADJOINT_EVAL " + json.dumps({
                "arm": name,
                "gaussians": result["arms"][name]["gaussians"],
                "mean": result["arms"][name]["mean"],
            }, sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
