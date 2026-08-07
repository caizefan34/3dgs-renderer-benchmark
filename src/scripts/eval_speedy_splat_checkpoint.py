#!/usr/bin/env python3
"""Score one Speedy-Splat checkpoint on the official test split.

Runs inside the pinned j-alex-hanson/speedy-splat checkout (via --source-dir)
so the metrics are computed with the official implementations:
utils.image_utils.psnr, utils.loss_utils.ssim, and the official VGG LPIPS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("-m", "--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    if not (source_dir / "train.py").is_file():
        raise SystemExit(f"missing pinned Speedy-Splat trainer in {source_dir}")
    sys.path.insert(0, str(source_dir))

    import torch

    try:
        from diff_gaussian_rasterization import SparseGaussianAdam

        SPARSE_ADAM_AVAILABLE = True
    except Exception:  # pragma: no cover - mirrors official train.py fallback
        SPARSE_ADAM_AVAILABLE = False

    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel, render
    from lpipsPyTorch import LPIPS
    from scene import Scene
    from utils.general_utils import safe_state
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    inner = argparse.ArgumentParser(description="official 3DGS paper eval")
    model_group = ModelParams(inner, sentinel=True)
    pipeline_group = PipelineParams(inner)
    inner.add_argument("--iteration", type=int, default=args.iteration)
    saved_argv = sys.argv
    sys.argv = [
        "eval_speedy_splat_checkpoint.py",
        "-m",
        str(args.model_path.resolve()),
        "--iteration",
        str(args.iteration),
    ]
    merged = get_combined_args(inner)
    sys.argv = saved_argv
    dataset = model_group.extract(merged)
    pipeline = pipeline_group.extract(merged)

    safe_state(True)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    criterion = LPIPS(net_type="vgg").to("cuda")

    cameras = scene.getTestCameras()
    if not cameras:
        raise SystemExit("official test split is empty")
    psnr_sum = ssim_sum = lpips_sum = 0.0
    with torch.no_grad():
        for view in cameras:
            image = torch.clamp(
                render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                )["render"],
                0.0,
                1.0,
            )
            gt = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            psnr_sum += float(psnr(image, gt).mean().double())
            ssim_sum += float(ssim(image, gt).mean().double())
            lpips_sum += float(criterion(image.unsqueeze(0), gt.unsqueeze(0)).mean().double())
    count = len(cameras)
    result = {
        "psnr": psnr_sum / count,
        "ssim": ssim_sum / count,
        "lpips": lpips_sum / count,
        "num_GS": int(len(gaussians.get_xyz)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
