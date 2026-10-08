"""Paired 64/512-step FastGS branch metrics on the same room test cameras."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from lpipsPyTorch.modules.lpips import LPIPS  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.image_utils import psnr  # noqa: E402


ROOT = REPO / "output"
CHILD_SEED = int(os.environ.get("FASTGS_CHILD_SEED", "20260927"))
RUNS = (
    ("control", 4164), ("veto", 4164),
    ("control", 4612), ("veto", 4612),
)
OUT = Path(f"/root/fastgs_gate_branch_64_512_eval_{CHILD_SEED}.json")


def model_file(arm, iteration):
    suffix = "" if CHILD_SEED == 20260927 else f"_seed{CHILD_SEED}"
    return ROOT / f"room_gate4100_{arm}_4612{suffix}/point_cloud/iteration_{iteration}/point_cloud.ply"


def main():
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(model_file("control", 4164).parents[2]),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=True,
    )
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=4164, shuffle=False)
    cameras = scene.getTestCameras()
    if len(cameras) != 39:
        raise AssertionError(len(cameras))
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    vgg = LPIPS("vgg", "0.1").cuda().eval()
    output = {"source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
              "child_seed": CHILD_SEED,
              "scene": "room", "effective_resolution": [1600, 1066],
              "camera_names": [camera.image_name for camera in cameras],
              "arms": {}}
    with torch.no_grad():
        for arm, iteration in RUNS:
            source = model_file(arm, iteration)
            if not source.exists():
                raise FileNotFoundError(source)
            model.load_ply(str(source))
            rows = []
            for camera in cameras:
                image = render_fastgs(camera, model, pipe, bg, 0.5)["render"].clamp(0, 1)
                gt = camera.original_image.cuda().clamp(0, 1)
                rows.append({
                    "camera": camera.image_name,
                    "psnr": float(psnr(image.unsqueeze(0), gt.unsqueeze(0)).mean()),
                    "ssim": float(fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))),
                    "lpips": float(vgg(image.unsqueeze(0), gt.unsqueeze(0))),
                })
            key = f"{arm}_{iteration}"
            output["arms"][key] = {
                "model_ply": str(source),
                "gaussians": int(model.get_xyz.shape[0]),
                "mean": {metric: sum(row[metric] for row in rows) / len(rows)
                         for metric in ("psnr", "ssim", "lpips")},
                "per_view": rows,
            }
            print("FASTGS_GATE_BRANCH_EVAL_ARM " + json.dumps({
                "arm": arm, "iteration": iteration,
                "gaussians": output["arms"][key]["gaussians"],
                "mean": output["arms"][key]["mean"],
            }, sort_keys=True), flush=True)
    for iteration in (4164, 4612):
        control = output["arms"][f"control_{iteration}"]
        veto = output["arms"][f"veto_{iteration}"]
        output[f"paired_{iteration}"] = {
            "veto_minus_control": {metric: veto["mean"][metric] - control["mean"][metric]
                                   for metric in ("psnr", "ssim", "lpips")},
            "views_better": {
                "psnr": sum(v["psnr"] > c["psnr"] for c, v in zip(control["per_view"], veto["per_view"])),
                "ssim": sum(v["ssim"] > c["ssim"] for c, v in zip(control["per_view"], veto["per_view"])),
                "lpips": sum(v["lpips"] < c["lpips"] for c, v in zip(control["per_view"], veto["per_view"])),
            },
        }
    output["scope"] = ("One native-selected 4100 split event from a common 4099 checkpoint; "
                       "both arms share pre-sampled child perturbations, full-resolution training loss, "
                       "camera RNG and no later densification. These room test views were used during "
                       "development; this is an exploratory 64/512-step result, not independent validation.")
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("FASTGS_GATE_BRANCH_EVAL_SUMMARY " + json.dumps({
        "paired_4164": output["paired_4164"], "paired_4612": output["paired_4612"]
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
