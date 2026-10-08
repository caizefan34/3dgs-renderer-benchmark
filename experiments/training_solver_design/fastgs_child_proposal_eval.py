"""Score the frozen child-choice branch on the room development cameras."""

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


RUNS = {
    "control": REPO / "output/room_child_choice_control_4612",
    "choice": REPO / "output/room_child_choice_choice_4612",
}
OUT = Path("/root/fastgs_child_proposal_eval.json")


def main():
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(RUNS["control"]), images="images", resolution=-1,
        white_background=False, data_device="cuda", eval=True)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=4164, shuffle=False)
    cameras = scene.getTestCameras()
    if len(cameras) != 39:
        raise AssertionError(len(cameras))
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    vgg = LPIPS("vgg", "0.1").cuda().eval()
    result = {"source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
              "camera_names": [c.image_name for c in cameras],
              "effective_resolution": [1600, 1066], "arms": {}}
    with torch.no_grad():
        for iteration in (4164, 4612):
            for arm, run in RUNS.items():
                source = run / f"point_cloud/iteration_{iteration}/point_cloud.ply"
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
                result["arms"][key] = {
                    "model_ply": str(source), "gaussians": int(model.get_xyz.shape[0]),
                    "mean": {metric: sum(row[metric] for row in rows) / len(rows)
                             for metric in ("psnr", "ssim", "lpips")},
                    "per_view": rows,
                }
                print("FASTGS_CHILD_CHOICE_EVAL " + json.dumps({
                    "arm": arm, "iteration": iteration, "gaussians": int(model.get_xyz.shape[0]),
                    "mean": result["arms"][key]["mean"],
                }, sort_keys=True), flush=True)
    result["scope"] = ("One room development event, ten training cameras used for selection, "
                       "39 previously consulted room development views for metrics. "
                       "No untouched confirmation or full training result.")
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
