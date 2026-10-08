"""Frozen FastGS full-resolution forward/backward capture for scalar-adjoint parity."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


REPO = Path("/root/codex_fastgs_audit_20260927")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from diff_gaussian_rasterization_fastgs import _C  # noqa: E402
from fused_ssim import fused_ssim  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from scene import GaussianModel, Scene, dataset_readers  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402


PARAMETERS = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")
CAMERAS = ("DSCF4668", "DSCF4687", "DSCF4729", "DSCF4806")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_name = "room_big_capture4100"
    ply = REPO / "output" / model_name / "point_cloud/iteration_4100/point_cloud.ply"
    expected = "64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc"
    assert sha256(ply) == expected
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output" / model_name), images="images", resolution=-1,
        white_background=False, data_device="cuda", eval=False,
    )
    original_reader = dataset_readers.readColmapCameras

    def selected_reader(cam_extrinsics, cam_intrinsics, images_folder):
        chosen = {key: item for key, item in cam_extrinsics.items()
                  if Path(item.name).stem in CAMERAS}
        assert {Path(item.name).stem for item in chosen.values()} == set(CAMERAS)
        return original_reader(chosen, cam_intrinsics, images_folder)

    model = GaussianModel(3, "default")
    dataset_readers.readColmapCameras = selected_reader
    try:
        scene = Scene(dataset, model, load_iteration=4100, shuffle=False)
    finally:
        dataset_readers.readColmapCameras = original_reader
    cameras = {camera.image_name: camera for camera in scene.getTrainCameras()}
    assert set(cameras) == set(CAMERAS)
    assert all(camera.image_width == 1600 for camera in cameras.values())
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    background = torch.zeros(3, device="cuda")
    rows = []
    for name in CAMERAS:
        for parameter in PARAMETERS:
            getattr(model, parameter).grad = None
        output = render_fastgs(cameras[name], model, pipe, background, 0.5)
        image = output["render"]
        gt = cameras[name].original_image
        loss = 0.8 * l1_loss(image, gt) + 0.2 * (1 - fused_ssim(image[None], gt[None]))
        loss.backward()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(loss))
        gradients = {parameter: getattr(model, parameter).grad.detach().cpu()
                     for parameter in PARAMETERS}
        screen = output["viewspace_points"].grad
        assert screen is not None
        rows.append({"camera": name, "loss": float(loss.detach()), "image": image.detach().cpu(),
                     "gradients": gradients, "screen_grad": screen.detach().cpu()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
                "ply_sha256": expected, "module_file": _C.__file__,
                "camera_names": CAMERAS, "rows": rows}, args.output)
    print(json.dumps({"module_file": _C.__file__, "losses": [r["loss"] for r in rows],
                      "output": str(args.output), "probe_sha256": sha256(__file__)}, sort_keys=True))


if __name__ == "__main__":
    main()
