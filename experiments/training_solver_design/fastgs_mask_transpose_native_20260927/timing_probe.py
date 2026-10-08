"""Time frozen FastGS full-image forward/loss/backward blocks without model updates."""

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch


REPO = Path("/root/codex_fastgs_audit_20260927")
EXTENSION_ROOT = os.environ.get("FASTGS_EXTENSION_ROOT")
if EXTENSION_ROOT:
    sys.path.insert(0, EXTENSION_ROOT)
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from diff_gaussian_rasterization_fastgs import _C  # noqa: E402
from fused_ssim import fused_ssim  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from scene import GaussianModel, Scene, dataset_readers  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402


CAMERAS = ("DSCF4668", "DSCF4770", "DSCF4873", "DSCF4976")
PARAMETERS = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
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

    def block(steps):
        torch.cuda.synchronize()
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter()
        event_start.record()
        for step in range(steps):
            for parameter in PARAMETERS:
                getattr(model, parameter).grad = None
            camera = cameras[CAMERAS[step % len(CAMERAS)]]
            image = render_fastgs(camera, model, pipe, background, 0.5)["render"]
            gt = camera.original_image
            loss = 0.8 * l1_loss(image, gt) + 0.2 * (1 - fused_ssim(image[None], gt[None]))
            loss.backward()
        event_end.record()
        torch.cuda.synchronize()
        return {"wall_seconds": time.perf_counter() - host_start,
                "cuda_seconds": event_start.elapsed_time(event_end) / 1000,
                "last_loss": float(loss.detach())}

    block(16)
    results = [block(128) for _ in range(5)]
    print(json.dumps({"module_file": _C.__file__, "ply_sha256": expected,
                      "probe_sha256": sha256(__file__), "camera_names": CAMERAS,
                      "warmup_steps": 16, "measured_steps_per_block": 128,
                      "blocks": results}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
