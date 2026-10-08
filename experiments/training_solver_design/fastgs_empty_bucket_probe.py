"""Measure zero-contribution pixel-bucket work on pinned full-resolution FastGS."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.cpp_extension import load


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/root")

from fastgs_real_split_shadow_probe import render_and_capture  # noqa: E402
from fastgs_real_tile_reference import sha256  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402


def main():
    source = REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    assert sha256(source) == "64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc"
    ext = load(name="fastgs_empty_bucket_diagnostic_v2",
               sources=["/root/fastgs_empty_bucket_cuda.cu"],
               extra_cuda_cflags=["-O3"], verbose=True)
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output/room_big_capture4100"),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=True)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False,
                           convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=4100, shuffle=False)
    cameras = scene.getTrainCameras()
    assert len(cameras) == 272
    expected_views = {row["camera"]: row["work"]["native_before"]["nonzero_alpha_accepted"]
                      for row in json.loads(Path("/root/fastgs_shadow_work_result.json").read_text())["views"]}
    records = []
    for camera_index in (0, 90, 180, 270):
        camera = cameras[camera_index]
        width, height = camera.image_width, camera.image_height
        _, rendered, geom, img, ids = render_and_capture(camera, model, pipe)
        tiles = ((width + 15) // 16) * ((height + 15) // 16)
        ranges = img["ranges"][:tiles].contiguous()
        lengths = (ranges[:, 1] - ranges[:, 0]).cpu().tolist()
        bucket_tile = []
        bucket_index = []
        for tile, length in enumerate(lengths):
            for bucket in range((length + 31) // 32):
                bucket_tile.append(tile)
                bucket_index.append(bucket)
        tile_gpu = torch.tensor(bucket_tile, device="cuda", dtype=torch.int32)
        index_gpu = torch.tensor(bucket_index, device="cuda", dtype=torch.int32)
        output = ext.count_empty(geom["mean2d"].contiguous(),
                                geom["conic"].contiguous(), ranges,
                                ids.contiguous(), img["n_contrib"].contiguous(),
                                tile_gpu, index_gpu, width, height)
        flags = output[:, :, 0]
        counts = output[:, :, 1]
        eligible = flags >= 1
        active = flags == 2
        eligible_per_bucket = eligible.sum(dim=1)
        active_per_bucket = active.sum(dim=1)
        executed = eligible_per_bucket > 0
        record = {
            "camera": camera.image_name,
            "resolution": [width, height],
            "gaussians": int(model.get_xyz.shape[0]),
            "tile_instances": rendered,
            "total_buckets": len(bucket_tile),
            "executed_buckets": int(executed.sum().item()),
            "eligible_pixel_bucket_pairs": int(eligible.sum().item()),
            "active_pixel_bucket_pairs": int(active.sum().item()),
            "active_gaussian_pixel_pairs": int(counts.sum().item()),
            "independent_reference_accepted": expected_views[camera.image_name],
            "empty_eligible_fraction": float((eligible & ~active).sum().item() / max(1, eligible.sum().item())),
            "executed_buckets_with_no_active_pixel": int((executed & (active_per_bucket == 0)).sum().item()),
            "active_pixels_per_executed_bucket_p50": float(active_per_bucket[executed].float().median().item()),
            "active_pixels_per_executed_bucket_p90": float(torch.quantile(active_per_bucket[executed].float(), .9).item()),
        }
        if record["active_gaussian_pixel_pairs"] != record["independent_reference_accepted"]:
            raise AssertionError(record)
        records.append(record)
        print("FASTGS_EMPTY_BUCKET_VIEW " + json.dumps(record, sort_keys=True), flush=True)
    payload = {"source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
               "ply_sha256": sha256(source), "views": records,
               "scope": "one room@4100 state, four 1600x1066 training views, author raster buffers; no GPU speed, VJP, or quality result"}
    Path("/root/fastgs_empty_bucket_result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
