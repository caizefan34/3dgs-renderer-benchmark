"""Bound the cost of author FastGS multiview topology scoring at saved states."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from scene import GaussianModel, Scene  # noqa: E402
from utils.fast_utils import compute_gaussian_score_fastgs  # noqa: E402


STATES = (
    ("room_big_capture4100", 4100, "64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc"),
    ("room_big_branch_base30k", 10000, "4927b6a3a2e8fcf8919ffba5f228ac810df1e16f890303b6620314dd8d3ac0ab"),
    ("room_big_branch_base30k", 30000, "c2afc615fd244d561bd790e7d7721af1b7a7515222bd5385650e7b886c58516a"),
)
GARDEN_STATE = ("garden_exchange_dev_4100", 4100,
                "d396092ca139fa0e6aea4d17a085ca0239bf2ff0ec2bbbe0ee3c5ab539feb5fb")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--garden", action="store_true")
    args = parser.parse_args()
    states = (GARDEN_STATE,) if args.garden else STATES
    rows = []
    for model_name, iteration, expected_hash in states:
        model_path = REPO / "output" / model_name
        ply = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if sha256(ply) != expected_hash:
            raise ValueError(f"PLY changed: {ply}")
        dataset = SimpleNamespace(
            source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/" +
                        ("garden" if args.garden else "room"),
            model_path=str(model_path), images="images", resolution=-1,
            white_background=False, data_device="cuda", eval=True)
        model = GaussianModel(3, "default")
        scene = Scene(dataset, model, load_iteration=iteration, shuffle=False)
        cameras = scene.getTrainCameras()
        if len(cameras) < 10 or cameras[0].image_width != 1600:
            raise ValueError("Camera protocol mismatch")
        chosen = [cameras[int(i)] for i in np.random.default_rng(20260927).choice(len(cameras), 10, replace=False)]
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        options = SimpleNamespace(mult=.5, loss_thresh=.1)
        background = torch.zeros(3, device="cuda")
        samples = []
        with torch.no_grad():
            for repeat in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                importance, pruning = compute_gaussian_score_fastgs(
                    chosen, model, pipe, background, options, DENSIFY=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if repeat:
                    samples.append(elapsed)
                if len(importance) != len(model.get_xyz) or len(pruning) != len(model.get_xyz):
                    raise ValueError("Score length mismatch")
        rows.append({"model_name": model_name, "iteration": iteration,
                     "model_ply_sha256": expected_hash, "gaussians": len(model.get_xyz),
                     "camera_names": [camera.image_name for camera in chosen],
                     "seconds_per_10_camera_score": samples,
                     "median_seconds_per_score": float(np.median(samples))})
        print("SCORE_COST " + json.dumps(rows[-1], sort_keys=True), flush=True)
        del scene, model, cameras, chosen, importance, pruning
        torch.cuda.empty_cache()
    result = {"source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
              "scene": "garden" if args.garden else "room", "rows": rows,
              "scope": "Frozen saved state(s), four timed repeats after one warmup, full 10-camera author scorer under no_grad. Excludes model loading, optimizer/step overlap, and complete training timing."}
    output = "/root/fastgs_score_cost_garden_result.json" if args.garden else "/root/fastgs_score_cost_state_result.json"
    Path(output).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
