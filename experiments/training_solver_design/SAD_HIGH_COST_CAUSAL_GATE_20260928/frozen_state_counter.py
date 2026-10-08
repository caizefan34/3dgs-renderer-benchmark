#!/usr/bin/env python3
"""Count native accepted Gaussian-pixel pairs on the frozen SAD room train views."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render_structgs
from scene import Scene
from utils.general_utils import safe_state


EXPECTED_SOURCE_HEAD = "029814fb5ed773e4a794cec5115951a64f610ab8"
EXPECTED_PLY_SHA256 = "3a088855de73bcd4e123d105812791ab75dd621bd19b0dc6cc89789b4c552894"
EXPECTED_INDICES = [0, 39, 77, 116, 155, 194, 232, 271]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--mult", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    model_path = Path(dataset.model_path).resolve()
    ply_path = model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if args.iteration != 5000 or args.mult != 0.25:
        raise RuntimeError(f"unexpected frozen settings: iteration={args.iteration}, mult={args.mult}")
    if sha256(ply_path) != EXPECTED_PLY_SHA256:
        raise RuntimeError(f"unexpected checkpoint PLY hash: {ply_path}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one CUDA-visible device; count={torch.cuda.device_count()}")

    gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = sorted(scene.getTrainCameras(), key=lambda camera: camera.image_name)
    if len(views) != 272:
        raise RuntimeError(f"expected 272 training cameras; found {len(views)}")
    if EXPECTED_INDICES != [round(i * (len(views) - 1) / 7) for i in range(8)]:
        raise RuntimeError("registered camera indices do not match the 272-view rule")
    selected = [(index, views[index]) for index in EXPECTED_INDICES]
    if {(int(camera.image_width), int(camera.image_height)) for _, camera in selected} != {(1600, 1066)}:
        raise RuntimeError("selected training camera dimensions differ from 1600x1066")

    device = torch.device("cuda:0")
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device=device)
    summed_counts = None
    camera_totals = []
    camera_names = []
    with torch.no_grad():
        for index, camera in selected:
            metric_map = torch.ones((int(camera.image_height), int(camera.image_width)), dtype=torch.int32, device=device)
            package = render_structgs(
                camera, gaussians, pipe, background, args.mult,
                get_flag=True, metric_map=metric_map.reshape(-1).contiguous(), compute_extra=False,
            )
            counts = package["accum_metric_counts"].reshape(-1).to(torch.int64)
            if counts.numel() != gaussians.get_xyz.shape[0]:
                raise RuntimeError(f"count vector length mismatch at training index {index}")
            summed_counts = counts.clone() if summed_counts is None else summed_counts + counts
            camera_totals.append(int(counts.sum().item()))
            camera_names.append(camera.image_name)
            print(f"counted sorted training index {index}: {camera.image_name}; accepted_pairs={camera_totals[-1]}", flush=True)

    counts_cpu = summed_counts.cpu()
    top_ids = torch.argsort(counts_cpu, descending=True, stable=True)[:32].tolist()
    total_pairs = int(counts_cpu.sum().item())
    top_pairs = int(counts_cpu[top_ids].sum().item())
    result = {
        "status": "passed",
        "scope": "frozen state and selection only; no held-out renders or edits; no training",
        "source_head": EXPECTED_SOURCE_HEAD,
        "model_path": str(model_path),
        "iteration": args.iteration,
        "ply_path": str(ply_path),
        "ply_sha256": sha256(ply_path),
        "source_path": dataset.source_path,
        "render_mult": args.mult,
        "resolution": [1600, 1066],
        "camera_order": "training cameras sorted by image_name",
        "selected_sorted_training_indices": EXPECTED_INDICES,
        "selected_camera_names": camera_names,
        "camera_accepted_pairs": camera_totals,
        "gaussian_count": int(counts_cpu.numel()),
        "total_accepted_gaussian_pixel_pairs": total_pairs,
        "top_k": 32,
        "top_32_accepted_pairs": top_pairs,
        "top_32_share": float(top_pairs / total_pairs) if total_pairs else None,
        "top_32": [
            {"rank": rank + 1, "gaussian_id": int(gid), "accepted_pairs": int(counts_cpu[gid].item())}
            for rank, gid in enumerate(top_ids)
        ],
        "counter": "author native render_structgs get_flag=True with all-one int32 metric_map; summed accum_metric_counts",
        "counter_semantics": "accepted Gaussian-pixel pairs passing the native forward compositor contribution checks; not elapsed time or total training cost",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "device_uuid": str(torch.cuda.get_device_properties(device).uuid) if hasattr(torch.cuda.get_device_properties(device), "uuid") else None,
            "visible_device_count": torch.cuda.device_count(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "decision": "continue only if top_32_share >= 0.05; otherwise stop before counterfactual",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(out), "top_32_share": result["top_32_share"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
