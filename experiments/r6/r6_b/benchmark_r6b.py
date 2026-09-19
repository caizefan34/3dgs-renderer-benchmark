#!/usr/bin/env python3
"""CUDA-event replay benchmark for R6-B modes on an existing R6 checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "experiments" / "r6"))
from runtime import configure, last_prepare_ms
from r6_1_bwd_decompose import (  # noqa: E402
    ReferenceV1Config, GTDataset, SepSSIM, load_model_from_ckpt, render_with_meta,
)


def summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()), "p50": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "n": int(a.size)}


def replay(model, cam, target, ssim, mode: str, warmup: int, measure: int) -> dict:
    configure(mode)
    bwd_start, bwd_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    iter_start, iter_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    bwd, iteration, prepare = [], [], []
    for step in range(warmup + measure):
        model.optimizer.zero_grad(set_to_none=True)
        iter_start.record()
        image, _, _ = render_with_meta(model, cam, model.active_sh_degree)
        loss = 0.8 * F.l1_loss(image, target) + 0.2 * ssim(image, target)
        bwd_start.record()
        loss.backward()
        bwd_end.record()
        model.optimizer.step()
        iter_end.record()
        torch.cuda.synchronize()
        if step >= warmup:
            bwd.append(bwd_start.elapsed_time(bwd_end))
            iteration.append(iter_start.elapsed_time(iter_end))
            if mode != "baseline":
                prepare.append(last_prepare_ms())
    result = {"T_bwd_ms": summary(bwd), "T_iter_ms": summary(iteration)}
    if prepare:
        result["T_raster_prepare_cuda_event_ms"] = summary(prepare)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("baseline", "b0", "b1"), required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--measure", type=int, default=100)
    p.add_argument("--camera-idx", type=int, default=0)
    args = p.parse_args()
    config = ReferenceV1Config()
    config.scene, config.repo_root = args.scene, str(REPO)
    dataset = GTDataset(scene=args.scene, repo_root=str(REPO), resolution=config.resolution,
                        device="cuda", background="black")
    model, _ = load_model_from_ckpt(args.ckpt, config, str(REPO))
    cam, target = dataset.get_item(args.camera_idx)
    result = replay(model, cam, target, SepSSIM(device="cuda"), args.mode, args.warmup, args.measure)
    result.update({"scene": args.scene, "checkpoint": args.ckpt, "mode": args.mode,
                   "warmup": args.warmup, "measure": args.measure,
                   "timing_source": "torch.cuda.Event"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
