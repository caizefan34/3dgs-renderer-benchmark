#!/usr/bin/env python3
"""CUDA-event replay benchmark for R6-B modes on an existing R6 checkpoint.

Reports per-mode:
  - T_bwd_ms:      backward wall time (CUDA events)
  - T_iter_ms:     full iteration (zero_grad + forward + loss + backward + optimizer)
  - T_prepare_ms:  rasterizer buffer preparation (B0: full clear; B1-v2: clear phase)
  - T_scatter_ms:  B1-v2 only: touched-mask scatter phase (in finish)
  - T_clear_ms:    B0/B1-v2: the clear kernel itself (subset of T_prepare)
  - metadata_bytes: B1-v2 only: persistent [C*N] bool touched mask
  - n_rows:        B1-v2 only: C*N row count
  - memory_delta_mb: peak VRAM delta relative to start-of-run baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "experiments" / "r6"))
from runtime import (configure, last_prepare_ms, last_scatter_ms,  # noqa: E402
                     last_clear_ms, metadata_bytes, prev_n_rows)
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
    bwd, iteration, prepare, scatter, clear = [], [], [], [], []
    meta_bytes, n_rows = 0, 0

    # Record baseline VRAM before measure phase
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

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
                clear.append(last_clear_ms())
                if mode == "b1":
                    scatter.append(last_scatter_ms())
                    meta_bytes = metadata_bytes()
                    n_rows = prev_n_rows()

    mem_after = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()

    result = {
        "T_bwd_ms": summary(bwd),
        "T_iter_ms": summary(iteration),
    }
    if prepare:
        result["T_prepare_ms"] = summary(prepare)
        result["T_clear_ms"] = summary(clear)
    if scatter:
        result["T_scatter_ms"] = summary(scatter)
        result["metadata_bytes"] = meta_bytes
        result["n_rows"] = n_rows
        result["metadata_MB"] = round(meta_bytes / 1048576, 3)
    result["memory_allocated_MB"] = round(mem_after / 1048576, 1)
    result["peak_memory_MB"] = round(peak_mem / 1048576, 1)
    result["memory_delta_vs_start_MB"] = round((mem_after - mem_before) / 1048576, 1)
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
                   "camera_idx": args.camera_idx,
                   "timing_source": "torch.cuda.Event"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
