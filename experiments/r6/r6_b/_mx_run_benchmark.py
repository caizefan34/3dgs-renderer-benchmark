#!/usr/bin/env python3
"""A100 benchmark runner for R6-B: runs baseline, B0, B1-v2 for a given scene/checkpoint.

Usage:
  python _mx_run_benchmark.py --scene room --ckpt-iter 15000 --output-dir /tmp/r6b_bench
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
sys.path.insert(0, str(REPO / "experiments" / "r6" / "r6_b"))
sys.path.insert(0, str(REPO / "experiments" / "r6"))
sys.path.insert(0, str(REPO / "baseline" / "reference_v1"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "epic05" / "phase7"))

from runtime import (configure, last_prepare_ms, last_scatter_ms,
                     last_clear_ms, metadata_bytes, prev_n_rows)
from r6_1_bwd_decompose import (ReferenceV1Config, GTDataset, SepSSIM,
                                 load_model_from_ckpt, render_with_meta)

CKPT_BASE = "/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts"


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "p50": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "n": int(a.size)}


def run_mode(model, cam, target, ssim, mode, warmup, measure):
    configure(mode)
    bwd_s, bwd_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    iter_s, iter_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    bwd, iteration, prepare, scatter, clear = [], [], [], [], []
    meta_b, n_rows = 0, 0

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    for step in range(warmup + measure):
        model.optimizer.zero_grad(set_to_none=True)
        iter_s.record()
        image, _, _ = render_with_meta(model, cam, model.active_sh_degree)
        loss = 0.8 * F.l1_loss(image, target) + 0.2 * ssim(image, target)
        bwd_s.record()
        loss.backward()
        bwd_e.record()
        model.optimizer.step()
        iter_e.record()
        torch.cuda.synchronize()
        if step >= warmup:
            bwd.append(bwd_s.elapsed_time(bwd_e))
            iteration.append(iter_s.elapsed_time(iter_e))
            if mode != "baseline":
                prepare.append(last_prepare_ms())
                clear.append(last_clear_ms())
                if mode == "b1":
                    scatter.append(last_scatter_ms())
                    meta_b = metadata_bytes()
                    n_rows = prev_n_rows()

    mem_after = torch.cuda.memory_allocated()
    result = {"T_bwd_ms": summary(bwd), "T_iter_ms": summary(iteration)}
    if prepare:
        result["T_prepare_ms"] = summary(prepare)
        result["T_clear_ms"] = summary(clear)
    if scatter:
        result["T_scatter_ms"] = summary(scatter)
        result["metadata_bytes"] = meta_b
        result["n_rows"] = n_rows
        result["metadata_MB"] = round(meta_b / 1048576, 3)
    result["peak_memory_MB"] = round(torch.cuda.max_memory_allocated() / 1048576, 1)
    result["memory_delta_MB"] = round((mem_after - mem_before) / 1048576, 1)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--ckpt-iter", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--measure", type=int, default=100)
    p.add_argument("--camera-idx", type=int, default=0)
    args = p.parse_args()

    ckpt = f"{CKPT_BASE}/{args.scene}/checkpoints/iter_{args.ckpt_iter}.pt"
    config = ReferenceV1Config()
    config.scene = args.scene
    config.repo_root = str(REPO)
    dataset = GTDataset(scene=args.scene, repo_root=str(REPO),
                        resolution=config.resolution, device="cuda",
                        background="black")
    cam, target = dataset.get_item(args.camera_idx)
    ssim = SepSSIM(device="cuda")

    results = {}
    for mode in ("baseline", "b0", "b1"):
        print(f"\n=== {args.scene} iter_{args.ckpt_iter} mode={mode} ===", flush=True)
        model, n_pts = load_model_from_ckpt(ckpt, config, str(REPO))
        print(f"  N={n_pts}, warmup={args.warmup}, measure={args.measure}", flush=True)
        t0 = time.time()
        r = run_mode(model, cam, target, ssim, mode, args.warmup, args.measure)
        t1 = time.time()
        r["scene"] = args.scene
        r["ckpt_iter"] = args.ckpt_iter
        r["mode"] = mode
        r["n_gaussians"] = n_pts
        r["warmup"] = args.warmup
        r["measure"] = args.measure
        r["wall_time_s"] = round(t1 - t0, 1)
        results[mode] = r
        print(f"  T_bwd mean={r['T_bwd_ms']['mean']:.2f}ms  T_iter mean={r['T_iter_ms']['mean']:.2f}ms", flush=True)
        del model
        torch.cuda.empty_cache()

    # Compute speedups
    base_bwd = results["baseline"]["T_bwd_ms"]["mean"]
    base_iter = results["baseline"]["T_iter_ms"]["mean"]
    for mode in ("b0", "b1"):
        r = results[mode]
        r["speedup_bwd_pct"] = round((base_bwd / r["T_bwd_ms"]["mean"] - 1) * 100, 2)
        r["speedup_iter_pct"] = round((base_iter / r["T_iter_ms"]["mean"] - 1) * 100, 2)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{args.scene}_{args.ckpt_iter}.json"
    outfile.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {outfile}")
    print(json.dumps({m: {"T_bwd": results[m]["T_bwd_ms"]["mean"],
                          "T_iter": results[m]["T_iter_ms"]["mean"],
                          "speedup_iter_pct": results[m].get("speedup_iter_pct", 0)}
                      for m in ("baseline", "b0", "b1")}, indent=2))


if __name__ == "__main__":
    main()
