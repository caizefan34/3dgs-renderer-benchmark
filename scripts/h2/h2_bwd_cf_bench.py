#!/usr/bin/env python3
"""CUDA-event H2-BWD-CF backward benchmark. Run this in a fresh process."""
import argparse, csv, os
from pathlib import Path
import numpy as np
import torch
from h2_bwd_cf_correctness import VARIANTS, bootstrap, load_fixture


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--source", default="/tmp/higs_h2_bwd_cf/source")
    p.add_argument("--core-so", default="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so")
    p.add_argument("--ply", default="/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply")
    p.add_argument("--cameras", default="/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json")
    p.add_argument("--max-long-side", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--measure", type=int, default=100)
    a = p.parse_args()
    backend = bootstrap(a.source, a.core_so)  # intentionally fresh experimental .so only
    values, vm, K, width, height = load_fixture(a.ply, a.cameras, a.max_long_side, "cuda:0")
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER

    def prepare(variant):
        os.environ["HIGS_BWD_CF_VARIANT"] = variant
        os.environ.pop("HIGS_BWD_CF_CAPTURE_RAW", None)
        leaves = tuple(x.detach().clone().requires_grad_(True) for x in values)
        _HIGS_FROZEN_TRACKER.reset(); handle = create_higs_renderer(*leaves, sh_degree=3)
        out = rasterize_gaussian_higs_frozen(*leaves, backward_mode="higs_native", scene=handle,
            freeze_topology=True, viewmats=vm, Ks=K, width=width, height=height, sh_degree=3,
            use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
        # Fixed nonzero VJP avoids a loss-dependent workload change.
        loss = out["frame"].float().sum() + out["alpha"].float().sum()
        return loss, handle

    rows = []
    for variant in VARIANTS:
        for _ in range(a.warmup):
            loss, handle = prepare(variant); loss.backward(); handle.release()
        torch.cuda.synchronize()
        samples = []
        for i in range(a.measure):
            loss, handle = prepare(variant)
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record(); loss.backward(); end.record(); torch.cuda.synchronize()
            ms = float(start.elapsed_time(end)); samples.append(ms); handle.release()
            rows.append({"variant": variant, "iteration": i, "ms": ms, "warmup": a.warmup,
                         "measure": a.measure, "timing": "CUDA Event backward only"})
        q = np.percentile(samples, [10, 90])
        rows.append({"variant": variant, "iteration": "summary", "median_ms": float(np.median(samples)),
                     "mean_ms": float(np.mean(samples)), "p10": float(q[0]), "p90": float(q[1]),
                     "std": float(np.std(samples)), "warmup": a.warmup, "measure": a.measure,
                     "timing": "CUDA Event backward only"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(x.keys() for x in rows)))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
