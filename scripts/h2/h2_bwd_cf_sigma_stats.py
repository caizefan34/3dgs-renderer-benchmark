#!/usr/bin/env python3
"""Optional H2-BWD-CF SIGMA_GATE counter run; never use for timing."""
import argparse, json, os
from pathlib import Path
import torch
from h2_bwd_cf_correctness import bootstrap, load_fixture


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--source", default="/tmp/higs_h2_bwd_cf/source")
    p.add_argument("--core-so", default="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so")
    p.add_argument("--ply", default="/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply")
    p.add_argument("--cameras", default="/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json")
    p.add_argument("--max-long-side", type=int, default=2048)
    a = p.parse_args()
    backend = bootstrap(a.source, a.core_so)
    values, vm, K, width, height = load_fixture(a.ply, a.cameras, a.max_long_side, "cuda:0")
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER
    os.environ["HIGS_BWD_CF_VARIANT"] = "sigma_gate"
    leaves = tuple(x.detach().clone().requires_grad_(True) for x in values)
    _HIGS_FROZEN_TRACKER.reset(); handle = create_higs_renderer(*leaves, sh_degree=3)
    out = rasterize_gaussian_higs_frozen(*leaves, backward_mode="higs_native", scene=handle,
        freeze_topology=True, viewmats=vm, Ks=K, width=width, height=height, sh_degree=3,
        use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
    (out["frame"].float().sum() + out["alpha"].float().sum()).backward(); torch.cuda.synchronize()
    n, drop, clamp, exp = [int(x) for x in backend.higs_bwd_cf_sigma_stats().cpu().tolist()]
    payload = {"candidate_pixel_gaussian_evaluations": n, "n_samples": n,
               "n_drop_pre_exp": drop, "n_clamp_pre_exp": clamp, "n_exp": exp,
               "drop_fraction": drop / n if n else 0.0, "clamp_fraction": clamp / n if n else 0.0,
               "exp_fraction": exp / n if n else 0.0, "timing_instrumentation": "disabled; counters only"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(payload, indent=2))
    handle.release()


if __name__ == "__main__":
    main()
