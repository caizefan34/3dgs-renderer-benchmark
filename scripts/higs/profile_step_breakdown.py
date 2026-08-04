#!/usr/bin/env python
"""Round-59 step-breakdown profiler (2026-08-04).

Runs N training steps of a given backend/recipe under torch.profiler and
prints the kernel-level CUDA-time breakdown (top kernels by self time) plus
the run's per-step totals. Purpose: after R57/R58 closed the sampling /
resolution / renderer-pixel levers, identify the dominant per-step kernel so
the next acceleration lever is evidence-based.

Usage (on EPIC-05):
  python scripts/higs/profile_step_breakdown.py \
    --base-dir /root/epic05-data/processed --scene mipnerf360/garden \
    --width 1280 --height 720 --ratio 0.35 --steps 40
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import importlib.util
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BENCH = os.path.join(_REPO, "benchmark", "run_higs_train_benchmark.py")
_spec = importlib.util.spec_from_file_location("run_higs_train_benchmark", _BENCH)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


class _FakeLPIPS(torch.nn.Module):
    def forward(self, x, y):
        return (x - y).pow(2).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--ratio", type=float, default=0.35)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    scene_dir = os.path.join(args.base_dir, args.scene)
    params0 = B.load_ply_scene(os.path.join(scene_dir, "point_cloud.ply"), device)
    print(f"scene={args.scene} n_gaussians={params0[0].shape[0]}")
    viewmats, Ks, train_idx, eval_idx = B.load_cameras(
        scene_dir, args.width, args.height, 4, 1, device,
    )
    with open(os.path.join(scene_dir, "eval_cameras.json")) as f:
        cams = json.load(f)
    refs_train = B.load_reference(
        os.path.join(scene_dir, "eval_images"),
        [cams[i] for i in train_idx], args.width, args.height, device,
    )
    refs_eval = B.load_reference(
        os.path.join(scene_dir, "eval_images"),
        [cams[i] for i in eval_idx], args.width, args.height, device,
    )
    lpips_model = _FakeLPIPS().to(device).eval()

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    )
    prof.start()
    t0 = time.perf_counter()
    r = B.run_backend(
        "higs_dynamic_ts", params0, viewmats, Ks, train_idx, refs_train,
        eval_idx, refs_eval, args.width, args.height, args.steps,
        seed=0, device=device, densify_every=(1 << 30),
        densify_threshold=5e-3, prune_threshold=0.01,
        lpips_model=lpips_model, radius_clip=0.0, fused_adam=True,
        tile_sampling_ratio=args.ratio, anchor_densify=False,
        anchor_densify_every=2, sampling_mode="uniform",
        error_alpha=1.0, error_refresh_every=25, error_lambda=0.7,
        eval_every=0, lr_decay=1.0, densify_window=None,
        lpips_loss_weight=0.0, lpips_loss_every=0, lpips_full_res=False,
        cull_interval=1, densify_grad_accum=False,
        res_schedule=None, res_schedule_full_signal=False,
        res_schedule_full_lpips=False,
        pixel_sampling_ratio=1.0, pixel_raster_ratio=1.0,
    )
    prof.stop()
    wall = time.perf_counter() - t0

    print("\n=== run totals ===")
    print(f"total_ms={r['total_ms']:.2f} fwd_ms={r['fwd_ms']:.2f} bwd_ms={r['bwd_ms']:.2f} "
          f"psnr={r['psnr']:.2f} sr={r['sampled_tile_ratio']:.3f} wall={wall:.1f}s")

    print("\n=== top CUDA kernels by self time (ms avg per event x count) ===")
    rows = []
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA or evt.self_device_time_total > 0:
            rows.append((evt.self_device_time_total, evt.count, evt.key))
    rows.sort(reverse=True)
    total_self = sum(x[0] for x in rows)
    for self_us, cnt, key in rows[:28]:
        pct = 100.0 * self_us / total_self if total_self else 0.0
        print(f"{key[:90]:90s} self={self_us/1e3:8.2f}ms count={cnt:6d} avg={self_us/cnt/1e3:7.3f}ms {pct:5.1f}%")
    print(f"TOTAL self cuda time: {total_self/1e3:.1f}ms over {args.steps} steps "
          f"= {total_self/1e3/args.steps:.2f}ms/step")


if __name__ == "__main__":
    main()