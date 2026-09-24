#!/usr/bin/env python3
"""Matched B1A/B2 room/train/bicycle reproduction harness for H1-B2-FREEZE."""
import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import runpy
import sys

import numpy as np
import torch
import torch.nn.functional as F


SCENES = {
    "train": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/cameras.json",
        "resolution": (1959, 1090),
    },
    "room": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
        "resolution": (2048, 1365),
    },
    "bicycle": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",
        "resolution": (2048, 1361),
    },
}
SOURCE = "/tmp/h1_b2_authoritative_source"
CORE_SO = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
WARMUP = 20
MEASURE = 100
NZ_EPS = 1e-10


def bootstrap():
    sys.path.insert(0, SOURCE)
    spec = importlib.util.spec_from_file_location("gsplat_cuda", CORE_SO)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.modules["gsplat.csrc"] = core
    build_path = os.path.join(SOURCE, "gsplat/experimental/render/kernels/cuda/build.py")
    exp_build = runpy.run_path(build_path)
    backend = exp_build["build_and_load_experimental_gaussian_render_inference_scene"]()
    return core, backend


def load_profile_helpers():
    path = "/tmp/h1_profile.py"
    spec = importlib.util.spec_from_file_location("h1_profile_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def b1_forward(args, vm, K, width, height):
    from gsplat.rendering import rasterization
    means, quats, scales, opacities, sh = args
    return rasterization(
        means=means.unsqueeze(0), quats=quats.unsqueeze(0), scales=scales.unsqueeze(0),
        opacities=opacities.unsqueeze(0), colors=sh, viewmats=vm, Ks=K,
        width=width, height=height, sh_degree=3, packed=False,
        radius_clip=0.0, eps2d=0.3,
    )


def b2_forward(args, vm, K, width, height, handle):
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    means, quats, scales, opacities, sh = args
    return rasterize_gaussian_higs_frozen(
        means, quats, scales, opacities, sh,
        backward_mode="higs_native", scene=handle, freeze_topology=True,
        viewmats=vm, Ks=K, width=width, height=height, sh_degree=3,
        use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0,
    )


def make_leaves(values):
    return tuple(x.detach().clone().requires_grad_(True) for x in values)


def loss_with_vjp(render, alpha, v_render, v_alpha):
    return (render.float() * v_render).sum() + (alpha.float() * v_alpha).sum()


def summarize_tensor(a, b):
    af, bf = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    diff = af - bf
    mse = diff.square().mean().item()
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "relative_l2": (diff.norm() / bf.norm().clamp_min(1e-12)).item(),
        "psnr_db": 60.0 if mse < 1e-12 else -10.0 * math.log10(mse),
    }


def grad_metrics(a, b):
    af, bf = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    diff = af - bf
    na, nb = af.abs() > NZ_EPS, bf.abs() > NZ_EPS
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "relative_l2": (diff.norm() / af.norm().clamp_min(1e-12)).item(),
        "cosine": F.cosine_similarity(af[None, :], bf[None, :]).item(),
        "zero_nonzero_disagreement": int((na != nb).sum().item()),
        "nonzero_a": int(na.sum().item()),
        "nonzero_b": int(nb.sum().item()),
        "nan": int((af.isnan() | bf.isnan()).sum().item()),
        "inf": int((af.isinf() | bf.isinf()).sum().item()),
    }


def time_events(fn, method, phase, scene, camera, raw):
    for _ in range(WARMUP):
        result = fn()
        del result
    torch.cuda.synchronize()
    times = []
    for i in range(MEASURE):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        value = float(start.elapsed_time(end))
        times.append(value)
        raw.append({"scene": scene, "camera": camera, "method": method,
                    "phase": phase, "iteration": i, "ms": value})
        del result
    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "n_warmup": WARMUP,
        "n_measure": MEASURE,
    }


def run_one(helper, scene, camera, values, viewmats, Ks, width, height, raw,
            do_timing=True):
    vm, K = viewmats[:, [camera]], Ks[:, [camera]]
    base_b1 = b1_forward(values, vm, K, width, height)
    r1, a1, info = base_b1[0], base_b1[1], base_b1[2]
    b1_render = r1.detach().clone()
    b1_alpha = a1.detach().clone()
    radii = info.get("radii")
    if radii is None:
        raise RuntimeError("B1A runtime did not expose radii in rasterization info")
    n_visible = int((radii > 0).any(dim=-1).sum().item())
    n_isects = int(info.get("n_isects", -1))
    if n_isects < 0:
        ids = info.get("flatten_ids")
        if ids is not None:
            n_isects = int(ids.numel())
    offsets = info.get("isect_offsets")
    active_tiles = -1
    mean_ipt = p95_ipt = -1.0
    if offsets is not None:
        counts = offsets.reshape(-1).to(torch.int64)
        counts = torch.diff(torch.cat((counts, counts.new_tensor([n_isects]))))
        counts = counts[counts >= 0]
        active_tiles = int((counts > 0).sum().item())
        if counts.numel():
            mean_ipt = counts.float().mean().item()
            p95_ipt = torch.quantile(counts.float(), 0.95).item()

    from gsplat.experimental.render.functional.gaussian_inference import (
        create_higs_renderer, _HIGS_FROZEN_TRACKER, _HigsAutogradFunction,
    )
    _HIGS_FROZEN_TRACKER.reset()
    handle = create_higs_renderer(*values, sh_degree=3)
    with torch.no_grad():
        base_b2 = b2_forward(values, vm, K, width, height, handle)
    b2_render = base_b2["frame"].detach().clone()
    b2_alpha = base_b2["alpha"].detach().clone()
    b2_meta = dict(_HigsAutogradFunction.last_forward_metadata)
    visible_ids = b2_meta.get("visible_gaussian_ids")
    n_visible_b2 = int(visible_ids.numel()) if isinstance(visible_ids, torch.Tensor) else int(b2_meta.get("n_visible", -1))

    # Non-zero fixed random VJP; identical upstream tensors for the two paths.
    gen = torch.Generator(device="cuda").manual_seed(4200 + camera)
    v_render = torch.randn(b1_render.shape, device="cuda", generator=gen)
    v_alpha = torch.randn(b1_alpha.shape, device="cuda", generator=gen)
    leaves1, leaves2 = make_leaves(values), make_leaves(values)
    out1 = b1_forward(leaves1, vm, K, width, height)
    loss1 = loss_with_vjp(out1[0], out1[1], v_render, v_alpha)
    loss1.backward()
    grad1 = [x.grad.detach().clone() for x in leaves1]
    leaves2 = make_leaves(values)
    _HIGS_FROZEN_TRACKER.reset()
    handle_grad = create_higs_renderer(*leaves2, sh_degree=3)
    out2 = b2_forward(leaves2, vm, K, width, height, handle_grad)
    loss2 = loss_with_vjp(out2["frame"], out2["alpha"], v_render, v_alpha)
    loss2.backward()
    grad2 = [x.grad.detach().clone() for x in leaves2]
    param_names = ["means", "quats", "scales", "opacities", "sh"]
    gradients = {name: grad_metrics(x, y) for name, x, y in zip(param_names, grad1, grad2)}
    grad_rows = [{"scene": scene, "camera": camera, "parameter": n, **m}
                 for n, m in gradients.items()]

    params_forward = values
    b1_forward_fn = lambda: tuple(t.detach() for t in b1_forward(params_forward, vm, K, width, height)[:2])
    # The B2 public result is a mapping; time frame/alpha only, not metadata copies.
    def b2_fwd_only():
        with torch.no_grad():
            out = b2_forward(values, vm, K, width, height, handle)
            return out["frame"], out["alpha"]
    def b1_fb():
        for t in leaves1:
            t.grad = None
        out = b1_forward(leaves1, vm, K, width, height)
        loss = loss_with_vjp(out[0], out[1], v_render, v_alpha)
        loss.backward()
        return out[0], out[1]
    def b2_fb():
        for t in leaves2:
            t.grad = None
        out = b2_forward(leaves2, vm, K, width, height, handle_grad)
        loss = loss_with_vjp(out["frame"], out["alpha"], v_render, v_alpha)
        loss.backward()
        return out["frame"], out["alpha"]

    timings = {}
    if do_timing:
        timings["B1A_forward"] = time_events(b1_forward_fn, "B1A", "forward", scene, camera, raw)
        timings["B2_forward"] = time_events(b2_fwd_only, "B2", "forward", scene, camera, raw)
        timings["B1A_fb"] = time_events(b1_fb, "B1A", "fb", scene, camera, raw)
        timings["B2_fb"] = time_events(b2_fb, "B2", "fb", scene, camera, raw)
        for method in ("B1A", "B2"):
            f = timings[method + "_forward"]["median_ms"]
            fb = timings[method + "_fb"]["median_ms"]
            timings[method + "_backward"] = {"median_ms": fb - f}

    visible_from_grad = int(torch.stack([g.reshape(g.shape[0], -1).abs().amax(dim=-1) > NZ_EPS for g in grad2[:4]]).any(dim=0).sum().item())
    health = {
        "scene": scene, "camera": camera, "N_total": int(values[0].shape[0]),
        "N_visible": n_visible, "N_nonzero_grad": visible_from_grad,
        "nonzero_grad_over_visible": visible_from_grad / max(n_visible, 1),
        "max_radius": float(radii.max().item()),
        "p99_radius": float(torch.quantile(radii.float().reshape(-1), 0.99).item()),
        "max_opacity": float(values[3].max().item()),
        "alpha_saturation_fraction": float((b1_alpha > 0.99).float().mean().item()),
        "classification": "BACKWARD_WORKLOAD_PATHOLOGICAL" if visible_from_grad / max(n_visible, 1) < 0.01 else "OK",
    }
    workload = {
        "scene": scene, "camera": camera, "N_total": int(values[0].shape[0]),
        "N_visible_B1A": n_visible, "N_visible_B2": n_visible_b2,
        "N_intersections_B1A": n_isects, "active_tiles_B1A": active_tiles,
        "mean_intersections_per_tile_B1A": mean_ipt,
        "p95_intersections_per_tile_B1A": p95_ipt,
        "tiles_per_gaussian_B1A": n_isects / max(n_visible, 1),
        "B2_metadata": {k: (v.detach().cpu().tolist() if isinstance(v, torch.Tensor) and v.numel() < 128 else str(v)) for k, v in b2_meta.items()},
    }
    correctness = {
        "scene": scene, "camera": camera,
        "rgb": summarize_tensor(b1_render, b2_render),
        "alpha": summarize_tensor(b1_alpha, b2_alpha),
        "gradients": gradients,
    }
    grad_rows.extend({"scene": scene, "camera": camera, "parameter": name,
                      "kind": "forward", **metrics}
                     for name, metrics in (("rgb", correctness["rgb"]),
                                           ("alpha", correctness["alpha"])))
    handle.release()
    handle_grad.release()
    del base_b1, base_b2, out1, out2, leaves1, leaves2
    gc.collect()
    torch.cuda.empty_cache()
    return timings, correctness, grad_rows, health, workload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenes", default="train,room,bicycle")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    core, backend = bootstrap()
    helpers = load_profile_helpers()
    from gsplat.experimental.render.functional.gaussian_inference import _HIGS_FROZEN_TRACKER

    raw, correctness_rows, gradient_rows, health_rows, workload_rows, run_rows = [], [], [], [], [], []
    summary = {}
    scenes = ["room"] if args.smoke else args.scenes.split(",")
    for scene in scenes:
        cfg = SCENES[scene]
        width, height = cfg["resolution"]
        device = torch.device("cuda:0")
        values = helpers.load_ply_scene(cfg["ply"], device)
        vm_all, Ks_all, cams = helpers.load_cameras(cfg["cams"], width, height, device)
        cam_indices = [0] if args.smoke else [0, len(cams) // 2, len(cams) - 1]
        for camera in cam_indices:
            _HIGS_FROZEN_TRACKER.reset()
            timings, correctness, grads, health, workload = run_one(
                helpers, scene, camera, values, vm_all, Ks_all, width, height, raw,
                do_timing=not args.smoke,
            )
            key = f"{scene}:cam{camera}"
            summary[key] = timings
            correctness_rows.append(correctness)
            gradient_rows.extend(grads)
            health_rows.append(health)
            workload_rows.append(workload)
            for method in ("B1A", "B2"):
                run_rows.append({"scene": scene, "camera_index": camera,
                                 "camera_id": cams[camera]["id"], "method": method,
                                 "width": width, "height": height,
                                 "warmup": WARMUP, "measure": MEASURE,
                                 "ply": cfg["ply"], "cameras": cfg["cams"]})
            print(json.dumps({"scene": scene, "camera": camera, "timings": timings,
                              "correctness": correctness, "health": health,
                              "workload": workload}, allow_nan=False), flush=True)
    write_csv(os.path.join(args.out_dir, "timings_raw.csv"), raw)
    write_csv(os.path.join(args.out_dir, "run_manifest.csv"), run_rows)
    write_csv(os.path.join(args.out_dir, "checkpoint_health.csv"), health_rows)
    write_csv(os.path.join(args.out_dir, "workload_metrics.csv"), workload_rows)
    write_csv(os.path.join(args.out_dir, "correctness.csv"), gradient_rows)
    with open(os.path.join(args.out_dir, "correctness.json"), "w") as f:
        json.dump(correctness_rows, f, indent=2, allow_nan=False)
    if args.smoke:
        return
    with open(os.path.join(args.out_dir, "timings_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    timing_summary_rows = []
    for key, measurements in summary.items():
        scene, camera_text = key.split(":cam")
        camera = int(camera_text)
        for phase_name, method, timing_key in (
            ("forward", "B1A", "B1A_forward"),
            ("forward", "B2", "B2_forward"),
            ("fb", "B1A", "B1A_fb"),
            ("fb", "B2", "B2_fb"),
            ("backward_subtracted", "B1A", "B1A_backward"),
            ("backward_subtracted", "B2", "B2_backward"),
        ):
            timing_summary_rows.append({"scene": scene, "camera": camera,
                                        "phase": phase_name, "method": method,
                                        **measurements[timing_key]})
    write_csv(os.path.join(args.out_dir, "timings_summary.csv"), timing_summary_rows)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
