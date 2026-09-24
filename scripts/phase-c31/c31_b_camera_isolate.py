#!/usr/bin/env python3
"""
C31-B: Static-vs-Rotating Camera Discrepancy — Controlled Isolation.

Question: why does the full training loop show backward ~97.8 ms while a
standalone diagnostic showed ~14.3 ms?

The two earlier measurements differed in TWO variables:
  (1) camera schedule (1 fixed camera vs 311 rotating)
  (2) optimizer stepping (diagnostic never stepped the optimizer; state frozen)

This experiment controls both: a 4 x 2 design.

Camera workloads (fresh identical Gaussian state per block):
  A_static_1cam    : camera 0 repeated
  B_alt_2cam       : cameras 0,1 alternating
  C_rr_8cam        : cameras 0..7 round-robin
  D_rr_311cam      : all 311 cameras round-robin (original schedule)

Modes:
  full   : optimizer.step() every iteration (matches 104ms baseline condition)
  frozen : no optimizer step, grads cleared (matches 14.3ms diagnostic condition)

Metrics per block:
  fwd/bwd/opt GPU time (CUDA events), iteration wall-clock,
  camera loading time, GPU memory (current/peak, peak reset per block),
  and a 5-step torch.profiler pass: CUDA kernel count, GPU busy time,
  GPU idle gap (wall - busy).
"""
from __future__ import annotations
import argparse, gc, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint


def run_block(dataset, model, opt, cameras, mode, device,
              n_warmup=15, n_measure=50, n_profile=5, tile_size=16, label=""):
    """One controlled measurement block. Returns dict of metrics."""
    ev_f_s = torch.cuda.Event(enable_timing=True)
    ev_f_e = torch.cuda.Event(enable_timing=True)
    ev_b_s = torch.cuda.Event(enable_timing=True)
    ev_b_e = torch.cuda.Event(enable_timing=True)
    ev_o_s = torch.cuda.Event(enable_timing=True)
    ev_o_e = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)

    total = n_warmup + n_measure
    records = []

    for step in range(total):
        cam_idx = cameras[step % len(cameras)]

        # Camera load + GT fetch timing (CPU side, includes H2D if uncached)
        t0 = time.perf_counter()
        camera = dataset.get_camera(cam_idx)
        gt_image = dataset.get_gt_image(cam_idx)
        torch.cuda.synchronize()
        cam_load_ms = (time.perf_counter() - t0) * 1000

        t_iter = time.perf_counter()

        data = model.forward()
        ev_f_s.record()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
            width=camera.image_width, height=camera.image_height,
            tile_size=tile_size, packed=True, sh_degree=model.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
            sparse_grad=False, absgrad=False,
        )
        ev_f_e.record()
        rendered = rendered[0].clamp(0, 1)

        loss = combined_loss(rendered, gt_image, lambda_dssim=0.2)["loss"]

        opt.zero_grad(set_to_none=True)
        ev_b_s.record()
        loss.backward()
        torch.cuda.synchronize()
        ev_b_e.record()

        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if mode == "full":
            ev_o_s.record()
            opt.step()
            ev_o_e.record()
        else:
            # Frozen mode: discard grads, no state change
            for p in model.parameters():
                p.grad = None
            ev_o_s.record()
            ev_o_e.record()

        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t_iter) * 1000

        with torch.no_grad():
            mse = torch.mean((rendered - gt_image) ** 2).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

        if step >= n_warmup:
            records.append({
                "wall_ms": wall_ms,
                "fwd_ms": ev_f_s.elapsed_time(ev_f_e),
                "bwd_ms": ev_b_s.elapsed_time(ev_b_e),
                "opt_ms": ev_o_s.elapsed_time(ev_o_e),
                "cam_load_ms": cam_load_ms,
                "psnr": psnr,
            })

        if step % 20 == 0 or step == total - 1:
            w = "WARM" if step < n_warmup else "MEAS"
            print(f"    [{w}] {label} step {step}: wall={wall_ms:.1f} "
                  f"fwd={ev_f_s.elapsed_time(ev_f_e):.2f} "
                  f"bwd={ev_b_s.elapsed_time(ev_b_e):.2f} "
                  f"opt={ev_o_s.elapsed_time(ev_o_e):.2f} "
                  f"load={cam_load_ms:.2f}", flush=True)

    # ── Profiler pass: kernel count + GPU busy time on n_profile steps ──
    prof_kernels = 0
    prof_gpu_busy_ms = 0.0
    prof_wall_ms = 0.0
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
        ) as prof:
            for step in range(n_profile):
                cam_idx = cameras[step % len(cameras)]
                camera = dataset.get_camera(cam_idx)
                gt_image = dataset.get_gt_image(cam_idx)
                data = model.forward()
                rendered, _, _ = rasterization(
                    means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                    opacities=data["opacity"], colors=data["shs"],
                    viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
                    width=camera.image_width, height=camera.image_height,
                    tile_size=tile_size, packed=True, sh_degree=model.sh_degree,
                    radius_clip=0.0, eps2d=0.1, render_mode="RGB",
                    sparse_grad=False, absgrad=False,
                )
                rendered = rendered[0].clamp(0, 1)
                loss = combined_loss(rendered, gt_image, lambda_dssim=0.2)["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.cuda.synchronize()
                if mode == "full":
                    opt.step()
                else:
                    for p in model.parameters():
                        p.grad = None
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        # Aggregate CUDA kernel events
        cuda_events = []
        for evt in prof.key_averages():
            if evt.device_type == torch.autograd.DeviceType.CUDA:
                cuda_events.append((evt.count, evt.self_device_time_total / 1000.0))
        prof_kernels = int(sum(c for c, _ in cuda_events))
        prof_gpu_busy_ms = float(sum(t for _, t in cuda_events))
    except Exception as e:
        print(f"    [WARN] profiler pass failed: {e}", flush=True)

    # Wall time of the profiler window (measured implicitly via events above;
    # re-measure wall for the gap calculation)
    if records:
        wall_mean = float(np.mean([r["wall_ms"] for r in records]))
        gpu_mean = float(np.mean([r["fwd_ms"] + r["bwd_ms"] + r["opt_ms"] for r in records]))
    else:
        wall_mean = gpu_mean = 0.0

    return {
        "mode": mode,
        "n_unique_cameras": len(set(cameras)) if len(set(cameras)) < 400 else len(cameras),
        "n_measure": len(records),
        "wall_ms": {"mean": wall_mean, "std": float(np.std([r["wall_ms"] for r in records]))},
        "fwd_ms": {"mean": float(np.mean([r["fwd_ms"] for r in records])),
                   "std": float(np.std([r["fwd_ms"] for r in records]))},
        "bwd_ms": {"mean": float(np.mean([r["bwd_ms"] for r in records])),
                   "std": float(np.std([r["bwd_ms"] for r in records]))},
        "opt_ms": {"mean": float(np.mean([r["opt_ms"] for r in records])),
                   "std": float(np.std([r["opt_ms"] for r in records]))},
        "cam_load_ms": {"mean": float(np.mean([r["cam_load_ms"] for r in records])),
                        "std": float(np.std([r["cam_load_ms"] for r in records]))},
        "final_psnr": records[-1]["psnr"] if records else None,
        "gpu_mem_current_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "gpu_mem_peak_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "profiler": {
            "n_steps": n_profile,
            "cuda_kernel_launches_total": prof_kernels,
            "cuda_kernel_launches_per_step": round(prof_kernels / n_profile, 1),
            "gpu_busy_ms_total": round(prof_gpu_busy_ms, 3),
            "gpu_busy_ms_per_step": round(prof_gpu_busy_ms / n_profile, 3),
            "gpu_gap_ms_per_step_est": round(max(0.0, wall_mean - (prof_gpu_busy_ms / n_profile)), 3),
            "gap_note": "wall - gpu_busy; gap > 0 means GPU idle waiting on CPU/launches",
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/phase-c31/c31_b_camera_isolate.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--n-profile", type=int, default=5)
    p.add_argument("--scene", default="room")
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    np.random.seed(42)

    print("=== C31-B: Camera Discrepancy Controlled Isolation ===")
    print(f"GPU: {torch.cuda.get_device_properties(args.gpu).name}")
    print(f"Design: 4 workloads x 2 modes (full/frozen), "
          f"warmup={args.warmup}, measure={args.n}, profile={args.n_profile}")

    dataset = GTDataset(scene=args.scene, repo_root=ROOT, resolution="1080p", device=device)
    sfm_data = load_initial_checkpoint(args.scene, ROOT, device=device)
    n_cam = len(dataset)
    all_cameras = list(range(n_cam))

    workloads = {
        "A_static_1cam": [0],
        "B_alt_2cam": [0, 1],
        "C_rr_8cam": list(range(8)),
        "D_rr_311cam": all_cameras,
    }

    sls = float(sfm_data["xyz"].norm(dim=-1).max().item())

    def make_fresh_model():
        m = GaussianModel(
            num_points=sfm_data["xyz"].shape[0],
            sh_degree=0, max_sh_degree=3, device=device,
        )
        m.init_from_sfm(
            xyz=sfm_data["xyz"],
            opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=device)),
            scales_log=sfm_data.get("scales"),
            rotations_raw=sfm_data.get("rotations"),
            shs=sfm_data.get("shs"),
        )
        return m

    def make_opt(m):
        return torch.optim.Adam([
            {"params": [m.xyz], "lr": 1.6e-4 * sls},
            {"params": [m.rotations], "lr": 1e-3},
            {"params": [m.scales], "lr": 5e-3},
            {"params": [m.opacity], "lr": 5e-2},
            {"params": [m.shs], "lr": 2.5e-3},
        ], eps=1e-15, betas=(0.9, 0.999))

    results = {}
    for wl_name, cameras in workloads.items():
        for mode in ["full", "frozen"]:
            label = f"{wl_name}/{mode}"
            print(f"\n--- Block {label} ---", flush=True)
            model = make_fresh_model()
            opt = make_opt(model)
            stats = run_block(
                dataset, model, opt, cameras, mode, device,
                n_warmup=args.warmup, n_measure=args.n, n_profile=args.n_profile,
                label=label,
            )
            results[label] = stats
            bwd = stats["bwd_ms"]["mean"]
            print(f"    -> bwd={bwd:.2f}ms wall={stats['wall_ms']['mean']:.2f}ms "
                  f"kernels/step={stats['profiler']['cuda_kernel_launches_per_step']}", flush=True)
            del model, opt
            gc.collect()
            torch.cuda.empty_cache()

    # ── Cross-analysis: isolate camera effect vs state-evolution effect ──
    def m(key, sub):
        v = results.get(key)
        return v[sub]["mean"] if v and isinstance(v[sub], dict) else None

    analysis = {
        "bwd_by_mode": {
            mode: {wl: m(f"{wl}/{mode}", "bwd_ms") for wl in workloads}
            for mode in ["full", "frozen"]
        },
        "wall_by_mode": {
            mode: {wl: m(f"{wl}/{mode}", "wall_ms") for wl in workloads}
            for mode in ["full", "frozen"]
        },
    }

    print(f"\n{'='*70}")
    print("  CROSS-ANALYSIS")
    print(f"{'='*70}")
    print(f"  {'workload':<16} {'bwd_full':>9} {'bwd_frozen':>11} {'wall_full':>10} {'wall_frozen':>12}")
    for wl in workloads:
        bf = m(f"{wl}/full", "bwd_ms") or 0
        bz = m(f"{wl}/frozen", "bwd_ms") or 0
        wf = m(f"{wl}/full", "wall_ms") or 0
        wz = m(f"{wl}/frozen", "wall_ms") or 0
        print(f"  {wl:<16} {bf:>8.2f}ms {bz:>10.2f}ms {wf:>9.2f}ms {wz:>11.2f}ms")

    # Interpretation rules (data-driven, stated after seeing results)
    cam_effect_full = (m("D_rr_311cam/full", "bwd_ms") or 0) / max(m("A_static_1cam/full", "bwd_ms") or 1e-9, 1e-9)
    cam_effect_frozen = (m("D_rr_311cam/frozen", "bwd_ms") or 0) / max(m("A_static_1cam/frozen", "bwd_ms") or 1e-9, 1e-9)
    state_effect_static = (m("A_static_1cam/full", "bwd_ms") or 0) / max(m("A_static_1cam/frozen", "bwd_ms") or 1e-9, 1e-9)
    state_effect_rotating = (m("D_rr_311cam/full", "bwd_ms") or 0) / max(m("D_rr_311cam/frozen", "bwd_ms") or 1e-9, 1e-9)
    analysis["ratios"] = {
        "cam_effect_full_mode": round(cam_effect_full, 3),
        "cam_effect_frozen_mode": round(cam_effect_frozen, 3),
        "state_effect_static_cam": round(state_effect_static, 3),
        "state_effect_rotating_cam": round(state_effect_rotating, 3),
        "interpretation": (
            "If cam_effect ~ 1 in both modes: camera schedule does not explain the gap; "
            "optimizer stepping / state evolution does (state_effect >> 1). "
            "If cam_effect >> 1: camera rotation itself drives backward cost. "
            "If both ~ 1: the earlier 14.3ms vs 97.8ms gap must come from a third variable "
            "(model state at measurement time, JIT warmup depth, or measurement context)."
        ),
    }
    print(f"\n  cam_effect (D/A bwd, full mode):    {cam_effect_full:.2f}x")
    print(f"  cam_effect (D/A bwd, frozen mode):  {cam_effect_frozen:.2f}x")
    print(f"  state_effect (full/frozen, A):      {state_effect_static:.2f}x")
    print(f"  state_effect (full/frozen, D):      {state_effect_rotating:.2f}x")

    summary = {
        "schema_version": 2,
        "phase": "C31-B",
        "question": "Why full-loop backward ~97.8ms vs diagnostic backward ~14.3ms?",
        "confounded_variables": ["camera schedule", "optimizer stepping (state evolution)"],
        "design": "4 workloads x 2 modes, fresh identical Gaussian state per block",
        "gpu": torch.cuda.get_device_properties(args.gpu).name,
        "warmup": args.warmup, "n_measure": args.n, "n_profile": args.n_profile,
        "workload_cameras": {k: v if len(v) < 20 else f"0..{len(v)-1}" for k, v in workloads.items()},
        "blocks": results,
        "analysis": analysis,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2, default=str)
    print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
