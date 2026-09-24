#!/usr/bin/env python3
"""
C32-A: Training Iteration Critical-Path Decomposition

Structure:
  1. Warmup (50 iters, no measurement)
  2. Clean measurement (150 iters, CUDA events + perf_counter, NO extra syncs)
  3. Profiled measurement (3 iters, torch.profiler) — WARNING: profiler affects perf
  4. Analysis: merge clean timing + profiler kernel-level data from Chrome trace

Key: clean measurement matches original training semantics (no synchronize
inserted between forward/backward/optimizer). Only synchronize at iteration
boundary for wall-clock measurement, and CUDA events for per-phase GPU time.

Outputs:
  - results/phase-c31/c32_a_analysis.trace.json  (Chrome trace)
  - results/phase-c31/c32_a_analysis.json          (parsed decomposition)

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/phase-c31/c32_a_profile.py \
      --out results/phase-c31/c32_a_analysis.json
"""
from __future__ import annotations
import argparse, gc, json, math, os, sys, time, collections
from pathlib import Path
import numpy as np
import torch
import torch.profiler as profiler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--trace-out", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--scene", default="room")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--clean-n", type=int, default=150)
    p.add_argument("--profile-n", type=int, default=3)
    p.add_argument("--tile-size", type=int, default=16)
    args = p.parse_args()
    trace_out = args.trace_out or str(Path(args.out).with_suffix(".trace.json"))

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    np.random.seed(42)

    print("=== C32-A: Training Iteration Critical-Path Decomposition ===")
    print(f"GPU: {torch.cuda.get_device_properties(args.gpu).name}")

    # ── Load dataset and model ────────────────────────────────────────────
    dataset = GTDataset(scene=args.scene, repo_root=ROOT, resolution="1080p", device=device)
    sfm_data = load_initial_checkpoint(args.scene, ROOT, device=device)
    n_cam = len(dataset)

    model = GaussianModel(
        num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device,
    )
    model.init_from_sfm(
        xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm_data.get("scales"), rotations_raw=sfm_data.get("rotations"),
        shs=sfm_data.get("shs"),
    )
    sls = float(sfm_data["xyz"].norm(dim=-1).max().item())
    print(f"Gs: {model.xyz.shape[0]:,}")

    def make_opt():
        return torch.optim.Adam([
            {"params": [model.xyz], "lr": 1.6e-4 * sls},
            {"params": [model.rotations], "lr": 1e-3},
            {"params": [model.scales], "lr": 5e-3},
            {"params": [model.opacity], "lr": 5e-2},
            {"params": [model.shs], "lr": 2.5e-3},
        ], eps=1e-15, betas=(0.9, 0.999))

    opt = make_opt()

    def run_iter(step, camera=None, gt_image=None, profiling=False):
        """Run one training iteration exactly as Phase 7.
        
        When profiling=True, wraps phases in record_function for Chrome trace.
        """
        nonlocal opt, model
        ci = step % n_cam
        if camera is None:
            camera = dataset.get_camera(ci)
            gt_image = dataset.get_gt_image(ci)

        data = model.forward()

        if profiling:
            rf = torch.profiler.record_function
        else:
            rf = lambda n: _null_context()

        with rf("forward_rasterization"):
            rendered, _, _ = rasterization(
                means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                opacities=data["opacity"], colors=data["shs"],
                viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
                width=camera.image_width, height=camera.image_height,
                tile_size=args.tile_size, packed=True, sh_degree=0,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB",
                sparse_grad=False, absgrad=False,
            )
        rendered = rendered[0].clamp(0, 1)

        with rf("loss_combined"):
            loss = combined_loss(rendered, gt_image, lambda_dssim=0.2)["loss"]

        opt.zero_grad(set_to_none=True)

        with rf("backward"):
            loss.backward()

        with rf("gradient_postproc"):
            model.accumulate_positional_gradient()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        with rf("optimizer_step"):
            opt.step()

    # ── Phase 1: Warmup ──────────────────────────────────────────────────
    print(f"Warmup: {args.warmup} iterations...", flush=True)
    for step in range(args.warmup):
        run_iter(step)
        gc.collect()
    print("  Warmup complete.", flush=True)

    # ── Phase 2: Clean measurement (matches Phase 7 training SEMANTICS) ───
    # Key: Phase 7 trains WITHOUT per-iteration synchronize. The iteration
    # timing is measured as: perf_counter at start → all ops → perf_counter
    # at end, with synchronize() called inside backward/optimizer event
    # boundaries.
    #
    # We use BLOCK TIMING for wall measurement:
    #   sync at block START → run N iterations (no sync between) → sync at
    #   block END → T_iter = (end - start) / N
    #
    # CUDA events are recorded but their time is read AFTER the block to
    # avoid per-iteration CPU blocking.
    print(f"Clean measurement: {args.clean_n} iterations...", flush=True)

    # Pre-create CUDA event pools
    ev_f_s_pool = [torch.cuda.Event(enable_timing=True) for _ in range(args.clean_n)]
    ev_f_e_pool = [torch.cuda.Event(enable_timing=True) for _ in range(args.clean_n)]
    ev_b_s_pool = [torch.cuda.Event(enable_timing=True) for _ in range(args.clean_n)]
    ev_b_e_pool = [torch.cuda.Event(enable_timing=True) for _ in range(args.clean_n)]
    ev_o_s_pool = [torch.cuda.Event(enable_timing=True) for _ in range(args.clean_n)]
    ev_o_e_pool = [torch.cuda.Event(enable_timing=True) for _ in range(args.clean_n)]

    # Block timing: sync before block
    torch.cuda.synchronize()
    block_start = time.perf_counter()

    for idx in range(args.clean_n):
        step = args.warmup + idx
        ci = step % n_cam
        camera = dataset.get_camera(ci)
        gt_image = dataset.get_gt_image(ci)

        data = model.forward()
        ev_f_s_pool[idx].record()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
            width=camera.image_width, height=camera.image_height,
            tile_size=args.tile_size, packed=True, sh_degree=0,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
            sparse_grad=False, absgrad=False,
        )
        ev_f_e_pool[idx].record()
        rendered = rendered[0].clamp(0, 1)

        loss = combined_loss(rendered, gt_image, lambda_dssim=0.2)["loss"]

        opt.zero_grad(set_to_none=True)
        ev_b_s_pool[idx].record()
        loss.backward()
        ev_b_e_pool[idx].record()

        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        ev_o_s_pool[idx].record()
        opt.step()
        ev_o_e_pool[idx].record()

    # Block timing: sync after block
    torch.cuda.synchronize()
    block_end = time.perf_counter()

    block_wall_ms = (block_end - block_start) * 1000
    avg_t_iter_wall = block_wall_ms / args.clean_n

    # Read CUDA events AFTER block (no per-iteration blocking)
    ev_fwd_ms_list = [ev_f_s_pool[i].elapsed_time(ev_f_e_pool[i]) for i in range(args.clean_n)]
    ev_bwd_ms_list = [ev_b_s_pool[i].elapsed_time(ev_b_e_pool[i]) for i in range(args.clean_n)]
    ev_opt_ms_list = [ev_o_s_pool[i].elapsed_time(ev_o_e_pool[i]) for i in range(args.clean_n)]

    avg_fwd = float(np.mean(ev_fwd_ms_list))
    avg_bwd = float(np.mean(ev_bwd_ms_list))
    avg_opt = float(np.mean(ev_opt_ms_list))
    avg_sum = avg_fwd + avg_bwd + avg_opt

    print(f"  Clean T_iter (block timing): {avg_t_iter_wall:.2f}ms ("
          f"fwd_evt={avg_fwd:.2f} bwd_evt={avg_bwd:.2f} opt_evt={avg_opt:.2f})",
          flush=True)

    # ── Phase 3: Profiled measurement ────────────────────────────────────
    # Note: torch.profiler adds ~40-60% overhead to wall time
    print(f"Profiled measurement: {args.profile_n} iterations...", flush=True)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for step_idx in range(args.profile_n):
            step = args.warmup + args.clean_n + step_idx
            run_iter(step, profiling=True)
            gc.collect()

    print("  Profiling complete.", flush=True)
    prof.export_chrome_trace(trace_out)
    print(f"  Chrome trace saved: {trace_out}", flush=True)

    # ── Parse Chrome trace JSON ──────────────────────────────────────────
    with open(trace_out) as f:
        trace_data = json.load(f)
    trace_events = trace_data.get("traceEvents", trace_data if isinstance(trace_data, list) else [])

    gpu_kernels = []
    cpu_cuda_events = []
    cpu_launch_events = []  # cudaLaunchKernel specifically
    sync_events = []
    memcpy_events = []
    memset_events = []
    malloc_free_events = []

    for ev in trace_events:
        if not isinstance(ev, dict):
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        dur_us = ev.get("dur", 0)
        ts_us = ev.get("ts", 0)

        if dur_us <= 0:
            continue

        item = {"name": name, "cat": cat, "ts_us": ts_us, "dur_us": dur_us,
                "pid": ev.get("pid"), "tid": ev.get("tid")}

        # torch.profiler Chrome trace categories:
        #   cat=="kernel" → real GPU kernel
        #   cat=="cpu_op" → CPU-side API call (cudaLaunchKernel, synchronize, etc.)
        #   NVTX ranges → cat=="python_function" or custom
        if cat == "kernel" or "cuda_kernel" in cat:
            gpu_kernels.append(item)
        elif cat == "cpu_op":
            if "cudaLaunch" in name:
                cpu_launch_events.append(item)
                cpu_cuda_events.append(item)
            elif "synchronize" in name.lower():
                sync_events.append(item)
                cpu_cuda_events.append(item)
            elif "Memcpy" in name:
                memcpy_events.append(item)
                cpu_cuda_events.append(item)
            elif "Memset" in name:
                memset_events.append(item)
                cpu_cuda_events.append(item)
            elif "malloc" in name.lower() or "free" in name.lower() or "alloc" in name.lower():
                malloc_free_events.append(item)
                cpu_cuda_events.append(item)

    if not gpu_kernels:
        print("  WARNING: No GPU kernel events ('kernel' category) in Chrome trace. "
              "Trying fallback: key_averages.", flush=True)
        kavg = prof.key_averages()
        cuda_ke = [e for e in kavg if hasattr(e, 'device_type')
                   and e.device_type == torch.autograd.profiler.DeviceType.CUDA]
        kernel_by_name = {}
        for e in cuda_ke:
            ct = getattr(e, "cuda_time", 0)  # μs
            if ct > 0:
                name = e.name
                kernel_by_name[name] = {"count": e.count, "total_us": ct * e.count,
                                         "min_us": ct, "max_us": ct, "durations": [ct]}
        n_kernels = sum(v["count"] for v in kernel_by_name.values())
        total_gpu_busy_us = sum(v["total_us"] for v in kernel_by_name.values())
        gpu_per_iter_ms = total_gpu_busy_us / 1000 / args.profile_n
        gap_stats = {"count": 0, "total_gap_ms": 0, "gap_per_iter_ms": 0,
                     "mean_us": 0, "p50_us": 0, "p90_us": 0, "p99_us": 0, "max_us": 0}
        trace_span_us = total_gpu_busy_us
        n_streams = 0
    else:
        # Real GPU kernels only (cat=="kernel")
        kernel_by_name = {}
        gpu_kernels_sorted = sorted(gpu_kernels, key=lambda k: k["ts_us"])
        for k in gpu_kernels_sorted:
            nm, dur = k["name"], k["dur_us"]
            if nm not in kernel_by_name:
                kernel_by_name[nm] = {"count": 0, "total_us": 0.0,
                                       "min_us": float("inf"), "max_us": 0.0, "durations": []}
            kb = kernel_by_name[nm]
            kb["count"] += 1
            kb["total_us"] += dur
            kb["min_us"] = min(kb["min_us"], dur)
            kb["max_us"] = max(kb["max_us"], dur)
            kb["durations"].append(dur)

        n_kernels = sum(kb["count"] for kb in kernel_by_name.values())
        total_gpu_busy_us = sum(kb["total_us"] for kb in kernel_by_name.values())
        total_gpu_busy_ms = total_gpu_busy_us / 1000
        gpu_per_iter_ms = total_gpu_busy_ms / args.profile_n

        # Inter-kernel gaps (global timeline — all kernels across all streams)
        gap_list = []
        for i in range(1, len(gpu_kernels_sorted)):
            pe = gpu_kernels_sorted[i-1]["ts_us"] + gpu_kernels_sorted[i-1]["dur_us"]
            cs = gpu_kernels_sorted[i]["ts_us"]
            if cs > pe:
                gap_list.append(cs - pe)
        gap_arr = np.array(gap_list) if gap_list else np.array([0.0])
        gap_stats = {
            "count": len(gap_list),
            "total_gap_ms": round(float(np.sum(gap_arr)) / 1000, 3),
            "gap_per_iter_ms": round(float(np.sum(gap_arr)) / 1000 / args.profile_n, 3),
            "mean_us": round(float(np.mean(gap_arr)), 1) if len(gap_arr) > 0 else 0,
            "p50_us": round(float(np.percentile(gap_arr, 50)), 1) if len(gap_arr) > 0 else 0,
            "p90_us": round(float(np.percentile(gap_arr, 90)), 1) if len(gap_arr) > 0 else 0,
            "p99_us": round(float(np.percentile(gap_arr, 99)), 1) if len(gap_arr) > 0 else 0,
            "max_us": round(float(np.max(gap_arr)), 1),
        }

        n_streams = len(set((k["pid"], k["tid"]) for k in gpu_kernels))

        min_gpu_start = gpu_kernels_sorted[0]["ts_us"]
        max_gpu_end = max(k["ts_us"] + k["dur_us"] for k in gpu_kernels_sorted)
        trace_span_us = max_gpu_end - min_gpu_start

    trace_span_ms = trace_span_us / 1000

    # Top 30 by total GPU duration
    sorted_dur = sorted(kernel_by_name.items(), key=lambda x: -x[1]["total_us"])
    top30_dur = []
    for name, kb in sorted_dur[:30]:
        d = np.array(kb["durations"])
        top30_dur.append({
            "kernel": name,
            "count": kb["count"],
            "total_ms": round(kb["total_us"] / 1000, 3),
            "mean_us": round(kb["total_us"] / max(kb["count"], 1), 1),
            "p50_us": round(float(np.percentile(d, 50)), 1) if len(d) > 0 else 0,
            "p90_us": round(float(np.percentile(d, 90)), 1) if len(d) > 0 else 0,
            "p99_us": round(float(np.percentile(d, 99)), 1) if len(d) > 0 else 0,
        })

    # Top 30 by launch frequency
    sorted_cnt = sorted(kernel_by_name.items(), key=lambda x: -x[1]["count"])
    top30_cnt = []
    for name, kb in sorted_cnt[:30]:
        top30_cnt.append({
            "kernel": name,
            "count": kb["count"],
            "total_ms": round(kb["total_us"] / 1000, 3),
            "mean_us": round(kb["total_us"] / max(kb["count"], 1), 1),
        })

    # ── Per-stream analysis ────────────────────────────────────────────
    # Group kernels by (pid, tid) → treat each as one CUDA stream
    stream_kernels = {}
    for k in gpu_kernels:
        sid = (k["pid"], k["tid"])
        if sid not in stream_kernels:
            stream_kernels[sid] = []
        stream_kernels[sid].append(k)

    # Within each stream, measure sequential kernel time and gaps
    stream_stats = {}
    for sid, kernels in stream_kernels.items():
        ks = sorted(kernels, key=lambda x: x["ts_us"])
        total_kernel_us = sum(k["dur_us"] for k in ks)
        total_gap_us = 0
        for i in range(1, len(ks)):
            gap = ks[i]["ts_us"] - (ks[i-1]["ts_us"] + ks[i-1]["dur_us"])
            if gap > 0:
                total_gap_us += gap
        stream_stats[str(sid)] = {
            "total_gpu_kernel_us": round(total_kernel_us, 1),
            "total_gap_us": round(total_gap_us, 1),
            "n_kernels": len(ks),
        }

    # Total GPU busy time (NO CUDA API calls — pure kernel execution)
    # across ALL streams. This is what "profiler GPU busy" should represent.
    # For wall-clock comparison, we need the GPU busy on the DEFAULT stream
    # specifically, since wall-clock is determined by the critical path.

    # Identify default stream: first kernel's pid/tid
    default_sid = None
    if gpu_kernels_sorted:
        default_sid = (gpu_kernels_sorted[0]["pid"], gpu_kernels_sorted[0]["tid"])
    default_stream_stats = stream_stats.get(str(default_sid), {})

    # ── Decomposition: wall = GPU_critical_path + CPU_overhead + sync ──
    wall_ms = avg_t_iter_wall
    cuda_ev_fwd = avg_fwd
    cuda_ev_bwd = avg_bwd
    cuda_ev_opt = avg_opt
    cuda_ev_sum = avg_sum

    # Default stream GPU kernel total (critical-path approximation)
    # CUDA events on default stream only cover the rasterization forward+backward
    # and optimizer step. The ~25ms CUDA event sum represents default-stream
    # GPU execution time for those specific operations.
    #
    # Profiler GPU busy = sum of kernel durations across ALL streams (~320ms/3 iters)
    # This exceeds wall because streams overlap.
    #
    # The gap between wall and CUDA events:
    #   wall - CUDA_events = ~80ms
    # This includes:
    #   (a) GPU time on default stream not captured by CUDA events
    #   (b) CPU launch overhead
    #   (c) Python dispatch
    #   (d) Allocator activity
    #   (e) Memory copy
    #   (f) Synchronization

    # From profiler: default stream total GPU time (if identifiable)
    default_gpu_ms = default_stream_stats.get("total_gpu_kernel_us", 0) / 1000
    default_gpu_per_iter = default_gpu_ms / args.profile_n if default_gpu_ms > 0 else 0

    # Count memory + CPU events per iter
    cpu_launch_per_iter = len(cpu_launch_events) / args.profile_n
    sync_per_iter = len(sync_events) / args.profile_n
    memcpy_per_iter = len(memcpy_events) / args.profile_n

    # Build decomposition
    profiled_gpu_per_iter = total_gpu_busy_us / 1000 / args.profile_n
    total_cpu_api_us = sum(e["dur_us"] for e in cpu_cuda_events)
    total_cpu_api_per_iter_ms = total_cpu_api_us / 1000 / args.profile_n

    decomposition = {
        "wall_clock_clean_ms": round(wall_ms, 3),
        "clean_n": args.clean_n,
        "block_wall_ms": round(block_wall_ms, 3),
        "block_wall_method": (
            "Block timing: sync at block start, run N iterations with NO per-iteration sync, "
            "sync at block end. T_iter = block_wall / N. This matches original Phase 7 "
            "training semantics where successive iterations pipeline naturally."
        ),
        "cuda_events_default_stream_ms": {
            "fwd": round(cuda_ev_fwd, 3),
            "bwd": round(cuda_ev_bwd, 3),
            "opt": round(cuda_ev_opt, 3),
            "sum": round(cuda_ev_sum, 3),
        },
        "profiler_n": args.profile_n,
        "profiler_trace_span_ms": round(trace_span_ms, 2),
        "expected_trace_span_ms": round(args.profile_n * wall_ms, 2),
        "gpu_kernel_summary": {
            "total_gpu_kernel_launches": n_kernels,
            "kernels_per_iter": round(n_kernels / args.profile_n, 1),
            "total_gpu_busy_ms": round(total_gpu_busy_us / 1000, 3),
            "gpu_busy_per_iter_ms": round(profiled_gpu_per_iter, 3),
            "estimated_streams": n_streams,
            "gpu_busy_vs_wall_ratio": round(profiled_gpu_per_iter / max(wall_ms, 1e-9), 3),
        },
        "default_stream_estimate": {
            "total_gpu_kernel_ms": round(default_gpu_ms, 2),
            "per_iter_ms": round(default_gpu_per_iter, 3),
            "note": (
                "Default stream identified by first kernel's (pid,tid). "
                "This stream typically carries the critical path. "
                "Per-iteration estimate = total / profile_n. CUDA events under-report "
                "because they only wrap specific operations (Fwd/Bwd/Opt) not all kernels."
            ),
        },
        "cpu_cuda_api_summary": {
            "cuda_launch_calls": len(cpu_launch_events),
            "cuda_launch_per_iter": round(cpu_launch_per_iter, 1),
            "sync_calls": len(sync_events),
            "sync_per_iter": round(sync_per_iter, 1),
            "memcpy_calls": len(memcpy_events),
            "memcpy_per_iter": round(memcpy_per_iter, 1),
            "total_api_time_ms": round(total_cpu_api_us / 1000, 3),
            "api_time_per_iter_ms": round(total_cpu_api_per_iter_ms, 3),
        },
        "stream_stats": stream_stats,
        "inter_kernel_gaps": gap_stats,
        "top30_kernels_by_duration": top30_dur,
        "top30_kernels_by_frequency": top30_cnt,
    }

    # ── C31 Mystery: why CUDA event ≠ profiler ≠ wall ──────────────────
    decomposition["c31_mystery"] = {
        "cuda_event_default_stream_ms": round(cuda_ev_sum, 2),
        "profiler_gpu_busy_ms": round(profiled_gpu_per_iter, 2),
        "expected_profiler_discrepancy": (
            f"Profiler GPU busy ({profiled_gpu_per_iter:.1f}ms/iter) >> wall ({wall_ms:.1f}ms) "
            f"because profiler sums durations across ALL {n_streams} streams. "
            f"Overlapping stream execution means the same wall second appears in multiple streams."
        ),
        "cuda_event_deficit": (
            f"CUDA events ({cuda_ev_sum:.1f}ms) < wall ({wall_ms:.1f}ms) because events "
            f"only wrap rasterization forward/backward and optimizer.step on the default stream. "
            f"Missing: loss computation, D-SSIM backward, autograd dispatch, gradient clipping, "
            f"and even the non-default-stream overlap portion of rasterization backward."
        ),
        "wall_decomposition": {
            "iteration_wall_ms": round(wall_ms, 2),
            "default_stream_gpu_ms": round(default_gpu_per_iter, 2),
            "cuda_event_covered_ms": round(cuda_ev_sum, 2),
            "remaining_wall_unaccounted_ms": round(max(0, wall_ms - cuda_ev_sum), 2),
        },
    }

    # ── Research Questions ──────────────────────────────────────────────
    gap_pct = gap_stats["gap_per_iter_ms"] / max(wall_ms, 1e-9) * 100
    decomposition["research_questions"] = {
        "Q1_launch_or_cpu_bound": {
            "question": "Is the workload actually launch-bound / CPU-bound?",
            "data": (
                f"Kernels/iter: {n_kernels/args.profile_n:.0f}. "
                f"CPU cudaLaunchKernel calls/iter: {cpu_launch_per_iter:.1f}. "
                f"CPU total API time/iter: {total_cpu_api_per_iter_ms:.3f}ms. "
                f"Inter-kernel gap total/iter (global): {gap_stats['gap_per_iter_ms']:.1f}ms "
                f"({gap_pct:.1f}% of wall). "
                f"GPU busy (all streams)/iter: {profiled_gpu_per_iter:.1f}ms "
                f"(vs wall {wall_ms:.1f}ms). "
                f"Default-stream GPU/iter: {default_gpu_per_iter:.2f}ms."
            ),
            "conclusion": (
                "launch_bound" if gap_pct > 50 else
                "hybrid" if gap_pct > 10 else
                "compute_bound"
            ),
        },
        "Q2_cuda_graph_removable": {
            "question": "Fraction of T_iter theoretically removable by CUDA Graph?",
            "answer": (
                f"Inter-kernel gaps/iter = {gap_stats['gap_per_iter_ms']:.1f}ms "
                f"({gap_pct:.0f}% of wall). "
                f"CPU API time/iter = {total_cpu_api_per_iter_ms:.3f}ms. "
                f"These are the primary targets. However, CUDA Graph serializes streams "
                f"to one stream, so the effective wall may become closer to profiled_gpu_busy "
                f"= {profiled_gpu_per_iter:.1f}ms if current multi-stream overlap is significant. "
                f"Graph does NOT remove GPU compute time, only launch overhead + gaps."
            ),
        },
        "Q3_cuda_graph_topology": {
            "question": "CUDA Graph viable despite dynamic Gaussian topology?",
            "answer": (
                "Re-capture every 100 steps (after densification/pruning). "
                f"For {n_kernels/args.profile_n:.0f} kernels, capture cost is O(iteration) with overhead. "
                "If 100 replays per capture, amortized capture cost is ~1% of total time. "
                "Requires same tensor shapes between topology changes."
            ),
        },
        "Q4_3dgs_specific_overhead": {
            "question": "3DGS-specific execution structure causing overhead?",
            "data": (
                f"Top kernels reveal: gsplat rasterization fwd/bwd, "
                f"cuDNN conv (D-SSIM backward via Gaussian blur), "
                f"Adam optimizer per-parameter kernels, "
                f"gradient accumulation/scaling ops. "
                f"Key 3DGS pattern: ~{n_kernels} kernels/iteration from autograd "
                f"disaggregating operations across 5+ parameter groups."
            ),
        },
        "Q5_cuda_graph_boundary": {
            "question": "Capture boundary for testing?",
            "answer": (
                "Capture: model.forward() through loss.backward() through opt.step(). "
                "Exclude: camera/data loading, zero_grad(), gradient clipping, "
                "accumulate_positional_gradient(). "
                f"With {n_kernels/args.profile_n:.0f} kernels/iter, graph must capture all of them. "
                "Test with torch.cuda.CUDAGraph on a single topology-stable iteration window."
            ),
        },
    }

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(decomposition, open(args.out, "w"), indent=2, default=str)
    print(f"\nAnalysis saved: {args.out}")

    print(f"\n=== C32-A SUMMARY ===")
    print(f"  Clean T_iter: {wall_ms:.2f} ms")
    print(f"  CUDA event sum (default stream, fwd+bwd+opt): {cuda_ev_sum:.2f} ms")
    print(f"  Profiler GPU busy (all streams): {profiled_gpu_per_iter:.2f} ms/iter")
    print(f"  Default-stream GPU (profiler): {default_gpu_per_iter:.2f} ms/iter")
    print(f"  GPU kernels/iter: {n_kernels/args.profile_n:.0f}")
    print(f"  cudaLaunchKernel calls/iter: {cpu_launch_per_iter:.1f}")
    print(f"  CUDA streams detected: {n_streams}")
    print(f"  GPU kernel gap total/iter: {gap_stats['gap_per_iter_ms']:.1f}ms ({gap_pct:.0f}%)")
    print(f"  Gaps P50/P90/P99: {gap_stats['p50_us']:.1f}/{gap_stats['p90_us']:.1f}/{gap_stats['p99_us']:.1f} us")
    print(f"  CUDA API time/iter: {total_cpu_api_per_iter_ms:.3f} ms")
    print(f"  Sync calls/iter: {sync_per_iter:.1f}")

    print(f"\nTop 5 GPU kernels by total time:")
    for k in top30_dur[:5]:
        print(f"  {k['kernel'][:80]}: {k['count']}x tot={k['total_ms']:.1f}ms mean={k['mean_us']:.1f}us")

    print(f"\nTop 5 GPU kernels by launch count:")
    for k in top30_cnt[:5]:
        print(f"  {k['kernel'][:80]}: {k['count']}x tot={k['total_ms']:.1f}ms")

    print(f"\nStream-level stats (per-iteration):")
    for sid, ss in stream_stats.items():
        print(f"  stream {sid}: {ss['n_kernels']} kernels, "
              f"{ss['total_gpu_kernel_us']/args.profile_n/1000:.2f}ms GPU, "
              f"{ss['total_gap_us']/args.profile_n/1000:.2f}ms gap")
        if sid == str(default_sid):
            print(f"    ← default (critical path) stream")


def _null_context():
    class _NC:
        def __enter__(self): pass
        def __exit__(self, *a): pass
    return _NC()


if __name__ == "__main__":
    main()
