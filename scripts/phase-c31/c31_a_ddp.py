#!/usr/bin/env python3
"""
C31-A: Real DDP Scaling Baseline.

True torch.distributed training on 1/2/4/8 GPUs.
Same Phase 7 stack, scene, initialization, seed, camera schedule, loss.

No topology changes during measurement (warmup+measured < 500 steps
so no densification/pruning/SH-degree-increase triggers).

For N=1, runs without DDP wrapper.
For N>1, cameras split round-robin across ranks; gradients averaged via DDP.

Usage:
  python3 scripts/phase-c31/c31_a_ddp.py
"""
from __future__ import annotations
import argparse, gc, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint


def worker(rank: int, world_size: int, args):
    """Single-rank DDP worker. Runs on GPU=rank."""
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = f"{29500 + args.port_offset + world_size}"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # ── Dataset (each rank loads independently) ──
    dataset = GTDataset(scene=args.scene, repo_root=ROOT, resolution="1080p", device=device)
    sfm_data = load_initial_checkpoint(args.scene, ROOT, device=device)
    num_cameras = len(dataset)

    # Round-robin camera assignment per rank
    rank_cameras = [c for c in range(num_cameras) if c % world_size == rank]
    if rank == 0:
        print(f"  World size {world_size}: {len(rank_cameras)} cam/rank ({num_cameras} total)", flush=True)

    # ── Model ──
    model = GaussianModel(
        num_points=sfm_data["xyz"].shape[0],
        sh_degree=0,
        max_sh_degree=3,
        device=device,
    )
    model.init_from_sfm(
        xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm_data.get("scales"),
        rotations_raw=sfm_data.get("rotations"),
        shs=sfm_data.get("shs"),
    )
    spatial_lr_scale = float(sfm_data["xyz"].norm(dim=-1).max().item())

    if world_size > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    def make_opt():
        return torch.optim.Adam([
            {"params": [model.xyz], "lr": args.lr_xyz * spatial_lr_scale},
            {"params": [model.rotations], "lr": args.lr_rotation},
            {"params": [model.scales], "lr": args.lr_scaling},
            {"params": [model.opacity], "lr": args.lr_opacity},
            {"params": [model.shs], "lr": args.lr_sh},
        ], eps=args.adam_eps, betas=(args.adam_beta1, args.adam_beta2))

    opt = make_opt()

    # DDP wrapper (N=1 is a no-op)
    if world_size > 1:
        model_ddp = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank,
        )
        model_local = model_ddp.module
    else:
        model_ddp = model
        model_local = model

    # ── CUDA events for GPU-only timing ──
    ev_fwd_s = torch.cuda.Event(enable_timing=True)
    ev_fwd_e = torch.cuda.Event(enable_timing=True)
    ev_bwd_s = torch.cuda.Event(enable_timing=True)
    ev_bwd_e = torch.cuda.Event(enable_timing=True)
    ev_opt_s = torch.cuda.Event(enable_timing=True)
    ev_opt_e = torch.cuda.Event(enable_timing=True)

    total = args.warmup + args.measured
    records = []
    n_gpu_info = torch.cuda.get_device_properties(rank)

    for step in range(total):
        local_idx = step % len(rank_cameras)
        cam_idx = rank_cameras[local_idx]
        camera = dataset.get_camera(cam_idx)
        gt_image = dataset.get_gt_image(cam_idx)

        # SH degree (step 0-999 all stay at degree 0; interval=1000)
        new_deg = min(3, step // args.sh_deg_interval)
        if new_deg != model_local.sh_degree:
            model_local.set_sh_degree(new_deg)
            opt = make_opt()
            if world_size > 1:
                model_ddp = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[rank], output_device=rank,
                )
                model_local = model_ddp.module
            else:
                model_ddp = model
                model_local = model

        # ── Iteration wall-clock start ──
        t_iter_start = time.perf_counter()

        # ── Forward ──
        data = model_local.forward()
        ev_fwd_s.record()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
            width=camera.image_width, height=camera.image_height,
            tile_size=args.tile_size, packed=True, sh_degree=model_local.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
            sparse_grad=False, absgrad=False,
        )
        ev_fwd_e.record()
        rendered = rendered[0].clamp(0, 1)

        # ── Loss + Backward ──
        loss = combined_loss(rendered, gt_image, lambda_dssim=args.lambda_dssim)["loss"]

        opt.zero_grad(set_to_none=True)
        ev_bwd_s.record()
        loss.backward()
        # DDP's backward hook allreduces on NCCL stream asynchronously.
        # Synchronize here so ev_bwd_e captures the complete GPU-side time.
        torch.cuda.synchronize()
        ev_bwd_e.record()

        model_local.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(
            model_ddp.parameters() if world_size > 1 else model.parameters(),
            max_norm=1.0,
        )

        # ── Optimizer ──
        ev_opt_s.record()
        opt.step()
        ev_opt_e.record()

        # ── Final sync ──
        torch.cuda.synchronize()
        iter_wall_ms = (time.perf_counter() - t_iter_start) * 1000
        fwd_gpu_ms = ev_fwd_s.elapsed_time(ev_fwd_e)
        bwd_gpu_ms = ev_bwd_s.elapsed_time(ev_bwd_e)
        opt_gpu_ms = ev_opt_s.elapsed_time(ev_opt_e)

        # No densification/pruning during measurement (step count < 500)

        with torch.no_grad():
            mse = torch.mean((rendered - gt_image) ** 2).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

        if step >= args.warmup:
            records.append({
                "iter_wall_ms": iter_wall_ms,
                "fwd_gpu_ms": fwd_gpu_ms,
                "bwd_gpu_ms": bwd_gpu_ms,
                "opt_gpu_ms": opt_gpu_ms,
                "psnr": psnr,
                "n_gaussians": model_local.xyz.shape[0],
                "cam_idx": cam_idx,
            })

        if rank == 0 and (step % max(1, total // 8) == 0 or step == total - 1):
            w = "WARM" if step < args.warmup else "MEAS"
            print(f"  [{w} R{rank}] Step {step}: PSNR={psnr:.2f} "
                  f"wall={iter_wall_ms:.1f} fwd={fwd_gpu_ms:.2f} "
                  f"bwd={bwd_gpu_ms:.2f} opt={opt_gpu_ms:.2f}", flush=True)

    if world_size > 1:
        dist.destroy_process_group()

    # ── Rank 0 produces results ──
    if rank == 0 and records:
        w_a = np.array([r["iter_wall_ms"] for r in records])
        f_a = np.array([r["fwd_gpu_ms"] for r in records])
        b_a = np.array([r["bwd_gpu_ms"] for r in records])
        o_a = np.array([r["opt_gpu_ms"] for r in records])

        results = {
            "schema_version": 2,
            "phase": "C31-A",
            "world_size": world_size,
            "warmup": args.warmup,
            "measured": args.measured,
            "scene": args.scene,
            "seed": args.seed,
            "n_cameras": num_cameras,
            "cameras_per_rank": len(rank_cameras),
            "gpu_name": n_gpu_info.name,
            "gpu_memory_gb": round(n_gpu_info.total_memory / 1e9, 1),
            "torch_version": torch.__version__,
            "gsplat_version": "1.5.3",
            "t_iter_ms": {
                "mean": float(np.mean(w_a)),
                "median": float(np.median(w_a)),
                "std": float(np.std(w_a)),
                "min": float(np.min(w_a)),
                "max": float(np.max(w_a)),
                "p5": float(np.percentile(w_a, 5)),
                "p95": float(np.percentile(w_a, 95)),
            },
            "fwd_gpu_ms": {
                "mean": float(np.mean(f_a)),
                "median": float(np.median(f_a)),
            },
            "bwd_gpu_ms": {
                "mean": float(np.mean(b_a)),
                "median": float(np.median(b_a)),
                "note": "N=1: pure backward; N>1: backward + DDP allreduce",
            },
            "opt_gpu_ms": {
                "mean": float(np.mean(o_a)),
                "median": float(np.median(o_a)),
            },
            "final_psnr": float(records[-1]["psnr"]),
            "final_gaussians": int(records[-1]["n_gaussians"]),
        }

        t_mean = float(np.mean(w_a))
        print(f"\n--- C31-A N={world_size} ---")
        print(f"  T_iter: {t_mean:.2f}ms ± {float(np.std(w_a)):.2f}")
        print(f"  Fwd: {float(np.mean(f_a)):.2f}ms  Bwd: {float(np.mean(b_a)):.2f}ms  "
              f"Opt: {float(np.mean(o_a)):.2f}ms")
        print(f"  PSNR: {records[-1]['psnr']:.2f}  Gs: {records[-1]['n_gaussians']:,}")

        out_path = args.out.replace("NVAL", str(world_size))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"  Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="C31-A: Real DDP scaling baseline for 3DGS training."
    )
    p.add_argument("--out", default="results/phase-c31/c31_a_nNVAL.json",
                   help="Output path; NVAL replaced by world_size")
    p.add_argument("--scene", default="room")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--measured", type=int, default=150)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port-offset", type=int, default=0)
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="GPU indices to use. Default: all available")
    p.add_argument("--lr-xyz", type=float, default=1.6e-4)
    p.add_argument("--lr-rotation", type=float, default=1e-3)
    p.add_argument("--lr-scaling", type=float, default=5e-3)
    p.add_argument("--lr-opacity", type=float, default=5e-2)
    p.add_argument("--lr-sh", type=float, default=2.5e-3)
    p.add_argument("--adam-eps", type=float, default=1e-15)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--lambda-dssim", type=float, default=0.2)
    p.add_argument("--sh-deg-interval", type=int, default=1000)
    p.add_argument("--denf-start", type=int, default=500)
    p.add_argument("--denf-interval", type=int, default=100)
    p.add_argument("--denf-threshold", type=float, default=0.0002)
    p.add_argument("--prune-start", type=int, default=500)
    p.add_argument("--prune-interval", type=int, default=100)
    p.add_argument("--prune-threshold", type=float, default=0.005)
    args = p.parse_args()

    n_avail = torch.cuda.device_count()
    print(f"Available GPUs: {n_avail}", flush=True)

    # Parent process has initialized CUDA (device_count above).
    # CUDA does not survive fork(); force spawn so child processes
    # start fresh CUDA contexts.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    gpuses = args.gpus if args.gpus is not None else list(range(n_avail))
    max_world = len(gpuses)
    results_n1 = None

    for world_size in [1, 2, 4, 8]:
        if world_size > max_world:
            print(f"  Skipping N={world_size} (only {max_world} GPUs available)", flush=True)
            continue

        print(f"\n{'='*60}")
        print(f"  Launching N={world_size} DDP run on GPUs {gpuses[:world_size]}")
        print(f"{'='*60}", flush=True)

        if world_size == 1:
            # No multiprocessing overhead for N=1
            worker(0, 1, args)
        else:
            mp.spawn(worker, args=(world_size, args), nprocs=world_size, join=True)

        print(f"  Completed N={world_size}", flush=True)

        # Compute speedup vs N=1 baseline
        out_path = args.out.replace("NVAL", str(world_size))
        try:
            with open(out_path) as f:
                result = json.load(f)
            if world_size == 1:
                results_n1 = result
            elif results_n1 is not None:
                n1_t = results_n1["t_iter_ms"]["mean"]
                n_t = result["t_iter_ms"]["mean"]
                result["baseline_t_iter_ms"] = n1_t
                result["speedup_vs_baseline"] = round(n1_t / max(n_t, 1e-9), 3)
                result["scaling_efficiency_pct"] = round(
                    n1_t / max(n_t * world_size, 1e-9) * 100, 1
                )
                # Also compute communication overhead estimate
                n1_bwd = results_n1["bwd_gpu_ms"]["mean"]
                n_bwd = result["bwd_gpu_ms"]["mean"]
                result["comm_overhead_ms_est"] = round(max(0, n_bwd - n1_bwd), 3)
                result["comm_overhead_note"] = (
                    "bwd(N_GPU) - bwd(N=1): includes allreduce. "
                    "Local backward may differ across ranks due to camera differences."
                )
                json.dump(result, open(out_path, "w"), indent=2)
                print(f"  Speedup vs N=1: {result['speedup_vs_baseline']}x, "
                      f"efficiency: {result['scaling_efficiency_pct']}%", flush=True)
        except FileNotFoundError:
            pass

        gc.collect()
        torch.cuda.empty_cache()
        # Brief pause to let NCCL port release
        time.sleep(2)

    print("\n=== C31-A DDP runs complete ===", flush=True)


if __name__ == "__main__":
    main()
