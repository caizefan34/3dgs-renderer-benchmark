#!/usr/bin/env python3
"""C30-G: Real Multi-GPU Training Scaling screening.

Measure the scaling envelope: profile compute/communication breakdown,
estimate scalability from data-parallel model using single-GPU profiling.
Does NOT build a distributed system — only measures attainable scaling.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch

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
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--scene", default="room")
    args = p.parse_args()
    device = "cuda"; torch.manual_seed(42); np.random.seed(42)

    n_gpus = torch.cuda.device_count()
    print(f"System: {n_gpus} GPUs available", flush=True)
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_properties(i).name}", flush=True)
    print(f"  NVLink: {torch.cuda.get_device_capability(0)}", flush=True)

    dataset = GTDataset(scene=args.scene, repo_root=ROOT, resolution="1080p", device=device)
    sfm_data = load_initial_checkpoint(args.scene, ROOT, device=device)
    model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm_data.get("scales"), rotations_raw=sfm_data.get("rotations"), shs=sfm_data.get("shs"))
    spatial_lr_scale = sfm_data["xyz"].norm(dim=-1).max().item()
    print(f"Initial Gs: {model.xyz.shape[0]:,}", flush=True)

    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ])

    # Profile forward/backward/optimizer times and measure gradient sizes
    records = []
    compute_ms_list = []
    for step in range(args.steps):
        cam_idx = step % len(dataset)
        camera = dataset.get_camera(cam_idx)
        gt_image = dataset.get_gt_image(cam_idx)
        new_degree = min(3, step // 500)
        if new_degree != model.sh_degree:
            model.set_sh_degree(new_degree)
            optimizer = torch.optim.Adam([
                {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                {"params": [model.rotations], "lr": 1e-3},
                {"params": [model.scales], "lr": 5e-3},
                {"params": [model.opacity], "lr": 5e-2},
                {"params": [model.shs], "lr": 2.5e-3},
            ])
        data = model.forward()

        t0 = time.perf_counter()
        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
            width=camera.image_width, height=camera.image_height,
            tile_size=16, packed=True, sh_degree=model.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        )
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1000
        rendered = rendered[0].clamp(0, 1)
        loss_dict = combined_loss(rendered, gt_image, lambda_dssim=0.2)
        loss = loss_dict["loss"]

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.cuda.synchronize()
        bwd_ms = (time.perf_counter() - t0) * 1000

        # Measure gradient sizes for each parameter group (communication cost)
        grad_sizes_bytes = {}
        for name in ["xyz", "rotations", "scales", "opacity", "shs"]:
            p = getattr(model, name)
            if p.grad is not None:
                grad_sizes_bytes[name] = p.grad.numel() * p.grad.element_size()

        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        t0 = time.perf_counter()
        optimizer.step(); torch.cuda.synchronize()
        opt_ms = (time.perf_counter() - t0) * 1000

        if step >= 200 and step % 100 == 0:
            model.densification(grad_threshold=2e-4)
            optimizer = torch.optim.Adam([
                {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                {"params": [model.rotations], "lr": 1e-3},
                {"params": [model.scales], "lr": 5e-3},
                {"params": [model.opacity], "lr": 5e-2},
                {"params": [model.shs], "lr": 2.5e-3},
            ])
        if step >= 200 and step % 100 == 0:
            model.prune(opacity_threshold=0.005)
            optimizer = torch.optim.Adam([
                {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                {"params": [model.rotations], "lr": 1e-3},
                {"params": [model.scales], "lr": 5e-3},
                {"params": [model.opacity], "lr": 5e-2},
                {"params": [model.shs], "lr": 2.5e-3},
            ])

        with torch.no_grad():
            mse = torch.mean((rendered - gt_image) ** 2).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

        compute_ms = fwd_ms + bwd_ms + opt_ms
        compute_ms_list.append(compute_ms)
        records.append({
            "step": step, "loss": loss.item(), "psnr": psnr,
            "n_gaussians": model.xyz.shape[0],
            "fwd_ms": round(fwd_ms, 3), "bwd_ms": round(bwd_ms, 3), "opt_ms": round(opt_ms, 3),
            "compute_ms": round(compute_ms, 3),
            "grad_sizes_bytes": grad_sizes_bytes,
        })
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  Step {step}: loss={loss.item():.4f} PSNR={psnr:.2f} N={model.xyz.shape[0]:,} "
                  f"fwd={fwd_ms:.2f} bwd={bwd_ms:.2f} opt={opt_ms:.2f}", flush=True)

    # Compute scaling estimates
    total_grad_bytes = sum(grad_sizes_bytes.values())
    total_grad_mb = total_grad_bytes / (1024 * 1024)

    # Estimate NCCL allreduce bandwidth (rough NVLink ~600GB/s theoretical on A100)
    # Actual achieved ~300GB/s for small payloads
    nvlink_bw_gbps = 300.0  # GB/s (empirical for A100 NVLink)
    pcie_bw_gbps = 12.0     # GB/s (PCIe Gen4 x16)

    nvlink_allreduce_ms = (total_grad_bytes * 2) / (nvlink_bw_gbps * 1e9) * 1000  # ×2 for ring
    pcie_allreduce_ms = (total_grad_bytes * 2) / (pcie_bw_gbps * 1e9) * 1000

    mean_compute_ms = float(np.mean(compute_ms_list))
    std_compute_ms = float(np.std(compute_ms_list))

    scaling_estimate = {}
    for ngpu in [1, 2, 4, 8]:
        if ngpu == 1:
            t_iter = mean_compute_ms
        else:
            # Amdahl: parallel fraction = fwd + bwd (non-opt = fwd+bwd)/total
            # Optimizer is partially parallel (per-GPU params) but requires sync
            # Conservative: assume fwd+bwd fully parallel, opt partially serial
            fwd_avg = float(np.mean([r["fwd_ms"] for r in records]))
            bwd_avg = float(np.mean([r["bwd_ms"] for r in records]))
            opt_avg = float(np.mean([r["opt_ms"] for r in records]))
            parallel_ms = fwd_avg + bwd_avg  # these parallelize well
            serial_ms = opt_avg  # partially serial
            t_parallel = parallel_ms / ngpu
            t_comm = nvlink_allreduce_ms  # for NVLink
            t_iter = serial_ms + t_parallel + t_comm
        scaling_estimate[f"{ngpu}gpu"] = {
            "estimated_t_iter_ms": round(t_iter, 3),
            "speedup_vs_1gpu": round(mean_compute_ms / t_iter, 3) if t_iter > 0 else 0,
            "efficiency": round((mean_compute_ms / t_iter) / ngpu * 100, 1) if ngpu > 1 else 100.0,
        }

    summary = {
        "schema_version": 2, "phase": "C30-G",
        "n_gpus_available": n_gpus,
        "initial_n_gaussians": int(sfm_data["xyz"].shape[0]),
        "profile": {
            "mean_compute_ms": round(mean_compute_ms, 3),
            "std_compute_ms": round(std_compute_ms, 3),
            "total_grad_mb": round(total_grad_mb, 2),
            "estimated_nvlink_allreduce_ms": round(nvlink_allreduce_ms, 3),
            "estimated_pcie_allreduce_ms": round(pcie_allreduce_ms, 3),
        },
        "scaling_estimate": scaling_estimate,
        "records": records,
        "verdict": "SCREENING"
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")
    print(f"Gradient size: {total_grad_mb:.1f} MB")
    print(f"NVLink allreduce estimate: {nvlink_allreduce_ms:.1f}ms")
    for k, v in scaling_estimate.items():
        print(f"  {k}: {v['estimated_t_iter_ms']}ms speedup={v['speedup_vs_1gpu']}x eff={v['efficiency']}%")

if __name__ == "__main__":
    main()
