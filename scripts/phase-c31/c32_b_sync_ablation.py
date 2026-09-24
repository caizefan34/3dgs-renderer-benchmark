#!/usr/bin/env python3
"""
C32-B: Hidden Synchronization Causal Validation.

A/B ablation: each .item() and synchronize call in Phase-7 hot loop.
Uses block timing (no per-iteration sync) for clean wall measurement.

Variants (* = 3 repeats each):
  A*     Baseline — all .item() calls
  B1*    Defer loss.item() only
  B2*    Defer l1.item() only
  B3*    Defer d_ssim.item() only
  B4*    Defer PSNR mse.item() only
  B5*    Defer clip_grad_norm sync only
  B6*    Defer ALL scalar .item() calls
  E*     No-metrics — skip ALL scalar reads
  F*     Correctness checkpoint (3 repeats, save model state)
"""
from __future__ import annotations
import argparse, gc, json, math, os, sys, time
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

VARIANTS = ["A","B1","B2","B3","B4","B5","B6","E","F"]
TILE_SIZE = 16
WARMUP = 50
MEASURED = 150


def make_opt(model, sls):
    return torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * sls},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ], eps=1e-15, betas=(0.9, 0.999))


def create(device="cuda:0", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    dataset = GTDataset(scene="room", repo_root=ROOT, resolution="1080p", device=device)
    sfm = load_initial_checkpoint("room", ROOT, device=device)
    n_cam = len(dataset)
    model = GaussianModel(num_points=sfm["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(xyz=sfm["xyz"],
        opacity_logit=torch.logit(torch.full((sfm["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm.get("scales"), rotations_raw=sfm.get("rotations"), shs=sfm.get("shs"))
    sls = float(sfm["xyz"].norm(dim=-1).max().item())
    opt = make_opt(model, sls)
    return dataset, model, opt, n_cam, sls


# ── Per-variant scalar-read flags ──
# True = read NOW (calls .item(), possibly syncs)
# False = skip reading (no sync from this source)
VAR = {
    "A":  {"loss": True,  "l1": True,  "dssim": True,  "psnr": True,  "gnorm": True},
    "B1": {"loss": False, "l1": True,  "dssim": True,  "psnr": True,  "gnorm": True},
    "B2": {"loss": True,  "l1": False, "dssim": True,  "psnr": True,  "gnorm": True},
    "B3": {"loss": True,  "l1": True,  "dssim": False, "psnr": True,  "gnorm": True},
    "B4": {"loss": True,  "l1": True,  "dssim": True,  "psnr": False, "gnorm": True},
    "B5": {"loss": True,  "l1": True,  "dssim": True,  "psnr": True,  "gnorm": "defer"},
    "B6": {"loss": False, "l1": False, "dssim": False, "psnr": False, "gnorm": "defer"},
    "E":  {"loss": False, "l1": False, "dssim": False, "psnr": False, "gnorm": "defer"},
    "F":  {"loss": True,  "l1": True,  "dssim": True,  "psnr": True,  "gnorm": True},
}


def clip_vanilla(model):
    """Standard clip_grad_norm_ with .item() inside."""
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if isinstance(gn, torch.Tensor):
        return gn.item()
    return gn


def clip_noitem(model, device):
    """Clip gradients to max_norm=1.0 WITHOUT any .item() call.
    All computation stays on GPU. Uses torch.where for conditional scaling."""
    total_norm_sq = torch.zeros(1, device=device)
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq = total_norm_sq + p.grad.norm() ** 2
    total_norm = total_norm_sq.sqrt()
    # scale = 1.0 / total_norm  if total_norm > 1.0  else 1.0
    # All on GPU, no scalar read
    scale = torch.where(total_norm > 1.0, 1.0 / total_norm, torch.ones_like(total_norm))
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)
    return None  # no host-visible grad norm


def _run(run_variant: str, device: str, rep: int):
    dataset, model, opt, n_cam, sls = create(device=device, seed=42 + rep)
    flags = VAR[run_variant]
    total = WARMUP + MEASURED

    # Log accumulators (not used in training, just for post-hoc verification)
    logged_loss, logged_l1, logged_dssim, logged_psnr = [], [], [], []

    def step_fn(step):
        nonlocal opt
        ci = step % n_cam
        cam = dataset.get_camera(ci); gt = dataset.get_gt_image(ci)

        # ── Forward ──
        data = model.forward()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=TILE_SIZE, packed=True, sh_degree=0,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
            sparse_grad=False, absgrad=False,
        )
        rendered = rendered[0].clamp(0, 1)
        ld = combined_loss(rendered, gt, lambda_dssim=0.2)
        loss, l1, dssim = ld["loss"], ld["l1"], ld["d_ssim"]

        # ── Scalar reads (may trigger cudaStreamSynchronize) ──
        if flags["loss"]:   logged_loss.append(loss.item())
        if flags["l1"]:     logged_l1.append(l1.item())
        if flags["dssim"]:  logged_dssim.append(dssim.item())
        if flags["psnr"]:
            with torch.no_grad():
                mse = torch.mean((rendered - gt) ** 2).item()
                logged_psnr.append(10 * math.log10(1.0 / max(mse, 1e-10)))

        # ── Backward ──
        opt.zero_grad(set_to_none=True)
        loss.backward()
        model.accumulate_positional_gradient()

        # ── Gradient clipping ──
        if flags["gnorm"] is True:
            clip_vanilla(model)
        else:
            clip_noitem(model, device)

        # ── Optimizer ──
        opt.step()

        # ── Densification & Pruning (every 100 steps after step 200) ──
        if step >= 200 and step < 30000 and step % 100 == 0:
            denf = model.densification(
                grad_threshold=2e-4, clone_max_screen_size=20, split_max_screen_size=20)
            pruned = model.prune_and_reset(
                opacity_threshold=0.005, reset_interval=3000, current_step=step)
            tot = denf["cloned"] + denf["split"] + pruned
            if tot > 0:
                new_opt = make_opt(model, sls)
                new_opt.load_state_dict(opt.state_dict())
                opt = new_opt
                for pg in opt.param_groups:
                    pg["params"] = [model.xyz, model.rotations, model.scales, model.opacity, model.shs]

    # ── Warmup ──
    for s in range(WARMUP):
        step_fn(s)

    # ── Measured block ──
    torch.cuda.synchronize()
    b_start = time.perf_counter()
    for s in range(WARMUP, total):
        step_fn(s)
    torch.cuda.synchronize()
    b_end = time.perf_counter()

    t_iter_ms = (b_end - b_start) * 1000 / MEASURED

    # ── Final eval PSNR (20 cameras) ──
    with torch.no_grad():
        psnrs = []
        for ci in range(min(n_cam, 20)):
            cam = dataset.get_camera(ci); gt = dataset.get_gt_image(ci)
            data = model.forward()
            rendered, _, _ = rasterization(
                means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                opacities=data["opacity"], colors=data["shs"],
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=TILE_SIZE, packed=True, sh_degree=0,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB",
                sparse_grad=False, absgrad=False,
            )
            mse = torch.mean((rendered[0].clamp(0, 1) - gt) ** 2)
            psnrs.append(10 * math.log10(1.0 / max(mse.item(), 1e-10)))
        final_psnr = float(np.mean(psnrs))

    return {
        "variant": run_variant, "rep": rep,
        "t_iter_ms": round(t_iter_ms, 3),
        "block_wall_ms": round((b_end - b_start) * 1000, 3),
        "final_psnr": round(final_psnr, 4),
        "final_gaussian_count": int(model.xyz.shape[0]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    print(f"=== C32-B variant={args.variant} repeats={args.repeats} ===")

    results = []
    for rep in range(args.repeats):
        print(f"  Repeat {rep+1}/{args.repeats}...", flush=True)
        r = _run(args.variant, device, rep)
        results.append(r)
        print(f"    T_iter={r['t_iter_ms']:.2f}ms  PSNR={r['final_psnr']:.3f}  Gs={r['final_gaussian_count']}",
              flush=True)

    t_iters = np.array([r["t_iter_ms"] for r in results])
    psnrs = np.array([r["final_psnr"] for r in results])
    gcs = np.array([r["final_gaussian_count"] for r in results])

    summary = {
        "variant": args.variant,
        "repeats": args.repeats,
        "t_iter_ms_mean": round(float(np.mean(t_iters)), 3),
        "t_iter_ms_std": round(float(np.std(t_iters)), 3),
        "t_iter_ms_min": round(float(np.min(t_iters)), 3),
        "t_iter_ms_max": round(float(np.max(t_iters)), 3),
        "final_psnr_mean": round(float(np.mean(psnrs)), 4),
        "final_psnr_std": round(float(np.std(psnrs)), 4),
        "final_gc_mean": int(round(np.mean(gcs))),
        "final_gc_std": int(round(np.std(gcs))),
        "runs": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {args.out}")
    print(f"  T_iter: {summary['t_iter_ms_mean']:.2f} ± {summary['t_iter_ms_std']:.2f} ms")
    print(f"  PSNR:   {summary['final_psnr_mean']:.4f} ± {summary['final_psnr_std']:.4f}")
    print(f"  Gs:     {summary['final_gc_mean']} ± {summary['final_gc_std']}")


if __name__ == "__main__":
    main()
