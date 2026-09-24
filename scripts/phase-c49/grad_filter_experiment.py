#!/usr/bin/env python3
"""
Phase C49 Experiment 2: Gradient Filtering Quality Validation (Track C Phase 1).

Tests whether zeroing gradients for low-contribution Gaussians hurts training quality.
The backward kernel still runs fully — this only tests the QUALITY IMPACT of gradient
filtering, not the speed benefit (which would require CUDA modification).

Configurations:
- baseline: No filtering (all gradients kept)
- top50: Keep top 50% of Gaussians by gradient norm (→ 97% of gradient)
- top32: Keep top 32% (→ 90% of gradient)
- top10: Keep top 10% (→ 60% of gradient)

If top32 maintains PSNR within 0.5 dB of baseline, Phase 2 (CUDA sparse backward) is justified.
"""
import json, math, sys, time, argparse, numpy as np
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/src")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
from loss import d_ssim_loss

DEVICE = "cuda"
SEED = 42
EVAL_INTERVAL = 1000
EVAL_CAMERAS = list(range(0, 311, 24))[:13]
PRUNE_THRESHOLD = 0.01
GRAD_THRESHOLD = 0.001


class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0,1,3,2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred**2, target**2, pred*target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p*mu_t
        sp2, st2, spt = bp2-mu_p2, bt2-mu_t2, bpt-mu_pt
        ssim_map = (2*mu_pt+self.C1)*(2*spt+self.C2) / ((mu_p2+mu_t2+self.C1)*(sp2+st2+self.C2))
        return 1.0 - ssim_map.mean()


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0


def render(model, cam, data=None):
    if data is None:
        data = model.forward()
    r, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    return r[0].clamp(0, 1)


def train_with_gradient_filtering(dataset, sfm_data, iters, filter_fraction):
    """Train with gradient filtering — zero gradients for bottom filter_fraction of Gaussians."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    n_gauss = sfm_data["xyz"].shape[0]
    model = GaussianModel(num_points=n_gauss, sh_degree=0, max_sh_degree=3, device=DEVICE)
    model.init_from_sfm(
        xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((n_gauss, 1), 0.1, device=DEVICE)),
        scales_log=sfm_data.get("scales"),
        rotations_raw=sfm_data.get("rotations"),
        shs=sfm_data.get("shs"))
    model.set_sh_degree(0)

    lr_params = [
        {"params": [model.xyz], "lr": 1.6e-4, "name": "xyz"},
        {"params": [model.rotations], "lr": 1e-3, "name": "rotations"},
        {"params": [model.scales], "lr": 5e-3, "name": "scales"},
        {"params": [model.opacity], "lr": 5e-2, "name": "opacity"},
        {"params": [model.shs], "lr": 2.5e-3, "name": "shs"},
    ]
    optimizer = torch.optim.Adam(lr_params, eps=1e-15)

    n_cams = len(dataset)
    cam_indices = list(range(n_cams))
    np.random.shuffle(cam_indices)

    sep_ssim = SepSSIM(device=DEVICE)

    results = {"eval_points": [], "gs_trajectory": [], "total_train_time_s": 0.0,
               "filter_fraction": filter_fraction}
    total_clone, total_split, total_prune = 0, 0, 0
    train_start = time.perf_counter()

    for iter_idx in range(iters):
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        data = model.forward()
        pred = render(model, cam, data)

        l1 = F.l1_loss(pred, gt)
        if iter_idx % 8 == 0:
            loss = 0.8 * l1 + 0.2 * sep_ssim(pred, gt)
        else:
            loss = l1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # === GRADIENT FILTERING ===
        if filter_fraction > 0 and model.xyz.grad is not None:
            grad_norms = model.xyz.grad.norm(dim=-1)  # [N]
            n = grad_norms.shape[0]
            keep_count = max(int(n * (1.0 - filter_fraction)), 1)
            if keep_count < n:
                threshold = torch.topk(grad_norms, keep_count, largest=True).values[-1]
                mask = (grad_norms >= threshold).float()  # [N]
                # Apply mask to ALL parameter gradients with proper shape expansion
                for p in [model.xyz, model.rotations, model.scales, model.opacity, model.shs]:
                    if p.grad is not None:
                        if p.grad.dim() == 1:
                            p.grad.mul_(mask)
                        else:
                            # Expand mask to match gradient shape: [N] -> [N, 1, 1, ...]
                            expand_shape = [n] + [1] * (p.grad.dim() - 1)
                            p.grad.mul_(mask.view(expand_shape))

        # Densification
        if iter_idx >= 500 and iter_idx < 15000 and iter_idx % 100 == 0:
            model.accumulate_positional_gradient()
            counts = model.densification(grad_threshold=GRAD_THRESHOLD)
            total_clone += counts["cloned"]
            total_split += counts["split"]
            removed = model.prune(opacity_threshold=PRUNE_THRESHOLD)
            total_prune += removed

        if iter_idx % 1000 == 0 and iter_idx > 0:
            if model.sh_degree < 3:
                model.set_sh_degree(model.sh_degree + 1)

        if iter_idx > 0 and iter_idx % 3000 == 0:
            removed = model.prune_and_reset(opacity_threshold=PRUNE_THRESHOLD, current_step=iter_idx)
            total_prune += removed

        optimizer.step()

        if iter_idx % EVAL_INTERVAL == 0 or iter_idx == iters - 1:
            model.eval()
            with torch.no_grad():
                data_eval = model.forward()
                psnrs = []
                for ci2 in EVAL_CAMERAS:
                    if ci2 >= len(dataset): break
                    cam2 = dataset.get_camera(ci2)
                    gt2 = dataset.get_gt_image(ci2)
                    pred2 = render(model, cam2, data_eval)
                    psnrs.append(compute_psnr(pred2, gt2))
                psnr = float(np.mean(psnrs))
            model.train()
            gs = model.xyz.shape[0]
            results["eval_points"].append({
                "iter": iter_idx, "psnr": psnr, "gaussians": gs,
                "clone": total_clone, "split": total_split, "prune": total_prune,
            })
            print(f"  [{iter_idx:>5}] PSNR={psnr:.2f}  GS={gs:,}", flush=True)

    results["total_train_time_s"] = time.perf_counter() - train_start
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="room")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--filter_fraction", type=float, default=0.0,
                        help="Fraction of Gaussians to filter (0=baseline, 0.5=top50, 0.68=top32, 0.9=top10)")
    parser.add_argument("--save_name", default="baseline")
    args = parser.parse_args()

    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Phase C49 Experiment 2: Gradient Filtering")
    print(f"  Scene: {args.scene}, Iters: {args.iters}")
    print(f"  Filter fraction: {args.filter_fraction} (keep top {(1-args.filter_fraction)*100:.0f}%)")

    dataset = GTDataset(scene=args.scene, repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint(args.scene, repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    results = train_with_gradient_filtering(dataset, sfm_data, args.iters, args.filter_fraction)

    final = results["eval_points"][-1]
    print(f"\nSUMMARY: {args.save_name}")
    print(f"  Total: {results['total_train_time_s']:.1f}s  PSNR: {final['psnr']:.2f}  GS: {final['gaussians']:,}")

    save_path = repo_root / "results" / "a100" / "phase-c49" / f"grad_filter_{args.save_name}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {save_path}")


if __name__ == "__main__":
    main()
