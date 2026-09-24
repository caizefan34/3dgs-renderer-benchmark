#!/usr/bin/env python3
"""Phase C51 Stage 4B — Recall@50 and mask overlap measurement.

Runs a short training sequence and records actual mask indices to compute:
  - Recall@50: fraction of top-50% current-grad Gaussians that are in top-50% prev-grad
  - Mask overlap: Jaccard similarity between consecutive masks
  - Mask churn: fraction of mask that changes between iterations
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/src")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
from loss import d_ssim_loss

SEED = 42
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


def render(model, cam, data=None, importance_mask=None, compute_densify_grad=False):
    if data is None:
        data = model.forward()
    r, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        importance_mask=importance_mask,
        compute_densify_grad=compute_densify_grad,
    )
    return r[0].clamp(0, 1)


def main():
    device = "cuda:0"
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=device)
    sfm_data = load_initial_checkpoint("room", repo_root, device=device)

    N = sfm_data['xyz'].shape[0]
    model = GaussianModel(num_points=N, sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(
        xyz=sfm_data['xyz'].clone().float().to(device),
        opacity_logit=torch.logit(torch.full((N,), 0.1, device=device)),
        scales_log=sfm_data['scales'].clone().float().to(device),
        rotations_raw=sfm_data['rotations'].clone().float().to(device),
        shs=sfm_data['shs'].clone().float().to(device),
    )
    model.set_sh_degree(0)

    lr_params = [
        {"params": [model.xyz], "lr": 1.6e-4, "name": "xyz"},
        {"params": [model.rotations], "lr": 1e-3, "name": "rotations"},
        {"params": [model.scales], "lr": 5e-3, "name": "scales"},
        {"params": [model.opacity], "lr": 5e-2, "name": "opacity"},
        {"params": [model.shs], "lr": 2.5e-3, "name": "shs"},
    ]
    optimizer = torch.optim.Adam(lr_params, eps=1e-15)
    sep_ssim = SepSSIM(device=device)

    n_cams = len(dataset)
    cam_indices = list(range(n_cams))
    np.random.shuffle(cam_indices)

    k_frac = 0.5
    n_keep = max(1, int(N * k_frac))
    prev_grad_norm = None
    prev_mask_indices = None

    # Run 500 iterations and record mask overlap stats
    overlaps = []
    recalls = []
    churns = []
    recording_start = 100  # start recording after warmup

    for iter_idx in range(500):
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)

        # Construct mask from previous gradient
        importance_mask = None
        if prev_grad_norm is not None:
            _, top_idx = torch.topk(prev_grad_norm, min(n_keep, len(prev_grad_norm)))
            importance_mask = torch.zeros(len(prev_grad_norm), dtype=torch.uint8, device=device)
            importance_mask[top_idx] = 1

        # Forward + backward (with mask if available)
        data = model.forward()
        pred = render(model, cam, data, importance_mask=importance_mask)
        l1 = F.l1_loss(pred, gt)
        dssim = sep_ssim(pred, gt)
        loss = 0.8 * l1 + 0.2 * dssim
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Record mask stats
        if iter_idx >= recording_start and importance_mask is not None and model.xyz.grad is not None:
            current_grad_norm = model.xyz.grad.detach().norm(dim=-1)
            _, current_top_idx = torch.topk(current_grad_norm, min(n_keep, len(current_grad_norm)))

            # Recall@50: fraction of current top-50% that are in prev top-50%
            prev_set = set(top_idx.cpu().numpy().tolist())
            curr_set = set(current_top_idx.cpu().numpy().tolist())
            recall = len(prev_set & curr_set) / len(curr_set) if len(curr_set) > 0 else 0
            recalls.append(recall)

            # Mask overlap (Jaccard)
            overlap = len(prev_set & curr_set) / len(prev_set | curr_set) if len(prev_set | curr_set) > 0 else 0
            overlaps.append(overlap)

            # Mask churn: fraction of mask that changed
            churn = 1.0 - recall  # fraction of current top-k not in prev top-k
            churns.append(churn)

        # Store gradient norm for next iteration
        if model.xyz.grad is not None:
            current_grad_norm = model.xyz.grad.detach().norm(dim=-1)
            eps = 1e-6
            current_grad_norm = current_grad_norm + eps
            if prev_grad_norm is not None and prev_grad_norm.shape[0] == model.xyz.shape[0]:
                ema_decay = 0.9
                prev_grad_norm = ema_decay * prev_grad_norm + (1 - ema_decay) * current_grad_norm
            else:
                prev_grad_norm = current_grad_norm.clone()

        # Densification
        if iter_idx >= 500 and iter_idx < 15000 and iter_idx % 100 == 0:
            model.accumulate_positional_gradient()
            model.densification(grad_threshold=GRAD_THRESHOLD)
            model.prune(opacity_threshold=PRUNE_THRESHOLD)
            optimizer = torch.optim.Adam([
                {"params": [model.xyz], "lr": 1.6e-4},
                {"params": [model.rotations], "lr": 1e-3},
                {"params": [model.scales], "lr": 5e-3},
                {"params": [model.opacity], "lr": 5e-2},
                {"params": [model.shs], "lr": 2.5e-3},
            ], eps=1e-15)
            prev_grad_norm = None

        optimizer.step()

        if iter_idx % 100 == 0:
            r_str = f"recall={np.mean(recalls[-20:]):.3f}" if recalls else "N/A"
            print(f"  Iter {iter_idx}: GS={model.xyz.shape[0]:,}, {r_str}")

    print(f"\n=== Predictor Validation Results ({len(recalls)} samples) ===")
    print(f"Recall@50:     mean={np.mean(recalls):.4f}, std={np.std(recalls):.4f}")
    print(f"               min={np.min(recalls):.4f}, max={np.max(recalls):.4f}")
    print(f"Mask overlap:  mean={np.mean(overlaps):.4f}, std={np.std(overlaps):.4f}")
    print(f"Mask churn:    mean={np.mean(churns):.4f}, std={np.std(churns):.4f}")
    print(f"\nInterpretation:")
    print(f"  Recall@50 = {np.mean(recalls):.1%} of current top-50% Gaussians were in prev top-50%")
    print(f"  Mask churn = {np.mean(churns):.1%} of the mask changes per iteration")
    print(f"  Jaccard overlap = {np.mean(overlaps):.1%}")

    # Save results
    results = {
        "recall_at_50": {"mean": float(np.mean(recalls)), "std": float(np.std(recalls)),
                         "min": float(np.min(recalls)), "max": float(np.max(recalls))},
        "mask_overlap": {"mean": float(np.mean(overlaps)), "std": float(np.std(overlaps))},
        "mask_churn": {"mean": float(np.mean(churns)), "std": float(np.std(churns))},
        "n_samples": len(recalls),
    }
    save_path = repo_root / "results" / "a100" / "phase-c51-stage4b" / "predictor_validation.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
