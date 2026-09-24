#!/usr/bin/env python3
"""
Phase C45 unified experiment for Tracks B, C, D.

Track B: Adaptive SSIM scheduling
  B1: dynamic frequency (early freq2, mid freq8, late freq16)
  B2: gradient-aware SSIM (SSIM when gradient norm is high)
  B3: PSNR-aware (SSIM when PSNR improvement stalls)

Track C: Alternative perceptual losses
  C1: Laplacian pyramid loss
  C2: FFT frequency loss
  C3: edge-aware loss

Track D: Multi-scene 30K validation
  baseline + best C44 method (separable SSIM + freq8)

All use separable SSIM from C44.
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

# Pruning configs
PRUNE_CONFIGS = {
    "aggressive": {"prune_threshold": 0.05, "grad_threshold": 0.002, "densify_end_frac": 1.0},
    "standard":   {"prune_threshold": 0.005, "grad_threshold": 0.0002, "densify_end_frac": 0.5},
    "moderate":   {"prune_threshold": 0.01, "grad_threshold": 0.001, "densify_end_frac": 0.5},
}

# ===== Separable SSIM =====
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

# ===== Alternative losses =====
class LaplacianPyramidLoss:
    """C1: Laplacian pyramid loss — multi-scale detail preservation."""
    def __init__(self, n_levels=3, device="cuda"):
        self.n_levels = n_levels
        k = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=device)
        k = k / k.sum()
        k2d = k[:, None] * k[None, :]
        self.kernel = k2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 5, 5).contiguous()
        self.padding = 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        loss = 0.0
        p, t = pred, target
        for i in range(self.n_levels):
            p_blur = F.conv2d(p, self.kernel, padding=self.padding, groups=3)
            t_blur = F.conv2d(t, self.kernel, padding=self.padding, groups=3)
            # Ensure same size before computing Laplacian
            min_h = min(p.shape[2], p_blur.shape[2])
            min_w = min(p.shape[3], p_blur.shape[3])
            loss = loss + F.l1_loss(p[:, :, :min_h, :min_w] - p_blur[:, :, :min_h, :min_w],
                                     t[:, :, :min_h, :min_w] - t_blur[:, :, :min_h, :min_w])
            # Downsample (use floor division to ensure even dims)
            h, w = p_blur.shape[2] // 2, p_blur.shape[3] // 2
            if h < 4 or w < 4: break
            p = F.interpolate(p_blur, size=(h, w), mode='area')
            t = F.interpolate(t_blur, size=(h, w), mode='area')
        min_h = min(p.shape[2], t.shape[2])
        min_w = min(p.shape[3], t.shape[3])
        loss = loss + F.l1_loss(p[:, :, :min_h, :min_w], t[:, :, :min_h, :min_w])
        return loss / (self.n_levels + 1)

class FFTLoss:
    """C2: FFT frequency loss — preserve frequency content."""
    def __init__(self, device="cuda"):
        pass

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        return F.l1_loss(pred_fft.abs(), target_fft.abs())

class EdgeAwareLoss:
    """C3: Edge-aware loss — weight L1 by edge presence."""
    def __init__(self, device="cuda"):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
        self.kx = sobel_x.unsqueeze(0).unsqueeze(0).repeat(3, 1, 3, 3).contiguous()
        self.ky = sobel_y.unsqueeze(0).unsqueeze(0).repeat(3, 1, 3, 3).contiguous()

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        # Ensure same size
        min_h = min(pred.shape[2], target.shape[2])
        min_w = min(pred.shape[3], target.shape[3])
        pred = pred[:, :, :min_h, :min_w]
        target = target[:, :, :min_h, :min_w]
        # Compute edge maps (padding=1 for 3x3 Sobel should preserve size)
        pred_edges = (F.conv2d(pred, self.kx, padding=1, groups=3)**2 +
                      F.conv2d(pred, self.ky, padding=1, groups=3)**2).sqrt()
        target_edges = (F.conv2d(target, self.kx, padding=1, groups=3)**2 +
                        F.conv2d(target, self.ky, padding=1, groups=3)**2).sqrt()
        # Crop edges to match pred size (in case conv changes size slightly)
        eh = min(pred_edges.shape[2], pred.shape[2])
        ew = min(pred_edges.shape[3], pred.shape[3])
        pred_edges = pred_edges[:, :, :eh, :ew]
        target_edges = target_edges[:, :, :eh, :ew]
        pred_c = pred[:, :, :eh, :ew]
        target_c = target[:, :, :eh, :ew]
        edge_weight = 1.0 + target_edges * 5.0
        return (edge_weight * (pred_c - target_c).abs()).mean()

# ===== Training =====
def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0

def compute_ssim_metric(pred, gt):
    return float(1.0 - d_ssim_loss(pred, gt))

def render(model, cam, data=None):
    if data is None: data = model.forward()
    r, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    return r[0].clamp(0, 1)

def evaluate(model, dataset):
    psnrs, ssims, lpipses = [], [], []
    model.eval()
    with torch.no_grad():
        data = model.forward()
        for ci in EVAL_CAMERAS:
            if ci >= len(dataset): break
            cam = dataset.get_camera(ci)
            gt = dataset.get_gt_image(ci)
            pred = render(model, cam, data)
            psnrs.append(compute_psnr(pred, gt))
            ssims.append(compute_ssim_metric(pred, gt))
            lpipses.append(float(F.l1_loss(pred, gt).item()))
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipses))

def get_freq_schedule(schedule, iter_idx, total_iters):
    """B1: Dynamic frequency schedule."""
    if schedule == "B1_early_mid_late":
        if iter_idx < total_iters * 0.2: return 2
        elif iter_idx < total_iters * 0.7: return 8
        else: return 16
    elif schedule == "B1_decreasing":
        if iter_idx < total_iters * 0.1: return 1
        elif iter_idx < total_iters * 0.3: return 4
        elif iter_idx < total_iters * 0.6: return 8
        else: return 16
    return 1

def train(args, dataset, sfm_data):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    prune_cfg = PRUNE_CONFIGS[args.prune_config]
    prune_threshold = prune_cfg["prune_threshold"]
    grad_threshold = prune_cfg["grad_threshold"]
    densify_end = int(args.iters * prune_cfg["densify_end_frac"])
    print(f"  Prune config: {args.prune_config} (threshold={prune_threshold}, grad={grad_threshold}, densify_end={densify_end})")
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

    # Initialize loss function
    sep_ssim = SepSSIM(device=DEVICE)
    alt_loss = None
    if args.track == "C":
        if args.config == "C1_laplacian":
            alt_loss = LaplacianPyramidLoss(device=DEVICE)
        elif args.config == "C2_fft":
            alt_loss = FFTLoss(device=DEVICE)
        elif args.config == "C3_edge":
            alt_loss = EdgeAwareLoss(device=DEVICE)

    results = {
        "track": args.track, "config": args.config, "scene": args.scene,
        "iters": args.iters,
        "per_iter_times": [], "eval_points": [], "gs_trajectory": [],
        "total_train_time_s": 0.0,
    }
    total_clone, total_split, total_prune = 0, 0, 0
    train_start = time.perf_counter()
    prev_psnr = 0.0
    psnr_history = []

    for iter_idx in range(args.iters):
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        data = model.forward()
        pred = render(model, cam, data)

        # Determine loss
        use_ssim = True
        use_alt = False
        freq = 1

        if args.track == "B":
            if args.config == "B1_early_mid_late" or args.config == "B1_decreasing":
                freq = get_freq_schedule(args.config, iter_idx, args.iters)
                use_ssim = (iter_idx % freq == 0)
            elif args.config == "B2_gradient_aware":
                # B2: Use SSIM when gradient norm is high (check every 10 iters)
                if iter_idx > 0 and iter_idx % 10 == 0:
                    total_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                    use_ssim = total_norm > 0.5  # threshold
                else:
                    use_ssim = True
            elif args.config == "B3_psnr_aware":
                # B3: Use SSIM when PSNR improvement stalls
                if iter_idx > 0 and iter_idx % EVAL_INTERVAL == 0:
                    psnr_history.append(prev_psnr)
                if len(psnr_history) >= 2:
                    improvement = psnr_history[-1] - psnr_history[-2]
                    use_ssim = improvement < 0.1  # use SSIM when improvement is small
                else:
                    use_ssim = True
                freq = 1
        elif args.track == "C":
            use_ssim = False
            use_alt = True
        elif args.track == "D":
            # D: baseline or best C44 method
            if args.config == "D_baseline":
                use_ssim = True  # original SSIM every iter
            elif args.config == "D_sep_ssim":
                use_ssim = True  # separable SSIM every iter
            elif args.config == "D_sep_freq8":
                use_ssim = (iter_idx % 8 == 0)

        # Compute loss
        if iter_idx % 100 == 0:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        l1 = F.l1_loss(pred, gt)
        if use_alt:
            loss = 0.8 * l1 + 0.2 * alt_loss(pred, gt)
        elif use_ssim:
            if args.track == "D" and args.config == "D_baseline":
                loss = 0.8 * l1 + 0.2 * d_ssim_loss(pred, gt)
            else:
                loss = 0.8 * l1 + 0.2 * sep_ssim(pred, gt)
        else:
            loss = l1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Densification
        if iter_idx >= 500 and iter_idx < densify_end and iter_idx % 100 == 0:
            model.accumulate_positional_gradient()
            counts = model.densification(grad_threshold=grad_threshold)
            total_clone += counts["cloned"]
            total_split += counts["split"]
            removed = model.prune(opacity_threshold=prune_threshold)
            total_prune += removed

        if iter_idx % 1000 == 0 and iter_idx > 0:
            if model.sh_degree < 3:
                model.set_sh_degree(model.sh_degree + 1)

        if iter_idx > 0 and iter_idx % 3000 == 0:
            removed = model.prune_and_reset(opacity_threshold=prune_threshold, current_step=iter_idx)
            total_prune += removed

        optimizer.step()

        if iter_idx % 100 == 0:
            torch.cuda.synchronize()
            iter_time = (time.perf_counter() - t0) * 1000
            results["per_iter_times"].append(iter_time)
            results["gs_trajectory"].append({"iter": iter_idx, "n_gaussians": model.xyz.shape[0]})

        if iter_idx % EVAL_INTERVAL == 0 or iter_idx == args.iters - 1:
            psnr, ssim_v, lpips_v = evaluate(model, dataset)
            prev_psnr = psnr
            gs_count = model.xyz.shape[0]
            results["eval_points"].append({
                "iter": iter_idx, "psnr": psnr, "ssim": ssim_v, "lpips": lpips_v,
                "gaussians": gs_count,
                "clone": total_clone, "split": total_split, "prune": total_prune,
            })
            print(f"  [{args.track}/{args.config}@{iter_idx}] PSNR={psnr:.2f} SSIM={ssim_v:.4f} GS={gs_count:,}")

    results["total_train_time_s"] = time.perf_counter() - train_start
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True, choices=["B", "C", "D"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", default="room")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--prune_config", default="aggressive", choices=["aggressive", "standard", "moderate"])
    parser.add_argument("--save_suffix", default="", help="Suffix to append to save filename")
    args = parser.parse_args()

    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Phase C45: track={args.track} config={args.config} scene={args.scene} iters={args.iters}")

    dataset = GTDataset(scene=args.scene, repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint(args.scene, repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    results = train(args, dataset, sfm_data)
    final = results["eval_points"][-1]
    total_time = results["total_train_time_s"]

    print(f"\nSUMMARY: {args.track}/{args.config}")
    print(f"  Total: {total_time:.1f}s  PSNR: {final['psnr']:.2f}  SSIM: {final['ssim']:.4f}  GS: {final['gaussians']:,}")

    config_str = args.config.replace(".", "")
    suffix = f"_{args.save_suffix}" if args.save_suffix else ""
    save_path = repo_root / "results" / "a100" / "phase-c45" / f"{args.track}_{config_str}_{args.scene}{suffix}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {save_path}")

if __name__ == "__main__":
    main()
