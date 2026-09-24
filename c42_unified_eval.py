#!/usr/bin/env python3
"""
C42 Unified Metric Evaluation — evaluates all 30K checkpoints with identical pipeline.
Uses baseline/reference_v1/gaussian_model.py (correct model for canonical checkpoints).
"""
import argparse, json, math, os, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
REFERENCE_V1_DIR = REPO_ROOT / "baseline" / "reference_v1"
sys.path.insert(0, str(REFERENCE_V1_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))
from gaussian_model import GaussianModel
from gsplat import rasterization
sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))
from dataset import GTDataset


class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + self.C1) * (2 * spt + self.C2) / \
                   ((mu_p2 + mu_t2 + self.C1) * (sp2 + st2 + self.C2))
        return 1.0 - ssim_map.mean()


def render(model, cam, sh_degree):
    data = {
        "xyz": model.get_xyz, "rotations": model.get_rotation,
        "scales": model.get_scaling, "opacity": model.get_opacity,
        "shs": model.get_features,
    }
    r, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
    )
    return r[0].clamp(0, 1)


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0


def evaluate_checkpoint(ckpt_path, scene, lpips_fn, ssim_fn):
    """Load checkpoint and evaluate PSNR/SSIM/LPIPS on ALL cameras."""
    print(f"\n  Evaluating: {Path(ckpt_path).name} (scene={scene})")

    dataset = GTDataset(scene=scene, repo_root=str(REPO_ROOT), resolution="1080p", device="cuda", background="black")
    n_cams = len(dataset)
    print(f"    {n_cams} cameras")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = GaussianModel(max_sh_degree=3)
    model.restore(ckpt, config={
        "position_lr_init": 0.00016, "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01, "position_lr_max_steps": 30000,
        "feature_lr": 0.0025, "opacity_lr": 0.025, "scaling_lr": 0.005,
        "rotation_lr": 0.001, "percent_dense": 0.01,
    })

    psnrs, ssims, lpips_vals = [], [], []
    for ci in range(n_cams):
        cam, gt = dataset.get_item(ci)
        with torch.no_grad():
            img = render(model, cam, model.active_sh_degree)
        psnrs.append(compute_psnr(img, gt))
        ssims.append(float(1.0 - ssim_fn(img, gt).item()))
        if lpips_fn is not None:
            pred_lp = img.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
            gt_lp = gt.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
            lpips_vals.append(float(lpips_fn(pred_lp, gt_lp).item()))
        if ci % 50 == 0:
            print(f"    cam {ci}/{n_cams}...", flush=True)

    result = {
        "scene": scene, "checkpoint": str(ckpt_path),
        "n_gaussians": model._xyz.shape[0], "n_cameras": n_cams,
        "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
    }
    if lpips_vals:
        result["lpips"] = float(np.mean(lpips_vals))
    lpips_str = f" LPIPS={result.get('lpips','N/A')}"
    if isinstance(result.get('lpips'), float):
        lpips_str = f" LPIPS={result['lpips']:.4f}"
    print(f"    PSNR={result['psnr']:.4f} SSIM={result['ssim']:.4f}{lpips_str} N={result['n_gaussians']:,}")

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--output", default="results/c42_adaptive/completion_batch/unified_metrics.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("=" * 72)
    print("C42 Unified Metric Evaluation (REFERENCE_V1 model)")
    print("=" * 72)

    import lpips
    lpips_fn = lpips.LPIPS(net="vgg").to("cuda")
    lpips_fn.eval()
    ssim_fn = SepSSIM(device="cuda")

    repo_root = REPO_ROOT
    rr = str(repo_root)

    # Checkpoint registry
    ckpt_dirs = {
        "room": {
            "1.0": repo_root / "results" / "reference_v1" / "room_30k" / "checkpoints" / "iter_30000.pt",
            "0.75": repo_root / "results" / "c42_adaptive" / "completion_batch" / "checkpoints" / "room_s0.75_iter_30000.pt",
            "0.5": repo_root / "results" / "reference_v1" / "c42_30k" / "checkpoints" / "iter_30000.pt",
        },
        "garden": {
            "1.0": repo_root / "results" / "reference_v1" / "s22" / "garden" / "checkpoints" / "baseline_iter_30000.pt",
            "0.75": repo_root / "results" / "reference_v1" / "s23" / "garden" / "checkpoints" / "s0.750_iter_30000.pt",
            "0.625": repo_root / "results" / "c42_adaptive" / "completion_batch" / "checkpoints" / "garden_s0.625_iter_30000.pt",
            "0.5": repo_root / "results" / "reference_v1" / "s22" / "garden" / "checkpoints" / "c42_iter_30000.pt",
        },
        "bicycle": {
            "1.0": repo_root / "results" / "reference_v1" / "s22" / "bicycle" / "checkpoints" / "baseline_iter_30000.pt",
            "0.75": repo_root / "results" / "c42_adaptive" / "completion_batch" / "checkpoints" / "bicycle_s0.75_iter_30000.pt",
            "0.625": repo_root / "results" / "reference_v1" / "s23" / "bicycle" / "checkpoints" / "s0.625_iter_30000.pt",
            "0.5": repo_root / "results" / "reference_v1" / "s22" / "bicycle" / "checkpoints" / "c42_iter_30000.pt",
        },
    }

    results = {"experiment": "C42 Unified Metric Evaluation", "checkpoints": {}}

    for scene, scales in ckpt_dirs.items():
        for scale, path in scales.items():
            key = f"{scene}_s{scale}"
            if path is None or not path.exists():
                results["checkpoints"][key] = {"status": "NOT_FOUND", "path": str(path), "scene": scene, "scale": scale}
                print(f"\n  NOT FOUND: {key} — {path}")
                continue
            try:
                result = evaluate_checkpoint(path, scene, lpips_fn, ssim_fn)
                result["scale"] = scale
                result["status"] = "EVALUATED"
                results["checkpoints"][key] = result
            except Exception as e:
                results["checkpoints"][key] = {"status": "ERROR", "error": str(e), "scene": scene, "scale": scale}
                print(f"\n  ERROR: {key} — {e}")

    # Provenance
    import gsplat
    results["provenance"] = {
        "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "gsplat": gsplat.__version__,
        "lpips": getattr(lpips, "__version__", "0.1.4"),
        "gpu": torch.cuda.get_device_name(0),
        "evaluation_code": "unified — same PSNR/SSIM/LPIPS pipeline for all checkpoints",
        "ssim_implementation": "SepSSIM window=11 sigma=1.5 C1=(0.01)^2 C2=(0.03)^2",
        "lpips_net": "vgg", "all_cameras": True,
        "gaussian_model": "baseline/reference_v1/gaussian_model.py",
    }

    # Discrepancy check with canonical values
    canonical = {
        "room_s1.0": {"psnr": 32.30, "ssim": 0.9263},
        "room_s0.5": {"psnr": 32.54, "ssim": 0.9185},
        "garden_s1.0": {"psnr": 29.63, "ssim": 0.8994, "lpips": 0.1400},
        "garden_s0.75": {"psnr": 29.28, "ssim": 0.8811, "lpips": 0.1613},
        "garden_s0.5": {"psnr": 29.23, "ssim": 0.8770, "lpips": 0.1655},
        "bicycle_s1.0": {"psnr": 26.47, "ssim": 0.8383, "lpips": 0.2436},
        "bicycle_s0.625": {"psnr": 26.14, "ssim": 0.8084, "lpips": 0.2750},
        "bicycle_s0.5": {"psnr": 26.43, "ssim": 0.8170, "lpips": 0.2633},
    }

    results["discrepancies"] = {}
    for key, canon in canonical.items():
        if key in results["checkpoints"] and results["checkpoints"][key].get("status") == "EVALUATED":
            recomputed = results["checkpoints"][key]
            disc = {}
            for metric in ["psnr", "ssim", "lpips"]:
                if metric in canon and metric in recomputed:
                    d = recomputed[metric] - canon[metric]
                    disc[metric] = {"canonical": canon[metric], "recomputed": recomputed[metric], "delta": round(d, 4)}
            results["discrepancies"][key] = disc

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n\nUnified metrics saved to {output_path}")

    # Print discrepancy summary
    print("\n=== DISCREPANCY SUMMARY ===")
    for key, disc in results["discrepancies"].items():
        for metric, vals in disc.items():
            if abs(vals["delta"]) > 0.01:
                print(f"  {key} {metric}: canonical={vals['canonical']} recomputed={vals['recomputed']} delta={vals['delta']:+.4f} ***")
            else:
                print(f"  {key} {metric}: delta={vals['delta']:+.4f} (OK)")


if __name__ == "__main__":
    main()
