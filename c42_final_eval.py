#!/usr/bin/env python3
"""
C42 Completion Batch — Final Eval from Checkpoints.
Loads 30K checkpoints and produces JSON output with PSNR/SSIM/LPIPS.
"""
import argparse, json, math, os, sys, time, hashlib, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
REFERENCE_V1_DIR = REPO_ROOT / "baseline" / "reference_v1"
sys.path.insert(0, str(REFERENCE_V1_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))
from gaussian_model import GaussianModel
from config import ReferenceV1Config
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
        "xyz": model.get_xyz,
        "rotations": model.get_rotation,
        "scales": model.get_scaling,
        "opacity": model.get_opacity,
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


def evaluate_all(model, dataset, ssim_fn, lpips_fn=None):
    n_cams = len(dataset)
    psnrs, ssims, l1s, lpips_vals = [], [], [], []
    for ci in range(n_cams):
        cam, gt = dataset.get_item(ci)
        with torch.no_grad():
            img = render(model, cam, model.active_sh_degree)
        psnrs.append(compute_psnr(img, gt))
        ssims.append(float(1.0 - ssim_fn(img, gt).item()))
        l1s.append(float(F.l1_loss(img, gt).item()))
        if lpips_fn is not None:
            pred_lp = img.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
            gt_lp = gt.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
            lp = float(lpips_fn(pred_lp, gt_lp).item())
            lpips_vals.append(lp)
        if ci % 50 == 0:
            print(f"    cam {ci}/{n_cams}...", flush=True)
    result = {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "l1": float(np.mean(l1s)),
        "n_cameras": n_cams,
        "n_gaussians": model._xyz.shape[0],
    }
    if lpips_vals:
        result["lpips"] = float(np.mean(lpips_vals))
    return result


def get_provenance(repo_root, scene, scale):
    import subprocess
    prov = {}
    try:
        prov["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
        prov["git_describe"] = subprocess.check_output(["git", "describe", "--tags"], cwd=str(repo_root), text=True).strip()
        prov["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root), text=True).strip())
    except:
        prov["git_head"] = "unknown"
        prov["git_describe"] = "unknown"
        prov["git_dirty"] = None
    prov["pytorch_version"] = torch.__version__
    prov["cuda_version"] = torch.version.cuda
    try:
        import gsplat
        prov["gsplat_version"] = gsplat.__version__
    except:
        prov["gsplat_version"] = "unknown"
    prov["gpu_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    prov["gpu_sms"] = props.multi_processor_count
    model_path = REFERENCE_V1_DIR / "gaussian_model.py"
    config_path = REFERENCE_V1_DIR / "config.py"
    prov["gaussian_model_hash"] = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    prov["config_hash"] = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]
    prov["gaussian_model_path"] = str(model_path.relative_to(repo_root))
    prov["semantic_label"] = f"REFERENCE_V1 + C42 (DS-SSIM {scale})"
    prov["config_base"] = "REFERENCE_V1_ABSGRAD"
    prov["scene"] = scene
    prov["scale"] = scale
    prov["dataset_resolution"] = "1080p"
    prov["loss_definition"] = f"(1-lambda)*L1 + lambda*d_ssim_downsampled(scale={scale}), lambda=0.2"
    prov["note"] = "Final eval from 30K checkpoint produced by c42_completion_train_v2.py"
    return prov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"C42 Final Eval: scene={args.scene}, scale={args.scale}, ckpt={args.ckpt}")

    # LPIPS
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to("cuda")
        lpips_fn.eval()
        print("  LPIPS initialized")
    except Exception as e:
        print(f"  LPIPS not available: {e}")
        lpips_fn = None

    ssim_fn = SepSSIM(device="cuda")

    # Load dataset
    print(f"  Loading dataset: {args.scene}")
    dataset = GTDataset(scene=args.scene, repo_root=str(REPO_ROOT), resolution="1080p", device="cuda", background="black")
    print(f"  {len(dataset)} cameras")

    # Load checkpoint
    ckpt_path = args.ckpt
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = GaussianModel(max_sh_degree=3)
    model.restore(ckpt, config={
        "position_lr_init": 0.00016, "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01, "position_lr_max_steps": 30000,
        "feature_lr": 0.0025, "opacity_lr": 0.025, "scaling_lr": 0.005,
        "rotation_lr": 0.001, "percent_dense": 0.01,
    })
    print(f"  Model restored: N={model._xyz.shape[0]}, SH degree={model.active_sh_degree}")

    # Evaluate
    print(f"  Evaluating on all {len(dataset)} cameras...")
    t0 = time.perf_counter()
    final_eval = evaluate_all(model, dataset, ssim_fn, lpips_fn)
    eval_time = time.perf_counter() - t0
    print(f"  Eval time: {eval_time:.1f}s")
    print(f"  PSNR={final_eval['psnr']:.4f} SSIM={final_eval['ssim']:.4f} "
          f"LPIPS={final_eval.get('lpips', 'N/A')} GS={final_eval['n_gaussians']:,}")

    # Save
    output = {
        "experiment": "C42 Completion Batch V2 (final eval from checkpoint)",
        "scene": args.scene,
        "scale": args.scale,
        "config": f"REFERENCE_V1 + C42 (DS-SSIM {args.scale})",
        "provenance": get_provenance(REPO_ROOT, args.scene, args.scale),
        "results": {
            "scene": args.scene,
            "scale": args.scale,
            "seed": 42,
            "final_eval": final_eval,
            "checkpoint_path": ckpt_path,
            "checkpoint_iteration": ckpt.get("iteration", 30000),
            "eval_time_s": eval_time,
        }
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
