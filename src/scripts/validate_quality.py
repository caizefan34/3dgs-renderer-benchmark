"""
Quality validation script for 3DGS renderer benchmark.

Compares rendered outputs from baseline (diff_gaussian) and optimized (speedy_splat + culling)
rasterizers using PSNR, SSIM, and LPIPS metrics.

Usage:
    python src/scripts/validate_quality.py [--frames N]

Requires:
    pip install torch diff-gaussian-rasterization speedy-gaussian-rasterization lpips scikit-image
"""

import sys, os, json, torch
import numpy as np

PROJECT_ROOT = r"C:\Users\36570\Documents\Codex\2026-07-12\caizefan34-3dgs-renderer-benchmark-https-github-2\work\repo"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, r'C:\Users\36570\Documents\Codex\2026-07-12\caizefan34-3dgs-renderer-benchmark-https-github-2\work\repo\src\benchmark_framework')
from benchmark_framework import load_ply, load_cameras_from_json

SCENE = os.path.join(PROJECT_ROOT, "data", "scene.ply")
CAMS_JSON = os.path.join(PROJECT_ROOT, "data", "cameras.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "quality_validation")
os.makedirs(OUT_DIR, exist_ok=True)
NUM_FRAMES = 10
WIDTH, HEIGHT = 1920, 1080


def compute_psnr(img_pred, img_gt):
    mse = torch.mean((img_pred - img_gt) ** 2)
    return float("inf") if mse < 1e-10 else (-10.0 * torch.log10(mse)).item()


def compute_ssim(img_pred, img_gt, ws=11):
    from torch.nn.functional import conv2d
    g1d = torch.tensor([0.000000, 0.000002, 0.000175, 0.005561, 0.056451,
                        0.183090, 0.189921, 0.063021, 0.006689, 0.000227, 0.000002],
                       dtype=torch.float32)[:ws]
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    w = g2d.expand(3, 1, ws, ws).contiguous().to(img_pred.device)

    def cv(x):
        return conv2d(x.unsqueeze(0), w, padding=ws // 2, groups=3)

    mu1, mu2 = cv(img_pred), cv(img_gt)
    s1 = cv(img_pred.pow(2)) - mu1.pow(2)
    s2 = cv(img_gt.pow(2)) - mu2.pow(2)
    s12 = cv(img_pred * img_gt) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu1 * mu2 + c1) * (2 * s12 + c2) / (
        (mu1.pow(2) + mu2.pow(2) + c1) * (s1 + s2 + c2))).mean().item()


def compute_lpips(img_pred, img_gt):
    try:
        import lpips
        fn = lpips.LPIPS(net="alex").to(img_pred.device)
        with torch.no_grad():
            return fn(img_pred.unsqueeze(0) * 2 - 1, img_gt.unsqueeze(0) * 2 - 1).item()
    except ImportError:
        print("  [WARN] lpips not installed; using 1 - SSIM proxy")
        return 1.0 - compute_ssim(img_pred, img_gt)


class Validator:
    def __init__(self):
        self.scene = load_ply(SCENE, device="cuda")
        self.cameras = load_cameras_from_json(CAMS_JSON, device="cuda")
        self.N_G = self.scene["num_points"]
        d = self.scene
        self.means3d = d["xyz"].contiguous()
        self.opacities = torch.sigmoid(d["opacity"]).contiguous()
        self.shs = d["shs"].contiguous()
        self.scales = d["scales"].contiguous()
        self.rotations = torch.nn.functional.normalize(d["rotations"], dim=-1).contiguous()
        print(f"  {self.N_G} gaussians, {len(self.cameras)} cameras")

    def _cull_mask(self, cam):
        # p_view.z = viewmatrix[2,0]*x + viewmatrix[2,1]*y + viewmatrix[2,2]*z + viewmatrix[2,3]
        view = cam.viewmatrix
        ms = self.means3d
        pz = view[2,0]*ms[:,0] + view[2,1]*ms[:,1] + view[2,2]*ms[:,2] + view[2,3]
        wvt = cam.world_view_transform
        px = wvt[0,0]*ms[:,0] + wvt[1,0]*ms[:,1] + wvt[2,0]*ms[:,2] + wvt[3,0]
        py = wvt[0,1]*ms[:,0] + wvt[1,1]*ms[:,1] + wvt[2,1]*ms[:,2] + wvt[3,1]
        proj_x = px / (pz.abs() * cam.tanfovx + 1e-8)
        proj_y = py / (pz.abs() * cam.tanfovy + 1e-8)
        return (pz > 0.1) & (proj_x >= -3.0) & (proj_x <= 3.0) & (proj_y >= -3.0) & (proj_y <= 3.0)

    def render(self, cam, use_opt=False):
        if use_opt:
            from speedy_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
            mask = self._cull_mask(cam)
            nv = mask.sum().item()
            settings = GaussianRasterizationSettings(
                image_height=HEIGHT, image_width=WIDTH,
                tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
                bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
                viewmatrix=cam.world_view_transform,
                projmatrix=cam.full_proj_transform,
                sh_degree=3, campos=cam.camera_center,
                prefiltered=False, debug=False,
            )
            rast = GaussianRasterizer(settings)
            with torch.no_grad():
                out, _, _ = rast(
                    means3D=self.means3d[mask], means2D=torch.zeros(nv, 2, device="cuda"),
                    opacities=self.opacities[mask], scores=torch.ones(nv, device="cuda"),
                    shs=self.shs[mask], colors_precomp=None,
                    scales=self.scales[mask], rotations=self.rotations[mask], cov3D_precomp=None,
                )
            return out, mask
        else:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
            settings = GaussianRasterizationSettings(
                image_height=HEIGHT, image_width=WIDTH,
                tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
                bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
                viewmatrix=cam.world_view_transform,
                projmatrix=cam.full_proj_transform,
                sh_degree=3, campos=cam.camera_center,
                prefiltered=False, debug=False, antialiasing=False,
            )
            rast = GaussianRasterizer(settings)
            with torch.no_grad():
                out, _, _ = rast(
                    means3D=self.means3d, means2D=torch.zeros_like(self.means3d[:, :2]),
                    opacities=self.opacities, shs=self.shs, colors_precomp=None,
                    scales=self.scales, rotations=self.rotations, cov3D_precomp=None,
                )
            return out, None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=NUM_FRAMES)
    args = parser.parse_args()

    print("Loading...")
    v = Validator()
    nf = min(args.frames, len(v.cameras))

    print(f"\n{'Frame':>6}  {'PSNR(dB)':>10}  {'SSIM':>8}  {'LPIPS':>6}  {'Visible':>8}")
    print("-" * 55)

    results = []
    for fi in range(nf):
        cam = v.cameras[fi]
        img_base, _ = v.render(cam, use_opt=False)
        img_opt, mask = v.render(cam, use_opt=True)

        psnr = compute_psnr(img_opt, img_base)
        ssim = compute_ssim(img_opt, img_base)
        lpips = compute_lpips(img_opt, img_base)
        vpct = mask.float().mean().item() * 100

        print(f"{fi:>6}  {psnr:>10.2f}  {ssim:>8.5f}  {lpips:>6.4f}  {vpct:>7.1f}%")
        results.append({
            "frame": fi, "psnr": round(psnr, 4),
            "ssim": round(ssim, 6), "lpips": round(lpips, 6),
            "visible_pct": round(vpct, 1),
        })

    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]
    lpipss = [r["lpips"] for r in results]

    print("\n" + "=" * 55)
    print("  QUALITY VALIDATION SUMMARY")
    print("=" * 55)
    print(f"  Frames:      {nf}")
    print(f"  Resolution:  {WIDTH}x{HEIGHT}")
    print(f"  Gaussians:   {v.N_G}")
    print(f"  ┌────────────┬──────────┬──────────┬──────────┐")
    print(f"  │ Metric     │    Mean  │     Min  │     Max  │")
    print(f"  ├────────────┼──────────┼──────────┼──────────┤")
    print(f"  │ PSNR (dB)  │ {np.mean(psnrs):>8.2f} │ {np.min(psnrs):>8.2f} │ {np.max(psnrs):>8.2f} │")
    print(f"  │ SSIM       │ {np.mean(ssims):>8.5f} │ {np.min(ssims):>8.5f} │ {np.max(ssims):>8.5f} │")
    print(f"  │ LPIPS      │ {np.mean(lpipss):>8.4f} │ {np.min(lpipss):>8.4f} │ {np.max(lpipss):>8.4f} │")
    print(f"  └────────────┴──────────┴──────────┴──────────┘")

    passed = True
    if np.min(psnrs) < 45:
        print(f"  ✗ PSNR < 45 dB (min {np.min(psnrs):.2f})"); passed = False
    else:
        print(f"  ✓ PSNR > 45 dB")
    if np.min(ssims) < 0.99:
        print(f"  ✗ SSIM < 0.99 (min {np.min(ssims):.5f})"); passed = False
    else:
        print(f"  ✓ SSIM > 0.99")
    if np.max(lpipss) > 0.02:
        print(f"  ✗ LPIPS > 0.02 (max {np.max(lpipss):.4f})"); passed = False
    else:
        print(f"  ✓ LPIPS < 0.02")

    if passed:
        print("\n  ✓ ALL CHECKS PASSED: Frustum Pre-Culling introduces no detectable quality loss.")
    else:
        print("\n  ✗ QUALITY DEGRADATION DETECTED")

    report = {"config": {"resolution": f"{WIDTH}x{HEIGHT}", "num_frames": nf, "num_gaussians": v.N_G},
              "summary": {"psnr_mean": round(float(np.mean(psnrs)), 4), "psnr_min": round(float(np.min(psnrs)), 4),
                          "psnr_max": round(float(np.max(psnrs)), 4), "ssim_mean": round(float(np.mean(ssims)), 6),
                          "ssim_min": round(float(np.min(ssims)), 6), "ssim_max": round(float(np.max(ssims)), 6),
                          "lpips_mean": round(float(np.mean(lpipss)), 6), "lpips_min": round(float(np.min(lpipss)), 6),
                          "lpips_max": round(float(np.max(lpipss)), 6), "passed": passed},
              "frames": results}
    rp = os.path.join(OUT_DIR, "quality_validation.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Results saved to {rp}")


if __name__ == "__main__":
    main()
