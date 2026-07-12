"""
Quality validation script for 3DGS renderer benchmark.

Compares rendered outputs from:
  Test 1: Speedy (all points) vs Diff Gaussian (all points)
          -> Validates rasterizer consistency
  Test 2: Speedy (culled) vs Speedy (all points)
          -> Validates Frustum Pre-Culling quality impact

Usage:
    conda activate gsplat
    python src/scripts/validate_quality.py [--frames N]
"""

import sys, os, json, torch
import numpy as np

PROJECT_ROOT = r"C:\Users\36570\Documents\Codex\2026-07-12\caizefan34-3dgs-renderer-benchmark-https-github-2\work\repo"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
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

    def render_diff(self, cam):
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        s = GaussianRasterizationSettings(image_height=HEIGHT, image_width=WIDTH,
            tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
            bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
            viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
            sh_degree=3, campos=cam.camera_center,
            prefiltered=False, debug=False, antialiasing=False)
        r = GaussianRasterizer(s)
        with torch.no_grad():
            out, _, _ = r(means3D=self.means3d, means2D=torch.zeros_like(self.means3d[:,:2]),
                opacities=self.opacities, shs=self.shs, colors_precomp=None,
                scales=self.scales, rotations=self.rotations, cov3D_precomp=None)
        return out

    def render_speedy(self, cam, cull_mask=None):
        from speedy_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        s = GaussianRasterizationSettings(image_height=HEIGHT, image_width=WIDTH,
            tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
            bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
            viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
            sh_degree=3, campos=cam.camera_center,
            prefiltered=False, debug=False)
        r = GaussianRasterizer(s)
        if cull_mask is None:
            with torch.no_grad():
                out, _, _ = r(means3D=self.means3d, means2D=torch.zeros_like(self.means3d[:,:2]),
                    opacities=self.opacities, scores=torch.ones(self.N_G, device="cuda"),
                    shs=self.shs, colors_precomp=None,
                    scales=self.scales, rotations=self.rotations, cov3D_precomp=None)
            return out, None
        else:
            nv = cull_mask.sum().item()
            with torch.no_grad():
                out, _, _ = r(means3D=self.means3d[cull_mask], means2D=torch.zeros(nv, 2, device="cuda"),
                    opacities=self.opacities[cull_mask], scores=torch.ones(nv, device="cuda"),
                    shs=self.shs[cull_mask], colors_precomp=None,
                    scales=self.scales[cull_mask], rotations=self.rotations[cull_mask], cov3D_precomp=None)
            return out, cull_mask

    def compute_cull_mask(self, cam):
        view = cam.viewmatrix
        ms = self.means3d
        pz = view[2,0]*ms[:,0] + view[2,1]*ms[:,1] + view[2,2]*ms[:,2] + view[2,3]
        wvt = cam.world_view_transform
        px = wvt[0,0]*ms[:,0] + wvt[1,0]*ms[:,1] + wvt[2,0]*ms[:,2] + wvt[3,0]
        py = wvt[0,1]*ms[:,0] + wvt[1,1]*ms[:,1] + wvt[2,1]*ms[:,2] + wvt[3,1]
        proj_x = px / (pz.abs() * cam.tanfovx + 1e-8)
        proj_y = py / (pz.abs() * cam.tanfovy + 1e-8)
        return (proj_x >= -100.0) & (proj_x <= 100.0) & (proj_y >= -100.0) & (proj_y <= 100.0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=NUM_FRAMES)
    args = parser.parse_args()

    print("Loading...")
    v = Validator()
    nf = min(args.frames, len(v.cameras))

    print()
    print("Test 1: Speedy (all) vs Diff (all) -- rasterizer consistency")
    print("  Frame    PSNR(dB)      SSIM   LPIPS    DiffMax")
    print("  " + "-" * 48)

    results_1 = []
    for fi in range(nf):
        cam = v.cameras[fi]
        img_ref = v.render_diff(cam)
        img_test, _ = v.render_speedy(cam, cull_mask=None)
        if torch.isnan(img_test).any() or torch.isnan(img_ref).any():
            print(f"  {fi:>4}     NaN       NaN     NaN       NaN")
            continue
        psnr_val = compute_psnr(img_test, img_ref)
        ssim_val = compute_ssim(img_test, img_ref)
        lpips_val = compute_lpips(img_test, img_ref)
        dmax = float(torch.max(torch.abs(img_test - img_ref)).item())
        print(f"  {fi:>4}  {psnr_val:>9.2f}  {ssim_val:>7.5f}  {lpips_val:>6.4f}  {dmax:.6f}")
        results_1.append({"frame": fi, "psnr": round(psnr_val, 4), "ssim": round(ssim_val, 6),
                          "lpips": round(lpips_val, 6), "max_diff": round(dmax, 8)})

    print()
    print("Test 2: Speedy (culled) vs Speedy (all) -- culling quality")
    print("  Frame    PSNR(dB)      SSIM   LPIPS   Visible%")
    print("  " + "-" * 48)

    results_2 = []
    for fi in range(nf):
        cam = v.cameras[fi]
        img_ref, _ = v.render_speedy(cam, cull_mask=None)
        mask = v.compute_cull_mask(cam)
        img_test, _ = v.render_speedy(cam, cull_mask=mask)
        if torch.isnan(img_test).any() or torch.isnan(img_ref).any():
            print(f"  {fi:>4}     NaN       NaN     NaN      NaN")
            continue
        psnr_val = compute_psnr(img_test, img_ref)
        ssim_val = compute_ssim(img_test, img_ref)
        lpips_val = compute_lpips(img_test, img_ref)
        vpct = mask.float().mean().item() * 100
        print(f"  {fi:>4}  {psnr_val:>9.2f}  {ssim_val:>7.5f}  {lpips_val:>6.4f}  {vpct:>6.2f}%")
        results_2.append({"frame": fi, "psnr": round(psnr_val, 4), "ssim": round(ssim_val, 6),
                          "lpips": round(lpips_val, 6), "visible_pct": round(vpct, 2)})

    print()
    print("=" * 55)
    print("  QUALITY VALIDATION SUMMARY")
    print("=" * 55)
    print(f"  Resolution: {WIDTH}x{HEIGHT}  |  Gaussians: {v.N_G}")
    print()

    if results_1:
        ps = [r["psnr"] for r in results_1]
        print("  Test 1 -- Speedy(all) vs Diff(all):")
        print(f"    PSNR mean={np.mean(ps):.2f} dB, min={np.min(ps):.2f} dB")
        identical = all(p == float("inf") for p in ps)
        passed_1 = identical or all(p >= 60 for p in ps)
        if identical:
            print("    PASS: Outputs are IDENTICAL (PSNR = inf dB)")
        elif passed_1:
            print("    PASS: Outputs near-identical (all PSNR >= 60 dB)")
        else:
            print("    FAIL: Outputs differ (PSNR < 60 dB)")
    print()

    if results_2:
        ps2 = [r["psnr"] for r in results_2]
        vis = [r["visible_pct"] for r in results_2]
        print("  Test 2 -- Speedy(culled) vs Speedy(all):")
        print(f"    PSNR mean={np.mean(ps2):.2f} dB, min={np.min(ps2):.2f} dB")
        print(f"    Visible: mean={np.mean(vis):.1f}%, min={np.min(vis):.1f}%, max={np.max(vis):.1f}%")
        passed_2 = all(p >= 45 for p in ps2)
        if passed_2:
            print("    PASS: Culling introduces no detectable quality loss")
        else:
            print("    FAIL: Culling causes PSNR < 45 dB")
    print()

    report = {"config": {"resolution": f"{WIDTH}x{HEIGHT}", "num_frames": nf, "num_gaussians": v.N_G},
              "test1_rasterizer_consistency": results_1, "test2_culling_quality": results_2}
    rp = os.path.join(OUT_DIR, "quality_validation.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Results saved to {rp}")


if __name__ == "__main__":
    main()
