#!/usr/bin/env python
"""CGT-PAPER13 canonical baseline (P1): full-test-set gsplat render timing.

Frozen canonical renderer = gsplat 1.5.3 rasterization (production path, packed=True).
Two tile sizes per scene (Baseline A tile16 / Baseline B tile32). NO CGT involvement.
Timing: warmup + CUDA-event repeats, median + spread. Clean-GPU guarded by caller.
Correctness: rendered RGB vs GT (PSNR/SSIM/L1) on test views (GT-compare; not CGT).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

SCRIPTS = "/home/liaoyuanjun/cgt_realscene/final/SCRIPTS"
sys.path.insert(0, SCRIPTS)
import torch
import gsplat
from g1_compressibility import read_cameras_bin, read_images_bin, q2R, TILE as G1TILE

def load_scene(ckpt, data_root, scene):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    means = ck["xyz"].cuda().float()
    scales = torch.exp(ck["scaling"].cuda().float())
    quats = ck["rotation"].cuda().float()
    opacities = torch.sigmoid(ck["opacity"].cuda().float())
    shs = ck["shs"].cuda().float()
    deg = int(ck["active_sh_degree"])
    cams = read_cameras_bin(f"{data_root}/{scene}/sparse/0/cameras.bin")
    imgs = read_images_bin(f"{data_root}/{scene}/sparse/0/images.bin")
    return dict(means=means, scales=scales, quats=quats, opacities=opacities,
                shs=shs, deg=deg, cams=cams, imgs=imgs, N=int(means.shape[0]))

def view_data(imgs, cams, vi, H=None, W=None):
    iid, name, cam, qv, tv = imgs[vi]
    mid, Wm, Hm, par = cams[cam]
    fx, fy, cx, cy = par
    W = W or Wm; H = H or Hm
    R_w2c = q2R(qv); t_w2c = np.array(tv); t_c2w = -R_w2c.T @ t_w2c
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")[None]
    viewmat[0, :3, :3] = torch.as_tensor(R_w2c, dtype=torch.float32, device="cuda")
    viewmat[0, :3, 3] = torch.as_tensor(t_w2c, dtype=torch.float32, device="cuda")
    K = torch.as_tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32, device="cuda")
    campos = torch.as_tensor(t_c2w, dtype=torch.float32, device="cuda")[None]
    return name, viewmat, K, W, H, campos

def render(sd, viewmat, K, W, H, tile_size):
    rc, ra, meta = gsplat.rasterization(
        sd["means"], sd["quats"], sd["scales"], sd["opacities"], sd["shs"],
        viewmat, K, width=W, height=H, sh_degree=sd["deg"],
        render_mode="RGB", packed=True, tile_size=tile_size)
    return rc  # (1,H,W,3)

def psnr_ssim_l1(pred, gt):
    # pred,gt in [0,1] float (H,W,3)
    mse = torch.mean((pred - gt) ** 2).item()
    ps = 10 * np.log10(1.0 / mse) if mse > 0 else 99.0
    l1 = torch.mean(torch.abs(pred - gt)).item()
    # SSIM (simple global, windowless)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_p = pred.mean(); mu_g = gt.mean()
    va_p = pred.var(); va_g = gt.var(); cov = ((pred - mu_p) * (gt - mu_g)).mean()
    ssim = ((2 * mu_p * mu_g + c1) * (2 * cov + c2)) / ((mu_p ** 2 + mu_g ** 2 + c1) * (va_p + va_g + c2))
    ssim = float(ssim.clamp(-1, 1)) if torch.is_tensor(ssim) else float(np.clip(ssim, -1, 1))
    return ps, ssim, l1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="/tmp/3dgs-renderer-benchmark/data/datasets/mipnerf360")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--tile-sizes", default="16,32")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n-test-views", type=int, default=0, help="0=all test views")
    ap.add_argument("--split-every", type=int, default=8, help="test = idx %% every == 0")
    ap.add_argument("--no-quality", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.cuda.set_device(args.device)
    np.seterr(divide="ignore", invalid="ignore")

    sd = load_scene(args.ckpt, args.data_root, args.scene)
    n_imgs = len(sd["imgs"])
    test_idx = [i for i in range(n_imgs) if i % args.split_every == 0]
    if args.n_test_views > 0:
        test_idx = test_idx[: args.n_test_views]
    tile_sizes = [int(x) for x in args.tile_sizes.split(",")]

    result = {"scene": args.scene, "N_gaussians": sd["N"], "sh_degree": sd["deg"],
              "n_images": n_imgs, "n_test_views": len(test_idx), "split_every": args.split_every,
              "tile_sizes": tile_sizes, "device": args.device,
              "gsplat": gsplat.__version__, "torch": torch.__version__,
              "per_tile": {}, "quality": {}}

    # preload GT for quality (optional)
    gt_cache = {}
    def get_gt(name):
        if name in gt_cache: return gt_cache[name]
        from PIL import Image
        p = f"{args.data_root}/{args.scene}/images/{name}"
        try:
            im = Image.open(p).convert("RGB")
            arr = torch.from_numpy(np.asarray(im)).float().cuda() / 255.0
        except Exception:
            arr = None
        gt_cache[name] = arr
        return arr

    for ts in tile_sizes:
        per_view = []
        q_res = []
        for vi in test_idx:
            name, viewmat, K, W, H, campos = view_data(sd["imgs"], sd["cams"], vi)
            # warmup
            for _ in range(args.warmup):
                _ = render(sd, viewmat, K, W, H, ts)
            torch.cuda.synchronize()
            # timed
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                _ = render(sd, viewmat, K, W, H, ts)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
            per_view.append(float(np.median(times)))
            # quality (one render)
            if not args.no_quality:
                rc = render(sd, viewmat, K, W, H, ts)
                pred = rc[0].clamp(0, 1)
                gt = get_gt(name)
                if gt is not None and gt.shape == pred.shape:
                    q_res.append({"view": name, **dict(zip(["psnr", "ssim", "l1"], psnr_ssim_l1(pred, gt)))})
        per_view = np.asarray(per_view)
        result["per_tile"][str(ts)] = {
            "median_ms": float(np.median(per_view)),
            "mean_ms": float(np.mean(per_view)),
            "min_ms": float(np.min(per_view)),
            "p95_ms": float(np.percentile(per_view, 95)),
            "fps_median": float(1000.0 / np.median(per_view)) if np.median(per_view) > 0 else None,
            "per_view_ms": per_view.tolist(),
        }
        if q_res:
            ps = np.array([q["psnr"] for q in q_res]); ss = np.array([q["ssim"] for q in q_res]); l1 = np.array([q["l1"] for q in q_res])
            result["quality"][str(ts)] = {"psnr_mean": float(ps.mean()), "ssim_mean": float(ss.mean()),
                                          "l1_mean": float(l1.mean()), "n": len(q_res)}
        print(f"[{args.scene}] tile{ts}: median={result['per_tile'][str(ts)]['median_ms']:.3f}ms fps={result['per_tile'][str(ts)]['fps_median']:.2f}", flush=True)

    outp = os.path.join(args.out, f"canonical_{args.scene}.json")
    open(outp, "w").write(json.dumps(result, indent=2))
    print(f"WROTE {outp}")

if __name__ == "__main__":
    main()