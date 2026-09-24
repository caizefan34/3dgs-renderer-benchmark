#!/usr/bin/env python3
"""
Phase 11 — M5 (eps2d) Evidence Chain — Steps 2-4

Measures how eps2d affects forward output, gradients, quality, and workload.
"""
import argparse, json, os, sys, time, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch, numpy as np
from benchmark_framework.scene import load_ply

def load_gt_images(scene_name, data_dir):
    from PIL import Image
    img_dir = Path(data_dir) / "datasets" / "mipnerf360" / scene_name / "images"
    if not img_dir.exists():
        return {}
    gt = {}
    for f in sorted(img_dir.iterdir()):
        if f.suffix.upper() in (".JPG", ".JPEG", ".PNG"):
            gt[f.stem] = np.array(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
    return gt

def load_cameras(json_path, target_h=1080):
    with open(json_path) as f:
        data = json.load(f)
    viewmats, Ks, names = [], [], []
    for cd in data:
        fx, fy = cd["fx"], cd["fy"]
        w_orig, h_orig = cd["width"], cd["height"]
        scale = target_h / h_orig
        w = int(round(w_orig * scale))
        h = target_h
        R = np.array(cd["rotation"]).reshape(3, 3)
        t = np.array(cd["position"])
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R.T
        viewmat[:3, 3] = -R.T @ t
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = fx * scale
        K[1, 1] = fy * scale
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        viewmats.append(torch.from_numpy(viewmat))
        Ks.append(torch.from_numpy(K))
        names.append(cd.get("img_name", f"cam_{cd['id']:03d}"))
    return torch.stack(viewmats).float().cuda(), torch.stack(Ks).float().cuda(), names

def render_with_meta(means, quats, scales, opacities, shs, viewmats, Ks, width, height,
                     sh_degree, eps2d=0.3):
    from gsplat import rasterization
    rendered, _, meta = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=shs, viewmats=viewmats, Ks=Ks,
        width=width, height=height,
        sh_degree=sh_degree, eps2d=eps2d,
        packed=True, tile_size=16, render_mode="RGB",
    )
    return rendered, meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--scene", default="room")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "epic05" / "phase11"))
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scene
    ply_path = Path(args.data_dir) / "official" / "mipnerf360" / args.scene / "point_cloud.ply"
    scene = load_ply(str(ply_path), device="cuda")
    means = scene["xyz"].contiguous()
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = torch.exp(scene["scales"]).contiguous()
    opacities = torch.sigmoid(scene["opacity"]).contiguous()
    shs = scene["shs"].contiguous()
    sh_degree = scene.get("sh_degree", 3)
    print(f"Loaded {means.shape[0]} Gaussians, SH={sh_degree}")

    cam_json = Path(args.data_dir) / "official" / "mipnerf360" / args.scene / "cameras.json"
    viewmats, Ks, img_names = load_cameras(str(cam_json), target_h=args.height)
    C = len(viewmats)
    print(f"Loaded {C} cameras")

    # Determine render size from camera JSON
    with open(cam_json) as f:
        cam0_cd = json.load(f)[0]
    scale = args.height / cam0_cd["height"]
    render_w = int(round(cam0_cd["width"] * scale))
    render_h = args.height
    print(f"Rendering at {render_w}x{render_h}")

    # GT images
    gt_images = load_gt_images(args.scene, args.data_dir)
    print(f"Loaded {len(gt_images)} GT images")

    eps_values = [0.0, 0.01, 0.05, 0.1, 0.3, 1.0]
    if args.smoke:
        eps_values = [0.0, 0.3, 1.0]
        print("*** SMOKE MODE ***")

    # ======== STEP 2: Forward Correctness ========
    print("\n" + "="*60)
    print("STEP 2: Forward Correctness + Workload Analysis")
    print("="*60)

    vm = viewmats[0:1]
    k = Ks[0:1]
    all_results = {}

    baseline_img = None
    baseline_nnz = None
    fwd_results = {}
    workload_results = {}

    for eps_val in eps_values:
        rendered, meta = render_with_meta(means, quats, scales, opacities, shs, vm, k, render_w, render_h, sh_degree, eps2d=eps_val)
        img = rendered[0]

        # Extract workload metrics
        nnz = meta.get("batch_ids").shape[0] if "batch_ids" in meta else 0
        n_isects = meta.get("flatten_ids").shape[0] if "flatten_ids" in meta else 0
        tiles_per_gauss = meta.get("tiles_per_gauss")
        if tiles_per_gauss is not None:
            mean_tpg = tiles_per_gauss.float().mean().item()
            max_tpg = tiles_per_gauss.max().item()
        else:
            mean_tpg = max_tpg = 0

        # Timing
        torch.cuda.synchronize()
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            render_with_meta(means, quats, scales, opacities, shs, vm, k, render_w, render_h, sh_degree, eps2d=eps_val)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        key = f"eps_{eps_val}"
        fwd_results[key] = {"eps2d": eps_val}
        workload_results[key] = {
            "eps2d": eps_val,
            "nnz": nnz,
            "n_isects": n_isects,
            "mean_tiles_per_gauss": round(mean_tpg, 4),
            "max_tiles_per_gauss": int(max_tpg),
            "fwd_ms_mean": round(np.mean(times), 4),
            "fwd_ms_median": round(np.median(times), 4),
        }

        if eps_val == 0.0:
            baseline_img = img.clone()
            baseline_nnz = nnz
            fwd_results[key]["mean"] = img.mean().item()
            fwd_results[key]["nnz"] = nnz
            print(f"  eps=0.0 (no blur): mean={img.mean():.6f}, nnz={nnz}, "
                  f"isects={n_isects}, TPG={mean_tpg:.2f}, fwd={np.median(times):.2f}ms")
        else:
            diff = (img - baseline_img).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            rel_err = diff.norm().item() / (baseline_img.norm().item() + 1e-8)
            affected = (diff > 1/255).sum().item()
            total = diff.numel()
            fwd_results[key].update({
                "max_abs_diff": round(max_abs, 8),
                "mean_abs_diff": round(mean_abs, 8),
                "relative_error": round(rel_err, 8),
                "affected_pixels": affected,
                "affected_pct": round(100*affected/total, 4),
            })
            nnz_change = ((nnz - baseline_nnz) / baseline_nnz * 100) if baseline_nnz else 0
            print(f"  eps={eps_val}: max_diff={max_abs:.6f}, mean_diff={mean_abs:.8f}, "
                  f"affected={100*affected/total:.2f}%, nnz={nnz}({nnz_change:+.1f}%), "
                  f"isects={n_isects}, fwd={np.median(times):.2f}ms")

    with open(output_dir / "m5_forward_results.json", "w") as f:
        json.dump(fwd_results, f, indent=2, default=str)
    with open(output_dir / "m5_workload_analysis.json", "w") as f:
        json.dump(workload_results, f, indent=2, default=str)

    # ======== STEP 3: Gradient Correctness ========
    print("\n" + "="*60)
    print("STEP 3: Gradient Correctness")
    print("="*60)

    grad_results = {}
    for eps_val in eps_values:
        m = means.clone().detach().requires_grad_(True)
        s = scales.clone().detach().requires_grad_(True)
        q = quats.clone().detach().requires_grad_(True)
        o = opacities.clone().detach().requires_grad_(True)
        sh = shs.clone().detach().requires_grad_(True)

        rendered, _ = render_with_meta(m, q, s, o, sh, vm, k, render_w, render_h, sh_degree, eps2d=eps_val)
        loss = rendered.sum()
        loss.backward()

        key = f"eps_{eps_val}"
        grad_results[key] = {"all_finite": True}
        nonzero = 0
        for pname, pt in [("xyz", m), ("scales", s), ("rotations", q), ("opacity", o), ("shs", sh)]:
            g = pt.grad
            gn = g.norm().item()
            af = bool(g.isfinite().all().item())
            gz = (g.abs() < 1e-30).all().item()
            grad_results[key][pname] = {"grad_norm": round(gn, 8), "all_finite": af, "all_zero": gz}
            if not af:
                grad_results[key]["all_finite"] = False
            if not gz:
                nonzero += 1
        grad_results[key]["nonzero_params"] = nonzero
        print(f"  eps={eps_val}: all_finite={grad_results[key]['all_finite']}, nonzero={nonzero}")

    with open(output_dir / "m5_gradient_results.json", "w") as f:
        json.dump(grad_results, f, indent=2, default=str)

    # ======== STEP 4: GT Quality ========
    print("\n" + "="*60)
    print("STEP 4: GT Quality (3 test views)")
    print("="*60)

    try:
        from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
    except ImportError:
        os.system(f"{sys.executable} -m pip install scikit-image -q")
        from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

    qual_results = {}
    test_indices = list(range(0, C, 8))[:3]
    for eps_val in eps_values:
        psnr_list, ssim_list = [], []
        for idx in test_indices:
            vm_i = viewmats[idx:idx+1]
            k_i = Ks[idx:idx+1]
            rendered, _ = render_with_meta(means, quats, scales, opacities, shs, vm_i, k_i, render_w, render_h, sh_degree, eps2d=eps_val)
            img_np = rendered[0].clamp(0, 1).cpu().numpy()
            gt = gt_images.get(img_names[idx])
            if gt is None:
                continue
            if gt.shape[:2] != (render_h, render_w):
                from skimage.transform import resize
                gt = resize(gt, (render_h, render_w), anti_aliasing=True, preserve_range=True)
            p = psnr(gt, img_np, data_range=1.0)
            s = ssim(gt, img_np, data_range=1.0, channel_axis=-1)
            psnr_list.append(p); ssim_list.append(s)

        key = f"eps_{eps_val}"
        qual_results[key] = {
            "eps2d": eps_val,
            "avg_psnr": round(float(np.mean(psnr_list)), 4) if psnr_list else 0,
            "avg_ssim": round(float(np.mean(ssim_list)), 4) if ssim_list else 0,
            "n_views": len(psnr_list),
        }
        print(f"  eps={eps_val}: PSNR={qual_results[key]['avg_psnr']:.4f}, SSIM={qual_results[key]['avg_ssim']:.4f}")

    with open(output_dir / "m5_gt_quality.json", "w") as f:
        json.dump(qual_results, f, indent=2, default=str)

    # Consolidate
    all_results = {
        "metadata": {"scene": args.scene, "n_gaussians": means.shape[0], "eps_values": eps_values},
        "forward": fwd_results,
        "workload": workload_results,
        "gradient": grad_results,
        "gt_quality": qual_results,
    }
    with open(output_dir / "m5_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDone. Results in {output_dir / 'm5_results.json'}")

if __name__ == "__main__":
    main()
