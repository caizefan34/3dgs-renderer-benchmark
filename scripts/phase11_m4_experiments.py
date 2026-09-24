#!/usr/bin/env python3
"""
Phase 11 — M4 (radius_clip) Full Evidence Chain — SIMPLIFIED
Uses gsplat directly without the benchmark Camera class.
"""
import argparse, json, os, sys, time, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch, numpy as np
from benchmark_framework.scene import load_ply

def load_gt_images(scene_name, data_dir):
    """Load GT images as numpy arrays keyed by image name (no ext)."""
    from PIL import Image
    img_dir = Path(data_dir) / "datasets" / "mipnerf360" / scene_name / "images"
    if not img_dir.exists():
        return {}
    gt = {}
    for f in sorted(img_dir.iterdir()):
        if f.suffix.upper() in (".JPG", ".JPEG", ".PNG"):
            name = f.stem  # no extension
            img = Image.open(f).convert("RGB")
            gt[name] = np.array(img, dtype=np.float32) / 255.0
    return gt

def load_cameras(json_path, target_h=1080):
    """Load cameras and return (viewmats_batch, Ks_batch, img_names, original_H)."""
    with open(json_path) as f:
        data = json.load(f)
    viewmats, Ks, names, orig_heights = [], [], [], []
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
        orig_heights.append((h_orig, w_orig))
    return torch.stack(viewmats).float().cuda(), torch.stack(Ks).float().cuda(), names, (target_h, w), orig_heights

def render(means, quats, scales, opacities, shs, viewmats, Ks, width, height,
           sh_degree, radius_clip=0.0, eps2d=0.3):
    from gsplat import rasterization
    rendered, _, meta = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=shs, viewmats=viewmats, Ks=Ks,
        width=width, height=height,
        sh_degree=sh_degree, radius_clip=radius_clip, eps2d=eps2d,
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
    parser.add_argument("--skip-training", action="store_true")
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

    # Load cameras (resized)
    cam_json = Path(args.data_dir) / "official" / "mipnerf360" / args.scene / "cameras.json"
    viewmats, Ks, img_names, (render_h, render_w), orig_sizes = load_cameras(str(cam_json), target_h=args.height)
    C = len(viewmats)
    print(f"Loaded {C} cameras, rendering at {render_w}x{render_h}")

    # GT images
    gt_images = load_gt_images(args.scene, args.data_dir)
    print(f"Loaded {len(gt_images)} GT images")

    clip_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    if args.smoke:
        clip_values = [0.0, 5.0]
        print("\n*** SMOKE MODE ***")

    # ======== STEP 2: Forward Correctness ========
    print("\n" + "="*60)
    print("STEP 2: Forward Correctness")
    print("="*60)

    fwd_results = {}
    # Single camera for forward comparison
    vm = viewmats[0:1]
    k = Ks[0:1]

    baseline_img = None
    for clip_val in clip_values:
        rendered, meta = render(means, quats, scales, opacities, shs, vm, k, render_w, render_h, sh_degree, radius_clip=clip_val)
        img = rendered[0]
        if clip_val == 0.0:
            baseline_img = img.clone()
            fwd_results["rclip_0.0"] = {"mean": img.mean().item(), "nnz": meta.get("n_isects", 0)}
            torch.save(img.cpu(), output_dir / "m4_fwd_baseline.pt")
            print(f"  baseline: mean={img.mean().item():.6f}")
        else:
            diff = (img - baseline_img).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            rel_err = diff.norm().item() / (baseline_img.norm().item() + 1e-8)
            affected = (diff > 1/255).sum().item()
            total = diff.numel()
            key = f"rclip_{clip_val}"
            fwd_results[key] = {
                "radius_clip": clip_val,
                "max_abs_diff": round(max_abs, 8),
                "mean_abs_diff": round(mean_abs, 8),
                "relative_error": round(rel_err, 8),
                "affected_pixels": affected,
                "affected_pct": round(100*affected/total, 4),
            }
            print(f"  rclip={clip_val}: max_diff={max_abs:.6f}, mean_diff={mean_abs:.6f}, affected={100*affected/total:.2f}%")

    with open(output_dir / "m4_forward_results.json", "w") as f:
        json.dump(fwd_results, f, indent=2, default=str)

    # ======== STEP 3: Gradient Correctness ========
    print("\n" + "="*60)
    print("STEP 3: Gradient Correctness")
    print("="*60)

    grad_results = {}
    for clip_val in clip_values:
        m = means.clone().detach().requires_grad_(True)
        s = scales.clone().detach().requires_grad_(True)
        q = quats.clone().detach().requires_grad_(True)
        o = opacities.clone().detach().requires_grad_(True)
        sh = shs.clone().detach().requires_grad_(True)

        rendered, _ = render(m, q, s, o, sh, vm, k, render_w, render_h, sh_degree, radius_clip=clip_val)
        loss = rendered.sum()
        loss.backward()

        key = f"rclip_{clip_val}"
        grad_results[key] = {"all_finite": True, "all_nonzero": False}
        for pname, pt in [("xyz", m), ("scales", s), ("rotations", q), ("opacity", o), ("shs", sh)]:
            g = pt.grad
            gn = g.norm().item()
            af = bool(g.isfinite().all().item())
            gz = (g.abs() < 1e-30).all().item()
            grad_results[key][pname] = {
                "grad_norm": round(gn, 8), "all_finite": af, "all_zero": gz,
            }
            if not af:
                grad_results[key]["all_finite"] = False
        nonzero_cnt = sum(1 for p in ["xyz","scales","rotations","opacity","shs"] if not grad_results[key][p].get("all_zero", False))
        grad_results[key]["nonzero_params"] = nonzero_cnt
        print(f"  rclip={clip_val}: gradients finite={grad_results[key]['all_finite']}, nonzero_params={nonzero_cnt}")

    with open(output_dir / "m4_gradient_results.json", "w") as f:
        json.dump(grad_results, f, indent=2, default=str)

    # ======== STEP 4: GT Quality ========
    print("\n" + "="*60)
    print("STEP 4: GT Quality (every 8th camera)")
    print("="*60)

    try:
        from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
    except ImportError:
        os.system(f"{sys.executable} -m pip install scikit-image -q")
        from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

    qual_results = {}
    test_indices = list(range(0, C, 8))
    for clip_val in clip_values:
        psnr_list, ssim_list = [], []
        for idx in test_indices[:3]:  # Limit to 3 test views for speed
            vm_i = viewmats[idx:idx+1]
            k_i = Ks[idx:idx+1]
            rendered, _ = render(means, quats, scales, opacities, shs, vm_i, k_i, render_w, render_h, sh_degree, radius_clip=clip_val)
            img_np = rendered[0].clamp(0, 1).cpu().numpy()
            gt = gt_images.get(img_names[idx])
            if gt is None:
                print(f"  WARNING: No GT for {img_names[idx]}")
                continue
            if gt.shape[:2] != (render_h, render_w):
                from skimage.transform import resize
                gt = resize(gt, (render_h, render_w), anti_aliasing=True, preserve_range=True)
            p = psnr(gt, img_np, data_range=1.0)
            s = ssim(gt, img_np, data_range=1.0, channel_axis=-1)
            psnr_list.append(p); ssim_list.append(s)

        key = f"rclip_{clip_val}"
        qual_results[key] = {
            "radius_clip": clip_val,
            "avg_psnr": round(float(np.mean(psnr_list)), 4) if psnr_list else 0,
            "avg_ssim": round(float(np.mean(ssim_list)), 4) if ssim_list else 0,
            "n_views": len(psnr_list),
        }
        print(f"  rclip={clip_val}: PSNR={qual_results[key]['avg_psnr']:.4f}, SSIM={qual_results[key]['avg_ssim']:.4f}")

    with open(output_dir / "m4_gt_quality.json", "w") as f:
        json.dump(qual_results, f, indent=2, default=str)

    # ======== STEP 5: 500-step Training Sanity ========
    if not args.skip_training and not args.smoke:
        print("\n" + "="*60)
        print("STEP 5: 500-step Training Sanity")
        print("="*60)

        train_results = {}
        for clip_val in clip_values:
            if clip_val == 0.0:
                continue  # skip baseline training
            print(f"\n  Training radius_clip={clip_val}...")

            m = means.clone().detach().requires_grad_(True)
            s = scales.clone().detach().requires_grad_(True)
            q = quats.clone().detach().requires_grad_(True)
            o = opacities.clone().detach().requires_grad_(True)
            sh = shs.clone().detach().requires_grad_(True)

            optim = torch.optim.Adam([
                {"params": [m], "lr": 1.6e-4},
                {"params": [s], "lr": 5e-3},
                {"params": [q], "lr": 1e-3},
                {"params": [o], "lr": 5e-2},
                {"params": [sh], "lr": 2.5e-3 / 20.0},
            ])

            train_indices = list(range(0, C, 4))[:16]  # subset for speed
            fwd_ms, bwd_ms, opt_ms = [], [], []
            losses, psnrs = [], []
            nan_flag, inf_flag = False, False

            for step in range(min(200, 500) if args.smoke else 500):
                idx = train_indices[step % len(train_indices)]
                vm_i = viewmats[idx:idx+1]; k_i = Ks[idx:idx+1]
                gt_np = gt_images.get(img_names[idx])
                if gt_np is None:
                    continue
                if gt_np.shape[:2] != (render_h, render_w):
                    from skimage.transform import resize
                    gt_np = resize(gt_np, (render_h, render_w), anti_aliasing=True, preserve_range=True)
                gt = torch.from_numpy(gt_np).cuda()

                t0 = time.perf_counter()
                rendered, _ = render(m, q, s, o, sh, vm_i, k_i, render_w, render_h, sh_degree, radius_clip=clip_val)
                img = rendered[0].clamp(0, 1)
                t1 = time.perf_counter(); fwd_ms.append((t1-t0)*1000)

                loss = ((img - gt)**2).mean()
                t1b = time.perf_counter()

                optim.zero_grad()
                loss.backward()
                t2 = time.perf_counter(); bwd_ms.append((t2-t1b)*1000)

                # Check gradients
                for pt in [m, s, q, o, sh]:
                    if not pt.grad.isfinite().all():
                        nan_flag = True
                    if torch.isinf(pt.grad).any():
                        inf_flag = True

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_([m, s, q, o, sh], 1.0)

                optim.step()
                t3 = time.perf_counter(); opt_ms.append((t3-t2)*1000)

                with torch.no_grad():
                    o.data = o.data.clamp(-10, 10)

                loss_val = loss.item()
                psnr_val = 10 * math.log10(1.0 / (loss_val + 1e-10))
                losses.append(loss_val); psnrs.append(psnr_val)

                if (step+1) % 50 == 0:
                    print(f"    step {step+1}: loss={loss_val:.6f}, PSNR={psnr_val:.2f}, fwd={fwd_ms[-1]:.2f}ms, bwd={bwd_ms[-1]:.2f}ms")

            key = f"rclip_{clip_val}"
            train_results[key] = {
                "radius_clip": clip_val,
                "final_loss": round(losses[-1], 8),
                "final_psnr": round(psnrs[-1], 4),
                "fwd_ms_mean": round(np.mean(fwd_ms), 4),
                "bwd_ms_mean": round(np.mean(bwd_ms), 4),
                "opt_ms_mean": round(np.mean(opt_ms), 4),
                "nan_detected": nan_flag,
                "inf_detected": inf_flag,
            }
            print(f"  => final PSNR={psnrs[-1]:.2f}, fwd={train_results[key]['fwd_ms_mean']:.2f}ms, nan={nan_flag}")

            del m, s, q, o, sh, optim
            torch.cuda.empty_cache()

        # Also run baseline training (radius_clip=0) for comparison
        print("\n  Training radius_clip=0 (baseline)...")
        m = means.clone().detach().requires_grad_(True)
        s = scales.clone().detach().requires_grad_(True)
        q = quats.clone().detach().requires_grad_(True)
        o = opacities.clone().detach().requires_grad_(True)
        sh = shs.clone().detach().requires_grad_(True)
        optim = torch.optim.Adam([
            {"params": [m], "lr": 1.6e-4},
            {"params": [s], "lr": 5e-3},
            {"params": [q], "lr": 1e-3},
            {"params": [o], "lr": 5e-2},
            {"params": [sh], "lr": 2.5e-3 / 20.0},
        ])
        train_indices = list(range(0, C, 4))[:16]
        for step in range(500):
            idx = train_indices[step % len(train_indices)]
            vm_i = viewmats[idx:idx+1]; k_i = Ks[idx:idx+1]
            gt_np = gt_images.get(img_names[idx])
            if gt_np is None: continue
            if gt_np.shape[:2] != (render_h, render_w):
                from skimage.transform import resize
                gt_np = resize(gt_np, (render_h, render_w), anti_aliasing=True, preserve_range=True)
            gt = torch.from_numpy(gt_np).cuda()
            rendered, _ = render(m, q, s, o, sh, vm_i, k_i, render_w, render_h, sh_degree, radius_clip=0.0)
            loss = ((rendered[0].clamp(0, 1) - gt)**2).mean()
            optim.zero_grad()
            loss.backward()
            for pt in [m, s, q, o, sh]:
                if not pt.grad.isfinite().all(): nan_flag = True
            torch.nn.utils.clip_grad_norm_([m, s, q, o, sh], 1.0)
            optim.step()
            with torch.no_grad(): o.data = o.data.clamp(-10, 10)
        train_results["rclip_0.0"] = {
            "radius_clip": 0.0, "final_loss": round(loss.item(), 8),
            "final_psnr": round(10*math.log10(1.0/(loss.item()+1e-10)), 4),
        }
        del m, s, q, o, sh, optim
        torch.cuda.empty_cache()

        with open(output_dir / "m4_sanity_training.json", "w") as f:
            json.dump(train_results, f, indent=2, default=str)
    else:
        train_results = {"skipped": True}

    # Consolidate
    all_results = {
        "metadata": {"scene": args.scene, "n_gaussians": means.shape[0], "clip_values": clip_values},
        "forward": fwd_results,
        "gradient": grad_results,
        "gt_quality": qual_results,
        "training": train_results,
    }
    with open(output_dir / "m4_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDone. Results in {output_dir / 'm4_results.json'}")

if __name__ == "__main__":
    main()
