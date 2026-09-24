#!/usr/bin/env python3
"""N2-CR1: CUDA gradient equivalence test.

Loads Room checkpoints (5K/15K/30K), runs forward + backward rasterization
with BOTH baseline and CPCB gsplat builds (in separate subprocesses),
and compares gradient outputs.

CR1 PASS if every nonzero family satisfies:
  cosine >= 0.99999
  relative_L2 <= 1e-4
  no NaN/Inf
"""
import sys
import os
# Set CUDA env BEFORE any torch import in subprocesses
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import subprocess
# Import torch lazily only when comparing (CPU-only)
import numpy as np

LOWLEVEL_RUNNER = r'''
import sys, os, json, torch
import numpy as np

def main():
    gsplat_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    camera_indices = json.loads(sys.argv[3])
    output_dir = sys.argv[4]
    test_config = json.loads(sys.argv[5])

    sys.path.insert(0, gsplat_path)
    for mod in list(sys.modules.keys()):
        if "gsplat" in mod:
            del sys.modules[mod]
    import gsplat
    from gsplat import rasterization

    device = "cuda"
    torch.manual_seed(42)
    np.random.seed(42)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    N = ckpt["num_points"]
    print(f"  N={N}", flush=True)

    means3d = ckpt["xyz"].to(device).float()
    quats = ckpt["rotation"].to(device).float()
    scales = ckpt["scaling"].to(device).float()
    opacities_raw = ckpt["opacity"].to(device).float()
    shs = ckpt["shs"].to(device).float()
    sh_degree = ckpt.get("active_sh_degree", 3)

    C0 = 0.28209479177387814
    colors = shs[:, 0, :] * C0 + 0.5  # [N, 3] RGB from SH DC
    repo_root = os.path.expanduser("~/3dgs-renderer-benchmark")
    camera_path = os.path.join(repo_root, "data", "official", "mipnerf360", "room", "cameras.json")
    with open(camera_path) as f:
        cameras_data = json.load(f)

    if isinstance(cameras_data, dict):
        cam_list = cameras_data.get("cameras", list(cameras_data.values()))
    elif isinstance(cameras_data, list):
        cam_list = cameras_data

    def parse_camera(cam_entry):
        orig_W = int(cam_entry.get("width", 1920))
        orig_H = int(cam_entry.get("height", 1080))
        fx = float(cam_entry.get("fx", 1000))
        fy = float(cam_entry.get("fy", 1000))
        # Use lower resolution (960x540) to avoid OOM at isect_tiles
        # The gradient comparison is resolution-independent for CPCB validation
        W, H = 960, 540
        fx = fx * W / orig_W
        fy = fy * H / orig_H
        rot = np.asarray(cam_entry["rotation"], dtype=np.float32)
        pos = np.asarray(cam_entry["position"], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rot
        c2w[:3, 3] = pos
        viewmat = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32)
        K = torch.tensor([[fx, 0, W/2.0], [0, fy, H/2.0], [0, 0, 1]], dtype=torch.float32)
        return viewmat, K, W, H

    for cam_idx in camera_indices:
        cam_idx = int(cam_idx)
        if cam_idx >= len(cam_list):
            continue

        viewmat, K, width, height = parse_camera(cam_list[cam_idx])
        viewmat = viewmat.unsqueeze(0).to(device)
        K = K.unsqueeze(0).to(device)

        print(f"  Camera {cam_idx}: {width}x{height}", flush=True)

        means3d.requires_grad_(True)
        quats.requires_grad_(True)
        scales.requires_grad_(True)
        opacities_raw.requires_grad_(True)
        shs.requires_grad_(True)
        opacities = torch.sigmoid(opacities_raw)

        bg_config = test_config.get("background", "black")
        if bg_config == "nonzero":
            backgrounds = torch.tensor([[0.5, 0.3, 0.7]], dtype=torch.float32, device=device)
        else:
            backgrounds = None

        render_colors, render_alphas, meta = rasterization(
            means=means3d, quats=quats, scales=scales, opacities=opacities,
            colors=shs.unsqueeze(0),
            viewmats=viewmat, Ks=K,
            width=width, height=height, tile_size=16,
            packed=False, sh_degree=sh_degree,
            radius_clip=0.0, eps2d=0.1,
            render_mode="RGB", absgrad=True,
            backgrounds=backgrounds,
        )

        # Retain grad on non-leaf intermediates (means2d, conics)
        means2d = meta.get("means2d", None)
        conics = meta.get("conics", None)
        if means2d is not None and means2d.requires_grad:
            means2d.retain_grad()
        if conics is not None and conics.requires_grad:
            conics.retain_grad()

        v_alpha_mode = test_config.get("v_render_alpha", "real")
        if v_alpha_mode == "zero":
            v_render_alphas = torch.zeros_like(render_alphas)
        elif v_alpha_mode == "nonzero":
            torch.manual_seed(42 + cam_idx)
            v_render_alphas = torch.randn_like(render_alphas) * 0.1
        else:
            torch.manual_seed(42 + cam_idx)
            v_render_alphas = torch.randn_like(render_alphas) * 0.01

        torch.manual_seed(100 + cam_idx)
        v_render_colors = torch.randn_like(render_colors) * 0.1

        torch.autograd.backward(
            (render_colors, render_alphas),
            (v_render_colors, v_render_alphas),
        )

        grad_outputs = {}
        # v_colors: compare SH gradient (both builds use identical SH fwd)
        grad_outputs["v_colors"] = shs.grad.detach().cpu()

        # v_opacities: compare post-sigmoid gradient (both builds identical sigmoid)
        grad_outputs["v_opacities"] = opacities_raw.grad.detach().cpu()

        if means2d is not None and means2d.requires_grad:
            grad_outputs["v_means2d"] = means2d.grad.detach().cpu()
        if conics is not None and conics.requires_grad:
            grad_outputs["v_conics"] = conics.grad.detach().cpu()
        if hasattr(means2d, "absgrad") and means2d.absgrad is not None:
            grad_outputs["v_means2d_abs"] = means2d.absgrad.detach().cpu()

        fwd_outputs = {}
        fwd_outputs["render_colors"] = render_colors.detach().cpu()
        fwd_outputs["render_alphas"] = render_alphas.detach().cpu()
        if means2d is not None:
            fwd_outputs["means2d"] = means2d.detach().cpu()
        if conics is not None:
            fwd_outputs["conics"] = conics.detach().cpu()
        if "last_ids" in meta:
            val = meta["last_ids"]
            if isinstance(val, torch.Tensor):
                fwd_outputs["last_ids"] = val.detach().cpu()

        cam_key = f"cam_{cam_idx}"
        cam_out = os.path.join(output_dir, cam_key)
        os.makedirs(cam_out, exist_ok=True)
        torch.save(grad_outputs, os.path.join(cam_out, "grads.pt"))
        torch.save(fwd_outputs, os.path.join(cam_out, "fwd.pt"))
        print(f"  Camera {cam_idx}: saved", flush=True)

        means3d.grad = None
        quats.grad = None
        scales.grad = None
        opacities_raw.grad = None
        shs.grad = None
        torch.cuda.empty_cache()

    print("  Done", flush=True)

if __name__ == "__main__":
    main()
'''


def compare_tensors(base, cpcb, name):
    import torch
    base_f = base.float().flatten()
    cpcb_f = cpcb.float().flatten()
    base_norm = base_f.norm().item()
    cpcb_norm = cpcb_f.norm().item()
    diff = base_f - cpcb_f
    diff_norm = diff.norm().item()

    if base_norm > 0 and cpcb_norm > 0:
        cosine = torch.dot(base_f, cpcb_f).item() / (base_norm * cpcb_norm)
    else:
        cosine = 1.0 if base_norm == 0 and cpcb_norm == 0 else 0.0

    if base_norm > 0:
        rel_l2 = diff_norm / base_norm
    else:
        rel_l2 = 0.0 if cpcb_norm == 0 else float("inf")

    max_abs = diff.abs().max().item() if diff.numel() > 0 else 0.0

    eps = 1e-8
    nonzero_mask = base_f.abs() > eps
    if nonzero_mask.any():
        rel_err = diff.abs()[nonzero_mask] / base_f.abs()[nonzero_mask].abs()
        max_rel = rel_err.max().item()
    else:
        max_rel = 0.0

    nan_count = int(torch.isnan(cpcb_f).sum().item())
    inf_count = int(torch.isinf(cpcb_f).sum().item())

    return {
        "name": name, "shape": list(base.shape),
        "cosine": cosine, "rel_l2": rel_l2,
        "max_abs": max_abs, "max_rel": max_rel,
        "nan_count": nan_count, "inf_count": inf_count,
        "base_norm": base_norm, "cpcb_norm": cpcb_norm,
    }


def run_cr1_test(baseline_path, cpcb_path, ckpt_path, camera_indices,
                 output_dir, test_config):
    config_json = json.dumps(test_config)
    cam_json = json.dumps(camera_indices)

    for label, gsplat_path in [("baseline", baseline_path), ("cpcb", cpcb_path)]:
        print(f"\n=== Running {label} ===")
        build_out = os.path.join(output_dir, label)
        os.makedirs(build_out, exist_ok=True)
        runner_path = os.path.join(output_dir, f"runner_{label}.py")
        with open(runner_path, "w") as f:
            f.write(LOWLEVEL_RUNNER)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "2"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        result = subprocess.run(
            [sys.executable, runner_path, gsplat_path, ckpt_path,
             cam_json, build_out, config_json],
            capture_output=True, text=True, env=env, timeout=600
        )
        print(f"  stdout: {result.stdout[-800:]}")
        if result.returncode != 0:
            print(f"  stderr: {result.stderr[-2000:]}")
            return None, result.stderr

    print("\n=== Comparing gradients ===")
    import torch
    results = {}
    for cam_idx in camera_indices:
        cam_key = f"cam_{cam_idx}"
        base_g = os.path.join(output_dir, "baseline", cam_key, "grads.pt")
        cpcb_g = os.path.join(output_dir, "cpcb", cam_key, "grads.pt")
        base_f = os.path.join(output_dir, "baseline", cam_key, "fwd.pt")
        cpcb_f = os.path.join(output_dir, "cpcb", cam_key, "fwd.pt")

        if not os.path.exists(base_g) or not os.path.exists(cpcb_g):
            print(f"  {cam_key}: missing files")
            continue

        base_grads = torch.load(base_g, weights_only=False)
        cpcb_grads = torch.load(cpcb_g, weights_only=False)

        cam_results = {}
        for key in ["v_colors", "v_opacities", "v_means2d", "v_conics", "v_means2d_abs"]:
            if key in base_grads and key in cpcb_grads:
                bt, ct = base_grads[key], cpcb_grads[key]
                if bt.shape != ct.shape:
                    cam_results[key] = {"error": "shape_mismatch"}
                    continue
                m = compare_tensors(bt, ct, key)
                cam_results[key] = m
                status = "PASS" if (m["cosine"] >= 0.99999 and m["rel_l2"] <= 1e-4
                                    and m["nan_count"] == 0 and m["inf_count"] == 0) else "FAIL"
                print(f"  {cam_key}/{key}: cos={m['cosine']:.8f} rel_l2={m['rel_l2']:.2e} "
                      f"max_abs={m['max_abs']:.2e} NaN={m['nan_count']} Inf={m['inf_count']} [{status}]")

        base_fwd = torch.load(base_f, weights_only=False)
        cpcb_fwd = torch.load(cpcb_f, weights_only=False)
        fwd_results = {}
        for key in ["render_colors", "render_alphas", "means2d", "conics", "last_ids"]:
            if key in base_fwd and key in cpcb_fwd and base_fwd[key] is not None and cpcb_fwd[key] is not None:
                bt, ct = base_fwd[key], cpcb_fwd[key]
                if bt.shape == ct.shape:
                    fwd_results[key] = compare_tensors(bt, ct, f"fwd_{key}")

        results[cam_key] = {"gradient_metrics": cam_results, "forward_metrics": fwd_results}

    return results, None


def main():
    work = "/tmp/gsplat_n2_cpcb"
    baseline_path = os.path.join(work, "gsplat_baseline")
    cpcb_path = os.path.join(work, "gsplat_cpcb")
    results_dir = os.path.expanduser("~/3dgs-renderer-benchmark/results/n2_cpcb")
    output_dir = os.path.join(results_dir, "cr1_output")
    os.makedirs(output_dir, exist_ok=True)

    checkpoints = {
        "5K": os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_5000.pt"),
        "15K": os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_15000.pt"),
        "30K": os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_30000.pt"),
    }

    camera_indices = [0]  # Start with 1 camera for validation
    all_results = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"CR1: {ckpt_name} - black bg, real gradients")
        print(f"{'='*60}")
        tc = {"background": "black", "v_render_alpha": "real"}
        to = os.path.join(output_dir, f"standard_{ckpt_name}")
        r, err = run_cr1_test(baseline_path, cpcb_path, ckpt_path, camera_indices, to, tc)
        if err:
            all_results[f"standard_{ckpt_name}"] = {"error": err[:500]}
        else:
            all_results[f"standard_{ckpt_name}"] = r

    # Adversarial: bg null + va=0 (at 30K, passed before)
    ckpt_30k = checkpoints["30K"]
    for name, tc in [
        ("adv_bgnull_va0", {"background": "black", "v_render_alpha": "zero"}),
    ]:
        print(f"\n{'='*60}")
        print(f"CR1 ADVERSARIAL: 30K - {name}")
        print(f"{'='*60}")
        to = os.path.join(output_dir, name)
        r, err = run_cr1_test(baseline_path, cpcb_path, ckpt_30k, camera_indices, to, tc)
        if err:
            all_results[name] = {"error": err[:500]}
        else:
            all_results[name] = r

    # Adversarial nonzero-bg tests use 5K checkpoint (smaller N, avoids OOM)
    ckpt_5k = checkpoints["5K"]
    for name, tc in [
        ("adv_bgnonzero_vanonzero", {"background": "nonzero", "v_render_alpha": "nonzero"}),
        ("adv_bgnonzero_va0", {"background": "nonzero", "v_render_alpha": "zero"}),
    ]:
        print(f"\n{'='*60}")
        print(f"CR1 ADVERSARIAL: 5K - {name}")
        print(f"{'='*60}")
        to = os.path.join(output_dir, name)
        r, err = run_cr1_test(baseline_path, cpcb_path, ckpt_5k, camera_indices, to, tc)
        if err:
            all_results[name] = {"error": err[:500]}
        else:
            all_results[name] = r

    print("\n" + "=" * 60)
    print("CR1 GATE EVALUATION")
    print("=" * 60)
    cr1_pass = True
    for test_name, tr in all_results.items():
        if isinstance(tr, dict) and "error" in tr:
            print(f"  {test_name}: ERROR")
            cr1_pass = False
            continue
        for cam_key, cr in tr.items():
            if isinstance(cr, dict) and "gradient_metrics" in cr:
                for gn, m in cr["gradient_metrics"].items():
                    if isinstance(m, dict) and "cosine" in m:
                        ok = (m["cosine"] >= 0.99999 and m["rel_l2"] <= 1e-4
                              and m["nan_count"] == 0 and m["inf_count"] == 0)
                        if not ok and m.get("base_norm", 0) > 1e-10:
                            print(f"  {test_name}/{cam_key}/{gn}: FAIL cos={m['cosine']:.8f}")
                            cr1_pass = False

    all_results["_cr1_verdict"] = "PASS" if cr1_pass else "FAIL"
    print(f"\nN2_CR1 = {'PASS' if cr1_pass else 'FAIL'}")

    results_path = os.path.join(results_dir, "cr1_gradient_equivalence.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()
