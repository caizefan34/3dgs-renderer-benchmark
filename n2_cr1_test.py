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
import json
import subprocess
import tempfile
import torch
import numpy as np
from pathlib import Path

# ====================================================================
# Single-build gradient computation script (run as subprocess)
# ====================================================================
GRAD_RUNNER = r"""
import sys, os, json, torch
import numpy as np

def main():
    # Args: gsplat_path, ckpt_path, camera_indices_json, output_dir, test_config_json
    gsplat_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    camera_indices = json.loads(sys.argv[3])
    output_dir = sys.argv[4]
    test_config = json.loads(sys.argv[5])

    # Force load the specific gsplat build
    sys.path.insert(0, gsplat_path)
    # Remove any cached gsplat
    for mod in list(sys.modules.keys()):
        if 'gsplat' in mod:
            del sys.modules[mod]
    import gsplat
    from gsplat import rasterization

    device = "cuda"
    torch.manual_seed(42)
    np.random.seed(42)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    N = ckpt['num_points']
    print(f"  Loaded checkpoint: N={N}", flush=True)

    # Extract Gaussian parameters
    means = ckpt['xyz'].to(device).float()
    quats = ckpt['rotation'].to(device).float()
    scales = ckpt['scaling'].to(device).float()
    opacities = ckpt['opacity'].to(device).float()
    # SH features: [N, K, 3] -> need [1, N, K, 3] for rasterization
    shs = ckpt['shs'].to(device).float()
    sh_degree = ckpt.get('active_sh_degree', 3)

    # Load cameras from the dataset
    # We need viewmats and Ks for each camera
    # Load from cameras.json
    repo_root = os.path.expanduser('~/3dgs-renderer-benchmark')
    camera_path = os.path.join(repo_root, 'data', 'official', 'mipnerf360', 'room', 'cameras.json')

    import json as jsonmod
    with open(camera_path) as f:
        cameras_data = jsonmod.load(f)

    # Parse cameras - need to understand the format
    # cameras.json should have camera intrinsics and extrinsics
    # Let's parse it
    if isinstance(cameras_data, dict):
        # Could be {"cameras": [...]} or {"<name>": {...}, ...}
        if 'cameras' in cameras_data:
            cam_list = cameras_data['cameras']
        else:
            cam_list = list(cameras_data.values())
    elif isinstance(cameras_data, list):
        cam_list = cameras_data
    else:
        raise ValueError(f"Unknown camera format: {type(cameras_data)}")

    print(f"  Loaded {len(cam_list)} cameras from cameras.json", flush=True)

    # Parse camera to viewmat [4,4] and K [3,3]
    def parse_camera(cam_entry):
        # 3DGS JSON format: rotation (3x3 c2w), position (3-vec), fx, fy, width, height
        # Resize to 1080p to match trainer and avoid OOM
        orig_W, orig_H = int(cam_entry.get('width', 1920)), int(cam_entry.get('height', 1080))
        fx, fy = float(cam_entry.get('fx', 1000)), float(cam_entry.get('fy', 1000))
        # Resize to 1920x1080
        W, H = 1920, 1080
        fx = fx * W / orig_W
        fy = fy * H / orig_H
        rot = np.asarray(cam_entry['rotation'], dtype=np.float32)
        pos = np.asarray(cam_entry['position'], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rot
        c2w[:3, 3] = pos
        viewmat = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32)
        K = torch.tensor([[fx, 0, W/2.0], [0, fy, H/2.0], [0, 0, 1]], dtype=torch.float32)
        return viewmat, K, W, H

    results = {}

    for cam_idx in camera_indices:
        cam_idx = int(cam_idx)
        if cam_idx >= len(cam_list):
            print(f"  Skipping camera {cam_idx} (out of range)", flush=True)
            continue

        viewmat, K, width, height = parse_camera(cam_list[cam_idx])
        viewmat = viewmat.unsqueeze(0).to(device)  # [1, 4, 4]
        K = K.unsqueeze(0).to(device)  # [1, 3, 3]

        print(f"  Camera {cam_idx}: {width}x{height}", flush=True)

        # Run forward rasterization
        # colors = SH coefficients [N, K, 3] -> need [1, N, K, 3]
        colors = shs.unsqueeze(0)  # [1, N, K, 3]

        # Determine background
        bg_config = test_config.get('background', 'black')
        if bg_config == 'black' or bg_config is None:
            backgrounds = None
        elif bg_config == 'nonzero':
            backgrounds = torch.tensor([0.5, 0.3, 0.7], dtype=torch.float32, device=device).unsqueeze(0)
        else:
            backgrounds = None

        # Forward
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            tile_size=16,
            packed=True,
            sh_degree=sh_degree,
            radius_clip=0.0,
            eps2d=0.1,
            render_mode="RGB",
            absgrad=True,
            backgrounds=backgrounds,
        )

        # render_colors: [1, H, W, 3], render_alphas: [1, H, W, 1]
        rendered = render_colors[0]  # [H, W, 3]

        # Create upstream gradients
        # v_render_colors: gradient w.r.t. rendered colors
        # v_render_alphas: gradient w.r.t. rendered alphas
        v_alpha_mode = test_config.get('v_render_alpha', 'real')
        if v_alpha_mode == 'zero':
            v_render_alphas = torch.zeros_like(render_alphas)
        elif v_alpha_mode == 'nonzero':
            torch.manual_seed(42 + cam_idx)
            v_render_alphas = torch.randn_like(render_alphas) * 0.1
        else:  # 'real' - use L1 loss gradient
            # Simulate L1 loss: v_render_colors = sign(render - target)
            # For alpha, use the SSIM/density gradient
            v_render_alphas = torch.randn_like(render_alphas) * 0.01

        # v_render_colors: use a deterministic gradient
        torch.manual_seed(100 + cam_idx)
        v_render_colors = torch.randn_like(render_colors) * 0.1

        # Backward
        # We need to call backward through the autograd graph
        # The rasterization function returns render_colors and render_alphas
        # which have requires_grad=True (through means, quats, scales, opacities, colors)

        # Make sure params require grad
        means.requires_grad_(True)
        quats.requires_grad_(True)
        scales.requires_grad_(True)
        opacities.requires_grad_(True)
        shs.requires_grad_(True)

        # Re-run forward to build autograd graph (with grad enabled)
        render_colors2, render_alphas2, meta2 = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=shs.unsqueeze(0),
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            tile_size=16,
            packed=True,
            sh_degree=sh_degree,
            radius_clip=0.0,
            eps2d=0.1,
            render_mode="RGB",
            absgrad=True,
            backgrounds=backgrounds,
        )

        # Apply backward with our synthetic gradients
        torch.autograd.backward(
            (render_colors2, render_alphas2),
            (v_render_colors, v_render_alphas),
        )

        # Collect gradient outputs
        # The rasterization backward returns:
        # v_means, v_quats, v_scales, v_opacities, v_colors (SH grads)
        # AND the rasterize_to_pixels_3dgs_bwd kernel computes:
        # v_colors (RGB channel grads), v_opacities, v_means2d, v_conics, v_means2d_abs
        # The meta dict should contain means2d with absgrad

        means2d = meta2.get('means2d', None)

        grad_outputs = {}
        grad_outputs['v_means_xyz'] = means.grad.detach().cpu()  # [N, 3]
        grad_outputs['v_quats'] = quats.grad.detach().cpu()  # [N, 4]
        grad_outputs['v_scales'] = scales.grad.detach().cpu()  # [N, 3]
        grad_outputs['v_opacities_raw'] = opacities.grad.detach().cpu()  # [N]
        grad_outputs['v_colors_sh'] = shs.grad.detach().cpu()  # [N, K, 3]

        # Extract means2d gradients (v_means2d, v_conics, v_means2d_abs)
        if means2d is not None and means2d.requires_grad:
            grad_outputs['v_means2d'] = means2d.grad.detach().cpu()  # [1, N, 2]

        # The conics gradient
        conics = meta2.get('conics', None)
        if conics is not None and conics.requires_grad:
            grad_outputs['v_conics'] = conics.grad.detach().cpu()  # [1, N, 3]

        # absgrad
        if hasattr(means2d, 'absgrad') and means2d.absgrad is not None:
            grad_outputs['v_means2d_abs'] = means2d.absgrad.detach().cpu()  # [1, N, 2]

        # Forward outputs for verification
        fwd_outputs = {}
        fwd_outputs['render_colors'] = render_colors2.detach().cpu()
        fwd_outputs['render_alphas'] = render_alphas2.detach().cpu()
        fwd_outputs['last_ids'] = meta2.get('last_ids', None)
        if fwd_outputs['last_ids'] is not None:
            if isinstance(fwd_outputs['last_ids'], torch.Tensor):
                fwd_outputs['last_ids'] = fwd_outputs['last_ids'].detach().cpu()
        if means2d is not None:
            fwd_outputs['means2d'] = means2d.detach().cpu()

        # Save to output dir
        cam_key = f"cam_{cam_idx}"
        cam_out = os.path.join(output_dir, cam_key)
        os.makedirs(cam_out, exist_ok=True)
        torch.save(grad_outputs, os.path.join(cam_out, 'grads.pt'))
        torch.save(fwd_outputs, os.path.join(cam_out, 'fwd.pt'))

        # Also save the rasterize_to_pixels-level gradients directly
        # by calling the low-level API
        # Actually, the high-level rasterization() already computes these
        # The v_opacities from rasterize_to_pixels_bwd is NOT the same as
        # opacities.grad (which goes through sigmoid). We need the raw
        # rasterize_to_pixels_bwd outputs.

        print(f"  Camera {cam_idx}: saved grads and fwd outputs", flush=True)

    print("  Done", flush=True)

if __name__ == '__main__':
    main()
"""

# ====================================================================
# Low-level gradient runner - directly calls rasterize_to_pixels
# to get the exact kernel-level gradients (v_colors, v_opacities,
# v_means2d, v_conics, v_means2d_abs)
# ====================================================================
LOWLEVEL_RUNNER = r"""
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
        if 'gsplat' in mod:
            del sys.modules[mod]
    import gsplat
    from gsplat import rasterization

    device = "cuda"
    torch.manual_seed(42)
    np.random.seed(42)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    N = ckpt['num_points']
    print(f"  Loaded checkpoint: N={N}", flush=True)

    means3d = ckpt['xyz'].to(device).float()
    quats = ckpt['rotation'].to(device).float()
    scales = ckpt['scaling'].to(device).float()
    opacities_raw = ckpt['opacity'].to(device).float()
    # Apply sigmoid to get [0,1] opacity
    opacities = torch.sigmoid(opacities_raw)
    shs = ckpt['shs'].to(device).float()
    sh_degree = ckpt.get('active_sh_degree', 3)

    # Compute SH colors (DC only for simplicity - gives us RGB)
    C0 = 0.28209479177387814
    colors_sh = shs[:, 0, :] * C0 + 0.5  # [N, 3] - SH to RGB
    colors = colors_sh  # [N, 3]

    # Load cameras
    repo_root = os.path.expanduser('~/3dgs-renderer-benchmark')
    camera_path = os.path.join(repo_root, 'data', 'official', 'mipnerf360', 'room', 'cameras.json')
    with open(camera_path) as f:
        cameras_data = json.load(f)

    if isinstance(cameras_data, dict):
        if 'cameras' in cameras_data:
            cam_list = cameras_data['cameras']
        else:
            cam_list = list(cameras_data.values())
    elif isinstance(cameras_data, list):
        cam_list = cameras_data

    def parse_camera(cam_entry):
        orig_W, orig_H = int(cam_entry.get('width', 1920)), int(cam_entry.get('height', 1080))
        fx, fy = float(cam_entry.get('fx', 1000)), float(cam_entry.get('fy', 1000))
        W, H = 1920, 1080
        fx = fx * W / orig_W
        fy = fy * H / orig_H
        rot = np.asarray(cam_entry['rotation'], dtype=np.float32)
        pos = np.asarray(cam_entry['position'], dtype=np.float32)
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

        # Use high-level rasterization with autograd to capture gradients
        means3d.requires_grad_(True)
        quats.requires_grad_(True)
        scales.requires_grad_(True)
        opacities.requires_grad_(True)
        colors.requires_grad_(True)

        # Use sh_degree=None with colors [1, N, 3]
        bg_config = test_config.get('background', 'black')
        if bg_config == 'black' or bg_config is None:
            backgrounds = None
        elif bg_config == 'nonzero':
            backgrounds = torch.tensor([[0.5, 0.3, 0.7]], dtype=torch.float32, device=device)
        else:
            backgrounds = None

        render_colors, render_alphas, meta = rasterization(
            means=means3d, quats=quats, scales=scales, opacities=opacities,
            colors=colors.unsqueeze(0),  # [1, N, 3]
            viewmats=viewmat, Ks=K, width=width, height=height,
            tile_size=16, packed=True, sh_degree=None,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
            backgrounds=backgrounds,
        )

        # Synthetic upstream gradients
        v_alpha_mode = test_config.get('v_render_alpha', 'real')
        if v_alpha_mode == 'zero':
            v_render_alphas = torch.zeros_like(render_alphas)
        elif v_alpha_mode == 'nonzero':
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

        # Collect the rasterize_to_pixels_bwd-level gradients:
        # v_colors (per-Gaussian RGB), v_opacities (per-Gaussian opacity),
        # v_means2d, v_conics, v_means2d_abs
        means2d = meta.get('means2d', None)
        conics = meta.get('conics', None)

        grad_outputs = {}
        # These are the raw rasterize_to_pixels_bwd outputs (before projection backward)
        # colors.grad = v_colors from rasterize_to_pixels_bwd (for the RGB channel)
        grad_outputs['v_colors'] = colors.grad.detach().cpu()  # [N, 3]
        # opacities.grad goes through sigmoid, but the rasterize_to_pixels_bwd
        # computes v_opacities_raw. The grad here is dL/d(sigmoid_input) = v_opacities * sigmoid * (1-sigmoid)
        # We want the RAW v_opacities from the kernel = dL/d(opacity)
        # So we need to divide by sigmoid derivative
        sig = torch.sigmoid(opacities_raw)
        grad_outputs['v_opacities'] = (opacities.grad / (sig * (1 - sig))).detach().cpu()  # [N]

        if means2d is not None and means2d.requires_grad:
            grad_outputs['v_means2d'] = means2d.grad.detach().cpu()  # [1, N, 2]
        if conics is not None and conics.requires_grad:
            grad_outputs['v_conics'] = conics.grad.detach().cpu()  # [1, N, 3]
        if hasattr(means2d, 'absgrad') and means2d.absgrad is not None:
            grad_outputs['v_means2d_abs'] = means2d.absgrad.detach().cpu()  # [1, N, 2]

        # Forward outputs for verification
        fwd_outputs = {}
        fwd_outputs['render_colors'] = render_colors.detach().cpu()
        fwd_outputs['render_alphas'] = render_alphas.detach().cpu()
        fwd_outputs['means2d'] = means2d.detach().cpu() if means2d is not None else None
        fwd_outputs['conics'] = conics.detach().cpu() if conics is not None else None

        # Also save the rasterization meta info
        for key in ['last_ids', 'tile_offsets', 'flatten_ids', 'radii', 'depths']:
            if key in meta:
                val = meta[key]
                if isinstance(val, torch.Tensor):
                    fwd_outputs[key] = val.detach().cpu()

        cam_key = f"cam_{cam_idx}"
        cam_out = os.path.join(output_dir, cam_key)
        os.makedirs(cam_out, exist_ok=True)
        torch.save(grad_outputs, os.path.join(cam_out, 'grads.pt'))
        torch.save(fwd_outputs, os.path.join(cam_out, 'fwd.pt'))
        print(f"  Camera {cam_idx}: saved lowlevel grads and fwd", flush=True)

    print("  Done", flush=True)

if __name__ == '__main__':
    main()
"""


def compare_tensors(base: torch.Tensor, cpcb: torch.Tensor, name: str) -> dict:
    """Compare two gradient tensors and return metrics."""
    base_f = base.float().flatten()
    cpcb_f = cpcb.float().flatten()

    # Handle zero tensors
    base_norm = base_f.norm().item()
    cpcb_norm = cpcb_f.norm().item()

    diff = (base_f - cpcb_f)
    diff_norm = diff.norm().item()

    # Cosine similarity
    if base_norm > 0 and cpcb_norm > 0:
        cosine = torch.dot(base_f, cpcb_f).item() / (base_norm * cpcb_norm)
    else:
        cosine = 1.0 if base_norm == 0 and cpcb_norm == 0 else 0.0

    # Relative L2
    if base_norm > 0:
        rel_l2 = diff_norm / base_norm
    else:
        rel_l2 = 0.0 if cpcb_norm == 0 else float('inf')

    # Max absolute error
    max_abs = diff.abs().max().item() if diff.numel() > 0 else 0.0

    # Max relative elementwise error
    eps = 1e-8
    nonzero_mask = base_f.abs() > eps
    if nonzero_mask.any():
        rel_err = (diff.abs()[nonzero_mask] / base_f.abs()[nonzero_mask].abs())
        max_rel = rel_err.max().item()
    else:
        max_rel = 0.0

    # NaN/Inf counts
    nan_count = int(torch.isnan(cpcb_f).sum().item())
    inf_count = int(torch.isinf(cpcb_f).sum().item())

    return {
        'name': name,
        'shape': list(base.shape),
        'cosine': cosine,
        'rel_l2': rel_l2,
        'max_abs': max_abs,
        'max_rel': max_rel,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'base_norm': base_norm,
        'cpcb_norm': cpcb_norm,
    }


def run_cr1_test(
    baseline_path: str,
    cpcb_path: str,
    ckpt_path: str,
    camera_indices: list,
    output_dir: str,
    test_config: dict,
    runner_script: str,
):
    """Run gradient computation for both builds and compare."""
    config_json = json.dumps(test_config)
    cam_json = json.dumps(camera_indices)

    results = {}

    for label, gsplat_path in [("baseline", baseline_path), ("cpcb", cpcb_path)]:
        print(f"\n=== Running {label} ===")
        build_out = os.path.join(output_dir, label)
        os.makedirs(build_out, exist_ok=True)

        # Write runner script
        runner_path = os.path.join(output_dir, f"runner_{label}.py")
        with open(runner_path, 'w') as f:
            f.write(runner_script)

        # Run as subprocess
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        result = subprocess.run(
            [sys.executable, runner_path, gsplat_path, ckpt_path, cam_json, build_out, config_json],
            capture_output=True, text=True, env=env, timeout=300
        )
        print(f"  stdout: {result.stdout[:500]}")
        if result.returncode != 0:
            print(f"  stderr: {result.stderr[:2000]}")
            return None, result.stderr

    # Compare
    print("\n=== Comparing gradients ===")
    for cam_idx in camera_indices:
        cam_key = f"cam_{cam_idx}"
        base_grads_path = os.path.join(output_dir, "baseline", cam_key, "grads.pt")
        cpcb_grads_path = os.path.join(output_dir, "cpcb", cam_key, "grads.pt")
        base_fwd_path = os.path.join(output_dir, "baseline", cam_key, "fwd.pt")
        cpcb_fwd_path = os.path.join(output_dir, "cpcb", cam_key, "fwd.pt")

        if not os.path.exists(base_grads_path) or not os.path.exists(cpcb_grads_path):
            print(f"  {cam_key}: missing gradient files")
            continue

        base_grads = torch.load(base_grads_path, weights_only=False)
        cpcb_grads = torch.load(cpcb_grads_path, weights_only=False)

        cam_results = {}
        for key in ['v_colors', 'v_opacities', 'v_means2d', 'v_conics', 'v_means2d_abs']:
            if key in base_grads and key in cpcb_grads:
                base_t = base_grads[key]
                cpcb_t = cpcb_grads[key]
                if base_t.shape != cpcb_t.shape:
                    print(f"  {cam_key}/{key}: SHAPE MISMATCH {base_t.shape} vs {cpcb_t.shape}")
                    cam_results[key] = {'error': 'shape_mismatch'}
                    continue
                metrics = compare_tensors(base_t, cpcb_t, key)
                cam_results[key] = metrics
                status = "PASS" if (metrics['cosine'] >= 0.99999 and metrics['rel_l2'] <= 1e-4 and metrics['nan_count'] == 0 and metrics['inf_count'] == 0) else "FAIL"
                print(f"  {cam_key}/{key}: cosine={metrics['cosine']:.8f} rel_l2={metrics['rel_l2']:.2e} max_abs={metrics['max_abs']:.2e} NaN={metrics['nan_count']} Inf={metrics['inf_count']} [{status}]")

        # Compare forward outputs
        base_fwd = torch.load(base_fwd_path, weights_only=False)
        cpcb_fwd = torch.load(cpcb_fwd_path, weights_only=False)
        fwd_results = {}
        for key in ['render_colors', 'render_alphas', 'means2d', 'conics']:
            if key in base_fwd and key in cpcb_fwd and base_fwd[key] is not None and cpcb_fwd[key] is not None:
                base_t = base_fwd[key]
                cpcb_t = cpcb_fwd[key]
                if base_t.shape == cpcb_t.shape:
                    fwd_metrics = compare_tensors(base_t, cpcb_t, f"fwd_{key}")
                    fwd_results[key] = fwd_metrics

        results[cam_key] = {
            'gradient_metrics': cam_results,
            'forward_metrics': fwd_results,
        }

    return results, None


def main():
    work = "/tmp/gsplat_n2_cpcb"
    baseline_path = os.path.join(work, "gsplat_baseline")
    cpcb_path = os.path.join(work, "gsplat_cpcb")
    results_dir = os.path.expanduser("~/3dgs-renderer-benchmark/results/n2_cpcb")
    output_dir = os.path.join(results_dir, "cr1_output")
    os.makedirs(output_dir, exist_ok=True)

    checkpoints = {
        '5K': os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_5000.pt"),
        '15K': os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_15000.pt"),
        '30K': os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_30000.pt"),
    }

    # 5 fixed cameras (deterministic selection)
    camera_indices = [0, 50, 100, 150, 200]

    all_results = {}

    # Test 1: Standard (black background, real gradients)
    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"CR1 TEST: {ckpt_name} â€?black bg, real gradients")
        print(f"{'='*60}")
        test_config = {'background': 'black', 'v_render_alpha': 'real'}
        test_out = os.path.join(output_dir, f"standard_{ckpt_name}")
        results, err = run_cr1_test(
            baseline_path, cpcb_path, ckpt_path, camera_indices,
            test_out, test_config, LOWLEVEL_RUNNER
        )
        if err:
            print(f"ERROR: {err[:500]}")
            all_results[f"standard_{ckpt_name}"] = {'error': err[:500]}
        else:
            all_results[f"standard_{ckpt_name}"] = results

    # Test 2: Adversarial â€?null/black background, v_render_alpha=0
    ckpt_path = checkpoints['30K']
    print(f"\n{'='*60}")
    print(f"CR1 ADVERSARIAL: 30K â€?black bg, v_render_alpha=0")
    print(f"{'='*60}")
    test_config = {'background': 'black', 'v_render_alpha': 'zero'}
    test_out = os.path.join(output_dir, "adv_bgnull_va0")
    results, err = run_cr1_test(
        baseline_path, cpcb_path, ckpt_path, camera_indices,
        test_out, test_config, LOWLEVEL_RUNNER
    )
    if err:
        all_results['adv_bgnull_va0'] = {'error': err[:500]}
    else:
        all_results['adv_bgnull_va0'] = results

    # Test 3: Adversarial â€?nonzero background, v_render_alpha=nonzero
    print(f"\n{'='*60}")
    print(f"CR1 ADVERSARIAL: 30K â€?nonzero bg, v_render_alpha=nonzero")
    print(f"{'='*60}")
    test_config = {'background': 'nonzero', 'v_render_alpha': 'nonzero'}
    test_out = os.path.join(output_dir, "adv_bgnonzero_vanonzero")
    results, err = run_cr1_test(
        baseline_path, cpcb_path, ckpt_path, camera_indices,
        test_out, test_config, LOWLEVEL_RUNNER
    )
    if err:
        all_results['adv_bgnonzero_vanonzero'] = {'error': err[:500]}
    else:
        all_results['adv_bgnonzero_vanonzero'] = results

    # Test 4: Adversarial â€?nonzero background, v_render_alpha=0
    print(f"\n{'='*60}")
    print(f"CR1 ADVERSARIAL: 30K â€?nonzero bg, v_render_alpha=0")
    print(f"{'='*60}")
    test_config = {'background': 'nonzero', 'v_render_alpha': 'zero'}
    test_out = os.path.join(output_dir, "adv_bgnonzero_va0")
    results, err = run_cr1_test(
        baseline_path, cpcb_path, ckpt_path, camera_indices,
        test_out, test_config, LOWLEVEL_RUNNER
    )
    if err:
        all_results['adv_bgnonzero_va0'] = {'error': err[:500]}
    else:
        all_results['adv_bgnonzero_va0'] = results

    # Save results
    results_path = os.path.join(results_dir, "cr1_gradient_equivalence.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # CR1 gate
    print("\n" + "=" * 60)
    print("CR1 GATE EVALUATION")
    print("=" * 60)
    cr1_pass = True
    for test_name, test_results in all_results.items():
        if isinstance(test_results, dict) and 'error' in test_results:
            print(f"  {test_name}: ERROR - {test_results['error'][:100]}")
            cr1_pass = False
            continue
        for cam_key, cam_results in test_results.items():
            if isinstance(cam_results, dict) and 'gradient_metrics' in cam_results:
                for grad_name, metrics in cam_results['gradient_metrics'].items():
                    if isinstance(metrics, dict) and 'cosine' in metrics:
                        passed = (metrics['cosine'] >= 0.99999 and
                                 metrics['rel_l2'] <= 1e-4 and
                                 metrics['nan_count'] == 0 and
                                 metrics['inf_count'] == 0)
                        if not passed and metrics.get('base_norm', 0) > 1e-10:
                            print(f"  {test_name}/{cam_key}/{grad_name}: FAIL")
                            cr1_pass = False

    print(f"\nN2_CR1 = {'PASS' if cr1_pass else 'FAIL'}")

    # Add CR1 verdict to results
    all_results['_cr1_verdict'] = 'PASS' if cr1_pass else 'FAIL'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
