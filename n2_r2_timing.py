#!/usr/bin/env python3
"""N2-R2: A100 raster-backward + E2E timing benchmark.

Measures the exact RasterizeToPixels3DGS backward kernel separately,
plus full iteration, for BASELINE vs CPCB on Room 5K/15K/30K.

Protocol:
- warmup >= 50 iterations
- timed >= 300 iterations
- >= 5 independent timing repetitions
- CUDA Events with explicit synchronization
- Randomized/interleaved baseline/CPCB order
"""
import sys
import os
import json
import subprocess
import torch
import numpy as np
from pathlib import Path

# ====================================================================
# Single-build timing runner (run as subprocess)
# ====================================================================
TIMING_RUNNER = r"""
import sys, os, json, time, torch
import numpy as np

def main():
    gsplat_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    camera_indices = json.loads(sys.argv[3])
    output_path = sys.argv[4]
    config_json = json.loads(sys.argv[5])

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
    print(f"  N={N}", flush=True)

    means3d = ckpt['xyz'].to(device).float()
    quats = ckpt['rotation'].to(device).float()
    scales = ckpt['scaling'].to(device).float()
    opacities_raw = ckpt['opacity'].to(device).float()
    shs = ckpt['shs'].to(device).float()
    sh_degree = ckpt.get('active_sh_degree', 3)

    # Load cameras
    repo_root = os.path.expanduser('~/3dgs-renderer-benchmark')
    camera_path = os.path.join(repo_root, 'data', 'official', 'mipnerf360', 'room', 'cameras.json')
    with open(camera_path) as f:
        cameras_data = json.load(f)

    if isinstance(cameras_data, dict):
        cam_list = cameras_data.get('cameras', list(cameras_data.values()))
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

    # Pre-load all cameras
    cams = []
    for cam_idx in camera_indices:
        cam_idx = int(cam_idx)
        if cam_idx < len(cam_list):
            viewmat, K, width, height = parse_camera(cam_list[cam_idx])
            cams.append((viewmat.unsqueeze(0).to(device), K.unsqueeze(0).to(device), width, height, cam_idx))

    print(f"  {len(cams)} cameras loaded", flush=True)

    warmup = config_json.get('warmup', 50)
    timed = config_json.get('timed', 300)
    n_reps = config_json.get('n_reps', 5)
    measure_e2e = config_json.get('measure_e2e', True)

    # Pre-compute synthetic upstream gradients (fixed for all runs)
    torch.manual_seed(123)
    v_render_colors_list = []
    v_render_alphas_list = []
    for viewmat, K, width, height, cam_idx in cams:
        v_rc = torch.randn(1, height, width, 3, device=device) * 0.1
        v_ra = torch.randn(1, height, width, 1, device=device) * 0.01
        v_render_colors_list.append(v_rc)
        v_render_alphas_list.append(v_ra)

    all_rep_results = []

    for rep in range(n_reps):
        print(f"  Rep {rep+1}/{n_reps}", flush=True)

        # Warmup
        for _ in range(warmup):
            for i, (viewmat, K, width, height, cam_idx) in enumerate(cams):
                means3d.grad = None
                quats.grad = None
                scales.grad = None
                opacities_raw.grad = None
                shs.grad = None

                means3d.requires_grad_(True)
                quats.requires_grad_(True)
                scales.requires_grad_(True)
                opacities_raw.requires_grad_(True)
                shs.requires_grad_(True)

                render_colors, render_alphas, meta = rasterization(
                    means=means3d, quats=quats, scales=scales, opacities=opacities_raw,
                    colors=shs.unsqueeze(0), viewmats=viewmat, Ks=K,
                    width=width, height=height, tile_size=16, packed=True,
                    sh_degree=sh_degree, radius_clip=0.0, eps2d=0.1,
                    render_mode="RGB", absgrad=True,
                )
                torch.autograd.backward(
                    (render_colors, render_alphas),
                    (v_render_colors_list[i], v_render_alphas_list[i]),
                )
                torch.cuda.synchronize()

        # Timed runs - measure raster backward kernel specifically
        # We use CUDA events around the backward call
        bwd_times = []
        e2e_times = []
        fwd_times = []

        for _ in range(timed):
            for i, (viewmat, K, width, height, cam_idx) in enumerate(cams):
                means3d.grad = None
                quats.grad = None
                scales.grad = None
                opacities_raw.grad = None
                shs.grad = None

                means3d.requires_grad_(True)
                quats.requires_grad_(True)
                scales.requires_grad_(True)
                opacities_raw.requires_grad_(True)
                shs.requires_grad_(True)

                # E2E timing (forward + backward)
                start_e2e = torch.cuda.Event(enable_timing=True)
                end_e2e = torch.cuda.Event(enable_timing=True)
                start_fwd = torch.cuda.Event(enable_timing=True)
                end_fwd = torch.cuda.Event(enable_timing=True)
                start_bwd = torch.cuda.Event(enable_timing=True)
                end_bwd = torch.cuda.Event(enable_timing=True)

                start_e2e.record()

                # Forward
                start_fwd.record()
                render_colors, render_alphas, meta = rasterization(
                    means=means3d, quats=quats, scales=scales, opacities=opacities_raw,
                    colors=shs.unsqueeze(0), viewmats=viewmat, Ks=K,
                    width=width, height=height, tile_size=16, packed=True,
                    sh_degree=sh_degree, radius_clip=0.0, eps2d=0.1,
                    render_mode="RGB", absgrad=True,
                )
                end_fwd.record()
                torch.cuda.synchronize()
                fwd_ms = start_fwd.elapsed_time(end_fwd)

                # Backward
                start_bwd.record()
                torch.autograd.backward(
                    (render_colors, render_alphas),
                    (v_render_colors_list[i], v_render_alphas_list[i]),
                )
                end_bwd.record()
                torch.cuda.synchronize()
                bwd_ms = start_bwd.elapsed_time(end_bwd)

                end_e2e.record()
                torch.cuda.synchronize()
                e2e_ms = start_e2e.elapsed_time(end_e2e)

                bwd_times.append(bwd_ms)
                fwd_times.append(fwd_ms)
                e2e_times.append(e2e_ms)

        bwd_times = np.array(bwd_times)
        fwd_times = np.array(fwd_times)
        e2e_times = np.array(e2e_times)

        rep_result = {
            'rep': rep,
            'bwd_mean_ms': float(bwd_times.mean()),
            'bwd_median_ms': float(np.median(bwd_times)),
            'bwd_std_ms': float(bwd_times.std()),
            'bwd_p5_ms': float(np.percentile(bwd_times, 5)),
            'bwd_p95_ms': float(np.percentile(bwd_times, 95)),
            'fwd_mean_ms': float(fwd_times.mean()),
            'fwd_median_ms': float(np.median(fwd_times)),
            'e2e_mean_ms': float(e2e_times.mean()),
            'e2e_median_ms': float(np.median(e2e_times)),
            'e2e_std_ms': float(e2e_times.std()),
            'e2e_p5_ms': float(np.percentile(e2e_times, 5)),
            'e2e_p95_ms': float(np.percentile(e2e_times, 95)),
            'n_warmup': warmup,
            'n_timed': timed,
            'n_cameras': len(cams),
            'total_timed_samples': len(bwd_times),
        }
        all_rep_results.append(rep_result)
        print(f"    bwd_mean={rep_result['bwd_mean_ms']:.3f}ms e2e_mean={rep_result['e2e_mean_ms']:.3f}ms", flush=True)

    # Aggregate across reps
    bwd_means = [r['bwd_mean_ms'] for r in all_rep_results]
    e2e_means = [r['e2e_mean_ms'] for r in all_rep_results]
    fwd_means = [r['fwd_mean_ms'] for r in all_rep_results]

    final = {
        'checkpoint': ckpt_path,
        'N': N,
        'n_cameras': len(cams),
        'camera_indices': [c[4] for c in cams],
        'warmup': warmup,
        'timed': timed,
        'n_reps': n_reps,
        'reps': all_rep_results,
        'bwd_mean_ms': float(np.mean(bwd_means)),
        'bwd_median_ms': float(np.median(bwd_means)),
        'bwd_std_ms': float(np.std(bwd_means)),
        'bwd_p5_ms': float(np.percentile(bwd_means, 5)),
        'bwd_p95_ms': float(np.percentile(bwd_means, 95)),
        'fwd_mean_ms': float(np.mean(fwd_means)),
        'fwd_median_ms': float(np.median(fwd_means)),
        'e2e_mean_ms': float(np.mean(e2e_means)),
        'e2e_median_ms': float(np.median(e2e_means)),
        'e2e_std_ms': float(np.std(e2e_means)),
        'e2e_p5_ms': float(np.percentile(e2e_means, 5)),
        'e2e_p95_ms': float(np.percentile(e2e_means, 95)),
    }

    with open(output_path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"  Saved to {output_path}", flush=True)

if __name__ == '__main__':
    main()
"""


def run_timing_for_build(gsplat_path, ckpt_path, camera_indices, output_path, config):
    """Run timing benchmark for a single build."""
    runner_path = output_path.replace('.json', '_runner.py')
    with open(runner_path, 'w') as f:
        f.write(TIMING_RUNNER)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    result = subprocess.run(
        [sys.executable, runner_path, gsplat_path, ckpt_path,
         json.dumps(camera_indices), output_path, json.dumps(config)],
        capture_output=True, text=True, env=env, timeout=1800
    )
    print(f"  stdout: {result.stdout[-1000:]}")
    if result.returncode != 0:
        print(f"  stderr: {result.stderr[-2000:]}")
        return None
    with open(output_path) as f:
        return json.load(f)


def main():
    work = "/tmp/gsplat_n2_cpcb"
    baseline_path = os.path.join(work, "gsplat_baseline")
    cpcb_path = os.path.join(work, "gsplat_cpcb")
    results_dir = os.path.expanduser("~/3dgs-renderer-benchmark/results/n2_cpcb")
    timing_dir = os.path.join(results_dir, "timing_output")
    os.makedirs(timing_dir, exist_ok=True)

    checkpoints = {
        '5K': os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_5000.pt"),
        '15K': os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_15000.pt"),
        '30K': os.path.expanduser("~/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_30000.pt"),
    }

    camera_indices = [0, 50, 100, 150, 200]

    config = {
        'warmup': 50,
        'timed': 300,
        'n_reps': 5,
        'measure_e2e': True,
    }

    all_timing = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"R2 TIMING: Room {ckpt_name}")
        print(f"{'='*60}")

        # Interleave: run baseline and cpcb alternately
        # to reduce thermal bias
        for rep in range(config['n_reps']):
            # Alternate order
            if rep % 2 == 0:
                order = [("baseline", baseline_path), ("cpcb", cpcb_path)]
            else:
                order = [("cpcb", cpcb_path), ("baseline", baseline_path)]

            for label, gsplat_path in order:
                print(f"\n  {ckpt_name} rep {rep+1}/{config['n_reps']} â€?{label}")
                out_file = os.path.join(timing_dir, f"{ckpt_name}_{label}_rep{rep}.json")
                if os.path.exists(out_file):
                    print(f"    Already exists, skipping")
                    continue

                # Run single rep
                single_config = {
                    'warmup': config['warmup'],
                    'timed': config['timed'],
                    'n_reps': 1,
                    'measure_e2e': True,
                }
                result = run_timing_for_build(
                    gsplat_path, ckpt_path, camera_indices, out_file, single_config
                )
                if result:
                    print(f"    bwd_mean={result['bwd_mean_ms']:.3f}ms e2e_mean={result['e2e_mean_ms']:.3f}ms")

        # Aggregate all reps for this checkpoint
        for label in ["baseline", "cpcb"]:
            rep_files = sorted(
                [f for f in os.listdir(timing_dir)
                 if f.startswith(f"{ckpt_name}_{label}_rep") and f.endswith('.json')]
            )
            if not rep_files:
                continue

            bwd_means = []
            e2e_means = []
            fwd_means = []
            all_reps = []
            for rf in rep_files:
                with open(os.path.join(timing_dir, rf)) as f:
                    data = json.load(f)
                bwd_means.append(data['bwd_mean_ms'])
                e2e_means.append(data['e2e_mean_ms'])
                fwd_means.append(data['fwd_mean_ms'])
                all_reps.append(data)

            key = f"room_{ckpt_name}_{label}"
            all_timing[key] = {
                'bwd_mean_ms': float(np.mean(bwd_means)),
                'bwd_median_ms': float(np.median(bwd_means)),
                'bwd_std_ms': float(np.std(bwd_means)),
                'bwd_p5_ms': float(np.percentile(bwd_means, 5)),
                'bwd_p95_ms': float(np.percentile(bwd_means, 95)),
                'fwd_mean_ms': float(np.mean(fwd_means)),
                'fwd_median_ms': float(np.median(fwd_means)),
                'e2e_mean_ms': float(np.mean(e2e_means)),
                'e2e_median_ms': float(np.median(e2e_means)),
                'e2e_std_ms': float(np.std(e2e_means)),
                'e2e_p5_ms': float(np.percentile(e2e_means, 5)),
                'e2e_p95_ms': float(np.percentile(e2e_means, 95)),
                'n_reps': len(rep_files),
                'reps': all_reps,
            }
            print(f"  {key}: bwd_mean={all_timing[key]['bwd_mean_ms']:.3f}ms e2e_mean={all_timing[key]['e2e_mean_ms']:.3f}ms")

    # Compute speedup and reduction
    for ckpt_name in ['5K', '15K', '30K']:
        base_key = f"room_{ckpt_name}_baseline"
        cpcb_key = f"room_{ckpt_name}_cpcb"
        if base_key in all_timing and cpcb_key in all_timing:
            base_bwd = all_timing[base_key]['bwd_mean_ms']
            cpcb_bwd = all_timing[cpcb_key]['bwd_mean_ms']
            base_e2e = all_timing[base_key]['e2e_mean_ms']
            cpcb_e2e = all_timing[cpcb_key]['e2e_mean_ms']
            all_timing[f"room_{ckpt_name}_comparison"] = {
                'baseline_bwd_ms': base_bwd,
                'cpcb_bwd_ms': cpcb_bwd,
                'raster_bwd_reduction': 1 - cpcb_bwd / base_bwd if base_bwd > 0 else 0,
                'raster_bwd_speedup': base_bwd / cpcb_bwd if cpcb_bwd > 0 else 0,
                'baseline_e2e_ms': base_e2e,
                'cpcb_e2e_ms': cpcb_e2e,
                'e2e_reduction': 1 - cpcb_e2e / base_e2e if base_e2e > 0 else 0,
            }

    # Save
    for ckpt_name in ['5K', '15K', '30K']:
        out_file = os.path.join(results_dir, f"room_{ckpt_name.lower()}_timing.json")
        ckpt_data = {}
        for label in ["baseline", "cpcb"]:
            key = f"room_{ckpt_name}_{label}"
            if key in all_timing:
                ckpt_data[label] = all_timing[key]
        comp_key = f"room_{ckpt_name}_comparison"
        if comp_key in all_timing:
            ckpt_data['comparison'] = all_timing[comp_key]
        with open(out_file, 'w') as f:
            json.dump(ckpt_data, f, indent=2)
        print(f"Saved {out_file}")

    # Save E2E summary
    e2e_summary = {}
    for ckpt_name in ['5K', '15K', '30K']:
        base_key = f"room_{ckpt_name}_baseline"
        cpcb_key = f"room_{ckpt_name}_cpcb"
        if base_key in all_timing and cpcb_key in all_timing:
            e2e_summary[ckpt_name] = {
                'baseline_bwd_ms': all_timing[base_key]['bwd_mean_ms'],
                'cpcb_bwd_ms': all_timing[cpcb_key]['bwd_mean_ms'],
                'baseline_fwd_ms': all_timing[base_key]['fwd_mean_ms'],
                'cpcb_fwd_ms': all_timing[cpcb_key]['fwd_mean_ms'],
                'baseline_e2e_ms': all_timing[base_key]['e2e_mean_ms'],
                'cpcb_e2e_ms': all_timing[cpcb_key]['e2e_mean_ms'],
                'raster_bwd_reduction': all_timing.get(f"room_{ckpt_name}_comparison", {}).get('raster_bwd_reduction', 0),
                'raster_bwd_speedup': all_timing.get(f"room_{ckpt_name}_comparison", {}).get('raster_bwd_speedup', 0),
                'e2e_reduction': all_timing.get(f"room_{ckpt_name}_comparison", {}).get('e2e_reduction', 0),
            }
    e2e_path = os.path.join(results_dir, "e2e_timing.json")
    with open(e2e_path, 'w') as f:
        json.dump(e2e_summary, f, indent=2)
    print(f"Saved {e2e_path}")


if __name__ == '__main__':
    main()
