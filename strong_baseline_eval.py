#!/usr/bin/env python3
"""
Unified evaluation for strong baselines.
Renders test views and computes PSNR/SSIM/LPIPS for trained models.
Works with both Speedy-Splat and FastGS checkpoints.
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

def evaluate_scene(baseline, dataset, scene, method, output_dir):
    """Run render.py + metrics.py for a trained model."""
    repo = f"/mnt/storage_pool/liaoyuanjun/strong_baselines/{baseline}"
    model_path = output_dir
    data_path = f"/mnt/storage_pool/liaoyuanjun/strong_baselines/{baseline}/datasets/{dataset}/{scene}"

    results = {"baseline": baseline, "scene": scene, "method": method}

    # Step 1: Render test views
    render_cmd = [sys.executable, "-u", os.path.join(repo, "render.py"),
                  "-m", model_path, "--skip_train"]
    env = os.environ.copy()
    torch_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                             "lib", "python3.10", "site-packages", "torch", "lib")
    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = repo

    print(f"  Rendering {scene}/{method}...", flush=True)
    render_log = os.path.join(model_path, "render.log")
    with open(render_log, "w") as f:
        ret = subprocess.run(render_cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=repo)
    if ret.returncode != 0:
        print(f"  Render FAILED: {scene}/{method}", flush=True)
        results["error"] = "render_failed"
        return results

    # Step 2: Compute metrics
    metrics_cmd = [sys.executable, "-u", os.path.join(repo, "metrics.py"),
                   "-m", model_path]
    print(f"  Computing metrics {scene}/{method}...", flush=True)
    metrics_log = os.path.join(model_path, "metrics.log")
    with open(metrics_log, "w") as f:
        ret = subprocess.run(metrics_cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=repo)
    if ret.returncode != 0:
        print(f"  Metrics FAILED: {scene}/{method}", flush=True)
        results["error"] = "metrics_failed"
        return results

    # Step 3: Read results — try results.json first, then parse metrics.log
    results_found = False
    results_dir = os.path.join(model_path, "test")
    # 3DGS saves results to test/ours_30000/results.json (some variants)
    if os.path.exists(results_dir):
        for d in os.listdir(results_dir):
            ours_dir = os.path.join(results_dir, d)
            results_json = os.path.join(ours_dir, "results.json")
            if os.path.exists(results_json):
                with open(results_json) as f:
                    metrics = json.load(f)
                results.update(metrics)
                results_found = True
                break
    
    # Fallback: parse metrics.log for PSNR/SSIM/LPIPS printed by metrics.py
    if not results_found and os.path.exists(metrics_log):
        import re
        with open(metrics_log) as f:
            log_content = f.read()
        psnr_match = re.search(r'PSNR\s*:\s*([\d.]+)', log_content)
        ssim_match = re.search(r'SSIM\s*:\s*([\d.]+)', log_content)
        lpips_match = re.search(r'LPIPS\s*:\s*([\d.]+)', log_content)
        if psnr_match:
            results["PSNR"] = float(psnr_match.group(1))
        if ssim_match:
            results["SSIM"] = float(ssim_match.group(1))
        if lpips_match:
            results["LPIPS"] = float(lpips_match.group(1))

    # Step 4: Count Gaussians from checkpoint
    ckpt_path = os.path.join(model_path, "point_cloud", "iteration_30000", "point_cloud.ply")
    if os.path.exists(ckpt_path):
        try:
            from plyfile import PlyData
            plydata = PlyData.read(ckpt_path)
            results["n_gaussians"] = len(plydata.elements[0].data)
        except:
            results["n_gaussians"] = None

    # Step 5: Read timing from training_summary.json
    summary_path = os.path.join(model_path, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        results["wall_time_s"] = summary.get("wall_time_s")
        results["wall_time_min"] = summary.get("wall_time_min")

    print(f"  {scene}/{method}: PSNR={results.get('PSNR', '?')} "
          f"SSIM={results.get('SSIM', '?')} LPIPS={results.get('LPIPS', '?')}", flush=True)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["speedy-splat", "fastgs"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    BASELINE = args.baseline
    RESULTS_BASE = f"/mnt/storage_pool/liaoyuanjun/strong_baseline_results/{BASELINE}"

    ALL_SCENES = [
        ("mipnerf360", "bicycle"), ("mipnerf360", "flowers"), ("mipnerf360", "garden"),
        ("mipnerf360", "stump"), ("mipnerf360", "treehill"), ("mipnerf360", "room"),
        ("mipnerf360", "counter"), ("mipnerf360", "kitchen"), ("mipnerf360", "bonsai"),
        ("tanksandtemples", "truck"), ("tanksandtemples", "train"),
        ("deepblending", "drjohnson"), ("deepblending", "playroom"),
    ]

    all_results = {}
    for dataset, scene in ALL_SCENES:
        for method in ["native", "c42"]:
            output_dir = os.path.join(RESULTS_BASE, dataset, scene, method)
            if not os.path.exists(output_dir):
                print(f"  SKIP (not found): {scene}/{method}", flush=True)
                continue
            results = evaluate_scene(BASELINE, dataset, scene, method, output_dir)
            all_results[f"{scene}_{method}"] = results

    # Save all results
    output_path = os.path.join(RESULTS_BASE, "all_metrics.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_path}", flush=True)
