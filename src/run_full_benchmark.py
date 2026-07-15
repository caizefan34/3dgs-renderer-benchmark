#!/usr/bin/env python
"""Comprehensive multi-scale benchmark for 3DGS Renderer Benchmark.
Runs each scene+renderer combo in an isolated subprocess."""
import sys, os, time, json, argparse, subprocess

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))

SCENE_CFG = [
    ("50K", "scene_50k.ply", 50000),
    ("200K", "scene_200k.ply", 200000),
    ("400K", "scene.ply", 400000),
]

def _load_synthetic_difficulty_catalog(repo_root=None):
    repo_root = repo_root or os.path.dirname(PROJECT_ROOT)
    path = os.path.join(repo_root, "data", "scenes", "synthetic_stress_suite.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        catalog = json.load(handle)
    normalization = catalog.get("normalization", {})
    formula = catalog.get("difficulty_formula")
    return {
        scene["gaussians"]: {
            "difficulty_score": scene.get("difficulty_score"),
            "difficulty_formula": formula,
            "difficulty_inputs": scene.get("inputs"),
            "difficulty_normalization": normalization,
            "synthetic_stress_class": scene.get("class"),
            "synthetic_stress_scene_id": scene.get("scene_id"),
        }
        for scene in catalog.get("scenes", [])
        if "gaussians" in scene
    }

def find_scene(scene_fname):
    for d in [os.path.join(os.path.dirname(PROJECT_ROOT), "data"), os.path.join(PROJECT_ROOT, "data")]:
        p = os.path.join(d, scene_fname)
        if os.path.exists(p):
            return p
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--renderers", type=str, nargs="+", default=["diff_gaussian"])
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    repo_root = os.path.dirname(PROJECT_ROOT)
    output_dir = args.output or os.path.join(repo_root, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  3DGS Renderer Comprehensive Benchmark")
    print("=" * 70)
    print(f"  Isolated subprocess benchmark")
    print(f"  Renderers: {args.renderers}")
    print(f"  Frames: {args.frames} (+ {args.warmup}) x {args.repeats}")
    print("=" * 70)

    available_scenes = [(n, find_scene(f), g) for n, f, g in SCENE_CFG]
    available_scenes = [(n, p, g) for n, p, g in available_scenes if p]

    if not available_scenes:
        print("No scene files found!")
        return

    all_results = {}
    difficulty_catalog = _load_synthetic_difficulty_catalog(repo_root)
    for renderer_name in args.renderers:
        for scene_name, scene_path, num_gaussians in available_scenes:
            output_json = os.path.join(output_dir, "_tmp_{}_{}.json".format(renderer_name, scene_name))
            difficulty_json = json.dumps(difficulty_catalog.get(num_gaussians, {}))

            with open(os.path.join(output_dir, "_script.py"), "w") as f:
                f.write("""import sys, json, os
sys.path.insert(0, r"{}")
from benchmark_framework import load_ply, load_cameras_from_json
from renderers import get_renderer
import torch, time, numpy as np

scene = load_ply(r"{}", device="cuda")
cameras = load_cameras_from_json(r"{}", device="cuda")
if not cameras:
    cameras = load_cameras_from_json(r"{}", device="cuda")

renderer = get_renderer("{}")
if not renderer:
    json.dump({{"error": "not available"}}, open(r"{}", "w"))
    sys.exit(1)

difficulty = json.loads(r'''{}''')
prep = renderer.prepare_scene(scene)
all_times = []
peak_mem = 0
mem_samples = []
W, H = {}, {}

for rep in range({}):
    for f in range({}):
        with torch.no_grad():
            renderer.render(prep, cameras[f % len(cameras)])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for f in range({}):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            renderer.render(prep, cameras[f % len(cameras)])
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        all_times.append(ms)
        mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        mem_samples.append(mem)
        if mem > peak_mem:
            peak_mem = mem

t = np.array(all_times)
result = {{
    "renderer_name": "{}",
    "num_gaussians": scene["num_points"],
    "num_frames": len(all_times),
    "mean_fps": round(float(1000.0 / t.mean()), 1),
    "mean_latency_ms": round(float(t.mean()), 3),
    "median_latency_ms": round(float(np.median(t)), 3),
    "p99_latency_ms": round(float(np.percentile(t, 99)), 3),
    "min_latency_ms": round(float(t.min()), 3),
    "max_latency_ms": round(float(t.max()), 3),
    "std_latency_ms": round(float(t.std()), 3),
    "coefficient_of_variation": round(float(t.std() / t.mean()), 6) if t.mean() > 0 else 0.0,
    "stability_score": round(float(np.median(t) / np.percentile(t, 99)), 6) if np.percentile(t, 99) > 0 else 0.0,
    "benchmark_type": "synthetic_stress",
    "difficulty_score": difficulty.get("difficulty_score"),
    "difficulty_formula": difficulty.get("difficulty_formula"),
    "difficulty_inputs": difficulty.get("difficulty_inputs"),
    "difficulty_normalization": difficulty.get("difficulty_normalization"),
    "synthetic_stress_class": difficulty.get("synthetic_stress_class"),
    "synthetic_stress_scene_id": difficulty.get("synthetic_stress_scene_id"),
    "peak_vram_mb": round(float(peak_mem), 1),
    "avg_vram_mb": round(float(np.mean(mem_samples)), 1),
    "gpu_name": torch.cuda.get_device_name(0),
    "image_width": W,
    "image_height": H,
}}
json.dump(result, open(r"{}", "w"), indent=2)
""".format(
    PROJECT_ROOT,
    scene_path,
    os.path.join(os.path.dirname(PROJECT_ROOT), "data", "camera_presets", "spiral.json"),
    os.path.join(os.path.dirname(PROJECT_ROOT), "data", "cameras.json"),
    renderer_name,
    output_json,
    difficulty_json,
    1920, 1080,
    args.repeats, args.warmup, args.frames,
    renderer_name,
    output_json,
))

            print(f"\n  {renderer_name} @ {scene_name} Gaussians ...", end=" ", flush=True)
            t0 = time.time()
            result = subprocess.run([sys.executable, os.path.join(output_dir, "_script.py")], capture_output=True, text=True, timeout=600)
            elapsed = time.time() - t0

            if os.path.exists(output_json):
                try:
                    with open(output_json) as f:
                        data = json.load(f)
                    if "error" in data:
                        print(f"FAILED: {data['error']}")
                    else:
                        all_results["{}_{}".format(renderer_name, scene_name)] = data
                        print(f"{data['mean_fps']:.0f} FPS, {data['mean_latency_ms']:.2f}ms, {data['peak_vram_mb']:.0f}MB ({elapsed:.0f}s)")
                except Exception as e:
                    print(f"ERROR: {e}")
            else:
                print(f"FAILED (rc={result.returncode})")
                for line in result.stderr.strip().split("\n")[-3:]:
                    print(f"  {line}")

            for f in [output_json, os.path.join(output_dir, "_script.py")]:
                if os.path.exists(f):
                    os.remove(f)

    # Summary
    print(f"\n\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Renderer':<20} {'GS Count':<10} {'FPS':>8} {'Latency':>10} {'P99':>8} {'VRAM':>8} {'Difficulty':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    for key in sorted(all_results.keys()):
        d = all_results[key]
        rname, sc = key.split("_", 1)
        difficulty = d["difficulty_score"] if d.get("difficulty_score") is not None else "N/A"
        print(f"  {rname:<20} {sc:<10} {d['mean_fps']:>8.1f} {d['mean_latency_ms']:>8.2f}ms {d['p99_latency_ms']:>6.2f}ms {d['peak_vram_mb']:>6.0f}MB {difficulty:>10}")

    with open(os.path.join(output_dir, "full_benchmark_results.json"), "w") as f:
        json.dump({
            "metadata": {
                "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU",
                "resolution": "1920x1080",
                "frames": args.frames, "warmup": args.warmup, "repeats": args.repeats,
                "date": "2026-07-13",
            },
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_dir}/full_benchmark_results.json")

if __name__ == "__main__":
    main()
