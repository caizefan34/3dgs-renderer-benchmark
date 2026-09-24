#!/usr/bin/env python3
"""Full 3-scene B1/A benchmark: bicycle and garden (Room already done)."""
import sys, os, json, subprocess

CKPT_BASE = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7"
CAM_BASE = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360"
RESULTS_DIR = "/tmp/accutile_a100_results"

BENCH_SCRIPT = "/tmp/accutile_bench_runner.py"

SCENES = {
    "bicycle": {
        "ckpt": f"{CKPT_BASE}/a100_30k_bicycle_t16_16/a100_30k_bicycle_t16_16_latest.pt",
        "cams": f"{CAM_BASE}/bicycle/cameras.json",
    },
    "garden": {
        "ckpt": f"{CKPT_BASE}/a100_30k_garden_t16_16/a100_30k_garden_t16_16_latest.pt",
        "cams": f"{CAM_BASE}/garden/cameras.json",
    },
}

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    variants = [
        ("B1", "INSTALLED", False),
        ("A",  "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153", True),
    ]

    all_results = {}
    for scene, paths in SCENES.items():
        print(f"\n{'='*70}")
        print(f"Scene: {scene}")
        print(f"{'='*70}")
        scene_results = {}
        for name, gsplat_path, accutile in variants:
            print(f"\n  Benchmarking {name}...")
            out_file = f"{RESULTS_DIR}/bench_{scene}_{name}.json"
            cmd = (f"source /home/liaoyuanjun/miniforge3/etc/profile.d/conda.sh && conda activate anysplat && "
                   f"rm -rf /home/liaoyuanjun/.cache/torch_extensions/py310_cu124/gsplat_cuda && "
                   f"CUDA_VISIBLE_DEVICES=0 python3 {BENCH_SCRIPT} "
                   f"--variant {name} --gsplat-path {gsplat_path} "
                   f"--accutile {'true' if accutile else 'false'} "
                   f"--ckpt {paths['ckpt']} --cams {paths['cams']} --warmup 20 --measure 100 --out {out_file}")
            r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=600)
            stdout = r.stdout.strip()
            idx = stdout.find('{')
            if idx >= 0:
                try:
                    result = json.loads(stdout[idx:])
                    scene_results[name] = result
                    print(f"    fwd={result['forward']['mean_ms']:.2f}ms bwd={result['backward']['mean_ms']:.2f}ms "
                          f"tot={result['total']['mean_ms']:.2f}ms isect={result['intersections']:,}")
                except json.JSONDecodeError as e:
                    print(f"    JSON parse error: {e}")
            else:
                print(f"    No JSON. stderr: {r.stderr[-300:]}")
        all_results[scene] = scene_results

    # Summary
    print(f"\n{'='*70}")
    print("3-SCENE BENCHMARK SUMMARY (B1 vs A)")
    print(f"{'='*70}")
    for scene, sr in all_results.items():
        b1 = sr.get("B1", {})
        a = sr.get("A", {})
        if not b1 or not a:
            continue
        b1_fwd = b1["forward"]["mean_ms"]
        b1_bwd = b1["backward"]["mean_ms"]
        b1_tot = b1["total"]["mean_ms"]
        a_fwd = a["forward"]["mean_ms"]
        a_bwd = a["backward"]["mean_ms"]
        a_tot = a["total"]["mean_ms"]
        print(f"\n{scene}:")
        print(f"  B1: fwd={b1_fwd:.2f}ms bwd={b1_bwd:.2f}ms tot={b1_tot:.2f}ms isect={b1['intersections']:,}")
        print(f"  A:  fwd={a_fwd:.2f}ms bwd={a_bwd:.2f}ms tot={a_tot:.2f}ms isect={a['intersections']:,}")
        print(f"  Speedup: fwd={b1_fwd/a_fwd:.4f}x bwd={b1_bwd/a_bwd:.4f}x tot={b1_tot/a_tot:.4f}x")
        print(f"  Reduction: {1-a['intersections']/b1['intersections']:.4f}")

    with open(f"{RESULTS_DIR}/accutile_b1a_3scene.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/accutile_b1a_3scene.json")

if __name__ == "__main__":
    main()
