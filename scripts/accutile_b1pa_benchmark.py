#!/usr/bin/env python3
"""B1/P/A benchmark: run each variant in a subprocess and collect timing."""
import sys, os, json, subprocess

CKPT = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7/a100_30k_room_t16_16/a100_30k_room_t16_16_latest.pt"
CAMS = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json"
RESULTS_DIR = "/tmp/accutile_a100_results"

BENCH_SCRIPT = "/tmp/accutile_bench_runner.py"

BENCH_CODE = r'''#!/usr/bin/env python3
import sys, os, json, argparse, math
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--gsplat-path", required=True)
    parser.add_argument("--accutile", type=str, default="false")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cams", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measure", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.gsplat_path != "INSTALLED":
        sys.path.insert(0, args.gsplat_path)
    import gsplat
    from gsplat import rasterization

    accutile = args.accutile == "true"
    device = "cuda"

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = state.get("model_state", state)
    with open(args.cams) as f:
        cam = json.load(f)[0]
    rotation = torch.tensor(cam["rotation"], dtype=torch.float32, device=device)
    position = torch.tensor(cam["position"], dtype=torch.float32, device=device)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -rotation @ position
    K = torch.tensor([[cam["fx"], 0., cam["width"] / 2],
                      [0., cam["fy"], cam["height"] / 2],
                      [0., 0., 1.]], dtype=torch.float32, device=device)[None]
    viewmats = viewmat[None]
    width, height = cam["width"], cam["height"]
    sh_degree = int(state.get("sh_degree", 3))

    def prep():
        means = state["xyz"].to(device).clone().requires_grad_(True)
        quats = state["rotations"].to(device).clone().requires_grad_(True)
        scales = torch.exp(state["scales"].to(device)).clone().requires_grad_(True)
        opacities = torch.sigmoid(state["opacity"].to(device).flatten()).clone().requires_grad_(True)
        colors = state["shs"].to(device).clone().requires_grad_(True)
        return means, quats, scales, opacities, colors

    kwargs = dict(sh_degree=sh_degree, absgrad=True, tile_size=16, packed=False)
    if accutile:
        kwargs["accutile"] = True

    means, quats, scales, opacities, colors = prep()

    # Warmup
    for _ in range(args.warmup):
        for t in (means, quats, scales, opacities, colors):
            t.grad = None
        r, a, m = rasterization(means, quats, scales, opacities, colors, viewmats, K,
                                width, height, **kwargs)
        (r.sum() + a.sum()).backward()
        torch.cuda.synchronize()

    fwd_times, bwd_times, tot_times = [], [], []
    for _ in range(args.measure):
        for t in (means, quats, scales, opacities, colors):
            t.grad = None
        s = torch.cuda.Event(enable_timing=True)
        fe = torch.cuda.Event(enable_timing=True)
        be = torch.cuda.Event(enable_timing=True)
        s.record()
        r, a, m = rasterization(means, quats, scales, opacities, colors, viewmats, K,
                                width, height, **kwargs)
        fe.record()
        (r.sum() + a.sum()).backward()
        be.record()
        torch.cuda.synchronize()
        fwd_times.append(s.elapsed_time(fe))
        bwd_times.append(fe.elapsed_time(be))
        tot_times.append(s.elapsed_time(be))

    result = {"variant": args.variant, "gsplat_version": gsplat.__version__, "accutile": accutile}
    for key, arr in [("forward", fwd_times), ("backward", bwd_times), ("total", tot_times)]:
        arr = np.array(arr)
        result[key] = {"mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)),
                       "std_ms": float(arr.std()), "p95_ms": float(np.percentile(arr, 95))}
    result["intersections"] = int(m["tiles_per_gauss"].sum())
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
'''

def main():
    with open(BENCH_SCRIPT, "w") as f:
        f.write(BENCH_CODE)
    os.chmod(BENCH_SCRIPT, 0o755)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    variants = [
        ("B1", "INSTALLED", False),
        ("P",  "/mnt/storage_pool/liaoyuanjun/gsplat-accutile-v153", True),
        ("A",  "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153", True),
    ]

    results = {}
    for name, path, accutile in variants:
        print(f"\n{'='*60}")
        print(f"Benchmarking {name}")
        print(f"{'='*60}")
        out_file = f"{RESULTS_DIR}/bench_{name}.json"
        cmd = (f"source /home/liaoyuanjun/miniforge3/etc/profile.d/conda.sh && conda activate anysplat && "
               f"CUDA_VISIBLE_DEVICES=0 python3 {BENCH_SCRIPT} "
               f"--variant {name} --gsplat-path {path} "
               f"--accutile {'true' if accutile else 'false'} "
               f"--ckpt {CKPT} --cams {CAMS} --warmup 20 --measure 100 --out {out_file}")
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=600)
        stdout = r.stdout.strip()
        # Find JSON in output
        idx = stdout.find('{')
        if idx >= 0:
            try:
                result = json.loads(stdout[idx:])
                results[name] = result
                print(f"  fwd={result['forward']['mean_ms']:.2f}ms bwd={result['backward']['mean_ms']:.2f}ms "
                      f"tot={result['total']['mean_ms']:.2f}ms isect={result['intersections']:,}")
            except json.JSONDecodeError as e:
                print(f"  JSON parse error: {e}")
                print(f"  stdout tail: {stdout[-200:]}")
                print(f"  stderr tail: {r.stderr[-200:]}")
        else:
            print(f"  No JSON output. stderr: {r.stderr[-300:]}")

    # Summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY (Room, 30K checkpoint, camera 0)")
    print(f"{'='*70}")
    print(f"\n{'Variant':<8} {'N_isect':>12} {'Fwd ms':>10} {'Bwd ms':>10} {'Total ms':>10} {'Fwd sp':>8} {'Bwd sp':>8} {'Tot sp':>8}")
    b1_fwd = results.get("B1", {}).get("forward", {}).get("mean_ms", 1)
    b1_bwd = results.get("B1", {}).get("backward", {}).get("mean_ms", 1)
    b1_tot = results.get("B1", {}).get("total", {}).get("mean_ms", 1)
    for name in ("B1", "P", "A"):
        if name not in results:
            continue
        s = results[name]
        isect = s.get("intersections", 0)
        fwd = s["forward"]["mean_ms"]
        bwd = s["backward"]["mean_ms"]
        tot = s["total"]["mean_ms"]
        fwd_sp = b1_fwd / fwd if name != "B1" else 1.0
        bwd_sp = b1_bwd / bwd if name != "B1" else 1.0
        tot_sp = b1_tot / tot if name != "B1" else 1.0
        print(f"{name:<8} {isect:>12,} {fwd:>10.2f} {bwd:>10.2f} {tot:>10.2f} {fwd_sp:>8.4f} {bwd_sp:>8.4f} {tot_sp:>8.4f}")

    with open(f"{RESULTS_DIR}/accutile_b1pa_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/accutile_b1pa_benchmark.json")

if __name__ == "__main__":
    main()
