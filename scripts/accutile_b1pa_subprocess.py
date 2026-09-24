#!/usr/bin/env python3
"""B1/P/A comparison: run each variant as a separate subprocess.

B1 = pip-installed gsplat 1.5.3 (AABB baseline, no accutile param)
P  = gsplat-accutile-v153 (per-tile predicate, accutile=True)
A  = gsplat-true-accutile-v153 (true strip-based AccuTile, accutile=True)

Each variant runs in its own process to avoid module contamination.
"""
import sys, os, json, subprocess, time

CKPT = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7/a100_30k_room_t16_16/a100_30k_room_t16_16_latest.pt"
CAMS = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json"
OUT = "/tmp/accutile_a100_results/accutile_b1pa_room.json"

RUNNER_SCRIPT = "/tmp/accutile_variant_runner.py"

RUNNER_CODE = r'''#!/usr/bin/env python3
"""Run a single variant and output JSON results."""
import sys, os, json, argparse, math
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["B1", "P", "A"])
    parser.add_argument("--gsplat-path", required=True)
    parser.add_argument("--accutile", type=str, default="false")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cams", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measure", type=int, default=100)
    args = parser.parse_args()

    # Set up gsplat import path
    if args.gsplat_path != "INSTALLED":
        sys.path.insert(0, args.gsplat_path)
    import gsplat
    from gsplat import rasterization

    accutile = args.accutile == "true"
    device = "cuda"

    # Load checkpoint
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = state.get("model_state", state)

    # Load camera
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
    Ks = K
    width, height = cam["width"], cam["height"]
    sh_degree = int(state.get("sh_degree", 3))

    def prep():
        means = state["xyz"].to(device).clone().requires_grad_(True)
        quats = state["rotations"].to(device).clone().requires_grad_(True)
        scales = torch.exp(state["scales"].to(device)).clone().requires_grad_(True)
        opacities = torch.sigmoid(state["opacity"].to(device).flatten()).clone().requires_grad_(True)
        colors = state["shs"].to(device).clone().requires_grad_(True)
        return means, quats, scales, opacities, colors

    kwargs = dict(
        sh_degree=sh_degree, absgrad=True, tile_size=16, packed=False,
    )
    if accutile:
        kwargs["accutile"] = True

    # ── Correctness run ──
    means, quats, scales, opacities, colors = prep()
    inputs = (means, quats, scales, opacities, colors)
    for t in inputs:
        t.grad = None
    rgb, alpha, meta = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks,
        width, height, **kwargs)
    (rgb.sum() + alpha.sum()).backward()
    torch.cuda.synchronize()

    correctness = {
        "intersections": int(meta["tiles_per_gauss"].sum()),
        "rgb_sum": float(rgb.sum()),
        "alpha_sum": float(alpha.sum()),
        "rgb": rgb.detach().clone(),
        "alpha": alpha.detach().clone(),
        "grads": {name: t.grad.detach().clone() for name, t in
                  zip(("means","quats","scales","opacities","colors"), inputs)
                  if t.grad is not None},
    }

    # ── Benchmark ──
    means, quats, scales, opacities, colors = prep()
    for _ in range(args.warmup):
        for t in (means, quats, scales, opacities, colors):
            t.grad = None
        r, a, m = rasterization(means, quats, scales, opacities, colors, viewmats, Ks,
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
        r, a, m = rasterization(means, quats, scales, opacities, colors, viewmats, Ks,
                                width, height, **kwargs)
        fe.record()
        (r.sum() + a.sum()).backward()
        be.record()
        torch.cuda.synchronize()
        fwd_times.append(s.elapsed_time(fe))
        bwd_times.append(fe.elapsed_time(be))
        tot_times.append(s.elapsed_time(be))

    benchmark = {}
    for key, arr in [("forward", fwd_times), ("backward", bwd_times), ("total", tot_times)]:
        arr = np.array(arr)
        benchmark[key] = {"mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)),
                          "std_ms": float(arr.std()), "p95_ms": float(np.percentile(arr, 95))}
    benchmark["intersections"] = int(m["tiles_per_gauss"].sum())

    # ── Save correctness tensors for comparison ──
    torch.save({
        "rgb": correctness["rgb"],
        "alpha": correctness["alpha"],
        "grads": correctness["grads"],
        "intersections": correctness["intersections"],
        "rgb_sum": correctness["rgb_sum"],
        "alpha_sum": correctness["alpha_sum"],
    }, f"/tmp/accutile_a100_results/variant_{args.variant}.pt")

    result = {
        "variant": args.variant,
        "gsplat_path": args.gsplat_path,
        "gsplat_version": gsplat.__version__,
        "accutile": accutile,
        "gaussian_count": int(state["xyz"].shape[0]),
        "resolution": [width, height],
        "sh_degree": sh_degree,
        "correctness": {
            "intersections": correctness["intersections"],
            "rgb_sum": correctness["rgb_sum"],
            "alpha_sum": correctness["alpha_sum"],
        },
        "benchmark": benchmark,
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
'''

def run_variant(variant, gsplat_path, accutile, warmup=20, measure=100):
    """Run a single variant as a subprocess and return its JSON output."""
    cmd = [
        "python3", RUNNER_SCRIPT,
        "--variant", variant,
        "--gsplat-path", gsplat_path,
        "--accutile", "true" if accutile else "false",
        "--ckpt", CKPT,
        "--cams", CAMS,
        "--warmup", str(warmup),
        "--measure", str(measure),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # For B1, we need the conda env activated
    conda_setup = "source /home/liaoyuanjun/miniforge3/etc/profile.d/conda.sh && conda activate anysplat"
    full_cmd = f'{conda_setup} && {" ".join(cmd)}'
    result = subprocess.run(["bash", "-c", full_cmd], capture_output=True, text=True, timeout=600, env=env)
    # The last non-empty line should be the closing brace of JSON
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    # Extract JSON from stdout (it's the last JSON block)
    json_start = stdout.rfind('{\n  "variant"')
    if json_start == -1:
        json_start = stdout.find('{')
    if json_start >= 0:
        # Find matching closing brace
        json_str = stdout[json_start:]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to find just the last JSON object
            lines = json_str.split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if line.strip().startswith('{') and '"variant"' in line:
                    in_json = True
                if in_json:
                    json_lines.append(line)
                    if line.strip() == '}':
                        break
            return json.loads('\n'.join(json_lines))

    print(f"ERROR for {variant}: stderr={stderr[-500:]}")
    return None

def main():
    # Write the runner script
    with open(RUNNER_SCRIPT, "w") as f:
        f.write(RUNNER_CODE)
    os.chmod(RUNNER_SCRIPT, 0o755)

    os.makedirs("/tmp/accutile_a100_results", exist_ok=True)

    variants = [
        ("B1", "INSTALLED", False),  # pip-installed gsplat 1.5.3
        ("P",  "/mnt/storage_pool/liaoyuanjun/gsplat-accutile-v153", True),
        ("A",  "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153", True),
    ]

    results = {}
    for name, path, accutile in variants:
        print(f"\n{'='*60}")
        print(f"Running variant {name} (path={path}, accutile={accutile})")
        print(f"{'='*60}")
        r = run_variant(name, path, accutile)
        if r:
            results[name] = r
        else:
            print(f"FAILED for {name}")

    # ── Compare correctness ──
    print(f"\n{'='*70}")
    print("CORRECTNESS COMPARISON (vs B1)")
    print(f"{'='*70}")
    if "B1" in results:
        b1 = results["B1"]
        for name in ("P", "A"):
            if name not in results:
                continue
            v = results[name]
            print(f"\n{name} vs B1:")
            print(f"  RGB sum: B1={b1['correctness']['rgb_sum']:.4f}, {name}={v['correctness']['rgb_sum']:.4f}, "
                  f"diff={abs(b1['correctness']['rgb_sum'] - v['correctness']['rgb_sum']):.4e}")
            print(f"  alpha sum: B1={b1['correctness']['alpha_sum']:.4f}, {name}={v['correctness']['alpha_sum']:.4f}, "
                  f"diff={abs(b1['correctness']['alpha_sum'] - v['correctness']['alpha_sum']):.4e}")

    # ── Compare tensor-level correctness ──
    print(f"\n{'='*70}")
    print("TENSOR-LEVEL CORRECTNESS (loaded from saved .pt files)")
    print(f"{'='*70}")
    b1_data = torch.load(f"/tmp/accutile_a100_results/variant_B1.pt", map_location="cpu")
    for name in ("P", "A"):
        pt_path = f"/tmp/accutile_a100_results/variant_{name}.pt"
        if not os.path.exists(pt_path):
            continue
        v_data = torch.load(pt_path, map_location="cpu")
        rgb_diff = (b1_data["rgb"] - v_data["rgb"]).abs()
        alpha_diff = (b1_data["alpha"] - v_data["alpha"]).abs()
        print(f"\n{name} vs B1:")
        print(f"  RGB: max_abs={float(rgb_diff.max()):.4e}, mean_abs={float(rgb_diff.mean()):.4e}")
        print(f"  alpha: max_abs={float(alpha_diff.max()):.4e}, mean_abs={float(alpha_diff.mean()):.4e}")
        for gname in ("means", "quats", "scales", "opacities", "colors"):
            if gname in b1_data["grads"] and gname in v_data["grads"]:
                g_diff = (b1_data["grads"][gname] - v_data["grads"][gname]).abs()
                b1_norm = b1_data["grads"][gname].flatten().norm().item()
                rel_l2 = g_diff.flatten().norm().item() / max(b1_norm, 1e-8)
                print(f"  grad {gname}: max_abs={float(g_diff.max()):.4e}, mean_abs={float(g_diff.mean()):.4e}, rel_L2={rel_l2:.4e}")

    # ── Benchmark summary ──
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Variant':<8} {'N_isect':>12} {'Fwd ms':>10} {'Bwd ms':>10} {'Total ms':>10} {'Fwd sp':>8} {'Bwd sp':>8} {'Tot sp':>8}")
    b1_fwd = results.get("B1", {}).get("benchmark", {}).get("forward", {}).get("mean_ms", 1)
    b1_bwd = results.get("B1", {}).get("benchmark", {}).get("backward", {}).get("mean_ms", 1)
    b1_tot = results.get("B1", {}).get("benchmark", {}).get("total", {}).get("mean_ms", 1)
    for name in ("B1", "P", "A"):
        if name not in results:
            continue
        s = results[name]["benchmark"]
        isect = s.get("intersections", 0)
        fwd = s["forward"]["mean_ms"]
        bwd = s["backward"]["mean_ms"]
        tot = s["total"]["mean_ms"]
        fwd_sp = b1_fwd / fwd if name != "B1" else 1.0
        bwd_sp = b1_bwd / bwd if name != "B1" else 1.0
        tot_sp = b1_tot / tot if name != "B1" else 1.0
        print(f"{name:<8} {isect:>12,} {fwd:>10.2f} {bwd:>10.2f} {tot:>10.2f} {fwd_sp:>8.4f} {bwd_sp:>8.4f} {tot_sp:>8.4f}")

    # Save combined results
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")

if __name__ == "__main__":
    main()
