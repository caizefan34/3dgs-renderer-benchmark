#!/usr/bin/env python3
"""H5-0R Section 6+7+8: Correctness, Timing, and Resources.

For each scene (room, bicycle, garden) and each variant (H5_FRONTIER=0,1,2,3):
  1. Correctness: Compare v_means2d/v_conics/v_colors/v_opacities against baseline
  2. Timing: 20w/100m/5r interleaved CUDA Events
  3. Resources: from cuobjdump (already collected)

Usage on mx: python3 /tmp/h5_0r_measure.py --out-dir /tmp/higs_h5_0r
"""
import argparse, hashlib, importlib.util, json, math, os, sys, time, uuid, gc
from pathlib import Path
import numpy as np
import torch

TILE_SIZE = 16
SH_DEGREE = 3
K_SH = (SH_DEGREE + 1) ** 2

SCENE_CONFIGS = {
    "room": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
        "native_w": 3114, "native_h": 2075,
    },
    "bicycle": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",
        "native_w": 4946, "native_h": 3286,
    },
    "garden": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/garden/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/garden/cameras.json",
        "native_w": 5187, "native_h": 3361,
    },
}

VARIANTS = {
    "baseline": {"H5_FRONTIER": "0", "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint", "HIGS_PX_RUNTIME": "2"},
    "H5A": {"H5_FRONTIER": "1", "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint", "HIGS_PX_RUNTIME": "2"},
    "H5B": {"H5_FRONTIER": "2", "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint", "HIGS_PX_RUNTIME": "2"},
    "H5C": {"H5_FRONTIER": "3", "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint", "HIGS_PX_RUNTIME": "2"},
}


def bootstrap(source, core_so, exp_so):
    """Bootstrap: load core .so, then access experimental backend via normal import.

    The experimental .so is loaded through the torch extension cache (JIT),
    NOT via importlib — loading it twice causes TORCH_LIBRARY re-registration
    errors. We only load the core .so via importlib, then access the
    experimental backend through the normal Python import path.
    """
    sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location("gsplat_cuda", core_so)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.modules["gsplat.csrc"] = core

    # Access experimental backend through normal import (uses JIT cache)
    from gsplat.experimental.render.kernels import _backend as _inference_backend
    exp = _inference_backend._C
    if exp is None:
        raise RuntimeError("Experimental backend not available")
    return exp


def load_ply_scene(ply_path, device):
    from plyfile import PlyData
    v = PlyData.read(ply_path)["vertex"]
    means = torch.tensor(np.column_stack([v["x"], v["y"], v["z"]]), device=device, dtype=torch.float32)
    quats = torch.tensor(np.column_stack([v[f"rot_{i}"] for i in range(4)]), device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.exp(torch.tensor(np.column_stack([v[f"scale_{i}"] for i in range(3)]), device=device, dtype=torch.float32))
    opacities = torch.sigmoid(torch.tensor(v["opacity"], device=device, dtype=torch.float32))
    sh = torch.zeros((len(v), K_SH, 3), device=device, dtype=torch.float32)
    sh[:, 0] = torch.tensor(np.column_stack([v[f"f_dc_{i}"] for i in range(3)]), device=device, dtype=torch.float32)
    rest = torch.stack([torch.tensor(v[f"f_rest_{i}"], device=device, dtype=torch.float32) for i in range(45)], 1)
    sh[:, 1:] = rest.reshape(len(v), 3, 15).permute(0, 2, 1)
    return means, quats, scales, opacities, sh


def load_cameras(cams_path, width, height, device):
    cams = json.loads(Path(cams_path).read_text())
    c = cams[0]
    native_w, native_h = int(c["width"]), int(c["height"])
    R = np.asarray(c["rotation"], dtype=np.float32).T
    p = np.asarray(c["position"], dtype=np.float32)
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ p
    K = np.array([[float(c["fx"]) * width / native_w, 0, (width - 1) / 2],
                  [0, float(c["fy"]) * width / native_w, (height - 1) / 2],
                  [0, 0, 1]], dtype=np.float32)
    return torch.tensor(vm, device=device)[None, None], torch.tensor(K, device=device)[None, None]


def capture_forward_state(scene, max_long_side, device, exp_mod):
    """Capture forward state needed for backward."""
    from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode
    from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched, _gather_visible_native
    from gsplat.rendering import _maybe_evaluate_sh

    cfg = SCENE_CONFIGS[scene]
    s = min(1.0, max_long_side / max(cfg["native_w"], cfg["native_h"]))
    width, height = round(cfg["native_w"] * s), round(cfg["native_h"] * s)
    tw, th = math.ceil(width / TILE_SIZE), math.ceil(height / TILE_SIZE)

    means, quats, scales, opacities, colors = load_ply_scene(cfg["ply"], device)
    vm, K = load_cameras(cfg["cams"], width, height, device)
    vm, K = vm[:, [0]], K[:, [0]]

    with torch.no_grad():
        ids, _, _ = _cull_gaussians_batched(
            means, quats, scales, vm, K, width, height,
            eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
            camera_model="pinhole")
        m, q, s_c, o, c = _gather_visible_native(means, quats, scales, opacities, colors, ids)
        radii, m2d, depths, conics, _ = fully_fused_projection(
            means=m.unsqueeze(0), covars=None, quats=q.unsqueeze(0), scales=s_c.unsqueeze(0),
            viewmats=vm, Ks=K, width=width, height=height,
            eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0,
            packed=False, calc_compensations=False, camera_model="pinhole")
        opa = o[None, None].expand(1, 1, -1).contiguous()
        try:
            _, isect, flat = torch.ops.gsplat.intersect_tile(
                m2d.contiguous(), radii.contiguous(), depths.contiguous(),
                conics.contiguous(), opa.contiguous(), None, None, 1,
                TILE_SIZE, tw, th, True, False, None)
        except RuntimeError:
            _, isect, flat = torch.ops.gsplat.intersect_tile(
                m2d.contiguous(), radii.contiguous(), depths.contiguous(),
                conics.contiguous(), opa.contiguous(), None, None, 1,
                TILE_SIZE, tw, th, True, False)
        offs = isect_offset_encode(isect, 1, tw, th).reshape(1, 1, th, tw)

        means_b = m[None]
        radii_b = radii
        colors_input = c  # [N, K, 3]
        colors_eval = _maybe_evaluate_sh(
            SH_DEGREE, colors_input, means_b, radii_b, vm,
            (1,), 1, len(o), True).contiguous()

        m2d_b = m2d[0, 0][None, None].contiguous()
        conics_b = conics[0, 0][None, None].contiguous()
        opa_b = o[None, None].contiguous()
        offs_b = offs.contiguous()
        flat_b = flat.contiguous()
        bg = torch.zeros((1, 1, 3), device=device, dtype=torch.float32)

        for _ in range(3):
            torch.ops.gsplat.rasterize_to_pixels_3dgs(
                m2d_b, conics_b, colors_eval, opa_b, bg, None,
                width, height, TILE_SIZE, offs_b, flat_b, False, False)
        torch.cuda.synchronize()

        result = torch.ops.gsplat.rasterize_to_pixels_3dgs(
            m2d_b, conics_b, colors_eval, opa_b, bg, None,
            width, height, TILE_SIZE, offs_b, flat_b, False, False)

        if len(result) == 4:
            render_colors, render_alphas, _absgrad, last_ids = result
        elif len(result) == 3:
            render_colors, render_alphas, last_ids = result
        else:
            raise RuntimeError(f"Unexpected return count: {len(result)}")

    # Prepare all tensors for backward call
    N_vis = len(o)
    # Use identity visible_ids so grad buffers only need N_vis rows
    vis_ids = torch.arange(N_vis, device=device, dtype=torch.int64)
    state = {
        "scene": scene, "width": width, "height": height, "tw": tw, "th": th,
        "N_vis": N_vis,
        # Forward state for backward (visible subset)
        "means2d": m2d[0, 0].contiguous(),          # [N_vis, 2]
        "conics": conics[0, 0].contiguous(),         # [N_vis, 3]
        "colors_eval": colors_eval[0, 0].contiguous(),  # [N_vis, 3]
        "opacities": o.contiguous(),                 # [N_vis]
        "backgrounds": bg[0, 0:1].contiguous(),      # [1, 3]
        "tile_offsets": offs[0, 0].contiguous(),     # [th, tw]
        "flatten_ids": flat.contiguous(),            # [n_isects]
        "active_tiles": None,
        "render_alphas": render_alphas[0, 0, :, :, 0].contiguous()[None],  # [1, H, W]
        "last_ids": last_ids[0, 0].contiguous()[None],     # [1, H, W]
        # For projection backward (visible subset)
        "means": m.contiguous(),                     # [N_vis, 3]
        "quats": q.contiguous(),                     # [N_vis, 4]
        "scales": s_c.contiguous(),                  # [N_vis, 3]
        "radii": radii[0].contiguous(),              # [1, N_vis, 2]
        "viewmats": vm.contiguous(),                 # [1, 1, 4, 4]
        "Ks": K.contiguous(),                         # [1, 1, 3, 3]
        "sh_coeffs": c.contiguous(),                 # [N_vis, K, 3]
        "visible_ids": vis_ids,                      # [N_vis] identity
    }
    return state


def make_grad_buffers(state, device):
    """Pre-allocate gradient output buffers."""
    N = state["N_vis"]
    return {
        "grad_means": torch.zeros((N, 3), device=device, dtype=torch.float32),
        "grad_quats": torch.zeros((N, 4), device=device, dtype=torch.float32),
        "grad_scales": torch.zeros((N, 3), device=device, dtype=torch.float32),
        "grad_opacities": torch.zeros((N,), device=device, dtype=torch.float32),
        "grad_colors": torch.zeros((N, K_SH, 3), device=device, dtype=torch.float32),
    }


def run_backward(exp_mod, state, v_render_colors, v_render_alphas, grads, device):
    """Call higs_rasterize_backward and return (v_means2d, v_conics, v_colors, v_opacities, ...)."""
    return exp_mod.higs_rasterize_backward(
        state["means2d"],
        state["conics"],
        state["colors_eval"],
        state["opacities"],
        state["backgrounds"],
        state["tile_offsets"],
        state["flatten_ids"],
        state["active_tiles"],
        state["render_alphas"],
        state["last_ids"],
        state["means"],
        state["quats"],
        state["scales"],
        state["radii"],
        state["viewmats"],
        state["Ks"],
        state["width"],
        state["height"],
        TILE_SIZE,
        0.3,     # eps2d
        0,       # camera_model (pinhole)
        v_render_colors,
        v_render_alphas,
        state["sh_coeffs"],
        SH_DEGREE,
        state["visible_ids"],
        grads["grad_means"],
        grads["grad_quats"],
        grads["grad_scales"],
        grads["grad_opacities"],
        grads["grad_colors"],
    )


def set_env_vars(variant_config):
    """Set env vars for a variant."""
    for key, val in variant_config.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def measure_correctness(exp_mod, state, device, seed=42):
    """Section 6: Correctness via noise-envelope protocol.

    Inject fixed random noise as v_render_colors/v_render_alphas,
    run backward with each variant, compare all 7 outputs.

    Return order: (grad_means, grad_quats, grad_scales, grad_opacities,
                   grad_colors, v_backgrounds, v_means2d)
    All outputs depend on the blend backward kernel, so comparing all 7
    is a strong correctness check covering blend + projection + SH backward.
    """
    H, W = state["height"], state["width"]

    # Fixed noise for reproducibility
    torch.manual_seed(seed)
    v_render_colors = torch.randn(1, H, W, 3, device=device, dtype=torch.float32)
    v_render_alphas = torch.randn(1, H, W, 1, device=device, dtype=torch.float32)

    OUTPUT_NAMES = [
        "grad_means", "grad_quats", "grad_scales", "grad_opacities",
        "grad_colors", "v_backgrounds", "v_means2d"
    ]

    results = {}
    baseline_outputs = None

    for vname, vcfg in VARIANTS.items():
        set_env_vars(vcfg)
        grads = make_grad_buffers(state, device)
        torch.cuda.synchronize()
        outputs = run_backward(exp_mod, state, v_render_colors, v_render_alphas, grads, device)
        torch.cuda.synchronize()

        entry = {}
        for i, name in enumerate(OUTPUT_NAMES):
            entry[f"{name}_shape"] = list(outputs[i].shape)

        if baseline_outputs is None:
            baseline_outputs = [o.clone() for o in outputs]
            for i, name in enumerate(OUTPUT_NAMES):
                entry[f"max_abs_{name}"] = float(outputs[i].abs().max())
        else:
            # Compare against baseline
            for i, name in enumerate(OUTPUT_NAMES):
                ref = baseline_outputs[i]
                out = outputs[i]
                diff = (out - ref).abs()
                entry[f"max_abs_diff_{name}"] = float(diff.max())
                entry[f"mean_abs_diff_{name}"] = float(diff.mean())
                ref_abs = ref.abs().clamp_min(1e-8)
                entry[f"max_rel_diff_{name}"] = float((diff / ref_abs).max())

        results[vname] = entry
        print(f"  {vname}:", flush=True)
        for name in OUTPUT_NAMES:
            if f"max_abs_diff_{name}" in entry:
                print(f"    {name}: max_diff={entry[f'max_abs_diff_{name}']:.2e} "
                      f"mean_diff={entry[f'mean_abs_diff_{name}']:.2e}", flush=True)
            elif f"max_abs_{name}" in entry:
                print(f"    {name}: max_abs={entry[f'max_abs_{name}']:.2e} (baseline)", flush=True)
        del outputs, grads
        gc.collect()

    return results


def measure_timing(exp_mod, state, device, n_warmup=20, n_measure=100, n_repeats=5):
    """Section 7: Timing with interleaved CUDA Events.

    20 warmup, 100 measured, 5 repeats. Interleaved: each repeat runs all variants.
    """
    H, W = state["height"], state["width"]

    # Fixed noise (same for all variants for fair comparison)
    torch.manual_seed(123)
    v_render_colors = torch.randn(1, H, W, 3, device=device, dtype=torch.float32)
    v_render_alphas = torch.randn(1, H, W, 1, device=device, dtype=torch.float32)

    # Pre-allocate grad buffers (reuse for all calls)
    base_grads = make_grad_buffers(state, device)

    variant_names = list(VARIANTS.keys())
    all_times = {v: [] for v in variant_names}

    for rep in range(n_repeats):
        # Warmup for all variants
        for vname in variant_names:
            set_env_vars(VARIANTS[vname])
            grads = make_grad_buffers(state, device)
            for _ in range(n_warmup):
                run_backward(exp_mod, state, v_render_colors, v_render_alphas, grads, device)
            torch.cuda.synchronize()

        # Measure: interleaved
        for vname in variant_names:
            set_env_vars(VARIANTS[vname])
            grads = make_grad_buffers(state, device)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Record all measurements in one batch
            times = []
            for _ in range(n_measure):
                start_event.record()
                run_backward(exp_mod, state, v_render_colors, v_render_alphas, grads, device)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))

            all_times[vname].append(times)
            print(f"  rep {rep} {vname}: median={np.median(times):.3f}ms mean={np.mean(times):.3f}ms std={np.std(times):.3f}ms", flush=True)
            del grads
            gc.collect()

    # Compute statistics
    results = {}
    for vname in variant_names:
        # Flatten all repeats
        all_flat = [t for rep_times in all_times[vname] for t in rep_times]
        # Per-repeat medians
        rep_medians = [np.median(rep_times) for rep_times in all_times[vname]]
        results[vname] = {
            "n_warmup": n_warmup,
            "n_measure": n_measure,
            "n_repeats": n_repeats,
            "median_ms": float(np.median(rep_medians)),
            "mean_ms": float(np.mean(rep_medians)),
            "std_ms": float(np.std(rep_medians)),
            "min_ms": float(np.min(rep_medians)),
            "max_ms": float(np.max(rep_medians)),
            "all_rep_medians": [float(x) for x in rep_medians],
        }

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", default="/tmp/higs_h3_fwd_1a_source")
    ap.add_argument("--core-so", default="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so")
    ap.add_argument("--exp-so", default="/tmp/higs_h3_fwd_1a_build/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so")
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--max-long-side", type=int, default=2048)
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-measure", type=int, default=100)
    ap.add_argument("--n-repeats", type=int, default=5)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PATH"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    run_id = str(uuid.uuid4())[:12]
    print(f"=== H5-0R Section 6+7: Correctness + Timing ===", flush=True)
    print(f"run_id={run_id} GPU={args.gpu}", flush=True)

    exp_mod = bootstrap(args.source, args.core_so, args.exp_so)
    print("Bootstrap complete.", flush=True)

    correctness_all = {}
    timing_all = {}

    for scene in ["room", "bicycle", "garden"]:
        print(f"\n{'='*60}", flush=True)
        print(f"=== Scene: {scene} ===", flush=True)
        print(f"{'='*60}", flush=True)

        state = capture_forward_state(scene, args.max_long_side, device, exp_mod)
        print(f"  N_vis={state['N_vis']} W={state['width']} H={state['height']} tw={state['tw']} th={state['th']}", flush=True)

        # Section 6: Correctness
        print(f"\n--- Section 6: Correctness ---", flush=True)
        correctness = measure_correctness(exp_mod, state, device)
        correctness_all[scene] = correctness

        # Section 7: Timing
        print(f"\n--- Section 7: Timing ---", flush=True)
        timing = measure_timing(exp_mod, state, device,
                                n_warmup=args.n_warmup, n_measure=args.n_measure, n_repeats=args.n_repeats)
        timing_all[scene] = timing

        # Print timing summary
        print(f"\n  Timing summary for {scene}:", flush=True)
        baseline_med = timing["baseline"]["median_ms"]
        for vname in VARIANTS:
            med = timing[vname]["median_ms"]
            speedup = (baseline_med / med - 1) * 100 if med > 0 else 0
            print(f"    {vname}: {med:.3f}ms ({speedup:+.2f}% vs baseline)", flush=True)

        del state
        gc.collect()
        torch.cuda.empty_cache()

    # Write artifacts
    (out_dir / "correctness.json").write_text(json.dumps(correctness_all, indent=2))
    print(f"\nWrote correctness.json", flush=True)

    # Write timing as CSV
    csv_lines = ["scene,variant,median_ms,mean_ms,std_ms,min_ms,max_ms,n_warmup,n_measure,n_repeats"]
    for scene in ["room", "bicycle", "garden"]:
        for vname in VARIANTS:
            t = timing_all[scene][vname]
            csv_lines.append(f"{scene},{vname},{t['median_ms']:.6f},{t['mean_ms']:.6f},{t['std_ms']:.6f},{t['min_ms']:.6f},{t['max_ms']:.6f},{t['n_warmup']},{t['n_measure']},{t['n_repeats']}")
    (out_dir / "timing.csv").write_text("\n".join(csv_lines))
    print(f"Wrote timing.csv", flush=True)

    # Write timing JSON too
    (out_dir / "timing.json").write_text(json.dumps(timing_all, indent=2))

    # Print final summary
    print(f"\n{'='*60}", flush=True)
    print(f"=== FINAL SUMMARY ===", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\nCorrectness (max abs diff vs baseline):", flush=True)
    OUTPUT_NAMES = ["grad_means", "grad_quats", "grad_scales", "grad_opacities",
                    "grad_colors", "v_backgrounds", "v_means2d"]
    for scene in ["room", "bicycle", "garden"]:
        print(f"  {scene}:", flush=True)
        for vname in ["H5A", "H5B", "H5C"]:
            c = correctness_all[scene][vname]
            diffs = " ".join(f"{n}={c.get(f'max_abs_diff_{n}', 0):.2e}" for n in OUTPUT_NAMES)
            print(f"    {vname}: {diffs}", flush=True)

    print(f"\nTiming (median ms, % vs baseline):", flush=True)
    for scene in ["room", "bicycle", "garden"]:
        print(f"  {scene}:", flush=True)
        baseline_med = timing_all[scene]["baseline"]["median_ms"]
        for vname in VARIANTS:
            med = timing_all[scene][vname]["median_ms"]
            speedup = (baseline_med / med - 1) * 100 if med > 0 else 0
            print(f"    {vname}: {med:.3f}ms ({speedup:+.2f}%)", flush=True)


if __name__ == "__main__":
    main()
