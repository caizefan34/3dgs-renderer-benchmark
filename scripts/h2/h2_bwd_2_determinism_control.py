#!/usr/bin/env python3
"""H2-BWD-2 determinism control: run baseline-vs-baseline twice and
scalar-vs-baseline multiple times to test whether the gradient
differences (esp. quats) are transformation error or atomicAdd order noise.

If baseline-vs-baseline shows comparable rel_L2 to scalar-vs-baseline, the
difference is atomicAdd non-determinism, NOT the scalar transformation.
"""
import argparse, importlib.util, json, os, runpy, sys
from pathlib import Path
import numpy as np, torch
from plyfile import PlyData

K_SH = 16
VARIANTS = ("baseline",)


def bootstrap(source, core_so):
    sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location("gsplat_cuda", core_so)
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    sys.modules["gsplat.csrc"] = core
    return runpy.run_path(str(Path(source) / "gsplat/experimental/render/kernels/cuda/build.py"))["build_and_load_experimental_gaussian_render_inference_scene"]()


def load_fixture(ply_path, cameras_path, max_long_side, device):
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
    cams = json.loads(Path(cameras_path).read_text()); c = cams[0]
    nw, nh = int(c["width"]), int(c["height"]); sc = min(1.0, max_long_side / max(nw, nh))
    w, h = int(round(nw * sc)), int(round(nh * sc))
    R = np.asarray(c["rotation"], dtype=np.float32).T; p = np.asarray(c["position"], dtype=np.float32)
    vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R; vm[:3, 3] = -R @ p
    K = np.array([[float(c["fx"]) * w / nw, 0, (w - 1) / 2], [0, float(c["fy"]) * w / nw, (h - 1) / 2], [0, 0, 1]], dtype=np.float32)
    return (means, quats, scales, opacities, sh), torch.tensor(vm, device=device)[None, None], torch.tensor(K, device=device)[None, None], w, h


def metrics(reference, value):
    a = reference.detach().float().reshape(-1); b = value.detach().float().reshape(-1)
    finite = torch.isfinite(a) & torch.isfinite(b)
    af, bf = a[finite], b[finite]; d = bf - af
    denom = af.norm().clamp_min(1e-30)
    cos = float(torch.dot(af, bf) / (af.norm().clamp_min(1e-30) * bf.norm().clamp_min(1e-30))) if af.numel() else float("nan")
    return {"rel_L2": float(d.norm() / denom) if d.numel() else float("nan"), "cosine": cos,
            "max_abs": float(d.abs().max()) if d.numel() else float("nan"), "norm_baseline": float(af.norm())}


def run_variant(values, vm, K, w, h, backend, variant, seed):
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER
    os.environ["HIGS_BWD_CF_VARIANT"] = variant
    os.environ["HIGS_BWD_CF_CAPTURE_RAW"] = "1"
    leaves = tuple(x.detach().clone().requires_grad_(True) for x in values)
    _HIGS_FROZEN_TRACKER.reset(); handle = create_higs_renderer(*leaves, sh_degree=3)
    out = rasterize_gaussian_higs_frozen(*leaves, backward_mode="higs_native", scene=handle, freeze_topology=True,
        viewmats=vm, Ks=K, width=w, height=h, sh_degree=3, use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    vr = torch.randn(out["frame"].shape, device="cuda", generator=gen)
    va = torch.randn(out["alpha"].shape, device="cuda", generator=gen)
    (out["frame"].float().mul(vr).sum() + out["alpha"].float().mul(va).sum()).backward()
    torch.cuda.synchronize()
    raw = backend.higs_bwd_cf_last_raw_grads()
    res = {"v_means2d": raw[0].detach().clone(), "v_conics": raw[1].detach().clone(),
           "v_colors": raw[2].detach().clone(), "v_opacities": raw[3].detach().clone(),
           "means": leaves[0].grad.detach().clone(), "quats": leaves[1].grad.detach().clone(),
           "scales": leaves[2].grad.detach().clone(), "opacities": leaves[3].grad.detach().clone(), "SH": leaves[4].grad.detach().clone()}
    handle.release(); return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="/tmp/higs_h2_bwd_cf/source")
    ap.add_argument("--core-so", default="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so")
    ap.add_argument("--ply", default="/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply")
    ap.add_argument("--cameras", default="/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json")
    ap.add_argument("--seed", type=int, default=4200)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    os.environ["HIGS_PX_RUNTIME"] = "2"
    backend = bootstrap(a.source, a.core_so)
    values, vm, K, w, h = load_fixture(a.ply, a.cameras, 2048, "cuda:0")

    # baseline-vs-baseline: run baseline N times, compare each to the first
    print("=== baseline-vs-baseline (N=%d) ===" % a.n, flush=True)
    b_runs = [run_variant(values, vm, K, w, h, backend, "baseline", a.seed) for _ in range(a.n)]
    bb = {}
    for i in range(1, a.n):
        bb[i] = {t: metrics(b_runs[0][t], b_runs[i][t]) for t in b_runs[0]}
        q = bb[i]["quats"]
        print(f"  baseline_run{i} vs baseline_run0: quats rel_L2={q['rel_L2']:.4e} cos={q['cosine']:.8f}", flush=True)

    # scalar-vs-baseline: run scalar N times, compare each to baseline_run0
    print("=== scalar_adjoint-vs-baseline (N=%d) ===" % a.n, flush=True)
    s_runs = [run_variant(values, vm, K, w, h, backend, "scalar_adjoint", a.seed) for _ in range(a.n)]
    sb = {}
    for i in range(a.n):
        sb[i] = {t: metrics(b_runs[0][t], s_runs[i][t]) for t in b_runs[0]}
        q = sb[i]["quats"]
        print(f"  scalar_run{i} vs baseline_run0: quats rel_L2={q['rel_L2']:.4e} cos={q['cosine']:.8f}", flush=True)

    # also all-tensors max for each comparison
    def allmax(d): return max(d[t]["rel_L2"] for t in d), [t for t in d if d[t]["rel_L2"] == max(d[x]["rel_L2"] for x in d)][0]
    out = {"baseline_vs_baseline": {str(i): {t: bb[i][t] for t in bb[i]} for i in bb},
           "scalar_vs_baseline": {str(i): {t: sb[i][t] for t in sb[i]} for i in sb}}
    print("\n=== max rel_L2 tensor per comparison ===", flush=True)
    for i in bb:
        m, t = allmax(bb[i]); print(f"  baseline_run{i} vs run0: max={m:.4e} on {t}", flush=True)
    for i in sb:
        m, t = allmax(sb[i]); print(f"  scalar_run{i} vs run0: max={m:.4e} on {t}", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2, allow_nan=False))
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
