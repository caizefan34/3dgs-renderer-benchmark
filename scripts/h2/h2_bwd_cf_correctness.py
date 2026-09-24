#!/usr/bin/env python3
"""H2-BWD-CF raw backward comparison; run in a fresh Python process."""
import argparse, importlib.util, json, math, os, runpy, sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

VARIANTS = ("baseline", "sigma_gate", "scalar_adjoint", "uv_reuse", "combined")
K_SH = 16


def bootstrap(source: str, core_so: str):
    sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location("gsplat_cuda", core_so)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.modules["gsplat.csrc"] = core
    backend = runpy.run_path(
        str(Path(source) / "gsplat/experimental/render/kernels/cuda/build.py")
    )["build_and_load_experimental_gaussian_render_inference_scene"]()
    return backend


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
    cams = json.loads(Path(cameras_path).read_text())
    c = cams[0]
    native_w, native_h = int(c["width"]), int(c["height"])
    scale = min(1.0, max_long_side / max(native_w, native_h))
    width, height = int(round(native_w * scale)), int(round(native_h * scale))
    R = np.asarray(c["rotation"], dtype=np.float32).T
    p = np.asarray(c["position"], dtype=np.float32)
    vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R; vm[:3, 3] = -R @ p
    K = np.array([[float(c["fx"]) * width / native_w, 0, (width - 1) / 2],
                  [0, float(c["fy"]) * width / native_w, (height - 1) / 2], [0, 0, 1]], dtype=np.float32)
    return (means, quats, scales, opacities, sh), torch.tensor(vm, device=device)[None, None], torch.tensor(K, device=device)[None, None], width, height


def metrics(reference, value):
    a, b = reference.detach().float().reshape(-1), value.detach().float().reshape(-1)
    d = b - a
    finite = torch.isfinite(a) & torch.isfinite(b)
    af, bf, df = a[finite], b[finite], d[finite]
    denom = af.norm().clamp_min(1e-30)
    cosine = float(torch.dot(af, bf) / (af.norm().clamp_min(1e-30) * bf.norm().clamp_min(1e-30))) if af.numel() else float("nan")
    support = (a.abs() > 1e-10) != (b.abs() > 1e-10)
    return {"max_abs": float(df.abs().max()) if df.numel() else float("nan"),
            "mean_abs": float(df.abs().mean()) if df.numel() else float("nan"),
            "rel_L2": float(df.norm() / denom), "cosine": cosine,
            "NaN_count": int((a.isnan() | b.isnan()).sum()),
            "Inf_count": int((a.isinf() | b.isinf()).sum()),
            "zero_nonzero_support_disagreement": int(support.sum())}


def run_variant(values, vm, K, width, height, backend, variant, seed):
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER
    os.environ["HIGS_BWD_CF_VARIANT"] = variant
    os.environ["HIGS_BWD_CF_CAPTURE_RAW"] = "1"
    leaves = tuple(x.detach().clone().requires_grad_(True) for x in values)
    _HIGS_FROZEN_TRACKER.reset()
    handle = create_higs_renderer(*leaves, sh_degree=3)
    out = rasterize_gaussian_higs_frozen(leaves[0], leaves[1], leaves[2], leaves[3], leaves[4],
        backward_mode="higs_native", scene=handle, freeze_topology=True, viewmats=vm, Ks=K,
        width=width, height=height, sh_degree=3, use_higs_culling=True, radius_clip=0.0,
        tile_sampling_ratio=1.0)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    vr = torch.randn(out["frame"].shape, device="cuda", generator=gen)
    va = torch.randn(out["alpha"].shape, device="cuda", generator=gen)
    (out["frame"].float().mul(vr).sum() + out["alpha"].float().mul(va).sum()).backward()
    torch.cuda.synchronize()
    raw = backend.higs_bwd_cf_last_raw_grads()
    result = {"v_means2d": raw[0].detach().clone(), "v_conics": raw[1].detach().clone(),
              "v_colors": raw[2].detach().clone(), "v_opacities": raw[3].detach().clone(),
              "means": leaves[0].grad.detach().clone(), "quats": leaves[1].grad.detach().clone(),
              "scales": leaves[2].grad.detach().clone(), "opacities": leaves[3].grad.detach().clone(),
              "SH": leaves[4].grad.detach().clone()}
    handle.release()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--source", default="/tmp/higs_h2_bwd_cf/source")
    p.add_argument("--core-so", default="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so")
    p.add_argument("--ply", default="/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply")
    p.add_argument("--cameras", default="/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json")
    p.add_argument("--max-long-side", type=int, default=2048)
    p.add_argument("--seed", type=int, default=4200)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    backend = bootstrap(a.source, a.core_so)
    values, vm, K, width, height = load_fixture(a.ply, a.cameras, a.max_long_side, "cuda:0")
    runs = {v: run_variant(values, vm, K, width, height, backend, v, a.seed) for v in VARIANTS}
    baseline = runs.pop("baseline")
    payload = {"fixture": {"scene": "room", "camera": 0, "width": width, "height": height,
               "seed": a.seed, "same_forward_state": "forward capture is recreated from immutable identical masters"},
               "metrics": {v: {name: metrics(baseline[name], got[name]) for name in baseline} for v, got in runs.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(payload, allow_nan=False))


if __name__ == "__main__":
    main()
