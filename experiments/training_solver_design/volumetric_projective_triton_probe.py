"""Isolated GPU arithmetic screen; not a Vol3DGS rasterizer benchmark."""

import json

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def precompute_b(A, MU, B, G: tl.constexpr, BLOCK: tl.constexpr):
    g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = g < G
    a00 = tl.load(A + 0 * G + g, ok, 0)
    a01 = tl.load(A + 1 * G + g, ok, 0)
    a02 = tl.load(A + 2 * G + g, ok, 0)
    a11 = tl.load(A + 3 * G + g, ok, 0)
    a12 = tl.load(A + 4 * G + g, ok, 0)
    a22 = tl.load(A + 5 * G + g, ok, 0)
    mx = tl.load(MU + 0 * G + g, ok, 0)
    my = tl.load(MU + 1 * G + g, ok, 0)
    mz = tl.load(MU + 2 * G + g, ok, 1)
    u0 = mx / mz
    v0 = my / mz
    aw0x = a00 * u0 + a01 * v0 + a02
    aw0y = a01 * u0 + a11 * v0 + a12
    aw0z = a02 * u0 + a12 * v0 + a22
    aw0 = u0 * aw0x + v0 * aw0y + aw0z
    z2 = mz * mz
    tl.store(B + 0 * G + g, z2 * (aw0 * a00 - aw0x * aw0x), ok)
    tl.store(B + 1 * G + g, z2 * (aw0 * a01 - aw0x * aw0y), ok)
    tl.store(B + 2 * G + g, z2 * (aw0 * a11 - aw0y * aw0y), ok)


@triton.jit
def ray_alpha(A, MU, B, DENSITY, U, V, NORM, OUT,
              G: tl.constexpr, N: tl.constexpr, P: tl.constexpr,
              PROJECTIVE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < N
    g = i // P
    a00 = tl.load(A + 0 * G + g, ok, 0)
    a01 = tl.load(A + 1 * G + g, ok, 0)
    a02 = tl.load(A + 2 * G + g, ok, 0)
    a11 = tl.load(A + 3 * G + g, ok, 0)
    a12 = tl.load(A + 4 * G + g, ok, 0)
    a22 = tl.load(A + 5 * G + g, ok, 0)
    mx = tl.load(MU + 0 * G + g, ok, 0)
    my = tl.load(MU + 1 * G + g, ok, 0)
    mz = tl.load(MU + 2 * G + g, ok, 1)
    kappa = tl.load(DENSITY + g, ok, 0)
    u = tl.load(U + i, ok, 0)
    v = tl.load(V + i, ok, 0)
    norm = tl.load(NORM + i, ok, 1)
    ax = a00 * u + a01 * v + a02
    ay = a01 * u + a11 * v + a12
    az = a02 * u + a12 * v + a22
    aw = u * ax + v * ay + az
    aunit = aw / (norm * norm)
    den = tl.maximum(1.0e-7, aunit)
    b = (mx * ax + my * ay + mz * az) / norm
    if PROJECTIVE:
        b00 = tl.load(B + 0 * G + g, ok, 0)
        b01 = tl.load(B + 1 * G + g, ok, 0)
        b11 = tl.load(B + 2 * G + g, ok, 0)
        du = u - mx / mz
        dv = v - my / mz
        q = (b00 * du * du + 2.0 * b01 * du * dv + b11 * dv * dv) / aw
    else:
        t = b / den
        rx = t * u / norm - mx
        ry = t * v / norm - my
        rz = t / norm - mz
        arx = a00 * rx + a01 * ry + a02 * rz
        ary = a01 * rx + a11 * ry + a12 * rz
        arz = a02 * rx + a12 * ry + a22 * rz
        q = rx * arx + ry * ary + rz * arz
    optical = 2.506 * kappa * tl.sqrt(1.0 / (den + 1.0e-7)) * tl.exp(-0.5 * q)
    alpha = tl.minimum(0.99, 1.0 - tl.exp(-optical))
    tl.store(OUT + i, tl.where(b <= 0.0, 0.0, alpha), ok)


def main():
    rng = np.random.default_rng(20260927)
    g = 4096
    z = np.exp(rng.uniform(np.log(1), np.log(50), g))
    mu = np.column_stack((rng.uniform(-.5, .5, g) * z,
                          rng.uniform(-.5, .5, g) * z, z)).astype(np.float32)
    scales = np.exp(rng.uniform(np.log(.001), np.log(.1), (g, 3)))
    rotation, _ = np.linalg.qr(rng.normal(size=(g, 3, 3)))
    invcov = rotation @ (np.eye(3)[None] / scales[:, None, :] ** 2) @ np.swapaxes(rotation, 1, 2)
    invcov = invcov.astype(np.float32)
    packed = np.stack((invcov[:, 0, 0], invcov[:, 0, 1], invcov[:, 0, 2],
                       invcov[:, 1, 1], invcov[:, 1, 2], invcov[:, 2, 2]))
    a_gpu = torch.tensor(packed, device="cuda")
    mu_gpu = torch.tensor(mu.T.copy(), device="cuda")
    b_gpu = torch.empty((3, g), device="cuda", dtype=torch.float32)
    density_gpu = torch.tensor(rng.uniform(.01, 2, g).astype(np.float32), device="cuda")
    precompute_b[(triton.cdiv(g, 256),)](a_gpu, mu_gpu, b_gpu, g, 256)
    rows = []
    for p in (16, 64, 256):
        n = g * p
        offsets = rng.normal(size=(g, p, 2)) * (scales.min(axis=1) / z)[:, None, None]
        uv = (mu[:, :2] / mu[:, 2, None])[:, None, :] + offsets
        uv = uv.reshape(n, 2).astype(np.float32)
        norm = np.sqrt(1 + np.sum(uv ** 2, axis=1)).astype(np.float32)
        u_gpu = torch.tensor(uv[:, 0].copy(), device="cuda")
        v_gpu = torch.tensor(uv[:, 1].copy(), device="cuda")
        norm_gpu = torch.tensor(norm, device="cuda")
        original = torch.empty(n, device="cuda")
        projective = torch.empty(n, device="cuda")

        def run(flag, out):
            ray_alpha[(triton.cdiv(n, 256),)](a_gpu, mu_gpu, b_gpu, density_gpu,
                    u_gpu, v_gpu, norm_gpu, out, g, n, p, flag, 256)

        run(False, original)
        run(True, projective)
        torch.cuda.synchronize()
        diff = (original - projective).abs()
        pairs = []
        for repeat in range(6):
            if repeat % 2:
                ms_projective = triton.testing.do_bench_cudagraph(
                    lambda: run(True, projective), rep=40)
                ms_original = triton.testing.do_bench_cudagraph(
                    lambda: run(False, original), rep=40)
            else:
                ms_original = triton.testing.do_bench_cudagraph(
                    lambda: run(False, original), rep=40)
                ms_projective = triton.testing.do_bench_cudagraph(
                    lambda: run(True, projective), rep=40)
            pairs.append((ms_original, ms_projective))
        ratios = [candidate / baseline for baseline, candidate in pairs]
        rows.append({"pixels_per_gaussian": p, "interactions": n,
                     "original_ms_median": float(np.median([x[0] for x in pairs])),
                     "projective_ms_median": float(np.median([x[1] for x in pairs])),
                     "paired_ratio_median": float(np.median(ratios)),
                     "paired_ratio_min": float(np.min(ratios)),
                     "paired_ratio_max": float(np.max(ratios)),
                     "alpha_abs_p99": float(torch.quantile(diff, .99).item()),
                     "alpha_abs_max": float(diff.max().item())})
    ms_precompute = triton.testing.do_bench_cudagraph(
        lambda: precompute_b[(triton.cdiv(g, 256),)](a_gpu, mu_gpu, b_gpu, g, 256), rep=40)
    print(json.dumps({"device": torch.cuda.get_device_name(), "gaussians": g,
                      "precompute_ms": ms_precompute, "rows": rows,
                      "scope": "CUDA-graph paired flat forward-only arithmetic kernel; excludes tile sorting, blending, backward, training, memory pressure"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
