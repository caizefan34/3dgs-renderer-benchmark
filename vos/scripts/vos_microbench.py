"""VOS Phase-6 — Performance Oracle: isolated CUDA microbenchmark (v2).

Compares, on one A100 (EPIC-05), an optimized FUSED dense Adam (the honest
best-case denominator) against VOS indexed optimizer kernels:

  DENSE  : full-N fused Adam — one pass over all N rows x D elements.
           Reads grad, m, v, theta; writes m, v, theta. (28 B/elem @ fp32)
  VOS v0 : indexed kernel over active rows; replays K deferred zero-grad
           transitions in registers; per-step scalars from 3 separate tables
           (lr[s], 1/(1-b1^s), sqrt(1/(1-b2^s))) — 3 scalar loads/step.
  VOS v1 : same, but scalars from ONE packed float4 table — 1 x 16B load/step
           (the implementation a production kernel would use; v0 kept to show
           the table-layout effect, which the SASS audit identified as the
           K>=~250 regime change: 3x128KB tables exceed L1 -> L2-latency-bound).

Scene configurations use the REAL measured (N, vis_frac, react_share) from the
Phase-4 trace analysis, and K values resampled from the REAL reactivation-gap
histogram of the matching 500-iter trace window when --trace-dir is given.

Correctness: --check-only runs both kernels against sequential torch references
(verified 2026-09-29: d_theta <= 7.5e-8, i.e. fast-math reassociation envelope,
untouched rows bit-identical; SASS audit confirms per-step loop execution).
"""
import argparse
import json
import os
import struct

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void dense_adam_kernel(
    const float* __restrict__ grad,
    float* __restrict__ theta,
    float* __restrict__ m,
    float* __restrict__ v,
    const float lr, const float beta1, const float beta2, const float eps,
    const float bc1, const float bc2,
    const long long nelem)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= nelem) return;
    float g = grad[i];
    float mm = m[i], vv = v[i], th = theta[i];
    mm += (1.0f - beta1) * (g - mm);
    vv = vv * beta2 + (1.0f - beta2) * g * g;
    m[i] = mm; v[i] = vv;
    float denom = sqrtf(vv) / bc2 + eps;
    theta[i] = th + (-lr / bc1) * mm / denom;
}

// VOS v0: three scalar tables
__global__ void vos_adam_kernel(
    const int* __restrict__ active_ids,
    const int* __restrict__ karr,
    const float* __restrict__ grad_rows,
    float* __restrict__ theta,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ lr_tab,
    const float* __restrict__ ibc1_tab,
    const float* __restrict__ ibc2_tab,
    const int D, const int n_active, const int t_now,
    const float beta1, const float beta2, const float eps)
{
    long long task = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long ntask = (long long)n_active * D;
    if (task >= ntask) return;
    int row = (int)(task / D);
    int elem = (int)(task % D);
    int master = active_ids[row];
    long long base = (long long)master * D + elem;
    float mm = m[base], vv = v[base], th = theta[base];

    int K = karr[row];
    #pragma unroll 2
    for (int j = K; j > 0; --j) {
        int s = t_now - j;
        float lr_s = lr_tab[s];
        float ibc1 = ibc1_tab[s];
        float ibc2 = ibc2_tab[s];
        mm += (1.0f - beta1) * (0.0f - mm);
        vv = vv * beta2 + (1.0f - beta2) * 0.0f;
        float denom = sqrtf(vv) * ibc2 + eps;
        th += (-lr_s * ibc1) * mm / denom;
    }

    float g = grad_rows[(long long)row * D + elem];
    float lr_s = lr_tab[t_now];
    float ibc1 = ibc1_tab[t_now];
    float ibc2 = ibc2_tab[t_now];
    mm += (1.0f - beta1) * (g - mm);
    vv = vv * beta2 + (1.0f - beta2) * g * g;
    float denom = sqrtf(vv) * ibc2 + eps;
    th += (-lr_s * ibc1) * mm / denom;

    m[base] = mm; v[base] = vv; theta[base] = th;
}

// VOS v1: packed float4 table {lr, ibc1, ibc2, pad} — one 16B load per step
__global__ void vos_adam_packed_kernel(
    const int* __restrict__ active_ids,
    const int* __restrict__ karr,
    const float* __restrict__ grad_rows,
    float* __restrict__ theta,
    float* __restrict__ m,
    float* __restrict__ v,
    const float4* __restrict__ tab,     // [T+1] {lr, ibc1, ibc2, 0}
    const int D, const int n_active, const int t_now,
    const float beta1, const float beta2, const float eps)
{
    long long task = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long ntask = (long long)n_active * D;
    if (task >= ntask) return;
    int row = (int)(task / D);
    int elem = (int)(task % D);
    int master = active_ids[row];
    long long base = (long long)master * D + elem;
    float mm = m[base], vv = v[base], th = theta[base];

    int K = karr[row];
    #pragma unroll 2
    for (int j = K; j > 0; --j) {
        float4 sc = tab[t_now - j];
        mm += (1.0f - beta1) * (0.0f - mm);
        vv = vv * beta2 + (1.0f - beta2) * 0.0f;
        float denom = sqrtf(vv) * sc.z + eps;
        th += (-sc.x * sc.y) * mm / denom;
    }

    float g = grad_rows[(long long)row * D + elem];
    float4 sc = tab[t_now];
    mm += (1.0f - beta1) * (g - mm);
    vv = vv * beta2 + (1.0f - beta2) * g * g;
    float denom = sqrtf(vv) * sc.z + eps;
    th += (-sc.x * sc.y) * mm / denom;

    m[base] = mm; v[base] = vv; theta[base] = th;
}

void dense_adam(torch::Tensor grad, torch::Tensor theta, torch::Tensor m,
                torch::Tensor v, double lr, double beta1, double beta2,
                double eps, double bc1, double bc2) {
    long long nelem = theta.numel();
    int block = 256;
    long long grid = (nelem + block - 1) / block;
    dense_adam_kernel<<<(unsigned)grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad.data_ptr<float>(), theta.data_ptr<float>(), m.data_ptr<float>(),
        v.data_ptr<float>(), (float)lr, (float)beta1, (float)beta2, (float)eps,
        (float)bc1, (float)bc2, nelem);
}

void vos_adam(torch::Tensor active_ids, torch::Tensor karr, torch::Tensor grad_rows,
              torch::Tensor theta, torch::Tensor m, torch::Tensor v,
              torch::Tensor lr_tab, torch::Tensor ibc1_tab, torch::Tensor ibc2_tab,
              int64_t D, int64_t t_now, double beta1, double beta2, double eps) {
    int n_active = active_ids.numel();
    if (n_active == 0) return;
    long long ntask = (long long)n_active * D;
    int block = 256;
    long long grid = (ntask + block - 1) / block;
    vos_adam_kernel<<<(unsigned)grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        active_ids.data_ptr<int>(), karr.data_ptr<int>(), grad_rows.data_ptr<float>(),
        theta.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(),
        lr_tab.data_ptr<float>(), ibc1_tab.data_ptr<float>(), ibc2_tab.data_ptr<float>(),
        (int)D, n_active, (int)t_now, (float)beta1, (float)beta2, (float)eps);
}

void vos_adam_packed(torch::Tensor active_ids, torch::Tensor karr,
                     torch::Tensor grad_rows, torch::Tensor theta, torch::Tensor m,
                     torch::Tensor v, torch::Tensor tab,
                     int64_t D, int64_t t_now, double beta1, double beta2,
                     double eps) {
    int n_active = active_ids.numel();
    if (n_active == 0) return;
    long long ntask = (long long)n_active * D;
    int block = 256;
    long long grid = (ntask + block - 1) / block;
    vos_adam_packed_kernel<<<(unsigned)grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        active_ids.data_ptr<int>(), karr.data_ptr<int>(), grad_rows.data_ptr<float>(),
        theta.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(),
        (const float4*)tab.data_ptr<float>(),
        (int)D, n_active, (int)t_now, (float)beta1, (float)beta2, (float)eps);
}
"""

CPP_SRC = """
void dense_adam(torch::Tensor grad, torch::Tensor theta, torch::Tensor m,
                torch::Tensor v, double lr, double beta1, double beta2,
                double eps, double bc1, double bc2);
void vos_adam(torch::Tensor active_ids, torch::Tensor karr, torch::Tensor grad_rows,
              torch::Tensor theta, torch::Tensor m, torch::Tensor v,
              torch::Tensor lr_tab, torch::Tensor ibc1_tab, torch::Tensor ibc2_tab,
              int64_t D, int64_t t_now, double beta1, double beta2, double eps);
void vos_adam_packed(torch::Tensor active_ids, torch::Tensor karr,
                     torch::Tensor grad_rows, torch::Tensor theta, torch::Tensor m,
                     torch::Tensor v, torch::Tensor tab,
                     int64_t D, int64_t t_now, double beta1, double beta2,
                     double eps);
"""


def build_ext(verbose=False):
    return load_inline(
        name="vos_microbench_ext_v2",
        cpp_sources=[CPP_SRC],
        cuda_sources=[CUDA_SRC],
        functions=["dense_adam", "vos_adam", "vos_adam_packed"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose)


def time_kernel(fn, warmup=20, reps=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {"mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
            "min_ms": float(np.min(times)), "max_ms": float(np.max(times))}


def load_gap_histogram(trace_dir, it_lo, it_hi, max_rows=None):
    """Empirical reactivation-gap histogram from a real trace segment."""
    meta = json.load(open(os.path.join(trace_dir, "vos_trace_meta.json")))
    recs = meta["records"]
    offs = {}
    off = 0
    for r in recs:
        offs[r[0]] = off
        off += 16 + r[2]
    N = recs[it_lo - 1][1]
    assert all(r[1] == N for r in recs[it_lo - 1:it_hi]), "N must be constant"
    if max_rows is not None and N > max_rows:
        sub = np.sort(np.random.RandomState(0).choice(N, size=max_rows, replace=False))
    else:
        sub = None
    seg = []
    with open(os.path.join(trace_dir, "vos_trace.bin"), "rb") as f:
        for it in range(it_lo, it_hi + 1):
            f.seek(offs[it])
            _it, _N, nb, _shd = struct.unpack("<4i", f.read(16))
            bits = np.unpackbits(np.frombuffer(f.read(nb), dtype=np.uint8),
                                 bitorder="big")[:N].astype(bool)
            seg.append(bits if sub is None else bits[sub])
    last_vis = np.full(len(seg[0]), -1, dtype=np.int64)
    gaps = []
    for t, b in enumerate(seg):
        idx = np.nonzero(b)[0]
        if len(idx):
            newly = idx[last_vis[idx] < t - 1]
            if len(newly):
                gaps.append(t - last_vis[newly] - 1)
        last_vis[idx] = t
    gaps = np.concatenate(gaps) if gaps else np.array([], dtype=np.int64)
    return {"gaps": gaps, "N": N}


def check_kernel_correctness(ext, dev, beta1, beta2, eps):
    """Correctness gate: VOS kernels (v0 + packed v1, K-step register replay +
    real step) vs a sequential torch reference; dense kernel vs torch Adam."""
    torch.manual_seed(123)
    N, D, K, t_now = 4096, 48, 17, 50000
    theta0 = torch.randn(N, D, device=dev) * 0.1
    m0 = torch.randn(N, D, device=dev) * 1e-3
    v0_ = torch.rand(N, D, device=dev) * 1e-5 + 1e-6
    grad = torch.randn(N, D, device=dev) * 1e-2
    n_active = 1024
    active_ids = torch.from_numpy(np.sort(np.random.RandomState(0).choice(
        N, size=n_active, replace=False)).astype(np.int32)).to(dev)
    karr = torch.from_numpy(
        np.random.RandomState(1).randint(0, K + 1, size=n_active).astype(np.int32)).to(dev)

    T_MAX = t_now + 1
    s = torch.arange(0, T_MAX, dtype=torch.float64)
    lr_t = torch.full((T_MAX,), 0.0025, dtype=torch.float64).to(dev).float().contiguous()
    ibc1 = (1.0 / (1.0 - beta1 ** s)).to(dev).float().contiguous()
    ibc2 = torch.sqrt(1.0 / (1.0 - beta2 ** s)).to(dev).float().contiguous()
    tab = torch.stack([lr_t, ibc1, ibc2, torch.zeros_like(lr_t)], dim=1).contiguous()

    # sequential torch reference
    th_r, m_r, v_r = theta0.clone(), m0.clone(), v0_.clone()
    for i in range(n_active):
        row = int(active_ids[i])
        Ki = int(karr[i])
        mrow, vrow, throw = m_r[row].clone(), v_r[row].clone(), th_r[row].clone()
        for sstep in range(t_now - Ki, t_now + 1):
            g = grad[row] if sstep == t_now else torch.zeros_like(mrow)
            mrow = mrow + (1.0 - beta1) * (g - mrow)
            vrow = vrow * beta2 + (1.0 - beta2) * g * g
            bc1 = 1.0 - beta1 ** sstep
            bc2 = 1.0 - beta2 ** sstep
            denom = (vrow.sqrt() / (bc2 ** 0.5)) + eps
            throw = throw + (-(0.0025 / bc1)) * mrow / denom
        th_r[row], m_r[row], v_r[row] = throw, mrow, vrow

    out = {}
    for name, run in (
        ("vos_v0", lambda th, m, v: ext.vos_adam(active_ids, karr, grad[active_ids],
            th, m, v, lr_t, ibc1, ibc2, D, t_now, beta1, beta2, eps)),
        ("vos_v1_packed", lambda th, m, v: ext.vos_adam_packed(active_ids, karr,
            grad[active_ids], th, m, v, tab, D, t_now, beta1, beta2, eps)),
    ):
        th_k, m_k, v_k = theta0.clone(), m0.clone(), v0_.clone()
        run(th_k, m_k, v_k)
        torch.cuda.synchronize()
        d_th = (th_k[active_ids] - th_r[active_ids]).abs().max().item()
        d_m = (m_k[active_ids] - m_r[active_ids]).abs().max().item()
        d_v = (v_k[active_ids] - v_r[active_ids]).abs().max().item()
        untouched = abs((th_k.sum() - th_k[active_ids].sum()).item()
                        - (theta0.sum() - theta0[active_ids].sum()).item())
        print(f"[check] {name}: d_theta={d_th:.3e} d_m={d_m:.3e} d_v={d_v:.3e} "
              f"untouched_delta={untouched:.3e}")
        assert d_th < 5e-6 and d_m < 5e-6 and d_v < 5e-6, f"{name} math mismatch"
        assert untouched < 1e-3, f"{name} touched non-active rows"
        out[name] = {"d_theta": d_th, "d_m": d_m, "d_v": d_v}

    p = torch.nn.Parameter(theta0.clone())
    opt = torch.optim.Adam([p], lr=0.0025, betas=(beta1, beta2), eps=eps,
                           foreach=False, fused=False)
    with torch.no_grad():
        opt.state[p]["exp_avg"] = m0.clone()
        opt.state[p]["exp_avg_sq"] = v0_.clone()
        opt.state[p]["step"] = torch.tensor(float(t_now))
    p.grad = grad.clone()
    opt.step()
    th_d, m_d, v_d = theta0.clone(), m0.clone(), v0_.clone()
    ext.dense_adam(grad, th_d, m_d, v_d, 0.0025, beta1, beta2, eps,
                   1 - beta1 ** t_now, 1 - beta2 ** t_now)
    torch.cuda.synchronize()
    dd_th = (th_d - p.detach()).abs().max().item()
    dd_m = (m_d - opt.state[p]["exp_avg"]).abs().max().item()
    dd_v = (v_d - opt.state[p]["exp_avg_sq"]).abs().max().item()
    print(f"[check] dense: d_theta={dd_th:.3e} d_m={dd_m:.3e} d_v={dd_v:.3e}")
    assert dd_th < 5e-6 and dd_m < 5e-6 and dd_v < 5e-6, "dense kernel mismatch"
    out["dense"] = {"d_theta": dd_th, "d_m": dd_m, "d_v": dd_v}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--trace-root", default="/root/vos_traces")
    ap.add_argument("--it-lo", type=int, default=29001)
    ap.add_argument("--it-hi", type=int, default=30000)
    ap.add_argument("--max-trace-rows", type=int, default=200000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--betas", type=str, default="0.9,0.999")
    ap.add_argument("--eps", type=float, default=1e-15)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--lock-clocks", action="store_true",
                    help="lock GPU clocks during the benchmark (root)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    beta1, beta2 = [float(x) for x in args.betas.split(",")]
    dev = "cuda"
    ext = build_ext()
    torch.manual_seed(0)

    if args.check_only:
        out = check_kernel_correctness(ext, dev, beta1, beta2, args.eps)
        print("KERNEL_CHECK_PASS")
        print(json.dumps(out, indent=2))
        return

    if args.lock_clocks:
        os.system(f"nvidia-smi -i {args.gpu} -lgc 1410,1410")
        os.system(f"nvidia-smi -i {args.gpu} --query-gpu=clocks.sm --format=csv")

    T_MAX = 32001
    s = torch.arange(0, T_MAX, dtype=torch.float64)
    lr_t = torch.full((T_MAX,), 0.0025, dtype=torch.float64)
    ibc1 = 1.0 / (1.0 - beta1 ** s)
    ibc2 = torch.sqrt(1.0 / (1.0 - beta2 ** s))
    lr_t = lr_t.to(dev).float().contiguous()
    ibc1_t = ibc1.to(dev).float().contiguous()
    ibc2_t = ibc2.to(dev).float().contiguous()
    tab = torch.stack([lr_t, ibc1_t, ibc2_t,
                       torch.zeros_like(lr_t)], dim=1).contiguous()
    bc1_now, bc2_now = float(1 - beta1 ** 30000), float(1 - beta2 ** 30000)

    # ---- real scene configs (Phase-4 analysis, 2026-09-29) -------------------
    scene_cfg = {
        "room":     {"N": 841806,  "vis_frac": 0.2682, "react_share": 0.63},
        "bicycle":  {"N": 3655232, "vis_frac": 0.2212, "react_share": 0.55},
        "garden":   {"N": 2558915, "vis_frac": 0.3152, "react_share": 0.45},
    }

    results = {"gpu": torch.cuda.get_device_name(0), "betas": [beta1, beta2],
               "eps": args.eps, "warmup": args.warmup, "reps": args.reps,
               "scenes": {}, "k_scaling": [], "sweeps": {}, "notes": [],
               "scene_cfg_source": "Phase-4 trace analysis vos_region_15K_30K"}

    D = 48
    BYTES_PER_ELEM = 28

    for scene, cfg in scene_cfg.items():
        N, frac, react_share = cfg["N"], cfg["vis_frac"], cfg["react_share"]
        trace_dir = os.path.join(args.trace_root, f"{scene}_T30")
        print(f"[mb] {scene}: loading real gaps from {trace_dir} "
              f"iters {args.it_lo}..{args.it_hi}", flush=True)
        gh = load_gap_histogram(trace_dir, args.it_lo, args.it_hi,
                                max_rows=args.max_trace_rows)
        gaps = gh["gaps"]
        print(f"[mb] {scene}: real gaps n={len(gaps)} mean={gaps.mean():.2f} "
              f"p50={np.percentile(gaps,50):.0f} p90={np.percentile(gaps,90):.0f} "
              f"p99={np.percentile(gaps,99):.0f} max={gaps.max()}", flush=True)

        theta = torch.randn(N, D, device=dev) * 0.1
        m = torch.randn(N, D, device=dev) * 1e-3
        v = torch.rand(N, D, device=dev) * 1e-5 + 1e-6
        grad = torch.randn(N, D, device=dev) * 1e-2

        td = time_kernel(lambda: ext.dense_adam(grad, theta, m, v, 0.0025,
                                                beta1, beta2, args.eps,
                                                bc1_now, bc2_now),
                         args.warmup, args.reps)
        dense_bytes = N * D * BYTES_PER_ELEM
        dense_gbs = dense_bytes / (td["median_ms"] * 1e-3) / 1e9

        n_active = max(1, int(N * frac))
        n_react = max(1, int(n_active * react_share))
        n_steady = n_active - n_react
        kvals = np.zeros(n_active, dtype=np.int32)
        kvals[n_steady:] = np.random.RandomState(1).choice(
            gaps, size=n_react).astype(np.int32) if len(gaps) else 1
        active_ids = torch.from_numpy(
            np.sort(np.random.RandomState(2).choice(
                N, size=n_active, replace=False)).astype(np.int32)).to(dev)
        karr = torch.from_numpy(kvals).to(dev)
        grad_rows = torch.randn(n_active, D, device=dev) * 1e-2

        tv0 = time_kernel(lambda: ext.vos_adam(active_ids, karr, grad_rows,
                                               theta, m, v, lr_t, ibc1_t, ibc2_t,
                                               D, 30000, beta1, beta2, args.eps),
                          args.warmup, args.reps)
        tv1 = time_kernel(lambda: ext.vos_adam_packed(active_ids, karr, grad_rows,
                                                      theta, m, v, tab,
                                                      D, 30000, beta1, beta2,
                                                      args.eps),
                          args.warmup, args.reps)
        vos_bytes = n_active * D * 6 * 4 + n_active * 3 * 4
        entry = {
            "N": N, "D": D, "vis_frac": frac, "n_active": n_active,
            "react_share": react_share, "mean_K": float(kvals[n_steady:].mean()),
            "sum_K": int(kvals.sum()),
            "gap_hist_source": {"it_lo": args.it_lo, "it_hi": args.it_hi,
                                "gap_count": int(len(gaps))},
            "dense_ms": td, "dense_bytes": dense_bytes, "dense_gbs": dense_gbs,
            "vos_v0_ms": tv0, "vos_v1_packed_ms": tv1,
            "vos_bytes": vos_bytes,
            "vos_v0_gbs": vos_bytes / (tv0["median_ms"] * 1e-3) / 1e9,
            "vos_v1_gbs": vos_bytes / (tv1["median_ms"] * 1e-3) / 1e9,
            "kernel_speedup_v0": td["median_ms"] / tv0["median_ms"],
            "kernel_speedup_v1": td["median_ms"] / tv1["median_ms"],
        }
        results["scenes"][scene] = entry
        print(f"[mb] {scene}: dense {td['median_ms']:.3f}ms ({dense_gbs:.0f} GB/s) | "
              f"vos_v0 {tv0['median_ms']:.3f}ms ({entry['kernel_speedup_v0']:.2f}x) | "
              f"vos_v1 {tv1['median_ms']:.3f}ms ({entry['kernel_speedup_v1']:.2f}x)",
              flush=True)

    # ---- K scaling with BOTH variants (locked to the room active size) -------
    N = scene_cfg["room"]["N"]
    frac = scene_cfg["room"]["vis_frac"]
    n_active = max(1, int(N * frac))
    theta = torch.randn(N, D, device=dev) * 0.1
    m = torch.randn(N, D, device=dev) * 1e-3
    v = torch.rand(N, D, device=dev) * 1e-5 + 1e-6
    grad_rows = torch.randn(n_active, D, device=dev) * 1e-2
    grad_full = torch.randn(N, D, device=dev) * 1e-2
    active_ids = torch.from_numpy(
        np.sort(np.random.RandomState(2).choice(
            N, size=n_active, replace=False)).astype(np.int32)).to(dev)
    t_dense = time_kernel(lambda: ext.dense_adam(grad_full, theta, m, v, 0.0025,
                                                 beta1, beta2, args.eps,
                                                 bc1_now, bc2_now),
                          args.warmup, args.reps)["median_ms"]
    results["k_scaling_dense_ms"] = t_dense
    for K in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        karr = torch.full((n_active,), K, dtype=torch.int32, device=dev)
        tv0 = time_kernel(lambda: ext.vos_adam(active_ids, karr, grad_rows,
                                               theta, m, v, lr_t, ibc1_t, ibc2_t,
                                               D, 30000, beta1, beta2, args.eps),
                          args.warmup, args.reps)
        tv1 = time_kernel(lambda: ext.vos_adam_packed(active_ids, karr, grad_rows,
                                                      theta, m, v, tab,
                                                      D, 30000, beta1, beta2,
                                                      args.eps),
                          args.warmup, args.reps)
        results["k_scaling"].append({
            "K": K, "n_active": n_active, "N": N,
            "vos_v0_ms": tv0["median_ms"], "vos_v1_ms": tv1["median_ms"],
            "ratio_v0_vs_dense": tv0["median_ms"] / t_dense,
            "ratio_v1_vs_dense": tv1["median_ms"] / t_dense,
        })
        print(f"[mb] K={K}: v0 {tv0['median_ms']:.3f}ms v1 {tv1['median_ms']:.3f}ms "
              f"(dense {t_dense:.3f}ms)", flush=True)

    # ---- speedup(active_frac, gap) surface, packed variant -------------------
    surface = []
    for frac in [0.05, 0.08, 0.13, 0.20, 0.27, 0.32, 0.40, 0.50]:
        row = []
        for gap in [1, 2, 5, 10, 20, 50, 100, 300, 1000]:
            n_active = max(1, int(N * frac))
            n_react = max(1, int(n_active * 0.5))
            kvals = np.zeros(n_active, dtype=np.int32)
            kvals[-n_react:] = gap
            karr = torch.from_numpy(kvals).to(dev)
            active_ids_l = torch.from_numpy(
                np.sort(np.random.RandomState(3).choice(
                    N, size=n_active, replace=False)).astype(np.int32)).to(dev)
            tv = time_kernel(lambda: ext.vos_adam_packed(
                active_ids_l, karr, grad_rows[:n_active], theta, m, v, tab,
                D, 30000, beta1, beta2, args.eps),
                max(5, args.warmup // 2), max(20, args.reps // 2))["median_ms"]
            row.append({"gap": gap, "frac": frac, "vos_ms": tv,
                        "speedup_vs_dense": t_dense / tv})
        surface.append(row)
        print(f"[mb] surface frac={frac}: " +
              " ".join(f"g{c['gap']}:{c['speedup_vs_dense']:.2f}x" for c in row),
              flush=True)
    results["sweeps"]["speedup_surface_v1_packed"] = surface

    # ---- break-even K (both variants) ----------------------------------------
    ks = [e["K"] for e in results["k_scaling"]]
    v0 = [e["vos_v0_ms"] for e in results["k_scaling"]]
    v1 = [e["vos_v1_ms"] for e in results["k_scaling"]]
    def fit_b(ks, vs, lo=3, hi=8):
        return np.polyfit(ks[lo:hi], vs[lo:hi], 1)
    b0, a0 = fit_b(ks, v0)
    b1, a1 = fit_b(ks, v1)
    dense_share = t_dense * (n_active / N)
    results["break_even"] = {
        "vos_v0_ms_at_K0": float(a0), "vos_v0_ms_per_K": float(b0),
        "vos_v1_ms_at_K0": float(a1), "vos_v1_ms_per_K": float(b1),
        "dense_ms_per_step_active_share": float(dense_share),
        "break_even_K_v0": float(dense_share / b0) if b0 > 0 else None,
        "break_even_K_v1": float(dense_share / b1) if b1 > 0 else None,
        "definition": ("K at which marginal register-replay time equals the "
                       "dense per-step time VOS avoids for the same row pop"),
    }
    print(f"[mb] break-even K: v0={results['break_even']['break_even_K_v0']:.0f} "
          f"v1={results['break_even']['break_even_K_v1']:.0f}", flush=True)

    if args.lock_clocks:
        os.system(f"nvidia-smi -i {args.gpu} -rgc")
        os.system(f"nvidia-smi -i {args.gpu} --query-gpu=clocks.sm --format=csv")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "VOS_MICROBENCH.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
