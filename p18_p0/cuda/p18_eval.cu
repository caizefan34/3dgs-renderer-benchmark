// P18-P0 Phase 3 — evaluator microbenchmark (float domain), per (gaussian, pixel-x).
//
// Workload: n_items rows, each (a, b, c, x). One lane = one (gaussian, pixel-x)
// = one baseline inner-loop G(x) evaluation.
//   DIRECT   : G(x) = exp(q(x)) per lane
//   QSCAN(SUB): lanes [SUB*k .. SUB*k+SUB-1] are assumed to be SUB consecutive
//               x-values of ONE gaussian (contiguous x). Leader computes
//               G0=exp(q(gs)), r0=exp(d0), k=exp(-dd); lane j: r_j=r0*k^j
//               (parallel pow); exclusive multiplicative prefix scan;
//               G_j=G0*prefix_j. gamma guard on the cull branch.
// The driver lays out the workload so consecutive lanes are consecutive x of
// the same gaussian (a row of 32 x-pixels = 1 gaussian), matching the real
// forward inner loop. Both forward exp2 and backward exp domains.
// No serial lane-0 walk.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr float GAMMA = 2.0e-6f;

struct EvParams
{
    const float *__restrict__ a, *b, *c, *x; // [n_items]
    float *__restrict__ out_g;                // [n_items]
    int n_items;
};

template <bool EXP2>
__device__ __forceinline__ float direct_G(float a, float b, float c, float x)
{
    const float q = a * x * x + b * x + c;
    if constexpr (EXP2) return exp2f(-q);
    else                return expf(-q);
}

template <int SUB>
__device__ __forceinline__ uint32_t sub_mask_of(int lane)
{
    const int gs = lane & ~(SUB - 1);
    uint32_t m = 0;
    #pragma unroll
    for (int q = 0; q < SUB; ++q) m |= (1u << (gs + q));
    return m;
}

template <int SUB>
__device__ __forceinline__ float excl_mul_scan(float in, int lane, uint32_t sub_mask)
{
    float prefix = 1.0f;
    #pragma unroll
    for (int off = 1; off < SUB; off <<= 1)
    {
        const int other = lane ^ off;
        const float m = __shfl_xor_sync(sub_mask, in, off);
        if (other < lane) in = m;
        else              prefix *= m;
    }
    return prefix;
}

__global__ void eval_direct_exp2(EvParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n_items) return;
    p.out_g[i] = direct_G<true>(p.a[i], p.b[i], p.c[i], p.x[i]);
}
__global__ void eval_direct_expf(EvParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n_items) return;
    p.out_g[i] = direct_G<false>(p.a[i], p.b[i], p.c[i], p.x[i]);
}

template <int SUB, bool EXP2>
__global__ void eval_qscan(EvParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n_items) return;
    const int lane = threadIdx.x & 31;
    const float av = p.a[i], bv = p.b[i], cv = p.c[i];
    const float x = p.x[i];
    const int gs = lane & ~(SUB - 1);
    const uint32_t sub_mask = sub_mask_of<SUB>(lane);
    const int pos = lane - gs;
    const float x0 = (float)gs; // group's first x (consecutive lanes => x0..x0+SUB-1)
    const float d0 = bv + 2.0f * av * x0;
    const float dd = 2.0f * av;
    const float G0 = direct_G<EXP2>(av, bv, cv, x0);
    const float r0 = (EXP2 ? exp2f(d0) : expf(d0));
    const float kpow = (EXP2 ? exp2f(-dd * (float)pos) : expf(-dd * (float)pos));
    const float r_pos = r0 * kpow;
    const float excl = excl_mul_scan<SUB>(r_pos, lane, sub_mask);
    float gq = G0 * excl;
    const float gd = direct_G<EXP2>(av, bv, cv, x);
    const float branch_d = (gd < (1.0f / 255.0f)) ? 1.0f : 0.0f;
    const float branch_q = (gq < (1.0f / 255.0f)) ? 1.0f : 0.0f;
    if (fabsf(gq - gd) > GAMMA || branch_d != branch_q) gq = gd;
    p.out_g[i] = gq;
}

} // namespace

void eval_direct(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor x,
                 torch::Tensor out, bool exp2, int64_t iters)
{
    const int n = a.size(0);
    EvParams p{a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), x.data_ptr<float>(), out.data_ptr<float>(), n};
    auto stream = at::cuda::getCurrentCUDAStream();
    int threads = 256, blocks = (n + threads - 1) / threads;
    for (int it = 0; it < iters; ++it)
        if (exp2) eval_direct_exp2<<<blocks, threads, 0, stream>>>(p);
        else      eval_direct_expf<<<blocks, threads, 0, stream>>>(p);
}

void eval_qscan(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor x,
                torch::Tensor out, int64_t sub, bool exp2, int64_t iters)
{
    const int n = a.size(0);
    EvParams p{a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), x.data_ptr<float>(), out.data_ptr<float>(), n};
    auto stream = at::cuda::getCurrentCUDAStream();
    int threads = 256, blocks = (n + threads - 1) / threads;
    for (int it = 0; it < iters; ++it)
    {
        if (sub == 4)      { if (exp2) eval_qscan<4,  true ><<<blocks, threads, 0, stream>>>(p); else eval_qscan<4,  false><<<blocks, threads, 0, stream>>>(p); }
        else if (sub == 8) { if (exp2) eval_qscan<8,  true ><<<blocks, threads, 0, stream>>>(p); else eval_qscan<8,  false><<<blocks, threads, 0, stream>>>(p); }
        else               { if (exp2) eval_qscan<16, true ><<<blocks, threads, 0, stream>>>(p); else eval_qscan<16, false><<<blocks, threads, 0, stream>>>(p); }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("eval_direct", &eval_direct);
    m.def("eval_qscan", &eval_qscan);
}
