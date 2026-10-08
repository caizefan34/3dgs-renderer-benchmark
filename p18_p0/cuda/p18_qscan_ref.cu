// P18-P0 QSCAN isolated reference-validation kernel (float domain).
//
// Validity: DIRECT vs QSCAN4/8/16 reconstruction of G(x)=exp(q(x)) over a
// contiguous run of pixels (x = 0..31, one row, one gaussian), for BOTH the
// forward exp2/half semantics and the backward exp/float semantics.
//
// Math (protocol section 3):
//  q(x) = a x^2 + b x + c  (1-D quadratic; dy,u1,u0_base constant per row)
//  d0 = q(x0+1)-q(x0) = b + 2a x0;  dd = 2a (constant second difference)
//  G0 = exp(q(x0));  r0 = exp(d0);  k = exp(-dd)   [same intrinsic as baseline]
//  r_j = r0 * k^j   (j = lane offset in group; parallel pow, no serial walk)
//  prefix_j = prod_{t<j} r_t   (exclusive multiplicative scan, Hillis-Steele)
//  G_j = G0 * prefix_j
//  gamma guard: if |alpha_q - alpha_direct| > gamma (2e-6) OR the cull branch
//  decision (alpha < 1/255) differs, the lane's QSCAN value is replaced by the
//  DIRECT value (the residual branch-mismatch is still counted for reporting).
//
// No approximate exp beyond baseline's own intrinsic (exp2f/expf, fast-math).
// No lane-0 serial walk: r_j via exp2(pow); prefix via warp shuffle scan.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr float GAMMA = 2.0e-6f;

template <bool EXP2>
__device__ __forceinline__ float direct_G(float a, float b, float c, float x)
{
    const float q = a * x * x + b * x + c;
    if constexpr (EXP2) return exp2f(-q);
    else                return expf(-q);
}

// exclusive multiplicative prefix scan of `in` over the SUB-lane subgroup
// anchored at a multiple of SUB. Returns prod_{t<j} r_t (Hillis-Steele).
template <int SUB>
__device__ __forceinline__ float excl_mul_scan(float in, int lane, uint32_t sub_mask)
{
    float prefix = 1.0f;
    #pragma unroll
    for (int off = 1; off < SUB; off <<= 1)
    {
        const int other = lane ^ off;
        const float m = __shfl_xor_sync(sub_mask, in, off);
        if (other < lane)
            in = m;          // lane > other: pull the lower half's product
        else
            prefix *= m;     // lane < other: multiply in the upper half's product
    }
    return prefix;
}

template <int SUB, bool EXP2>
__device__ __forceinline__ float qscan_G(float a, float b, float c, float x,
                                         int lane, int gs, uint32_t sub_mask)
{
    const int pos = lane - gs;
    if (pos < 0 || pos >= SUB) return 0.0f;
    const float x0 = (float)gs;
    const float d0 = b + 2.0f * a * x0;   // q(x0+1)-q(x0)
    const float dd = 2.0f * a;
    const float G0 = direct_G<EXP2>(a, b, c, x0);
    const float r0 = (EXP2 ? exp2f(d0) : expf(d0));
    // r_j = r0 * k^j, k = exp(-dd). Compute k^pos in parallel (no serial walk):
    const float kpow = (EXP2 ? exp2f(-dd * (float)pos) : expf(-dd * (float)pos));
    const float r_pos = r0 * kpow;
    const float excl = excl_mul_scan<SUB>(r_pos, lane, sub_mask);
    return G0 * excl;
}

struct Params
{
    const float *__restrict__ a, *b, *c;
    const int *__restrict__ variant;
    float *__restrict__ out_direct, *out_qscan;
    int *__restrict__ out_mm, *out_nan;
    int n_rows;
};

template <bool EXP2>
__global__ void qscan_row_kernel(Params p)
{
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= p.n_rows) return;
    const int lane = threadIdx.x & 31;
    const float av = p.a[row], bv = p.b[row], cv = p.c[row];
    const float x = (float)lane;
    const float gd = direct_G<EXP2>(av, bv, cv, x);
    const int v = p.variant[row];
    int gs, SUB;
    if (v == 0) { gs = lane & ~3;  SUB = 4; }
    else if (v == 1) { gs = lane & ~7; SUB = 8; }
    else { gs = lane & ~15; SUB = 16; }
    uint32_t sub_mask = 0;
    for (int q = 0; q < SUB; ++q) sub_mask |= (1u << (gs + q));
    float gq;
    if (SUB == 4)      gq = qscan_G<4,  EXP2>(av, bv, cv, x, lane, gs, sub_mask);
    else if (SUB == 8) gq = qscan_G<8,  EXP2>(av, bv, cv, x, lane, gs, sub_mask);
    else               gq = qscan_G<16, EXP2>(av, bv, cv, x, lane, gs, sub_mask);
    const float branch_d = (gd < (1.0f / 255.0f)) ? 1.0f : 0.0f;
    const float branch_q = (gq < (1.0f / 255.0f)) ? 1.0f : 0.0f;
    bool nanq = !isfinite(gq);
    if (nanq || fabsf(gq - gd) > GAMMA || branch_d != branch_q) gq = gd;
    p.out_direct[(int64_t)row * 32 + lane] = gd;
    p.out_qscan[(int64_t)row * 32 + lane]  = gq;
    if (branch_d != branch_q) atomicAdd(&p.out_mm[row], 1);
    if (nanq) atomicAdd(&p.out_nan[row], 1);
}

} // namespace

std::vector<torch::Tensor> qscan_ref(
    torch::Tensor a, torch::Tensor b, torch::Tensor c,
    torch::Tensor variant, bool exp2, int64_t iters)
{
    const int n_rows = a.size(0);
    auto opts = a.options();
    auto out_direct = torch::zeros({n_rows, 32}, opts);
    auto out_qscan  = torch::zeros({n_rows, 32}, opts);
    auto out_mm     = torch::zeros({n_rows}, opts.dtype(torch::kInt));
    auto out_nan    = torch::zeros({n_rows}, opts.dtype(torch::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    Params p{a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
             variant.data_ptr<int>(), out_direct.data_ptr<float>(),
             out_qscan.data_ptr<float>(), out_mm.data_ptr<int>(), out_nan.data_ptr<int>(),
             n_rows};
    int threads = 256, blocks = (n_rows + (threads >> 5) - 1) / (threads >> 5);
    for (int it = 0; it < iters; ++it)
    {
        if (exp2) qscan_row_kernel<true ><<<blocks, threads, 0, stream>>>(p);
        else      qscan_row_kernel<false><<<blocks, threads, 0, stream>>>(p);
    }
    return {out_direct, out_qscan, out_mm, out_nan};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("qscan_ref", &qscan_ref); }
