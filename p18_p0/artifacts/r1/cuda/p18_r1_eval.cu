// P18-P0-R1 corrected evaluator microbenchmark (float domain), spec items (2)(3)(4).
//
// Workload: n_items lanes, each (a, b, c, x) = one baseline inner-loop G(x)
// evaluation over ONE real pixel x of a real 16-px trace row.
//
//   DIRECT      : G(x) = exp(-q(x)) per lane, q = a x^2 + b x + c. ONE exp2f/expf
//                 per pixel (baseline per-pixel exp preserved). Same direct path
//                 as P18-P0 (gate continuity).
//   QSCAN(SUB)  : lanes [gs .. gs+SUB-1] hold SUB CONSECUTIVE real pixels of one
//                 gaussian row. ALLOWED exps (leader lane only, pos==0):
//                     G0 = exp(-q0), r0 = exp(-d0), k = exp(-dd)
//                 with q0 = q(xs), d0 = q(xs+1)-q(xs) = a(2xs+1)+b, dd = 2a,
//                 xs = subgroup's first real pixel. 3 exps per subgroup.
//                 k^j by PURE MULTIPLICATION (bits of j: k, k^2=k*k, k^4, k^8);
//                 u_j = r0*k^j; exclusive multiplicative prefix scan (Hillis-
//                 Steele, shfl_up); G_j = G0 * prefix_j.
//                 FORBIDDEN and absent: exp(k*j), exp2(dd*pos), any position-
//                 dependent exponential; per-lane direct recompute; guard fallback.
//
// The driver lays the workload out record-major: item = 16*record + t, so any
// SUB in {4,8,16} divides the 16-px record and subgroups never straddle records.
// n_items MUST be a multiple of the block size (256) so every lane is active in
// the shfl_sync collectives (host asserts; workload is trimmed accordingly).

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

struct EvParams
{
    const float *__restrict__ a, *b, *c, *x; // [n_items]
    float *__restrict__ out_g;               // [n_items]
    int n_items;
};

#ifdef P18_EXP_COUNTER
__device__ unsigned long long g_p18_exp_ctr = 0ull;
__device__ __forceinline__ float p18_exp2f(float v)
{
    atomicAdd(&g_p18_exp_ctr, 1ull);
    return exp2f(v);
}
__device__ __forceinline__ float p18_expf(float v)
{
    atomicAdd(&g_p18_exp_ctr, 1ull);
    return expf(v);
}
#else
__device__ __forceinline__ float p18_exp2f(float v) { return exp2f(v); }
__device__ __forceinline__ float p18_expf(float v) { return expf(v); }
#endif

template <bool EXP2>
__device__ __forceinline__ float exp_neg(float v)
{
    return EXP2 ? p18_exp2f(v) : p18_expf(v);
}

template <bool EXP2>
__device__ __forceinline__ float direct_G(float a, float b, float c, float x)
{
    const float q = a * x * x + b * x + c;
    return exp_neg<EXP2>(-q);
}

__global__ void eval_direct_exp2(EvParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    p.out_g[i] = direct_G<true>(p.a[i], p.b[i], p.c[i], p.x[i]);
}
__global__ void eval_direct_expf(EvParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    p.out_g[i] = direct_G<false>(p.a[i], p.b[i], p.c[i], p.x[i]);
}

// Diagnostic control (NOT timed by the gate): exp-FREE direct path with identical
// loads/store. Bounds the proxy: no exp-elimination technique can beat
// S_eval_noexp = T_direct / T_noexp in this microbenchmark (the SASS audit's
// check F proves this kernel is genuinely exp-free: 0 MUFU, 0 dynamic exp calls).
__global__ void eval_noexp_kernel(EvParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float q = fmaf(fmaf(p.a[i], p.x[i], p.b[i]), p.x[i], p.c[i]);
    p.out_g[i] = q;
}

// QSCAN: 3 leader exps + pure-multiplication k^j + multiplicative scan.
template <int SUB, bool EXP2>
__global__ void eval_qscan(EvParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int gs = lane & ~(SUB - 1);
    const int pos = lane - gs;

    const float av = p.a[i], bv = p.b[i], cv = p.c[i];
    const float xs = __shfl_sync(0xFFFFFFFFu, p.x[i], gs); // subgroup's first REAL pixel

    float G0 = 1.0f, r0 = 1.0f, k = 1.0f;
    if (pos == 0)
    {
        const float q0 = fmaf(fmaf(av, xs, bv), xs, cv);
        const float d0 = fmaf(2.0f * av, xs, av + bv); // a(2xs+1)+b
        const float dd = 2.0f * av;
        G0 = exp_neg<EXP2>(-q0);
        r0 = exp_neg<EXP2>(-d0);
        k  = exp_neg<EXP2>(-dd);
    }
    G0 = __shfl_sync(0xFFFFFFFFu, G0, gs);
    r0 = __shfl_sync(0xFFFFFFFFu, r0, gs);
    k  = __shfl_sync(0xFFFFFFFFu, k, gs);

    // k^pos by pure multiplication on the bits of pos
    float kj = 1.0f, kp = k;
#pragma unroll
    for (int bit = 0; (1 << bit) < SUB; ++bit)
    {
        if (pos & (1 << bit))
        {
            kj *= kp;
        }
        kp *= kp;
    }
    const float u = r0 * kj;

    // exclusive multiplicative prefix scan within the subgroup (Hillis-Steele)
    float s = u;
#pragma unroll
    for (int off = 1; off < SUB; off <<= 1)
    {
        const float t = __shfl_up_sync(0xFFFFFFFFu, s, off);
        if (pos >= off)
        {
            s *= t;
        }
    }
    float pref = __shfl_up_sync(0xFFFFFFFFu, s, 1);
    if (pos == 0)
    {
        pref = 1.0f;
    }
    p.out_g[i] = G0 * pref;
}

// FORBIDDEN control kernel: exp2f(k*pos) in the PER-LANE path. Never timed; it
// exists so the SASS auditor can positively detect the pattern it must reject.
template <int SUB, bool EXP2>
__global__ void eval_qscan_forbidden(EvParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int gs = lane & ~(SUB - 1);
    const int pos = lane - gs;
    const float av = p.a[i], bv = p.b[i], cv = p.c[i];
    const float xs = __shfl_sync(0xFFFFFFFFu, p.x[i], gs);
    float G0 = 1.0f, r0 = 1.0f, k = 1.0f;
    if (pos == 0)
    {
        const float q0 = fmaf(fmaf(av, xs, bv), xs, cv);
        const float d0 = fmaf(2.0f * av, xs, av + bv);
        G0 = exp_neg<EXP2>(-q0);
        r0 = exp_neg<EXP2>(-d0);
        k  = exp_neg<EXP2>(-2.0f * av);
    }
    G0 = __shfl_sync(0xFFFFFFFFu, G0, gs);
    r0 = __shfl_sync(0xFFFFFFFFu, r0, gs);
    k  = __shfl_sync(0xFFFFFFFFu, k, gs);
    const float u = r0 * exp_neg<EXP2>(-k * static_cast<float>(pos)); // FORBIDDEN
    float s = u;
#pragma unroll
    for (int off = 1; off < SUB; off <<= 1)
    {
        const float t = __shfl_up_sync(0xFFFFFFFFu, s, off);
        if (pos >= off)
        {
            s *= t;
        }
    }
    float pref = __shfl_up_sync(0xFFFFFFFFu, s, 1);
    if (pos == 0)
    {
        pref = 1.0f;
    }
    p.out_g[i] = G0 * pref;
}

} // namespace

void eval_direct(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor x,
                 torch::Tensor out, bool exp2, int64_t iters)
{
    const int n = static_cast<int>(a.size(0));
    TORCH_CHECK(n > 0 && n % 256 == 0, "n_items must be a positive multiple of 256");
    EvParams p{a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
               x.data_ptr<float>(), out.data_ptr<float>(), n};
    auto stream = at::cuda::getCurrentCUDAStream();
    int threads = 256, blocks = n / threads;
    for (int it = 0; it < iters; ++it)
    {
        if (exp2) eval_direct_exp2<<<blocks, threads, 0, stream>>>(p);
        else      eval_direct_expf<<<blocks, threads, 0, stream>>>(p);
    }
}

void eval_noexp(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor x,
                torch::Tensor out, int64_t iters)
{
    const int n = static_cast<int>(a.size(0));
    TORCH_CHECK(n > 0 && n % 256 == 0, "n_items must be a positive multiple of 256");
    EvParams p{a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
               x.data_ptr<float>(), out.data_ptr<float>(), n};
    auto stream = at::cuda::getCurrentCUDAStream();
    int threads = 256, blocks = n / threads;
    for (int it = 0; it < iters; ++it)
        eval_noexp_kernel<<<blocks, threads, 0, stream>>>(p);
}

void eval_qscan(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor x,
                torch::Tensor out, int64_t sub, bool exp2, int64_t iters)
{
    const int n = static_cast<int>(a.size(0));
    TORCH_CHECK(n > 0 && n % 256 == 0, "n_items must be a positive multiple of 256");
    EvParams p{a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
               x.data_ptr<float>(), out.data_ptr<float>(), n};
    auto stream = at::cuda::getCurrentCUDAStream();
    int threads = 256, blocks = n / threads;
    for (int it = 0; it < iters; ++it)
    {
        if (sub == 4)      { if (exp2) eval_qscan<4,  true ><<<blocks, threads, 0, stream>>>(p); else eval_qscan<4,  false><<<blocks, threads, 0, stream>>>(p); }
        else if (sub == 8) { if (exp2) eval_qscan<8,  true ><<<blocks, threads, 0, stream>>>(p); else eval_qscan<8,  false><<<blocks, threads, 0, stream>>>(p); }
        else               { if (exp2) eval_qscan<16, true ><<<blocks, threads, 0, stream>>>(p); else eval_qscan<16, false><<<blocks, threads, 0, stream>>>(p); }
    }
}

// Host launcher for the FORBIDDEN control kernel. NEVER called by the timing
// driver; it exists only to force the positive-control instantiations (per-lane
// position-dependent exponential, i.e. the old P18-P0 defect) into the module
// so the SASS auditor can prove its detection is sensitive.
void eval_qscan_forbidden(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor x,
                          torch::Tensor out, int64_t sub, bool exp2, int64_t iters)
{
    const int n = static_cast<int>(a.size(0));
    TORCH_CHECK(n > 0 && n % 256 == 0, "n_items must be a positive multiple of 256");
    EvParams p{a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
               x.data_ptr<float>(), out.data_ptr<float>(), n};
    auto stream = at::cuda::getCurrentCUDAStream();
    int threads = 256, blocks = n / threads;
    for (int it = 0; it < iters; ++it)
    {
        if (sub == 4)      { if (exp2) eval_qscan_forbidden<4,  true ><<<blocks, threads, 0, stream>>>(p); else eval_qscan_forbidden<4,  false><<<blocks, threads, 0, stream>>>(p); }
        else if (sub == 8) { if (exp2) eval_qscan_forbidden<8,  true ><<<blocks, threads, 0, stream>>>(p); else eval_qscan_forbidden<8,  false><<<blocks, threads, 0, stream>>>(p); }
        else               { if (exp2) eval_qscan_forbidden<16, true ><<<blocks, threads, 0, stream>>>(p); else eval_qscan_forbidden<16, false><<<blocks, threads, 0, stream>>>(p); }
    }
}

int64_t exp_ctr()
{
#ifdef P18_EXP_COUNTER
    unsigned long long v = 0ull;
    cudaMemcpyFromSymbol(&v, g_p18_exp_ctr, sizeof(v));
    return static_cast<int64_t>(v);
#else
    return -1;
#endif
}

void exp_ctr_reset()
{
#ifdef P18_EXP_COUNTER
    unsigned long long z = 0ull;
    cudaMemcpyToSymbol(g_p18_exp_ctr, &z, sizeof(z));
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("eval_direct", &eval_direct);
    m.def("eval_noexp", &eval_noexp);
    m.def("eval_qscan", &eval_qscan);
    m.def("eval_qscan_forbidden", &eval_qscan_forbidden);
    m.def("exp_ctr", &exp_ctr);
    m.def("exp_ctr_reset", &exp_ctr_reset);
}
