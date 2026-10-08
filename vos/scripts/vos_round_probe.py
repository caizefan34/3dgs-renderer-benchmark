"""Probe torch's exact addcmul_(g, g, value=w) rounding pattern for f32 CUDA.

Candidates:
  A: fma(round(w*g), g, v)          (my current kernel form)
  B: fma(w, round(g*g), v)
  C: round(v + round(round(w*g)*g)) (fully unfused)
  D: round(v + round(w*round(g*g))) (unfused, g*g first)
Also probes lerp and addcdiv forms for completeness.
"""
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void probe_kernel(const float* v, const float* g, const float w,
                             float* A, float* B, float* C, float* D,
                             const long long n)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float vv = v[i], gg = g[i];
    // A: fma(round(w*g), g, v)
    A[i] = __fmaf_rn(__fmul_rn(w, gg), gg, vv);
    // B: fma(w, round(g*g), v)
    B[i] = __fmaf_rn(w, __fmul_rn(gg, gg), vv);
    // C: fully unfused, (w*g) first
    float t2 = __fmul_rn(w, gg);
    float t3 = __fmul_rn(t2, gg);
    C[i] = __fadd_rn(vv, t3);
    // D: fully unfused, (g*g) first
    float u2 = __fmul_rn(gg, gg);
    float u3 = __fmul_rn(w, u2);
    D[i] = __fadd_rn(vv, u3);
}

__global__ void lerp_probe(const float* a, const float* b, const float w,
                           float* L1, float* L2, const long long n)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float aa = a[i], bb = b[i];
    L1[i] = __fmaf_rn(w, __fsub_rn(bb, aa), aa);   // fused
    L2[i] = __fadd_rn(aa, __fmul_rn(w, __fsub_rn(bb, aa)));  // unfused
}

__global__ void addcdiv_probe(const float* th, const float* m,
                              const float* d, const float ss,
                              float* R1, float* R2, const long long n)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = th[i], mm = m[i], dd = d[i];
    R1[i] = __fmaf_rn(-ss, __fdiv_rn(mm, dd), t);              // fused
    R2[i] = __fadd_rn(t, __fmul_rn(-ss, __fdiv_rn(mm, dd)));   // unfused
}

void probe(torch::Tensor v, torch::Tensor g, double w,
           torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D) {
    long long n = v.numel();
    probe_kernel<<<(unsigned)((n+255)/256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        v.data_ptr<float>(), g.data_ptr<float>(), (float)w,
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        D.data_ptr<float>(), n);
}
void lerp_probe_h(torch::Tensor a, torch::Tensor b, double w,
                  torch::Tensor L1, torch::Tensor L2) {
    long long n = a.numel();
    lerp_probe<<<(unsigned)((n+255)/256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), (float)w,
        L1.data_ptr<float>(), L2.data_ptr<float>(), n);
}
void addcdiv_probe_h(torch::Tensor th, torch::Tensor m, torch::Tensor d,
                     double ss, torch::Tensor R1, torch::Tensor R2) {
    long long n = th.numel();
    addcdiv_probe<<<(unsigned)((n+255)/256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        th.data_ptr<float>(), m.data_ptr<float>(), d.data_ptr<float>(), (float)ss,
        R1.data_ptr<float>(), R2.data_ptr<float>(), n);
}
"""
CPP = """
void probe(torch::Tensor v, torch::Tensor g, double w, torch::Tensor A,
           torch::Tensor B, torch::Tensor C, torch::Tensor D);
void lerp_probe_h(torch::Tensor a, torch::Tensor b, double w, torch::Tensor L1,
                  torch::Tensor L2);
void addcdiv_probe_h(torch::Tensor th, torch::Tensor m, torch::Tensor d,
                     double ss, torch::Tensor R1, torch::Tensor R2);
"""

ext = load_inline(name="vos_round_probe", cpp_sources=[CPP], cuda_sources=[SRC],
                  functions=["probe", "lerp_probe_h", "addcdiv_probe_h"],
                  extra_cuda_cflags=["-O3"], verbose=False)

torch.manual_seed(0)
dev = "cuda"
N = 1 << 20
v = torch.rand(N, device=dev) * 1e-5 + 1e-6
g = torch.randn(N, device=dev) * 1e-2
w = 1 - 0.999  # python float (f64), exactly like the oracle passes it

# torch reference: v.mul_(b2).addcmul_(g, g, value=w)
b2 = 0.999
v_ref = v.clone()
v_ref.mul_(b2)
v_ref.addcmul_(g, g, value=w)

A, B, C, D = (torch.empty(N, device=dev) for _ in range(4))
ext.probe(v, g, w, A, B, C, D)
torch.cuda.synchronize()
# NOTE: probe forms must include the prior mul_(b2): apply round(v*b2) first
# in each candidate. Redo with the b2 product folded in.
SRC2 = None
vb = (v * b2)  # rounded like mul_
vb = vb.clone()  # materialize the rounding
ext.probe(vb, g, w, A, B, C, D)
torch.cuda.synchronize()
for name, cand in zip("ABCD", (A, B, C, D)):
    print(f"addcmul form {name}: bitwise={bool(torch.equal(cand, v_ref))} "
          f"maxdiff={(cand - v_ref).abs().max().item():.3e}")

# lerp
a = torch.rand(N, device=dev)
bb = torch.rand(N, device=dev)
w1 = 1 - 0.9
lerp_ref = a.clone()
lerp_ref.lerp_(bb, w1)
L1, L2 = torch.empty(N, device=dev), torch.empty(N, device=dev)
ext.lerp_probe_h(a, bb, w1, L1, L2)
torch.cuda.synchronize()
print(f"lerp fused:   bitwise={bool(torch.equal(L1, lerp_ref))}")
print(f"lerp unfused: bitwise={bool(torch.equal(L2, lerp_ref))}")

# addcdiv
th = torch.rand(N, device=dev)
m = torch.randn(N, device=dev)
d = torch.rand(N, device=dev) + 0.5
ss = 0.0025 / (1 - 0.9 ** 25000)  # realistic magnitude
th_ref = th.clone()
th_ref.addcdiv_(m, d, value=-ss)
R1, R2 = torch.empty(N, device=dev), torch.empty(N, device=dev)
ext.addcdiv_probe_h(th, m, d, ss, R1, R2)
torch.cuda.synchronize()
print(f"addcdiv fused:   bitwise={bool(torch.equal(R1, th_ref))}")
print(f"addcdiv unfused: bitwise={bool(torch.equal(R2, th_ref))}")

# also check: does (v.sqrt()/bs).add_(eps) match __fdiv/__fsqrt/__fadd chain?
print("denom: torch two-op vs kernel chain assumed identical (div+add, both IEEE)")
