"""Fused cull-masked Adam step for HiGS training.

Round-60 lever: the stock fused Adam (torch.optim.Adam(fused=True)) applies
its multi-tensor update to ALL N Gaussians every step, even though the HiGS
union-visibility cull renders only a subset (round-59 profile: fused Adam is
17.5% of the 39 ms/step at the 720p r0.35 operating point; culling ratios are
52% garden / 69% bicycle / 17% train).  This module provides a small fused
CUDA kernel that runs the exact torch fused-Adam math (m = b1*m+(1-b1)*g,
v = b2*v+(1-b2)*g^2, p -= lr/bc1 * m / (sqrt(v)/sqrt(bc2)+eps)) but only on
rows where a per-Gaussian mask is True, so memory traffic scales with the
visible set instead of N.

The mask is the train-forward union-visibility mask (same one the renderer
used); culled Gaussians are frozen completely (params and moments unchanged),
which changes dynamics vs stock Adam only for out-of-view rows (stock Adam
keeps decaying their moments with zero grads).  Quality impact is measured
empirically in the round-60 sweep.

Float math mirrors torch's fused Adam (fused_adam_utils.cuh): bias
corrections computed as ``1 - pow(beta, step)`` in double and the bias-
correction-2 sqrt in double, step_size = float(lr/bc1) with a double
division, denom = sqrt(v)/sqrt(bc2)+eps in double rounded to float, and m/v
updates in double rounded to float.  Measured equivalence (probe): params
within 1-2 float32 ulps of torch's fused step (max rel diff ~2e-7), state
within ~1 ulp on a small fraction of elements (torch's own nvcc build rounds
a few m/v updates differently), frozen rows bit-identical.

Memory layout: one thread per flat element with coalesced access (the mask
check is warp-uniform since a row spans D consecutive elements), matching
torch's flat elementwise traversal instead of a row-per-thread layout.
"""
import os

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void masked_adam_kernel(
    float* __restrict__ p, float* __restrict__ m, float* __restrict__ v,
    const float* __restrict__ g, const bool* __restrict__ mask,
    const long numel, const long D,
    const double beta1, const double beta2, const double eps,
    const float bias_correction2_sqrt,
    const float step_size) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  // Row of this element: D consecutive elements share one mask bit, so the
  // branch is warp-uniform (threads of a row-spanning warp hit the same
  // mask value) and the flat access pattern stays fully coalesced.
  const long row = i / D;
  if (!mask[row]) return;
  const float gv = g[i];
  const float m_new = (float)(beta1 * (double)m[i] + (1.0 - beta1) * (double)gv);
  const float v_new = (float)(beta2 * (double)v[i] + (1.0 - beta2) * (double)gv * (double)gv);
  m[i] = m_new;
  v[i] = v_new;
  // torch: denom = float(sqrt(v)/bc2_sqrt + eps) with the division done in
  // float (bc2_sqrt arrives as opmath_t=float) and the +eps in double.
  const float denom = (float)((sqrtf(v_new) / bias_correction2_sqrt) + eps);
  p[i] -= (step_size * m_new) / denom;
}

void masked_adam_step(
    torch::Tensor p, torch::Tensor m, torch::Tensor v, torch::Tensor g,
    torch::Tensor mask,
    double beta1, double beta2, double eps,
    float bias_correction2_sqrt, float step_size) {
  TORCH_CHECK(p.is_cuda() && m.is_cuda() && v.is_cuda() && g.is_cuda(),
              "masked_adam_step requires CUDA tensors");
  TORCH_CHECK(p.dim() == 2, "params must be [N, D] (reshape at call site)");
  TORCH_CHECK(mask.is_cuda() && mask.scalar_type() == torch::kBool,
              "mask must be a CUDA bool tensor");
  const long N = p.size(0);
  const long D = p.size(1);
  TORCH_CHECK(m.size(0) == N && v.size(0) == N && g.size(0) == N,
              "m/v/g must have N rows");
  TORCH_CHECK(mask.numel() == N, "mask must have N elements");
  const long numel = p.numel();
  const int threads = 256;
  const long blocks = (numel + threads - 1) / threads;
  masked_adam_kernel<<<blocks, threads>>>(
      p.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(),
      g.data_ptr<float>(), mask.data_ptr<bool>(),
      numel, D, beta1, beta2, eps,
      bias_correction2_sqrt, step_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP_SRC = r"""
void masked_adam_step(
    torch::Tensor p, torch::Tensor m, torch::Tensor v, torch::Tensor g,
    torch::Tensor mask,
    double beta1, double beta2, double eps,
    float bias_correction2_sqrt, float step_size);
"""

_ext = None
_ext_error = None


def _load_ext():
    global _ext, _ext_error
    if _ext is not None:
        return _ext
    if _ext_error is not None:
        raise _ext_error
    if not torch.cuda.is_available():
        _ext_error = RuntimeError("masked_adam requires CUDA")
        raise _ext_error
    old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if old_arch is None:
        # Default to the actual device so the JIT'd kernel runs on
        # anything (A100 sm_80, RTX 50-series sm_120, ...) instead of
        # hard-coding sm_80.
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
    # CUDA 13.x bundles CCCL 3.0 whose headers hard-error when MSVC is used
    # with the traditional preprocessor; gsplat's own build.py passes this
    # flag on Windows for the same reason.
    extra_cuda_cflags = []
    if os.name == "nt":
        # nvcc treats a bare /Zc:preprocessor as an input file; it must be
        # forwarded to the MSVC host compiler via -Xcompiler.
        extra_cuda_cflags += ["-Xcompiler", "/Zc:preprocessor"]
    try:
        _ext = load_inline(
            name="higs_masked_adam_cuda",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["masked_adam_step"],
            with_cuda=True,
            verbose=False,
            extra_cuda_cflags=extra_cuda_cflags,
        )
    except Exception as e:  # pragma: no cover - toolchain-dependent
        _ext_error = e
        raise
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch
    return _ext


def masked_adam_step(opt, mask):
    """Apply torch-Adam-equivalent updates only to rows where ``mask`` is True.

    ``mask``: CUDA bool [N] (True = update this Gaussian).  State tensors
    (exp_avg / exp_avg_sq) and per-param step counters are created lazily and
    stored in the optimizer state so densify/prune topology sync keeps working.
    """
    ext = _load_ext()
    for group in opt.param_groups:
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = opt.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            step = state["step"]
            # Bias corrections hoisted to the host: the value is identical
            # for every element/block (torch computes it per block from the
            # same step tensor), and it removes one pow() per element.  Torch
            # casts both corrections to opmath_t=float before the kernel math,
            # so we mirror that cast here for bit-level agreement.
            bc1 = float(1.0 - pow(beta1, float(step)))
            bc2_sqrt = float((1.0 - pow(beta2, float(step))) ** 0.5)
            step_size = float(lr / bc1)
            n = p.shape[0]
            d = p.numel() // n
            p2 = p if p.dim() == 2 else p.view(n, d)
            m2 = state["exp_avg"] if state["exp_avg"].dim() == 2 else state["exp_avg"].view(n, d)
            v2 = state["exp_avg_sq"] if state["exp_avg_sq"].dim() == 2 else state["exp_avg_sq"].view(n, d)
            g2 = p.grad.detach()
            if g2.dim() != 2:
                g2 = g2.reshape(n, d)
            if not g2.is_contiguous():
                g2 = g2.contiguous()
            ext.masked_adam_step(
                p2, m2, v2, g2, mask,
                float(beta1), float(beta2), float(eps),
                bc2_sqrt, step_size,
            )
