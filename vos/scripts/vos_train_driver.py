"""VOS Phase-8 — SH-only virtual-optimizer-state trainer driver (post-PROMOTE).

ZERO source diff on the frozen trainer: imports publication_trainer_p13x.py as a
module (same proven pattern as vos_trace_driver.py), then patches:

  1. gsplat.experimental.rasterize_gaussian_higs_dynamic
       -> keeps the last grad-enabled iteration's visibility bitmap on GPU
          (no copy, no packbits, no RNG; evaluate() runs under no_grad so it
          never overwrites the training bitmap before step()).
  2. GaussianModel.training_setup (reference_v1)
       -> wraps the constructed torch.optim.Adam in a VOSAdamProxy.

VOSAdamProxy.step(t):
  t <= vos_start            -> stock inner.step() (bitwise C0).
  t >  vos_start (v1 scope: 15001..30000, no topology events there):
      a. pop the shs param group; inner.step() updates xyz/opacity/scaling/
         rotation via the same foreach path (per-element math independent of
         list composition); restore the group.
      b. VOS shs step on the ACTIVE rows only (bitmap from this iteration):
         gather m/v/theta, replay K = (t-1) - tau zero-grad Adam transitions
         (exact torch op order, per-step scalars from a packed float4 table:
         {step_size_s, bc2_sqrt_s} computed in host float64 then cast to f32,
         exactly like torch's host-scalar path), apply the real step-t update,
         scatter, tau[active] = t.
      c. fences (materialize-all, zero-grad replay only): at eval iterations
         (before evaluate() reads params), at the final iteration (before the
         ckpt save), and at the end of any step whose NEXT iteration runs a
         densification event (forced-early tests only; there are no events in
         the 15K->30K region).

Exactness model (Phase-5 oracle: 7/7 bitwise vs A_s):
  kernel replicates the single-tensor torch Adam op sequence
    m  = m + w1*(g - m)                       lerp_
    v  = v*b2 + w2*(g*g)                      mul_, addcmul_
    d  = (sqrt(v) / bc2_sqrt_s) + eps
    th = th + (-step_size_s) * (m / d)        addcdiv_
  with every scalar rounded exactly as torch's host-scalar path rounds it.
  A startup gate builds fmad=true then fmad=false variants and keeps the first
  that is BITWISE vs a torch-op reference; otherwise falls back to
  --vos-mode torchop (gather + vectorized torch ops + scatter: the exact
  oracle-B path, slower).

The trainer's own phase_ms["optimizer"] (pev[4]..pev[5]) wraps the whole proxy
step, so timing.json already holds the honest A/B optimizer-stage metric; this
driver additionally splits inner vs VOS-stage time via CUDA event pairs into
vos_stats.json.

Usage (EPIC-05):
  python vos_train_driver.py --scene room --iterations 30000 --gpu 0 \
      --outdir /root/vos_runs/vos_room_30k --vos-start 15000
  python vos_train_driver.py --kernel-check --outdir /tmp/vos_kcheck
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ void adam_transition(
    float& mm, float& vv, float& th, const float g,
    const float step_size, const float bc2_sqrt,
    const float w1, const float w2, const float b2, const float epsf)
{
    // torch op forms, empirically pinned by the bitwise gate + rounding probe:
    //   lerp_(grad, w1):            fma(w1, round(g - m), m)
    //   mul_(b2):                   round(v*b2)
    //   addcmul_(g, g, w2):         fma(w2, round(g*g), round(v*b2))
    //   (sqrt(v)/bc2).add_(eps):    round(sqrt(v) * f32(1/f32(bc2)) + eps)
    //     — torch's CUDA div-by-CPU-scalar is MULTIPLY-BY-RECIPROCAL
    //       (div_true_kernel_cuda's is_cpu_scalar fast path), NOT true
    //       division; at bc2 != 1.0 the two differ by 1-2 ulp.
    //   addcdiv_(m, d, -ss):        fma(-ss, round(m/d), th)   (d is a TENSOR:
    //                               true division here)
    mm = __fmaf_rn(w1, __fsub_rn(g, mm), mm);
    float t1 = __fmul_rn(vv, b2);
    float t2 = __fmul_rn(g, g);
    vv = __fmaf_rn(w2, t2, t1);
    float denom = __fadd_rn(__fmul_rn(__fsqrt_rn(vv), bc2_sqrt), epsf);
    th = __fmaf_rn(-step_size, __fdiv_rn(mm, denom), th);
}

__global__ void vos_shs_step_kernel(
    const int* __restrict__ active_ids,
    const int* __restrict__ karr,
    const float* __restrict__ grad_rows,
    float* __restrict__ theta,
    float* __restrict__ m,
    float* __restrict__ v,
    const float4* __restrict__ tab,
    const int D, const int n_active, const int t_now,
    const float w1, const float w2, const float b2, const float epsf)
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
    int s0 = t_now - K;
    for (int j = 0; j < K; ++j) {
        float4 sc = tab[s0 + j];
        adam_transition(mm, vv, th, 0.0f, sc.x, sc.y, w1, w2, b2, epsf);
    }
    float4 sc = tab[t_now];
    float g = grad_rows[(long long)row * D + elem];
    adam_transition(mm, vv, th, g, sc.x, sc.y, w1, w2, b2, epsf);

    m[base] = mm; v[base] = vv; theta[base] = th;
}

__global__ void vos_fence_kernel(
    const int* __restrict__ stale_ids,
    const int* __restrict__ karr,
    float* __restrict__ theta,
    float* __restrict__ m,
    float* __restrict__ v,
    const float4* __restrict__ tab,
    const int D, const int n_stale, const int t_now,
    const float w1, const float w2, const float b2, const float epsf)
{
    long long task = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long ntask = (long long)n_stale * D;
    if (task >= ntask) return;
    int row = (int)(task / D);
    int elem = (int)(task % D);
    int master = stale_ids[row];
    long long base = (long long)master * D + elem;
    float mm = m[base], vv = v[base], th = theta[base];

    int K = karr[row];
    int s0 = t_now + 1 - K;
    for (int j = 0; j < K; ++j) {
        float4 sc = tab[s0 + j];
        adam_transition(mm, vv, th, 0.0f, sc.x, sc.y, w1, w2, b2, epsf);
    }
    m[base] = mm; v[base] = vv; theta[base] = th;
}

void vos_shs_step(torch::Tensor active_ids, torch::Tensor karr,
                  torch::Tensor grad_rows, torch::Tensor theta, torch::Tensor m,
                  torch::Tensor v, torch::Tensor tab, int64_t D, int64_t t_now,
                  double w1, double w2, double b2, double epsf) {
    int n_active = active_ids.numel();
    if (n_active == 0) return;
    long long ntask = (long long)n_active * D;
    int block = 256;
    long long grid = (ntask + block - 1) / block;
    vos_shs_step_kernel<<<(unsigned)grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        active_ids.data_ptr<int>(), karr.data_ptr<int>(), grad_rows.data_ptr<float>(),
        theta.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(),
        (const float4*)tab.data_ptr<float>(), (int)D, n_active, (int)t_now,
        (float)w1, (float)w2, (float)b2, (float)epsf);
}

void vos_fence(torch::Tensor stale_ids, torch::Tensor karr, torch::Tensor theta,
               torch::Tensor m, torch::Tensor v, torch::Tensor tab, int64_t D,
               int64_t t_now, double w1, double w2, double b2, double epsf) {
    int n_stale = stale_ids.numel();
    if (n_stale == 0) return;
    long long ntask = (long long)n_stale * D;
    int block = 256;
    long long grid = (ntask + block - 1) / block;
    vos_fence_kernel<<<(unsigned)grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        stale_ids.data_ptr<int>(), karr.data_ptr<int>(),
        theta.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(),
        (const float4*)tab.data_ptr<float>(), (int)D, n_stale, (int)t_now,
        (float)w1, (float)w2, (float)b2, (float)epsf);
}
"""

CPP_SRC = """
void vos_shs_step(torch::Tensor active_ids, torch::Tensor karr,
                  torch::Tensor grad_rows, torch::Tensor theta, torch::Tensor m,
                  torch::Tensor v, torch::Tensor tab, int64_t D, int64_t t_now,
                  double w1, double w2, double b2, double epsf);
void vos_fence(torch::Tensor stale_ids, torch::Tensor karr, torch::Tensor theta,
               torch::Tensor m, torch::Tensor v, torch::Tensor tab, int64_t D,
               int64_t t_now, double w1, double w2, double b2, double epsf);
"""

BETAS = (0.9, 0.999)
EPS = 1e-15
LR_SHS = 0.0025


def build_ext(fmad):
    name = f"vos_train_ext_{'fm' if fmad else 'nofm'}"
    flags = ["-O3"] + ([] if fmad else ["-fmad=false"])
    return load_inline(name=name, cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                       functions=["vos_shs_step", "vos_fence"],
                       extra_cuda_cflags=flags, verbose=False)


def build_scalar_table(max_step, lr=LR_SHS, betas=BETAS, device="cuda"):
    """tab[s] = {step_size_s, inv_bc2_sqrt_s, bc2_sqrt_s, 0}.
    step_size_s = f32(f64(lr / (1-b1^s)))  — exactly the f64->f32 cast torch's
    addcdiv value undergoes. inv_bc2_sqrt_s = f32(1.0f / f32(sqrt(1-b2^s))) —
    replicates torch's CUDA div-by-CPU-scalar fast path (multiply by the f32
    reciprocal of the f32-rounded scalar). bc2_sqrt_s (col 2) is for the
    torchop real-step path, which passes a python float and lets torch apply
    its own (identical) reciprocal conversion."""
    b1, b2 = betas
    n = max_step + 1
    ss = np.empty(n, dtype=np.float64)
    bs = np.empty(n, dtype=np.float64)
    ss[0] = bs[0] = 0.0
    for s in range(1, n):
        ss[s] = lr / (1.0 - b1 ** s)
        bs[s] = math.sqrt(1.0 - b2 ** s)
    bs32 = bs.astype(np.float32)
    inv32 = (np.float32(1.0) / bs32).astype(np.float32)  # f32 IEEE division
    tab = torch.zeros(n, 4, dtype=torch.float32)
    tab[1:, 0] = torch.from_numpy(ss[1:])
    tab[1:, 1] = torch.from_numpy(inv32[1:])
    tab[1:, 2] = torch.from_numpy(bs32[1:])
    return tab.to(device).contiguous()


def torch_adam_ops(m, v, th, grad, step_size, bc2_sqrt,
                   w1=1 - BETAS[0], w2=1 - BETAS[1], b2=BETAS[1], eps=EPS):
    """Exact torch single-tensor Adam op sequence (the oracle-validated B path).
    step_size/bc2_sqrt: python float OR broadcastable f32 tensor.
    Float scalars use addcdiv_ (the exact op the oracle proved bitwise);
    tensor scalars use add_((-ss)*(m/denom)), which performs the identical
    three roundings (q=m/d, w=ss*q, th+w)."""
    m.lerp_(grad, w1)
    v.mul_(b2).addcmul_(grad, grad, value=w2)
    denom = (v.sqrt() / bc2_sqrt).add_(eps)
    if isinstance(step_size, float):
        th.addcdiv_(m, denom, value=-step_size)
    else:
        th.add_((-step_size) * (m / denom))
    return denom


def _torchop_row(m, v, th, grad, step_size, inv_bc2,
                 w1=1 - BETAS[0], w2=1 - BETAS[1], b2=BETAS[1], eps=EPS):
    """Per-row transition with TENSOR scalars. The historical torch op was
    div-by-CPU-scalar == multiply by f32(1/f32(bc2)); we pass that reciprocal
    directly. The theta update uses the unfused product form — 1 ulp off the
    kernel's fma on rare elements (torchop is the fallback path only; the
    kernel path is the bitwise-exact primary)."""
    m.lerp_(grad, w1)
    v.mul_(b2).addcmul_(grad, grad, value=w2)
    denom = (v.sqrt() * inv_bc2).add_(eps)
    th.add_((-step_size) * (m / denom))
    return denom


def kernel_bitwise_check(ext, tab, device, N=4096, D=48, n_active=1024,
                         t_now=None, seed=0):
    """Kernel vs per-row torch-op reference on a random state; BITWISE gate.
    Returns True iff m/v/theta of active rows and untouched rows all match."""
    if t_now is None:
        t_now = min(5000, tab.shape[0] - 1)
    assert t_now < tab.shape[0], "t_now beyond scalar table"
    gcpu = torch.Generator(device="cpu").manual_seed(seed)
    theta0 = (torch.randn(N, D, generator=gcpu) * 0.1).to(device)
    m0 = (torch.randn(N, D, generator=gcpu) * 1e-3).to(device)
    v0 = (torch.rand(N, D, generator=gcpu) * 1e-5 + 1e-6).to(device)
    grad = (torch.randn(N, D, generator=gcpu) * 1e-2).to(device)
    ids = np.sort(np.random.RandomState(1).choice(N, n_active, replace=False))
    active = torch.from_numpy(ids.astype(np.int64)).to(device)
    karr_np = np.random.RandomState(2).randint(0, 30, size=n_active)
    b1, b2 = BETAS
    w1, w2 = 1 - b1, 1 - b2

    # per-row torch-op reference (results stored)
    m_ref = torch.empty(n_active, D, device=device)
    v_ref = torch.empty(n_active, D, device=device)
    th_ref = torch.empty(n_active, D, device=device)
    for r in range(n_active):
        i = int(ids[r])
        K = int(karr_np[r])
        mr = m0[i].clone()
        vr = v0[i].clone()
        thr = theta0[i].clone()
        for s in range(t_now - K, t_now + 1):
            ss = float(LR_SHS / (1 - b1 ** s))
            bs = math.sqrt(1 - b2 ** s)
            torch_adam_ops(mr, vr, thr,
                           grad[i] if s == t_now else torch.zeros_like(mr),
                           ss, bs)
        m_ref[r], v_ref[r], th_ref[r] = mr, vr, thr

    # kernel arm on clones
    th_k = theta0.clone()
    m_k = m0.clone()
    v_k = v0.clone()
    karr = torch.from_numpy(karr_np.astype(np.int64)).to(device)
    ext.vos_shs_step(active.int(), karr.int(), grad[active].contiguous(),
                     th_k, m_k, v_k, tab, D, t_now, w1, w2, b2, EPS)
    torch.cuda.synchronize()

    ok_m = bool(torch.equal(m_k[active], m_ref))
    ok_v = bool(torch.equal(v_k[active], v_ref))
    ok_th = bool(torch.equal(th_k[active], th_ref))
    inactive = torch.ones(N, dtype=torch.bool, device=device)
    inactive[active] = False
    ok_untouched = bool(torch.equal(th_k[inactive], theta0[inactive])
                        and torch.equal(m_k[inactive], m0[inactive])
                        and torch.equal(v_k[inactive], v0[inactive]))
    dmax = (th_k[active] - th_ref).abs().max().item()
    print(f"[vos] kernel bitwise check: m={ok_m} v={ok_v} th={ok_th} "
          f"untouched={ok_untouched} max|dth|={dmax:.3e}", flush=True)
    return ok_m and ok_v and ok_th and ok_untouched


class VOSState:
    def __init__(self, max_iters, device, mode):
        self.ext = None
        self.fmad = None
        self.mode = mode
        # Build on CPU: touching CUDA here initializes the context BEFORE the
        # trainer sets CUDA_VISIBLE_DEVICES, pinning the whole run to GPU 0
        # (same bug class as the Phase-4 trace driver). Moved to self.device
        # lazily in ensure_ext(), which runs after the trainer's main().
        self.tab = build_scalar_table(max_iters, device="cpu")
        self.tau = None
        self.device = device
        self.stats = {"n_vos_steps": 0, "n_active_rows_sum": 0,
                      "replay_steps_sum": 0, "fences": []}
        self.timing = {"t": [], "inner_ms": [], "vos_ms": []}

    def ensure_ext(self):
        if self.ext is not None:
            return
        if self.tab.device.type == "cpu":
            # safe now: the trainer has set CUDA_VISIBLE_DEVICES
            self.tab = self.tab.to(self.device)
        if self.mode == "torchop":
            return
        for fmad in (True, False):
            try:
                ext = build_ext(fmad)
                # clamp t_now to the table built for THIS run's iteration
                # count; prefer ~5000 where bc2 < 1.0 in f32 so the
                # div-by-scalar reciprocal path is actually exercised
                # (at t=25000 bc2 rounds to 1.0 and masks it)
                t_now = min(5000, self.tab.shape[0] - 1)
                if kernel_bitwise_check(ext, self.tab, self.device,
                                        t_now=t_now):
                    self.ext, self.fmad, self.mode = ext, fmad, "kernel"
                    return
                print(f"[vos] fmad={fmad} variant NOT bitwise", flush=True)
            except Exception as e:
                print(f"[vos] ext build fmad={fmad} failed: {e}", flush=True)
        print("[vos] no bitwise kernel variant -> torchop mode", flush=True)
        self.mode = "torchop"


_BITMAP = {"bm": None, "n": 0}


class VOSAdamProxy:
    """Delegates everything to the inner torch Adam; intercepts step()."""

    def __init__(self, inner, vos, cfg):
        self.inner = inner
        self.vos = vos
        self.vos_start = cfg["vos_start"]
        self.n_iters = cfg["n_iters"]
        self.eval_iterations = set(cfg["eval_iterations"])
        self.dens = cfg["dens"]
        self.t = 0
        self.c = None      # shs step COUNTER, mirrors C0's state["step"]
                           # (event iterations are no-ops: no update, no
                           # increment — counter lags iteration by #events)
        self.shs_idx = None
        for i, grp in enumerate(inner.param_groups):
            if grp.get("name") == "shs":
                self.shs_idx = i
        assert self.shs_idx is not None, "shs param group not found"
        self.e0 = torch.cuda.Event(enable_timing=True)
        self.e1 = torch.cuda.Event(enable_timing=True)
        self.e2 = torch.cuda.Event(enable_timing=True)
        self.e3 = torch.cuda.Event(enable_timing=True)
        if self.vos_start < self.n_iters:
            vos.ensure_ext()   # build + bitwise gate before training starts

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def param_groups(self):
        return self.inner.param_groups

    @property
    def state(self):
        return self.inner.state

    def _fence_needed(self, t):
        if t in self.eval_iterations or t == self.n_iters:
            return True
        nxt = t + 1
        return (nxt > self.dens["from"] and nxt < self.dens["until"]
                and nxt % self.dens["interval"] == 0)

    def _shs_views(self):
        grp = self.inner.param_groups[self.shs_idx]
        p = grp["params"][0]
        theta = p.data
        N = theta.shape[0]
        D = theta[0].numel()
        state = self.inner.state[p]
        grad = p.grad.view(N, D) if p.grad is not None else None
        return (p, theta.view(N, D), state["exp_avg"].view(N, D),
                state["exp_avg_sq"].view(N, D), grad, N, D, state)

    def _vos_shs_step(self, t):
        """One VOS shs update at iteration t. All table/tau arithmetic is in
        C0's step-COUNTER space (c), which equals t only when no topology
        event has occurred; event iterations are grad-None no-ops in C0 and
        therefore also here."""
        vos = self.vos
        p, th2, m, v, grad, N, D, state = self._shs_views()

        if self.c is None:
            # first VOS step: adopt C0's own counter (authoritative)
            self.c = int(state["step"].item())

        if grad is None:
            # event iteration: C0's Adam skips every replaced param (no
            # update, no counter increment). Rebuild tau in the NEW indexing
            # (valid: we fenced before the event, so every row is advanced
            # through self.c; newborn rows have zero state and all their
            # missed transitions are exact no-ops on zero state).
            vos.tau = torch.full((N,), self.c, dtype=torch.int32,
                                 device=th2.device)
            return 0

        c = self.c + 1
        state["step"].fill_(float(c))          # logical step count (host tensor)

        bm = _BITMAP["bm"]
        assert bm is not None and bm.shape[0] == N, "missing visibility bitmap"
        active = torch.nonzero(bm, as_tuple=False).squeeze(1)
        n_active = int(active.shape[0])

        if vos.tau is None or vos.tau.shape[0] != N:
            vos.tau = torch.full((N,), c - 1, dtype=torch.int32,
                                 device=th2.device)
        tau = vos.tau
        tau_a = tau[active].long()
        karr = (c - 1) - tau_a                 # deferred zero-grad transitions

        if n_active:
            if vos.mode == "kernel":
                vos.ext.vos_shs_step(
                    active.int(), karr.int(), grad[active].contiguous(),
                    th2, m, v, vos.tab, D, c,
                    1 - BETAS[0], 1 - BETAS[1], BETAS[1], EPS)
            else:
                ma = m[active].clone()
                va = v[active].clone()
                tha = th2[active].clone()
                ga = grad[active]
                Kmax = int(karr.max().item())
                for j in range(Kmax):
                    mask = karr > j
                    if not bool(mask.any()):
                        break
                    idx = torch.nonzero(mask).squeeze(1)
                    s_r = tau_a[idx] + 1 + j
                    ss = vos.tab[s_r, 0].view(-1, 1)
                    inv = vos.tab[s_r, 1].view(-1, 1)
                    z = torch.zeros_like(ma[idx])
                    # tensor-scalar path: multiply by the SAME f32 reciprocal
                    # the historical torch div-by-scalar used (bitwise)
                    _torchop_row(ma[idx], va[idx], tha[idx], z, ss, inv)
                ss = float(vos.tab[c, 0])
                bs = float(vos.tab[c, 2])
                torch_adam_ops(ma, va, tha, ga, ss, bs)
                m[active] = ma
                v[active] = va
                th2[active] = tha
        tau[active] = c
        self.c = c

        vos.stats["n_vos_steps"] += 1
        vos.stats["n_active_rows_sum"] += n_active
        vos.stats["replay_steps_sum"] += int(karr.sum().item())
        return n_active

    def _fence(self, t):
        """Materialize every stale row through the current step (zero-grad
        replay only). Runs AFTER the active update of iteration t."""
        vos = self.vos
        if vos.tau is None or self.c is None:
            return
        p, th2, m, v, grad, N, D, state = self._shs_views()
        tau = vos.tau
        c = self.c
        stale = torch.nonzero(tau < c, as_tuple=False).squeeze(1)
        n_stale = int(stale.shape[0])
        replay = 0
        if n_stale:
            karr = c - tau[stale].long()
            replay = int(karr.sum().item())
            if vos.mode == "kernel":
                vos.ext.vos_fence(stale.int(), karr.int(), th2, m, v,
                                  vos.tab, D, c, 1 - BETAS[0], 1 - BETAS[1],
                                  BETAS[1], EPS)
            else:
                ma = m[stale].clone()
                va = v[stale].clone()
                tha = th2[stale].clone()
                Kmax = int(karr.max().item())
                for j in range(Kmax):
                    mask = karr > j
                    if not bool(mask.any()):
                        break
                    idx = torch.nonzero(mask).squeeze(1)
                    s_r = tau[stale[idx]].long() + 1 + j
                    ss = vos.tab[s_r, 0].view(-1, 1)
                    inv = vos.tab[s_r, 1].view(-1, 1)
                    z = torch.zeros_like(ma[idx])
                    _torchop_row(ma[idx], va[idx], tha[idx], z, ss, inv)
                m[stale] = ma
                v[stale] = va
                th2[stale] = tha
            tau[stale] = c
        vos.stats["fences"].append({"t": t, "n_stale": n_stale,
                                    "replay_steps": replay})

    def step(self, closure=None):
        t = self.t + 1
        self.t = t
        vos = self.vos
        if t <= self.vos_start:
            self.e0.record()
            self.inner.step()
            self.e1.record()
            self.e1.synchronize()
            vos.timing["t"].append(t)
            vos.timing["inner_ms"].append(self.e0.elapsed_time(self.e1))
            vos.timing["vos_ms"].append(0.0)
            return

        groups = self.inner.param_groups
        shs_grp = groups.pop(self.shs_idx)
        try:
            self.e0.record()
            self.inner.step()
            self.e1.record()
        finally:
            groups.insert(self.shs_idx, shs_grp)
        self.e2.record()
        self._vos_shs_step(t)
        if self._fence_needed(t):
            self._fence(t)
        self.e3.record()
        self.e3.synchronize()
        vos.timing["t"].append(t)
        vos.timing["inner_ms"].append(self.e0.elapsed_time(self.e1))
        vos.timing["vos_ms"].append(self.e2.elapsed_time(self.e3))


def patch_render():
    import gsplat.experimental as E
    real_fn = E.rasterize_gaussian_higs_dynamic

    def capturing_render(means, quats, scales, opacities, colors, *a, **kw):
        out = real_fn(means, quats, scales, opacities, colors, *a, **kw)
        try:
            if torch.is_grad_enabled():
                r = out["densification_info"]["radii"][0]
                _BITMAP["bm"] = (r > 0).any(dim=-1)   # stays on GPU, no copy
                _BITMAP["n"] += 1
        except Exception:
            pass
        return out

    E.rasterize_gaussian_higs_dynamic = capturing_render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--trainer",
                    default="/root/epic05_p13x/publication_trainer_p13x.py")
    ap.add_argument("--vos-start", type=int, default=15000,
                    help="VOS active for iterations > this; set >= iterations "
                         "to run a pure-C0 instrumented baseline")
    ap.add_argument("--vos-mode", choices=["kernel", "torchop"],
                    default="kernel")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--final-eval", default="subset")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--kernel-check", action="store_true")
    ap.add_argument("--as-mode", action="store_true",
                    help="replace the trainer's foreach Adam with the "
                         "single-tensor (A_s) path, NO VOS — reference arm "
                         "for isolating the torch-internal A_f-vs-A_s "
                         "envelope from VOS exactness")
    ap.add_argument("--eager-single-tensor", action="store_true",
                    help="VOS arm with the eager-attr inner optimizer on the "
                         "single-tensor path too — envelope-free vs --as-mode "
                         "(the decisive VOS-exactness test)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    # Pin the GPU from the driver side, BEFORE anything can touch CUDA:
    # module-level code in the imported trainer (or any pre-main CUDA use)
    # otherwise initializes the context on GPU 0 and the trainer's own
    # CUDA_VISIBLE_DEVICES setting becomes a no-op.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"[vos-dbg] CVD={os.environ['CUDA_VISIBLE_DEVICES']} "
          f"cuda_init={torch.cuda.is_initialized()}", flush=True)

    vos = VOSState(args.iterations + 100, "cuda", args.vos_mode)
    if args.kernel_check:
        vos.ensure_ext()
        print(json.dumps({"mode": vos.mode, "fmad": vos.fmad}))
        return

    spec = importlib.util.spec_from_file_location("vos_trainer", args.trainer)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    print(f"[vos-dbg] post-exec-module cuda_init="
          f"{torch.cuda.is_initialized()}", flush=True)

    orig_bootstrap = T.bootstrap_c0

    cfg = {
        "vos_start": args.vos_start,
        "n_iters": args.iterations,
        # frozen C0 recipe (baseline/reference_v1/config.py L75)
        "eval_iterations": [500, 1000, 2000, 5000, 10000, 15000,
                            20000, 25000, 30000],
        "dens": {"from": 500, "until": 15000, "interval": 100},
    }

    def bootstrap_wrapped():
        ident = orig_bootstrap()
        print(f"[vos-cuda] device_count={torch.cuda.device_count()} "
              f"current={torch.cuda.current_device()} "
              f"free/total={torch.cuda.mem_get_info()}", flush=True)
        patch_render()
        # gaussian_model is importable only AFTER bootstrap_c0() set up
        # sys.path (the reference_v1 dir is inserted there, not at module
        # import time).
        import gaussian_model as GM
        if not getattr(GM.GaussianModel.training_setup, "_vos_wrapped", False):
            orig_ts = GM.GaussianModel.training_setup

            def ts_wrapped(self, config, _orig_ts=orig_ts):
                _orig_ts(self, config)
                if args.as_mode:
                    # A_s reference arm: identical optimizer, single-tensor
                    # path (the exact semantics VOS's kernel reproduces)
                    groups = self.optimizer.param_groups
                    self.optimizer = torch.optim.Adam(
                        groups, lr=0.0, eps=1e-15, foreach=False)
                elif args.eager_single_tensor:
                    # VOS with the eager inner optimizer on the single-tensor
                    # path: envelope-free against the --as-mode arm
                    groups = self.optimizer.param_groups
                    inner = torch.optim.Adam(
                        groups, lr=0.0, eps=1e-15, foreach=False)
                    self.optimizer = VOSAdamProxy(inner, vos, cfg)
                else:
                    self.optimizer = VOSAdamProxy(self.optimizer, vos, cfg)

            ts_wrapped._vos_wrapped = True
            GM.GaussianModel.training_setup = ts_wrapped
        return ident

    T.bootstrap_c0 = bootstrap_wrapped

    targv = [args.trainer, "--arm", "c0", "--scene", args.scene,
             "--loss", "fused", "--gpu", str(args.gpu), "--eps2d", "0.3",
             "--iterations", str(args.iterations), "--checkpoints",
             "--outdir", args.outdir, "--final-eval", args.final_eval,
             "--timing-grade", "FUNCTIONAL_ONLY"]
    if args.seed is not None:
        targv += ["--seed", str(args.seed)]
    targv += ["--lpips"]
    targv += list(args.extra)
    sys.argv = targv

    t0 = time.time()
    try:
        T.main()
    finally:
        vos.stats.update({
            "wall_s": time.time() - t0,
            "bitmaps_captured": _BITMAP["n"],
            "vos_mode": vos.mode,
            "vos_fmad": vos.fmad,
            "vos_start": args.vos_start,
            "scene": args.scene,
            "iterations": args.iterations,
        })
        tarr = np.array(vos.timing["t"])
        if len(tarr):
            reg = tarr > args.vos_start
            vos.stats["timing_region"] = {
                "t_lo": int(tarr[reg].min()) if reg.any() else None,
                "t_hi": int(tarr[reg].max()) if reg.any() else None,
                "mean_inner_ms": float(np.mean(
                    np.array(vos.timing["inner_ms"])[reg])) if reg.any() else None,
                "mean_vos_ms": float(np.mean(
                    np.array(vos.timing["vos_ms"])[reg])) if reg.any() else None,
                "mean_total_ms_pre": float(np.mean(
                    np.array(vos.timing["inner_ms"])[~reg])) if (~reg).any() else None,
                "n_iters_timed": int(len(tarr)),
            }
        np.savez_compressed(
            os.path.join(args.outdir, "vos_timing.npz"),
            t=tarr,
            inner_ms=np.array(vos.timing["inner_ms"]),
            vos_ms=np.array(vos.timing["vos_ms"]))
        with open(os.path.join(args.outdir, "vos_stats.json"), "w") as f:
            json.dump(vos.stats, f, indent=2, default=str)
        print(f"VOS_DRIVER_DONE mode={vos.mode} fmad={vos.fmad} "
              f"vos_steps={vos.stats['n_vos_steps']} "
              f"fences={len(vos.stats['fences'])}", flush=True)


if __name__ == "__main__":
    main()
