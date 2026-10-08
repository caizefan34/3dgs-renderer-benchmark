#!/usr/bin/env python3
"""g5_cgt_renderer.py - differentiable CGT student (PyTorch port of frozen g1.predict_and_vjp).

This is the FOUNDATIONAL native component for every G5 stage (A/B/C/D). It is a
purely differentiable reimplementation of the F0 student render path:
  per pixel q (P,2), per-component coeffs p (R,10):
    dx,dy = q - p[:, :2]
    l00=exp(p2), l10=p3, l11=exp(p4)
    z0 = dx*l00 + dy*l10 ; z1 = dy*l11
    w  = softmax(p5 - 0.5*(z0^2+z1^2))            (over R)
    t  = sigmoid(p6) ; color = sigmoid(p7:10)
    expert = [(1-t)*color (3), t]                 (R,4)
    pred = w @ expert                              (P,4) = [E_rgb, T]
Gradients flow to all 10 coeffs per component. No teacher, no certificate here -
those are added by the stage harness. Math is bit-faithful to predict_and_vjp.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def cgt_student_render(coeffs: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
    """Differentiable CGT student.

    coeffs: (R, 10) float tensor (requires_grad for training)
    px:     (P, 2) float tensor of CENTERED pixel coords
    returns (P, 4) = [E_rgb (3), T (1)]
    """
    R = coeffs.shape[0]
    mu = coeffs[:, 0:2]                       # (R,2)
    l00 = torch.exp(coeffs[:, 2])             # (R,)
    l10 = coeffs[:, 3]
    l11 = torch.exp(coeffs[:, 4])
    logit_w = coeffs[:, 5]                    # (R,)
    t = torch.sigmoid(coeffs[:, 6:7])         # (R,1)
    color = torch.sigmoid(coeffs[:, 7:10])    # (R,3)

    dx = px[:, 0:1] - mu[None, :, 0]          # (P,R)
    dy = px[:, 1:2] - mu[None, :, 1]          # (P,R)
    z0 = dx * l00[None, :] + dy * l10[None, :]
    z1 = dy * l11[None, :]
    logits = logit_w[None, :] - 0.5 * (z0 * z0 + z1 * z1)   # (P,R)
    w = torch.softmax(logits, dim=1)                              # (P,R)

    expert = torch.cat([(1.0 - t) * color, t], dim=1)            # (R,4)
    pred = w @ expert                                             # (P,4)
    return pred


def cgt_student_render_batched(coeffs: torch.Tensor, px: torch.Tensor, tile_off: torch.Tensor) -> torch.Tensor:
    """Batched over tiles: coeffs (n_tiles, R, 10), px (P,2), tile_off (n_tiles+1,)
    returns (sum_tiles*P, 4). One tile's pixels are px (shared centered grid)."""
    n_tiles = coeffs.shape[0]
    outs = []
    for t in range(n_tiles):
        outs.append(cgt_student_render(coeffs[t], px))
    return torch.cat(outs, dim=0)


if __name__ == "__main__":
    # CPU self-check vs frozen numpy predict_and_vjp (no GPU needed).
    import sys, numpy as np
    sys.path.insert(0, "/home/liaoyuanjun/cgt_realscene/final/SCRIPTS")
    from g1_compressibility import predict_and_vjp, initialize, fit_cgt

    rng = np.random.default_rng(0)
    R = 4
    # random centered pixel grid
    P = 256
    xs = np.arange(16) + 0.5; ys = np.arange(16) + 0.5
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    px = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    px -= px.mean(0)
    # random teacher-like target y (P,4) and fit coeffs with the FROZEN fit_cgt
    y = rng.random((P, 4))
    p_np = fit_cgt(px, y, R, seed=0, iters=100)     # (R,10)

    # numpy reference
    ref = predict_and_vjp(p_np, px)                 # (P,4)

    # torch render
    px_t = torch.tensor(px, dtype=torch.float64)
    p_t = torch.tensor(p_np, dtype=torch.float64, requires_grad=True)
    out = cgt_student_render(p_t, px_t)
    out_np = out.detach().numpy()

    max_abs = float(np.max(np.abs(out_np - ref)))
    cos = float(np.sum(out_np * ref) / (np.linalg.norm(out_np) * np.linalg.norm(ref) + 1e-30))
    print(f"[selfcheck] R={R} P={P} max_abs_diff={max_abs:.3e} cosine={cos:.12f}")

    # gradient check: analytic vs central finite difference on a random direction
    out.sum().backward()
    g_analytic = p_t.grad.clone()
    g_fd = torch.zeros_like(p_t)
    eps = 1e-6
    flat = p_t.detach().numpy().ravel().copy()
    for i in range(flat.size):
        fp = flat.copy(); fp[i] += eps
        fm = flat.copy(); fm[i] -= eps
        pp = torch.tensor(fp.reshape(R, 10), dtype=torch.float64)
        pm = torch.tensor(fm.reshape(R, 10), dtype=torch.float64)
        op = cgt_student_render(pp, px_t).sum().item()
        om = cgt_student_render(pm, px_t).sum().item()
        g_fd.view(-1)[i] = (op - om) / (2 * eps)
    g_fd = g_fd.reshape(R, 10)
    rel_l2 = float(np.linalg.norm((g_analytic - g_fd).numpy()) / (np.linalg.norm(g_fd.numpy()) + 1e-30))
    print(f"[gradcheck] rel_L2(analytic vs FD)={rel_l2:.3e} (should be <1e-5)")
    ok = max_abs < 1e-9 and cos > 0.999999 and rel_l2 < 1e-4
    print("SELFCHECK_" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
