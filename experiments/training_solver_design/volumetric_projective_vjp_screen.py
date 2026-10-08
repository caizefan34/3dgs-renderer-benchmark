"""Autograd VJP screen for original versus projective Vol3DGS alpha."""

import json

import numpy as np
import torch


def alpha_original(a, mu, w, kappa):
    d = w / torch.linalg.vector_norm(w, dim=-1, keepdim=True)
    ad = torch.einsum("nij,nj->ni", a, d)
    den = torch.clamp_min((d * ad).sum(-1), 1e-7)
    num = (mu * ad).sum(-1)
    t = num / den
    residual = t[:, None] * d - mu
    power = -.5 * torch.einsum("ni,nij,nj->n", residual, a, residual)
    optical = 2.506 * kappa * torch.rsqrt(den + 1e-7) * torch.exp(power)
    return torch.where(num <= 0, torch.zeros_like(optical),
                       torch.clamp_max(1 - torch.exp(-optical), .99))


def alpha_projective(a, mu, w, kappa):
    z = mu[:, 2]
    w0 = mu / z[:, None]
    v = torch.einsum("nij,nj->ni", a, w0)
    a0 = (w0 * v).sum(-1)
    b = z[:, None, None] ** 2 * (a0[:, None, None] * a[:, :2, :2]
                                - v[:, :2, None] * v[:, None, :2])
    delta = w[:, :2] - w0[:, :2]
    aw = torch.einsum("ni,nij,nj->n", w, a, w)
    q = torch.einsum("ni,nij,nj->n", delta, b, delta) / aw
    norm = torch.linalg.vector_norm(w, dim=-1)
    den = torch.clamp_min(aw / norm.square(), 1e-7)
    num = torch.einsum("ni,nij,nj->n", mu, a, w) / norm
    optical = 2.506 * kappa * torch.rsqrt(den + 1e-7) * torch.exp(-.5 * q)
    return torch.where(num <= 0, torch.zeros_like(optical),
                       torch.clamp_max(1 - torch.exp(-optical), .99))


def run(dtype, a0, mu0, w0, kappa0, weights):
    outputs = {}
    for name, implementation in (("original", alpha_original),
                                  ("projective", alpha_projective)):
        a_raw = torch.tensor(a0, dtype=dtype, requires_grad=True)
        a = .5 * (a_raw + a_raw.transpose(-1, -2))
        mu = torch.tensor(mu0, dtype=dtype, requires_grad=True)
        kappa = torch.tensor(kappa0, dtype=dtype, requires_grad=True)
        w = torch.tensor(w0, dtype=dtype)
        alpha = implementation(a, mu, w, kappa)
        gradients = torch.autograd.grad(alpha, (a_raw, mu, kappa),
                                        torch.tensor(weights, dtype=dtype))
        outputs[name] = (alpha.detach().numpy(),
                         [item.detach().numpy() for item in gradients])
    alpha_a, grad_a = outputs["original"]
    alpha_b, grad_b = outputs["projective"]
    result = {"alpha_abs_p99": float(np.percentile(np.abs(alpha_a - alpha_b), 99)),
              "alpha_abs_max": float(np.max(np.abs(alpha_a - alpha_b))),
              "gradient": {}}
    for name, ga, gb in zip(("A", "mu", "kappa"), grad_a, grad_b):
        diff = ga - gb
        norm_a = np.linalg.norm(ga.reshape(len(ga), -1), axis=1)
        norm_diff = np.linalg.norm(diff.reshape(len(diff), -1), axis=1)
        ratio = norm_diff / np.maximum(norm_a, 1e-8)
        result["gradient"][name] = {
            "relative_norm_p50": float(np.median(ratio)),
            "relative_norm_p99": float(np.percentile(ratio, 99)),
            "relative_norm_max": float(np.max(ratio)),
            "absolute_max": float(np.max(np.abs(diff))),
        }
    return result


def main():
    rng = np.random.default_rng(20260927)
    n = 4096
    z = np.exp(rng.uniform(np.log(1), np.log(50), n))
    mu = np.column_stack((rng.uniform(-.5, .5, n) * z,
                          rng.uniform(-.5, .5, n) * z, z))
    scales = np.exp(rng.uniform(np.log(.001), np.log(.1), (n, 3)))
    rotation, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    a = rotation @ (np.eye(3)[None] / scales[:, None, :] ** 2) @ np.swapaxes(rotation, 1, 2)
    w0 = mu / z[:, None]
    w = w0 + np.column_stack((rng.normal(size=(n, 2)) *
                              (scales.min(axis=1) / z)[:, None], np.zeros(n)))
    kappa = np.exp(rng.uniform(np.log(.01), np.log(2), n))
    weights = rng.normal(size=n)
    print(json.dumps({"cases": n, "seed": 20260927,
                      "scope": "independent alpha VJP on symmetric-matrix parameter manifold; not Slang custom backward or full rasterizer",
                      "float64": run(torch.float64, a, mu, w, kappa, weights),
                      "float32": run(torch.float32, a, mu, w, kappa, weights)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
