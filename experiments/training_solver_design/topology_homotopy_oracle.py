"""Scalar optical-depth topology homotopy checks, not a 3DGS renderer test."""

import json

import torch


def gated_alpha(alpha: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return -torch.expm1(weight * torch.log1p(-alpha))


def composite(layers: list[tuple[float, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    out = torch.zeros((), dtype=torch.float64)
    transmittance = torch.ones((), dtype=torch.float64)
    for _, alpha, color in sorted(layers, key=lambda layer: layer[0]):
        out = out + transmittance * alpha * color
        transmittance = transmittance * (1 - alpha)
    return out


def main() -> None:
    torch.manual_seed(20260927)
    max_birth_error = 0.0
    max_release_error = 0.0
    one = torch.tensor(1.0, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)
    for _ in range(10_000):
        depth = torch.rand(5, dtype=torch.float64).tolist()
        alpha = torch.rand(5, dtype=torch.float64) * 0.98
        color = torch.rand(5, dtype=torch.float64)
        parent = (depth[0], alpha[0], color[0])
        children = [(depth[i], alpha[i], color[i]) for i in (1, 2)]
        others = [(depth[i], alpha[i], color[i]) for i in (3, 4)]
        birth = composite([(depth[0], gated_alpha(alpha[0], one), color[0])]
                          + [(depth[i], gated_alpha(alpha[i], zero), color[i]) for i in (1, 2)]
                          + others)
        release = composite([(depth[0], gated_alpha(alpha[0], zero), color[0])]
                            + [(depth[i], gated_alpha(alpha[i], one), color[i]) for i in (1, 2)]
                            + others)
        max_birth_error = max(max_birth_error, abs((birth - composite([parent] + others)).item()))
        max_release_error = max(max_release_error, abs((release - composite(children + others)).item()))

    # At zero activation, dropping the zero-opacity children from a hard-cutoff
    # rasterizer also drops their activation gradient. A constructed case shows
    # that this can reverse the accept/reject direction for an image loss.
    parent_alpha = torch.tensor(0.5, dtype=torch.float64)
    child_alpha = torch.tensor(0.25, dtype=torch.float64)
    parent_color = torch.tensor(0.2, dtype=torch.float64)
    child_color = torch.tensor(1.0, dtype=torch.float64)
    target = torch.tensor(0.3, dtype=torch.float64)
    activation = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    parent = gated_alpha(parent_alpha, 1 - activation)
    child = gated_alpha(child_alpha, activation)
    full = composite([(0.0, parent, parent_color), (1.0, child, child_color), (2.0, child, child_color)])
    full_loss = 0.5 * (full - target).square()
    full_gradient = torch.autograd.grad(full_loss, activation, retain_graph=True)[0]
    cutoff = 1 / 255
    culled_layers = [(0.0, parent, parent_color)]
    if child.item() > cutoff:
        culled_layers += [(1.0, child, child_color), (2.0, child, child_color)]
    culled = composite(culled_layers)
    culled_gradient = torch.autograd.grad(0.5 * (culled - target).square(), activation, retain_graph=True)[0]
    fixed_support_layers = [(0.0, parent, parent_color)]
    if child_alpha.item() > cutoff:
        fixed_support_layers += [(1.0, child, child_color), (2.0, child, child_color)]
    fixed_support = composite(fixed_support_layers)
    fixed_support_gradient = torch.autograd.grad(0.5 * (fixed_support - target).square(), activation)[0]

    result = {
        "trials": 10_000,
        "max_birth_forward_error": max_birth_error,
        "max_release_forward_error": max_release_error,
        "cutoff": cutoff,
        "zero_gate_full_loss_gradient": full_gradient.item(),
        "zero_gate_culled_loss_gradient": culled_gradient.item(),
        "zero_gate_fixed_support_loss_gradient": fixed_support_gradient.item(),
        "cutoff_reverses_acceptance_sign": bool(full_gradient.item() < 0 < culled_gradient.item()),
        "fixed_support_restores_gradient": bool(torch.allclose(full_gradient, fixed_support_gradient, atol=1e-12, rtol=0)),
    }
    if max_birth_error > 1e-12 or max_release_error > 1e-12 or not result["cutoff_reverses_acceptance_sign"] or not result["fixed_support_restores_gradient"]:
        raise AssertionError(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
