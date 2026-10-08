"""Small FP64 3DGS oracle for design validation, not a production rasterizer.

Uses perspective EWA projection, front-to-back alpha composition and degree-1
SH. No tile culling/early termination. The linear operator assumes that color
clamp masks stay inactive; tests deliberately exercise violation of this rule.
"""

import torch


def fixture(seed=7):
    rng = torch.Generator().manual_seed(seed)
    means = torch.rand(10, 3, generator=rng, dtype=torch.float64)
    means = means * torch.tensor([1.2, 0.8, 0.7]) + torch.tensor([-0.6, -0.4, 2.5])
    scales = 0.16 + 0.10 * torch.rand(10, 3, generator=rng, dtype=torch.float64)
    opacity = 0.25 + 0.35 * torch.rand(10, generator=rng, dtype=torch.float64)
    colors = 0.08 * torch.randn(40, 3, generator=rng, dtype=torch.float64)
    cameras = torch.tensor([[-0.35, 0.0, 0.0], [0.0, 0.15, 0.0], [0.35, 0.0, 0.0]], dtype=torch.float64)
    return means, scales, opacity, colors, cameras


def operator(means, scales, opacity, cameras, size=12):
    """Return A, offset, weights and SH bases in original Gaussian order."""
    rows, cols = torch.meshgrid(
        torch.arange(size, dtype=means.dtype),
        torch.arange(size, dtype=means.dtype), indexing="ij",
    )
    pixels = torch.stack((cols.flatten(), rows.flatten()), dim=-1) - (size - 1) / 2
    all_a, all_b, all_weights, all_bases = [], [], [], []
    for camera in cameras:
        p = means - camera
        x, y, z = p.unbind(-1)
        f = 22.0
        zero = torch.zeros_like(z)
        jac = torch.stack((f/z, zero, -f*x/z.square(), zero, f/z, -f*y/z.square()), -1).reshape(-1, 2, 3)
        cov = (jac * scales.square().unsqueeze(1)) @ jac.transpose(-1, -2)
        cov = cov + 0.3 * torch.eye(2, dtype=means.dtype)
        xy = f * p[:, :2] / z[:, None]
        delta = pixels[:, None, :] - xy[None, :, :]
        power = torch.einsum("pni,nij,pnj->pn", delta, torch.linalg.inv(cov), delta)
        alpha = (opacity[None, :] * torch.exp(-0.5 * power)).clamp(max=0.99)
        order = z.argsort(stable=True)
        alpha_sorted = alpha[:, order]
        trans = torch.cumprod(torch.cat((torch.ones_like(alpha_sorted[:, :1]), 1-alpha_sorted), 1), 1)
        weights = (alpha_sorted * trans[:, :-1])[:, order.argsort()]
        direction = p / torch.linalg.vector_norm(p, dim=-1, keepdim=True)
        basis = torch.stack((torch.full_like(z, 0.28209479177387814),
                             -0.4886025119029199*direction[:, 1],
                             0.4886025119029199*direction[:, 2],
                             -0.4886025119029199*direction[:, 0]), -1)
        all_a.append((weights[:, :, None] * basis[None, :, :]).flatten(1))
        all_b.append((0.5*weights.sum(1) + 0.03*trans[:, -1])[:, None].expand(-1, 3))
        all_weights.append(weights)
        all_bases.append(basis)
    return torch.cat(all_a), torch.cat(all_b), all_weights, all_bases


def render(means, scales, opacity, colors, cameras, clamp_colors=True, size=12):
    _, _, weights, bases = operator(means, scales, opacity, cameras, size)
    outputs = []
    for w, basis in zip(weights, bases):
        rgb = torch.einsum("nl,nlc->nc", basis, colors.reshape(-1, 4, 3)) + 0.5
        if clamp_colors:
            rgb = rgb.clamp_min(0)
        outputs.append(w @ rgb + 0.03*(1-w.sum(1))[:, None])
    return torch.cat(outputs)


def solve_appearance(a, target_minus_offset, prior, damping):
    h = a.T @ a + damping*torch.eye(a.shape[1], dtype=a.dtype, device=a.device)
    return prior + torch.linalg.solve(h, a.T @ (target_minus_offset-a @ prior))


def objective(a, offset, colors, target, prior, damping):
    return 0.5*(a @ colors+offset-target).square().sum() + 0.5*damping*(colors-prior).square().sum()


def reduced_objective(means, scales, opacity, cameras, target, prior, damping, detach_solution=False):
    a, offset, _, _ = operator(means, scales, opacity, cameras)
    colors = solve_appearance(a, target-offset, prior, damping)
    if detach_solution:
        colors = colors.detach()
    return objective(a, offset, colors, target, prior, damping)
