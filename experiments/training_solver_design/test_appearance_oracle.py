"""Mathematical correctness checks, not training performance claims."""

import torch

from appearance_oracle import fixture, objective, operator, reduced_objective, render, solve_appearance


def test_fixed_geometry_linear_operator_matches_alpha_compositing():
    means, scales, opacity, colors, cameras = fixture()
    a, offset, _, _ = operator(means, scales, opacity, cameras)
    reference = render(means, scales, opacity, colors, cameras)
    torch.testing.assert_close(a @ colors+offset, reference, atol=1e-13, rtol=1e-13)


def test_color_clamp_invalidates_unconditional_linearity():
    means, scales, opacity, colors, cameras = fixture()
    colors[::4] = -4.0
    a, offset, _, _ = operator(means, scales, opacity, cameras)
    actual = render(means, scales, opacity, colors, cameras)
    assert (actual-(a @ colors+offset)).abs().max().item() > 0.1


def test_exact_inner_solution_stationarity():
    means, scales, opacity, colors, cameras = fixture()
    a, offset, _, _ = operator(means, scales, opacity, cameras)
    target = a @ colors+offset
    prior = torch.zeros_like(colors)
    damping = 0.02
    solved = solve_appearance(a, target-offset, prior, damping)
    gradient = a.T @ (a @ solved+offset-target)+damping*(solved-prior)
    assert torch.linalg.vector_norm(gradient).item() < 1e-12
    assert objective(a, offset, solved, target, prior, damping) < objective(a, offset, prior, target, prior, damping)


def test_envelope_gradient_matches_differentiated_solve_and_finite_difference():
    means, scales, opacity, colors, cameras = fixture()
    target = render(means, scales, opacity, colors, cameras).detach()
    shifted = (means+torch.tensor([0.06, -0.025, 0.01])).requires_grad_()
    prior = torch.zeros_like(colors)
    args = (scales, opacity, cameras, target, prior, 0.03)
    exact = torch.autograd.grad(reduced_objective(shifted, *args), shifted)[0]
    envelope = torch.autograd.grad(reduced_objective(shifted, *args, detach_solution=True), shifted)[0]
    torch.testing.assert_close(exact, envelope, atol=1e-10, rtol=1e-9)
    direction = torch.linspace(-1, 1, shifted.numel(), dtype=shifted.dtype).reshape_as(shifted)
    direction /= direction.norm()
    h = 1e-5
    fd = (reduced_objective(shifted.detach()+h*direction, *args)-reduced_objective(shifted.detach()-h*direction, *args))/(2*h)
    torch.testing.assert_close((envelope*direction).sum(), fd, atol=1e-7, rtol=1e-5)


def test_inexact_inner_solution_cannot_use_exact_envelope_gradient():
    means, scales, opacity, colors, cameras = fixture()
    target = render(means, scales, opacity, colors, cameras).detach()
    shifted = (means+torch.tensor([0.08, -0.04, 0.0])).requires_grad_()
    prior = torch.zeros_like(colors)
    exact = torch.autograd.grad(reduced_objective(shifted, scales, opacity, cameras, target, prior, 0.03), shifted)[0]
    a, offset, _, _ = operator(shifted, scales, opacity, cameras)
    partial = torch.autograd.grad(objective(a, offset, prior, target, prior, 0.03), shifted)[0]
    assert (partial-exact).norm().item() / exact.norm().item() > 0.01


def test_schur_elimination_preserves_coupled_step():
    means, scales, opacity, colors, cameras = fixture()
    means.requires_grad_()
    a, offset, _, _ = operator(means, scales, opacity, cameras)
    target = (a @ colors+offset).detach()+0.005
    geom_j = torch.autograd.functional.jacobian(
        lambda m: (operator(m, scales, opacity, cameras)[0] @ colors + operator(m, scales, opacity, cameras)[1])[:, 0], means,
        vectorize=True,
    ).flatten(1)
    color_j = a
    residual = (a @ colors+offset-target)[:, 0]
    hgg = geom_j.T @ geom_j+0.03*torch.eye(geom_j.shape[1])
    hcc = color_j.T @ color_j+0.02*torch.eye(color_j.shape[1])
    hgc = geom_j.T @ color_j
    bg, bc = geom_j.T @ residual, color_j.T @ residual
    dg = torch.linalg.solve(hgg-hgc @ torch.linalg.solve(hcc, hgc.T), -bg+hgc @ torch.linalg.solve(hcc, bc))
    dc = torch.linalg.solve(hcc, -bc-hgc.T @ dg)
    full_h = torch.cat((torch.cat((hgg, hgc), 1), torch.cat((hgc.T, hcc), 1)), 0)
    full = torch.linalg.solve(full_h, -torch.cat((bg, bc)))
    torch.testing.assert_close(torch.cat((dg, dc)), full, atol=1e-10, rtol=1e-8)


def test_independent_gaussian_solves_miss_overlap_coupling():
    means, scales, opacity, _, cameras = fixture()
    a, _, _, _ = operator(means, scales, opacity, cameras)
    # DC columns only; neighboring Gaussians overlap in actual projected images.
    a = a[:, ::4]
    h = a.T @ a+0.01*torch.eye(a.shape[1])
    rhs = a.T @ (a @ torch.ones(a.shape[1], dtype=a.dtype))
    coupled = torch.linalg.solve(h, rhs)
    diagonal = rhs/h.diag()
    assert (h @ coupled-rhs).norm().item() < 1e-12
    assert (h @ diagonal-rhs).norm().item() / rhs.norm().item() > 0.1


def test_unregularized_appearance_can_erase_geometry_signal():
    # A local geometry direction already in the appearance span is unidentifiable.
    means, scales, opacity, _, cameras = fixture()
    a, _, _, _ = operator(means, scales, opacity, cameras)
    q, _ = torch.linalg.qr(a)
    direction = a @ torch.ones(a.shape[1], dtype=a.dtype)
    reduced = direction-q @ (q.T @ direction)
    assert reduced.norm().item() / direction.norm().item() < 1e-12
