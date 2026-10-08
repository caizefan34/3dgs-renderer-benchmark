"""Nonlinear unbiased gradient identity and a variance failure counterexample."""

import torch
import torch.nn.functional as F

from multilevel_loss import block_means, corrected_loss


def test_complete_partition_preserves_original_nonlinear_gradient():
    torch.manual_seed(51)
    parameter = torch.randn(12, 16, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(parameter)
    fine = block_means((parameter.sigmoid()-target).square(), 3, 4)
    coarse_image = F.avg_pool2d(parameter[None, None].exp(), 2)[0, 0]
    coarse = block_means((coarse_image-.3).square(), 3, 4)
    assignment = torch.randperm(12).reshape(3, 4) % 4
    estimates = [corrected_loss(fine, coarse, (assignment == i).double()*4) for i in range(4)]
    expected, = torch.autograd.grad(fine.mean(), parameter, retain_graph=True)
    actual, = torch.autograd.grad(torch.stack(estimates).mean(), parameter)
    torch.testing.assert_close(actual, expected)


def test_unbiased_control_variate_can_increase_variance():
    fine = torch.tensor([-1., 1.], dtype=torch.float64)
    coarse = -fine
    weights = [torch.tensor([2., 0.]), torch.tensor([0., 2.])]
    base = torch.stack([(w*fine).mean() for w in weights])
    corrected = torch.stack([corrected_loss(fine, coarse, w) for w in weights])
    assert corrected.mean() == fine.mean()
    torch.testing.assert_close(corrected.square().mean(), 4*base.square().mean())
