import torch

from shared_proxy_renderer import rescale_conics


def test_conic_value_and_derivative_against_covariance():
    torch.manual_seed(183)
    q = torch.randn(21, 2, 2, dtype=torch.double)
    cov = (q @ q.transpose(-1, -2) + .3*torch.eye(2)).requires_grad_()
    inv = torch.linalg.inv(cov)
    packed = torch.stack((inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]), -1)
    visible = torch.ones(21, dtype=torch.bool)
    actual = rescale_conics(packed, visible)
    expected_matrix = torch.linalg.inv((cov-.3*torch.eye(2))/4+.3*torch.eye(2))
    expected = torch.stack((expected_matrix[:, 0, 0], expected_matrix[:, 0, 1], expected_matrix[:, 1, 1]), -1)
    torch.testing.assert_close(actual, expected)
    adjoint = torch.randn_like(actual)
    ga, = torch.autograd.grad((actual*adjoint).sum(), cov, retain_graph=True)
    gb, = torch.autograd.grad((expected*adjoint).sum(), cov)
    # Off-diagonal entries represent a symmetric covariance.
    torch.testing.assert_close(ga+ga.transpose(-1, -2), gb+gb.transpose(-1, -2))


def test_inactive_nan_values_and_relu_boundary():
    conics = torch.tensor([[float('nan')]*3, [1., 0., 1.]], requires_grad=True)
    out = rescale_conics(conics, torch.tensor([False, True]))
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(conics.grad).all()
    assert torch.equal(conics.grad[0], torch.zeros(3))
    x = torch.tensor([0.], requires_grad=True)
    clamp_grad, = torch.autograd.grad(x.clamp_min(0).sum(), x)
    relu_grad, = torch.autograd.grad(x.relu().sum(), x)
    assert clamp_grad.item() == 1 and relu_grad.item() == 0
