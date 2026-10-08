"""Independent checks of chain rule and differential-only Adam updates."""

import torch

from sibling_coordinates import decode_raw, encode_children
from newborn_updates import step_with_newborn_policy


def test_nonlinear_physical_loss_chain_rule():
    torch.manual_seed(3)
    raw = torch.randn(13, 7, dtype=torch.float64)
    encoded = raw.clone()
    encoded[-8:] = encode_children(raw, 4)
    encoded.requires_grad_()
    physical = decode_raw(encoded.detach(), 4).requires_grad_()
    weights = torch.randn_like(raw)

    def loss(x):
        return ((x.sigmoid() * weights).sum(dim=0).square()).sum()

    gp, = torch.autograd.grad(loss(physical), physical)
    ge, = torch.autograd.grad(loss(decode_raw(encoded, 4)), encoded)
    expected = torch.cat((gp[:-8], gp[-8:-4]+gp[-4:], gp[-8:-4]-gp[-4:]))
    torch.testing.assert_close(ge, expected)


def test_fresh_relative_matches_independent_adam_and_preserves_shared():
    p = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64))
    shared = torch.nn.Parameter(p[:1].detach().clone())
    relative = torch.nn.Parameter(p[1:].detach().clone())
    joined = torch.optim.Adam([p], lr=.01)
    old = torch.optim.Adam([shared], lr=.01)
    fresh = torch.optim.Adam([relative], lr=.01)
    for opt, param in ((joined, p), (old, shared)):
        opt.state[param] = {'step': torch.tensor(1985.),
                            'exp_avg': torch.zeros_like(param),
                            'exp_avg_sq': torch.zeros_like(param)}
        opt.state[param]['exp_avg'][:1] = .2
        opt.state[param]['exp_avg_sq'][:1] = .1
    for age in range(1, 17):
        p.grad = torch.tensor([[.3, -.2], [.1*age, -.4]], dtype=p.dtype)
        shared.grad, relative.grad = p.grad[:1].clone(), p.grad[1:].clone()
        step_with_newborn_policy(joined, 1, age, 'fresh_age', 1.)
        old.step()
        fresh.step()
        torch.testing.assert_close(p, torch.cat((shared, relative)))
