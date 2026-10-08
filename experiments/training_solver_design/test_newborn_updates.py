"""Compare shared-tensor newborn correction with independent fresh Adam."""

import torch

from newborn_updates import step_with_newborn_policy


def test_fresh_age_matches_independent_adam_and_preserves_survivor():
    dtype = torch.float64
    shared = torch.nn.Parameter(torch.tensor([[0.3, -0.5], [0.8, 0.2]], dtype=dtype))
    survivor = torch.nn.Parameter(shared[:1].detach().clone())
    child = torch.nn.Parameter(shared[1:].detach().clone())
    options = dict(lr=0.002, eps=1e-8, betas=(0.9, 0.999))
    optim = torch.optim.Adam([shared], **options)
    old_optim = torch.optim.Adam([survivor], **options)
    child_optim = torch.optim.Adam([child], **options)
    for p, opt in [(shared, optim), (survivor, old_optim)]:
        m = torch.zeros_like(p)
        v = torch.zeros_like(p)
        m[0] = torch.tensor([0.08, -0.03])
        v[0] = torch.tensor([0.04, 0.02])
        opt.state[p] = {'step': torch.tensor(1985.0), 'exp_avg': m, 'exp_avg_sq': v}
    rng = torch.Generator().manual_seed(20260926)
    for age in range(1, 31):
        grad = torch.randn(shared.shape, generator=rng, dtype=dtype)
        if age % 5 == 0:
            grad[1] = 0  # Dense Adam still ages/decays on zero gradients.
        shared.grad = grad.clone()
        survivor.grad, child.grad = grad[:1].clone(), grad[1:].clone()
        lr = 0.002 * (0.99 ** age)
        for opt in (optim, old_optim, child_optim):
            opt.param_groups[0]['lr'] = lr
        step_with_newborn_policy(optim, 1, age, 'fresh_age', 1.0)
        old_optim.step()
        child_optim.step()
        torch.testing.assert_close(shared[:1], survivor, atol=0, rtol=0)
        torch.testing.assert_close(shared[1:], child, atol=1e-14, rtol=1e-14)
        for key in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(optim.state[shared][key][1:], child_optim.state[child][key], atol=0, rtol=0)


def test_constant_step_scale_matches_lower_lr_child_with_same_old_clock():
    p = torch.nn.Parameter(torch.tensor([0.2, 0.7], dtype=torch.float64))
    reference = torch.nn.Parameter(p[1:].detach().clone())
    scale = 0.34
    opt = torch.optim.Adam([p], lr=0.001, eps=1e-15)
    ref = torch.optim.Adam([reference], lr=0.001 * scale, eps=1e-15)
    for param, optimizer in [(p, opt), (reference, ref)]:
        optimizer.state[param] = {'step': torch.tensor(1985.0),
                                  'exp_avg': torch.zeros_like(param), 'exp_avg_sq': torch.zeros_like(param)}
    for age in range(1, 11):
        p.grad = torch.tensor([0.1, (-1) ** age * 0.2], dtype=p.dtype)
        reference.grad = p.grad[1:].clone()
        step_with_newborn_policy(opt, 1, age, 'lr_scale', scale)
        ref.step()
        torch.testing.assert_close(p[1:], reference, atol=1e-14, rtol=1e-14)
