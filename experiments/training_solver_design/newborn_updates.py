"""Isolated dense-Adam newborn controls, not a new optimizer claim."""

import torch


@torch.no_grad()
def step_with_newborn_policy(optimizer, first_newborn, age, policy, lr_scale):
    if policy in ('native', 'inherit_02') or first_newborn is None:
        optimizer.step()
        return
    assert policy in ('fresh_age', 'lr_scale')
    old_rows = []
    for group in optimizer.param_groups:
        assert len(group['params']) == 1 and not group['weight_decay'] and not group['amsgrad']
        p = group['params'][0]
        assert p.grad is not None, 'This diagnostic uses dense updates and a common birth age'
        old_rows.append(p[first_newborn:].clone())
    optimizer.step()
    for group, old in zip(optimizer.param_groups, old_rows):
        p = group['params'][0]
        if policy == 'lr_scale':
            p[first_newborn:] = old + lr_scale * (p[first_newborn:] - old)
        else:
            state = optimizer.state[p]
            beta1, beta2 = group['betas']
            m = state['exp_avg'][first_newborn:] / (1 - beta1 ** age)
            v = state['exp_avg_sq'][first_newborn:] / (1 - beta2 ** age)
            p[first_newborn:] = old - group['lr'] * m / (v.sqrt() + group['eps'])
