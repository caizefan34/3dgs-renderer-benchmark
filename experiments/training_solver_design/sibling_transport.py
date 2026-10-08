"""Raw-moment handoff under an explicit zero-extended relative prehistory.

Both modes must use identical EMA betas and dense update counts. A fresh bias
clock may be used for the relative *update*, but its raw pre-birth gradients are
defined as zero. Handoff uses the old physical group clock, not the fresh clock.
This transports gradient history; it does not preserve Adam's next update.
"""

import torch


@torch.no_grad()
def accumulate_cross(optimizer, count, cross):
    """Call exactly once before each paired optimizer step, after backward."""
    for group in optimizer.param_groups:
        assert len(group['params']) == 1 and not group['amsgrad'] and not group['weight_decay']
        p = group['params'][0]
        assert p.grad is not None
        beta2 = group['betas'][1]
        cross[group['name']].mul_(beta2).addcmul_(
            p.grad[-2*count:-count], p.grad[-count:], value=1-beta2)


@torch.no_grad()
def fold_to_physical(optimizer, count, cross):
    """Decode active pairs and transport raw moments before any topology edit.

    Requires zeroed parameter gradients. Clears no optimizer histories and leaves
    survivor rows/group clocks unchanged. Caller must stop using SiblingView.
    The cross state can be released after this call.
    """
    roundoff = {}
    for group in optimizer.param_groups:
        assert len(group['params']) == 1 and not group['amsgrad'] and not group['weight_decay']
        p = group['params'][0]
        assert p.grad is None, 'Fold only between updates with cleared gradients'
        state = optimizer.state[p]
        mu, delta = p[-2*count:-count], p[-count:]
        physical = torch.cat((mu+delta, mu-delta))
        m = state['exp_avg']
        v = state['exp_avg_sq']
        physical_m = torch.cat(((m[-2*count:-count]+m[-count:])*.5,
                                (m[-2*count:-count]-m[-count:])*.5))
        total = v[-2*count:-count]+v[-count:]
        c = cross[group['name']]
        physical_v = torch.cat(((total+2*c)*.25, (total-2*c)*.25))
        minimum = float(physical_v.min())
        tolerance = 32*torch.finfo(v.dtype).eps*float(v[-2*count:].abs().max())
        assert minimum >= -tolerance, ('Invalid cross-moment covariance', minimum, tolerance)
        roundoff[group['name']] = {'minimum_preclamp_v': minimum, 'roundoff_tolerance': tolerance}
        p[-2*count:].copy_(physical)
        m[-2*count:].copy_(physical_m)
        v[-2*count:].copy_(physical_v.clamp_min(0))
    return roundoff
