"""State transfer versus independent physical gradient histories."""

import torch

from newborn_updates import step_with_newborn_policy
from sibling_coordinates import decode_raw
from sibling_transport import accumulate_cross, fold_to_physical


def test_old_clock_handoff_matches_zero_extended_physical_history():
    torch.manual_seed(19)
    count, survivors = 3, 4
    p = torch.nn.Parameter(torch.randn(survivors+2*count, 5, dtype=torch.float64))
    q = torch.nn.Parameter(decode_raw(p.detach(), count).clone())
    joined = torch.optim.Adam([{'params': [p], 'name': 'raw'}], lr=.03)
    reference = torch.optim.Adam([q], lr=.03)
    m = torch.randn_like(p)*.1
    v = m.square()+.2
    m[-count:] = 0
    v[-count:] = 0
    mq, vq = m.clone(), v.clone()
    mq[-2*count:] = (m[-2*count:-count]*.5).repeat(2, 1)
    vq[-2*count:] = (v[-2*count:-count]*.25).repeat(2, 1)
    for opt, param, avg, square in ((joined, p, m, v), (reference, q, mq, vq)):
        opt.state[param] = {'step': torch.tensor(1985.), 'exp_avg': avg, 'exp_avg_sq': square}
    cross = {'raw': torch.zeros_like(p[-count:])}
    for age in range(1, 33):
        p.grad = torch.randn_like(p)
        g = p.grad
        q.grad = torch.cat((g[:-2*count], (g[-2*count:-count]+g[-count:])*.5,
                            (g[-2*count:-count]-g[-count:])*.5))
        accumulate_cross(joined, count, cross)
        # Distinct bias clocks alter parameter trajectories, not EMA algebra.
        step_with_newborn_policy(joined, survivors+count, age, 'fresh_age', 1.)
        reference.step()
    joined.zero_grad(set_to_none=True)
    before = decode_raw(p.detach(), count).clone()
    survivor_m = joined.state[p]['exp_avg'][:survivors].clone()
    fold_to_physical(joined, count, cross)
    torch.testing.assert_close(p, before, rtol=0, atol=0)
    torch.testing.assert_close(joined.state[p]['exp_avg'][:survivors], survivor_m, rtol=0, atol=0)
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(joined.state[p][key], reference.state[q][key])


def test_diagonal_moments_cannot_determine_physical_variances():
    # Equal diagonal second moments, opposite cross signs: different child states.
    plus = torch.tensor([1., 1.], dtype=torch.float64)
    same, opposite = plus.clone(), -plus
    assert torch.equal(same.square(), opposite.square())
    v_same = ((plus+same)*.5).square()
    v_opposite = ((plus+opposite)*.5).square()
    assert not torch.equal(v_same, v_opposite)
