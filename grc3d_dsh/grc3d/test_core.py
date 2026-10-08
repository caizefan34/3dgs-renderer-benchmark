import itertools
import unittest
import torch
from .core import Anchor, blocks, block_losses, corrected_loss, patch_loss, reference_step, probabilities
from .renderers import synthetic, render_reference

torch.set_num_threads(1)


class EstimatorTests(unittest.TestCase):
    def test_enumerated_nonuniform_gradient_and_variance(self):
        torch.manual_seed(9)
        theta = torch.tensor([.08, -.03, .11], dtype=torch.float64, requires_grad=True)
        af = torch.rand(7, 9, 3, dtype=theta.dtype)*.4+.2
        bf = torch.randn(7, 9, 3, 3, dtype=theta.dtype)*.1
        bc = torch.randn(3, 4, 3, 3, dtype=theta.dtype)*.1
        ac = torch.rand(3, 4, 3, dtype=theta.dtype)*.4+.2
        anchor = Anchor.capture(af, ac)
        target = torch.rand_like(af)*.4+.2
        partition = blocks(7, 9, 5)
        ff = block_losses(af+bf@theta, target, partition)
        pp = block_losses(anchor.proxy(ac+bc@theta), target, partition)
        pi = torch.tensor([.12, .37, .68, 1.], dtype=theta.dtype)
        grad = lambda loss: torch.autograd.grad(loss, theta, retain_graph=True)[0]
        gf, gp = grad(ff.sum()), grad(pp.sum())
        ds = torch.stack([grad(ff[i]-pp[i]) for i in range(len(partition))])
        expected = torch.zeros_like(theta); variance = torch.zeros((), dtype=theta.dtype)
        for bits in itertools.product([False, True], repeat=4):
            selected = torch.tensor(bits)
            mass = torch.where(selected, pi, 1-pi).prod()
            g = grad(corrected_loss(ff, pp, selected, pi))
            expected += mass*g
            variance += mass*(g-gf).square().sum()
        torch.testing.assert_close(expected, gf, atol=1e-12, rtol=1e-10)
        theory = (((1-pi)/pi)[:, None]*ds.square()).sum()
        torch.testing.assert_close(variance, theory, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(grad(corrected_loss(ff, pp, torch.ones(4, dtype=torch.bool), 1.)), gf)
        torch.testing.assert_close(grad(corrected_loss(ff, pp, torch.zeros(4, dtype=torch.bool), pi)), gp)

    def test_halo_loss_and_gradient_including_edges(self):
        torch.manual_seed(10)
        a = torch.rand(17, 19, 3, dtype=torch.float64, requires_grad=True)
        target = torch.rand_like(a); partition = blocks(17, 19, 7)
        all_parts = block_losses(a, target, partition)
        for i, b in enumerate(partition):
            s = b.halo(17, 19)
            local = patch_loss(a[s.y0:s.y1, s.x0:s.x1], target[s.y0:s.y1, s.x0:s.x1], b, s, 17, 19)
            torch.testing.assert_close(local, all_parts[i], atol=1e-13, rtol=1e-12)
            g1 = torch.autograd.grad(local, a, retain_graph=True)[0]
            g2 = torch.autograd.grad(all_parts[i], a, retain_graph=True)[0]
            torch.testing.assert_close(g1, g2, atol=1e-13, rtol=1e-11)

    def test_anchor_detached_and_probabilities_validated(self):
        f = torch.rand(8, 9, 3, requires_grad=True)
        c = torch.rand(2, 3, 3, requires_grad=True)
        a = Anchor.capture(f, c)
        torch.testing.assert_close(a.proxy(c), f)
        a.proxy(c+.01).sum().backward()
        self.assertIsNone(f.grad)
        self.assertIsNotNone(c.grad)
        for p in (0., -1., 1.1, float("nan"), [0.5]):
            with self.assertRaises(ValueError):
                probabilities(p, 4, f)

    def test_full_reference_step_and_3d_directional_derivative(self):
        p, old, camera, target = synthetic(height=12, width=16)
        h, w = target.shape[:2]
        render = lambda state, H, W: render_reference(state, camera, H, W)
        a = Anchor.capture(render(old, h, w), render(old, h//4, w//4))
        objective, info = reference_step(lambda: render(p, h, w), render(p, h//4, w//4),
                                          target, a, blocks(h, w, 8), 1.)
        baseline = block_losses(render(p, h, w), target, blocks(h, w, 8)).sum()
        keys = list(p); values = tuple(p.values())
        g = torch.autograd.grad(objective, values)
        gf = torch.autograd.grad(baseline, values)
        for x, y in zip(g, gf):
            torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-10)
        torch.manual_seed(3)
        directions = {k: torch.randn_like(v)*.05 for k, v in p.items()}
        def loss(sign):
            state = {k: v.detach()+sign*1e-5*directions[k] for k, v in p.items()}
            return block_losses(render(state, h, w), target, blocks(h, w, 8)).sum()
        fd = (loss(1)-loss(-1))/(2e-5)
        analytic = sum((g*directions[k]).sum() for k, g in zip(keys, gf))
        torch.testing.assert_close(fd, analytic, atol=1e-8, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
