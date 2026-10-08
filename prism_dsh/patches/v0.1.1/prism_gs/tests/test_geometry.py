import unittest
import torch
from prism.geometry import *
from prism.oracle import fixture,render,loss


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x,self.v,self.k,self.c,self.o,self.w,self.h=fixture()

    def test_roundtrip_spd_and_screen(self):
        m,c=unpack_geometry(self.x)
        torch.testing.assert_close(pack_geometry(m,c),self.x)
        z,_=project(self.x,self.v,self.k,self.w,self.h)
        torch.testing.assert_close(screen_pack(*screen_unpack(z)),z)
        cov=conic_to_cov(screen_unpack(z)[1])
        torch.testing.assert_close(conic_to_cov(cov_to_conic(cov)),cov)

    def test_jacobian_finite_difference(self):
        z,j=jacobian(self.x,self.v,self.k,self.w,self.h)
        direction=torch.randn_like(self.x)*.01; eps=1e-5
        plus=project(self.x+eps*direction,self.v,self.k,self.w,self.h)[0]
        minus=project(self.x-eps*direction,self.v,self.k,self.w,self.h)[0]
        torch.testing.assert_close((plus-minus)/(2*eps),torch.einsum('vnkp,np->vnk',j,direction),atol=1e-8,rtol=1e-5)

    def test_consensus_recovers_feasible_targets(self):
        truth=self.x.clone();truth[:,:3]+=torch.tensor([.012,-.008,.015]);truth[:,3:]+=.005
        target=project(truth,self.v,self.k,self.w,self.h)[0]
        initial=project(self.x,self.v,self.k,self.w,self.h)[0]
        fit,hist=fit_consensus(self.x,target,self.v,self.k,self.w,self.h,torch.ones_like(target),steps=8,damping=1e-7)
        after=project(fit,self.v,self.k,self.w,self.h)[0]
        self.assertLess(float((after-target).square().sum()),float((initial-target).square().sum())*1e-5)
        self.assertTrue(all(h['objective_after']<=h['objective_before']+1e-12 for h in hist))

    def test_incompatible_demand_has_normal_residual(self):
        z,j=jacobian(self.x,self.v,self.k,self.w,self.h)
        d=torch.randn_like(z); weights=torch.ones_like(d)
        delta,residual,s=decompose(j,d,weights,damping=1e-10)
        self.assertGreater(float(residual.square().sum()),.1)
        orth=torch.einsum('vnkp,vnk->np',j,residual)
        self.assertLess(float(orth.abs().max()),1e-6)

    def test_zero_weight_views_do_not_affect_fit(self):
        target=project(self.x,self.v,self.k,self.w,self.h)[0]
        target[-1]+=1000
        weights=torch.ones_like(target);weights[-1]=0
        result,_=fit_consensus(self.x,target,self.v,self.k,self.w,self.h,weights)
        torch.testing.assert_close(result,self.x)

    def test_lifted_and_direct_chain_gradient(self):
        x=self.x.clone().requires_grad_()
        z,dep=project(x,self.v,self.k,self.w,self.h)
        truth=render(z[0].detach()+.01,dep[0],self.c,self.o,self.h,self.w).detach()
        l=loss(render(z[0],dep[0],self.c,self.o,self.h,self.w),truth)
        direct=torch.autograd.grad(l,x,retain_graph=True)[0]
        screen=torch.autograd.grad(l,z,retain_graph=True)[0]
        chain=torch.autograd.grad(z,x,grad_outputs=screen)[0]
        torch.testing.assert_close(direct,chain,atol=1e-12,rtol=1e-10)

    def test_same_camera_geometry_is_rank_deficient(self):
        _,j=jacobian(self.x,self.v[:1].repeat(4,1,1),self.k[:4],self.w,self.h)
        s=torch.linalg.svdvals(j.permute(1,0,2,3).reshape(3,20,9))
        self.assertLess(float(s[:,-1].max()),1e-10)


if __name__=='__main__': unittest.main()
