# -*- coding: utf-8 -*-
"""Shared training infrastructure for C29 screens."""
from __future__ import annotations
import math, torch
from gsplat import rasterization
W, H, TILE = 1920, 1080, 16
TW, TH = math.ceil(W / TILE), math.ceil(H / TILE)
DEV = "cuda"

class GaussianModel:
    def __init__(self, scene_data, sh_degree, device=DEV):
        self.sh_degree = sh_degree; self.device = device
        self.means = torch.nn.Parameter(scene_data["xyz"].detach().clone().to(device))
        self.quats = torch.nn.Parameter(torch.nn.functional.normalize(scene_data["rotations"].detach().clone().to(device), dim=-1))
        self.scales = torch.nn.Parameter(scene_data["scales"].detach().clone().to(device))
        self.opacities = torch.nn.Parameter(scene_data["opacity"].detach().clone().to(device))
        K = (sh_degree + 1) ** 2
        shs = scene_data["shs"].detach().clone().to(device)
        if shs.shape[-2] > K: shs = shs[..., :K, :].contiguous()
        self.shs = torch.nn.Parameter(shs)
    @property
    def n(self): return self.means.shape[0]
    def render(self, camera):
        sa, oa = torch.exp(self.scales), torch.sigmoid(self.opacities)
        r, _, _ = rasterization(means=self.means, quats=self.quats, scales=sa, opacities=oa, colors=self.shs,
            viewmats=camera.world_view_transform[None].contiguous(), Ks=camera.K[None].contiguous(),
            width=camera.image_width, height=camera.image_height, near_plane=.01, far_plane=1e10,
            radius_clip=0., eps2d=.3, sh_degree=self.sh_degree, packed=True, render_mode="RGB")
        return r[0].clamp(0, 1)
    def get_optimizer(self):
        return torch.optim.Adam([{"params":[self.means],"lr":1.6e-3},{"params":[self.quats],"lr":1e-3},
            {"params":[self.scales],"lr":5e-3},{"params":[self.opacities],"lr":5e-2},{"params":[self.shs],"lr":2.5e-3}])
    def densify_and_prune(self, step, ds=100, de=500, di=100, pi=100, gt=0.0002, po=0.005):
        if step<ds or step>de or step%di!=0: return {"densified":0,"pruned":0}
        nb=self.n
        with torch.no_grad():
            g=self.means.grad
            if g is None: return {"densified":0,"pruned":0}
            gn=g.norm(dim=-1); cm=gn>=gt; nc=cm.sum().item()
            if nc>0 and step<de//2:
                sn=self.scales.norm(dim=-1); sp=(sn>=0.01)&cm; sc=(sn<0.01)&cm
                if sc.any():
                    for k in ["means","quats","scales","opacities","shs"]:
                        p=getattr(self,k); setattr(self,k,torch.nn.Parameter(torch.cat([p,p[sc].detach()],0)))
                if sp.any():
                    n=sp.sum().item(); sm=self.means[sp].detach(); ss=self.scales[sp].detach()
                    sq=self.quats[sp].detach(); so=self.opacities[sp].detach(); sh=self.shs[sp].detach()
                    ns=torch.log(torch.exp(ss)/1.6); d=torch.randn(n,3,device=self.device); d=d/(d.norm(dim=-1,keepdim=True)+1e-8)
                    off=d*torch.exp(ss)*0.01
                    self.means=torch.nn.Parameter(torch.cat([self.means,sm+off,sm-off],0))
                    self.scales=torch.nn.Parameter(torch.cat([self.scales,ns,ns],0))
                    self.quats=torch.nn.Parameter(torch.cat([self.quats,sq,sq],0))
                    self.opacities=torch.nn.Parameter(torch.cat([self.opacities,so,so],0))
                    self.shs=torch.nn.Parameter(torch.cat([self.shs,sh,sh],0))
            if step%pi==0:
                op=torch.sigmoid(self.opacities).squeeze(-1); keep=op>=po
                if keep.any():
                    for k in ["means","quats","scales","opacities","shs"]:
                        p=getattr(self,k); setattr(self,k,torch.nn.Parameter(p[keep].detach()))
        return {"densified":max(0,self.n-nb),"pruned":max(0,nb-self.n)}
