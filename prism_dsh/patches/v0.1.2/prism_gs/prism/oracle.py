"""Tiny full-support renderer for CPU math tests, NOT gsplat parity/performance."""
import torch
import torch.nn.functional as F
from .geometry import project,screen_unpack


def render(z, depth, colors, opacity, height, width):
    uv,conic=screen_unpack(z)
    yy,xx=torch.meshgrid(torch.arange(height,device=z.device,dtype=z.dtype)+.5,
        torch.arange(width,device=z.device,dtype=z.dtype)+.5,indexing='ij')
    d=torch.stack((xx,yy),-1)[None]-uv[:,None,None]
    power=(d[...,0].square()*conic[:,0,None,None]
           +2*d[...,0]*d[...,1]*conic[:,1,None,None]
           +d[...,1].square()*conic[:,2,None,None])
    alpha=(opacity[:,None,None]*(-.5*power).exp()).clamp_max(.99)
    t=torch.ones_like(xx); out=torch.zeros(height,width,3,device=z.device,dtype=z.dtype)
    for i in depth.detach().argsort().tolist():
        out=out+(t*alpha[i])[...,None]*colors[i]
        t=t*(1-alpha[i])
    return out


def loss(pred,target):
    """CPU oracle: L1 + Gaussian-window SSIM. Not the production fused kernel."""
    p=pred.permute(2,0,1)[None]; t=target.permute(2,0,1)[None]
    v=torch.arange(11,device=p.device,dtype=p.dtype)-5
    g=torch.exp(-v.square()/(2*1.5**2));g=g/g.sum()
    kernel=(g[:,None]*g[None,:]).expand(3,1,11,11)
    conv=lambda a:F.conv2d(a,kernel,padding=5,groups=3)
    mp,mt=conv(p),conv(t)
    vp,vt=conv(p*p)-mp*mp,conv(t*t)-mt*mt
    ct=conv(p*t)-mp*mt
    ssim=((2*mp*mt+.01**2)*(2*ct+.03**2)/
          ((mp*mp+mt*mt+.01**2)*(vp+vt+.03**2))).mean()
    return .8*(p-t).abs().mean()+.2*(1-ssim)


def fixture(seed=7):
    torch.manual_seed(seed); dtype=torch.float64
    means=torch.tensor([[-.25,0,2.5],[.28,.12,2.9],[0,-.24,3.3]],dtype=dtype)
    cov=torch.diag_embed(torch.tensor([[.10,.12,.13],[.14,.10,.12],[.12,.16,.10]],dtype=dtype).square())
    from .geometry import pack_geometry
    x=pack_geometry(means,cov)
    views=torch.eye(4,dtype=dtype).repeat(5,1,1)
    views[:,:3,3]=torch.tensor([[.4,0,0],[-.4,0,0],[0,.3,0],[0,-.3,0],[.2,.2,.05]],dtype=dtype)
    ks=torch.tensor([[32.,0,16],[0,32,12],[0,0,1]],dtype=dtype).repeat(5,1,1)
    colors=torch.tensor([[.9,.2,.1],[.2,.8,.3],[.1,.3,.9]],dtype=dtype)
    opacity=torch.tensor([.7,.6,.8],dtype=dtype)
    return x,views,ks,colors,opacity,32,24
