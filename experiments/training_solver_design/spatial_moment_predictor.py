"""Gaussian-Hermite spatial gradient predictor, not a deployed training method.

Reconstruct a quadratic effective adjoint from opacity/mean/conic gradient
moments. A 3x3 quadrature preserves these total moments while distributing signed
contributions spatially. Fresh full-image moments are only an opportunity oracle.
"""

import math

import torch


@torch.no_grad()
def spatial_nodes(conics, opacity, mean_gradient, conic_gradient, opacity_gradient, visible):
    a,b,c=torch.where(visible[:,None],conics,conics.new_tensor([1.,0.,1.])).unbind(-1)
    valid=visible & (a>0) & (a*c-b*b>0) & torch.isfinite(conics).all(-1)
    a=torch.where(valid,a,1.)
    b=torch.where(valid,b,0.)
    c=torch.where(valid,c,1.)
    l11=a.sqrt()
    l21=b/l11
    pivot=c-l21.square()
    valid=valid & (pivot>0)
    l22=torch.where(valid,pivot,1.).sqrt()
    r11=1/l11
    r12=-l21/(l11*l22)
    r22=1/l22
    gm=torch.where(valid[:,None],mean_gradient,0.)
    gq=torch.where(valid[:,None],conic_gradient,0.)
    go=torch.where(valid,opacity_gradient,0.)
    safe_o=torch.where(valid,opacity,1.)
    m0=safe_o*go
    b0=r11*gm[:,0]
    b1=r12*gm[:,0]+r22*gm[:,1]
    mxx,mxy,myy=-2*gq[:,0],-gq[:,1],-2*gq[:,2]
    h00=.5*(l11*l11*mxx+2*l11*l21*mxy+l21*l21*myy-m0)
    h01=.5*l22*(l11*mxy+l21*myy)
    h11=.5*(l22*l22*myy-m0)
    nodes=[]
    for zx,wx in ((-math.sqrt(3),1/6),(0.,2/3),(math.sqrt(3),1/6)):
        for zy,wy in ((-math.sqrt(3),1/6),(0.,2/3),(math.sqrt(3),1/6)):
            weight=wx*wy
            dx,dy=r11*zx+r12*zy,r22*zy
            mass=weight*(m0+b0*zx+b1*zy+h00*(zx*zx-1)+2*h01*zx*zy+h11*(zy*zy-1))
            gmean=torch.stack((mass*l11*zx,mass*(l21*zx+l22*zy)),-1)
            gconic=torch.stack((-.5*mass*dx*dx,-mass*dx*dy,-.5*mass*dy*dy),-1)
            gopacity=mass/safe_o
            nodes.append((torch.stack((dx,dy),-1),gmean,gconic,gopacity,weight))
    return nodes,valid


@torch.no_grad()
def allocate_nodes(projected, nodes, color_gradient, valid, labels, height, width):
    rows,columns=labels.shape
    out=[(torch.zeros_like(projected),torch.zeros_like(nodes[0][2]),
          torch.zeros_like(nodes[0][3]),torch.zeros_like(color_gradient)) for _ in range(4)]
    xy=torch.where(valid[:,None],projected,0.)
    gc=torch.where(valid[:,None],color_gradient,0.)
    for delta,gm,gq,go,weight in nodes:
        location=xy+delta
        col=(location[:,0]/(width/columns)).floor().long().clamp(0,columns-1)
        row=(location[:,1]/(height/rows)).floor().long().clamp(0,rows-1)
        group=labels[row,col]
        for k in range(4):
            mask=(group==k) & valid
            out[k][0].add_(gm*mask[:,None])
            out[k][1].add_(gq*mask[:,None])
            out[k][2].add_(go*mask)
            out[k][3].add_(weight*gc*mask[:,None])
    return out
