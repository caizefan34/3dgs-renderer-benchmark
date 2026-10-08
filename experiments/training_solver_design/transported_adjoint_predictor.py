"""Frozen screen-space polynomial adjoint, reintegrated at current Gaussian shape.

Stores reference normalizing mass and whitening frame; this is an oracle helper,
not a memory-optimized cache implementation. No hidden current-image refresh.
"""

import math

import torch


@torch.no_grad()
def frame(conics, visible):
    a,b,c=torch.where(visible[:,None],conics,conics.new_tensor([1.,0.,1.])).unbind(-1)
    valid=visible & (a>0) & (a*c-b*b>0) & torch.isfinite(conics).all(-1)
    a,b,c=torch.where(valid,a,1.),torch.where(valid,b,0.),torch.where(valid,c,1.)
    l11=a.sqrt()
    l21=b/l11
    pivot=c-l21.square()
    valid=valid & (pivot>0)
    l22=torch.where(valid,pivot,1.).sqrt()
    return l11,l21,l22,valid


@torch.no_grad()
def capture_field(projected,conics,opacity,mean_gradient,conic_gradient,opacity_gradient,color_gradient,visible):
    l11,l21,l22,valid=frame(conics,visible)
    r11,r12,r22=1/l11,-l21/(l11*l22),1/l22
    gm,gq=torch.where(valid[:,None],mean_gradient,0.),torch.where(valid[:,None],conic_gradient,0.)
    m0=torch.where(valid,opacity*opacity_gradient,0.)
    b0,b1=r11*gm[:,0],r12*gm[:,0]+r22*gm[:,1]
    mxx,mxy,myy=-2*gq[:,0],-gq[:,1],-2*gq[:,2]
    h00=.5*(l11*l11*mxx+2*l11*l21*mxy+l21*l21*myy-m0)
    h01=.5*l22*(l11*mxy+l21*myy)
    h11=.5*(l22*l22*myy-m0)
    # The common 2*pi cancels in current/reference mass ratios.
    mass=torch.where(valid,opacity/(l11*l22),1.)
    return {'anchor':torch.where(valid[:,None],projected,0.).clone(),
            'precision_frame':(l11.clone(),l21.clone(),l22.clone()),
            'polynomial':tuple(x.clone() for x in (m0,b0,b1,h00,h01,h11)),
            'mass':mass.clone(),'color_gradient':torch.where(valid[:,None],color_gradient,0.).clone(),
            'valid':valid.clone()}


@torch.no_grad()
def transported_nodes(field,projected,conics,opacity,visible):
    l11,l21,l22,valid=frame(conics,visible)
    valid=valid & field['valid'] & (opacity>0)
    r11,r12,r22=1/l11,-l21/(l11*l22),1/l22
    ratio=torch.where(valid,(opacity/(l11*l22))/field['mass'],0.)
    xy=torch.where(valid[:,None],projected,0.)
    shift=xy-field['anchor']
    a0,a1,a2=field['precision_frame']
    m0,b0,b1,h00,h01,h11=field['polynomial']
    safe_o=torch.where(valid,opacity,1.)
    nodes=[]
    for zx,wx in ((-math.sqrt(3),1/6),(0.,2/3),(math.sqrt(3),1/6)):
        for zy,wy in ((-math.sqrt(3),1/6),(0.,2/3),(math.sqrt(3),1/6)):
            weight=wx*wy
            dx,dy=r11*zx+r12*zy,r22*zy
            old_zx=a0*(shift[:,0]+dx)+a1*(shift[:,1]+dy)
            old_zy=a2*(shift[:,1]+dy)
            mass=weight*ratio*(m0+b0*old_zx+b1*old_zy+h00*(old_zx.square()-1)+2*h01*old_zx*old_zy+h11*(old_zy.square()-1))
            mass=torch.where(valid,mass,0.)
            gm=torch.stack((mass*l11*zx,mass*(l21*zx+l22*zy)),-1)
            gq=torch.stack((-.5*mass*dx*dx,-mass*dx*dy,-.5*mass*dy*dy),-1)
            nodes.append((torch.stack((dx,dy),-1),gm,gq,mass/safe_o,weight))
    gc=field['color_gradient']*ratio[:,None]
    return nodes,gc,valid
