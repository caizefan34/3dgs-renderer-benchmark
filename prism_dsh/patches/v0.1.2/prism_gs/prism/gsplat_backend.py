"""GPU diagnostic adapter. Every installed backend MUST pass run_gpu M0.

Supports only one pinhole camera, classic RGB, packed=False, black background.
Projection/SH semantic drift is detected, not patched by changing the baseline.
Reference implementation rebuilds intersections on every lifted render.
"""
import inspect
import math
import torch
import torch.nn.functional as F
from .geometry import screen_unpack,conic_to_cov


def baseline(params,camera,degree,covars=None,means=None):
    from gsplat import rasterization
    kw=dict(means=params['means'] if means is None else means,
        quats=F.normalize(params['quats'],dim=-1),scales=params['log_scales'].exp(),
        opacities=params['opacity_logits'].sigmoid(),
        colors=torch.cat((params['sh0'],params['shN']),1),
        viewmats=camera['viewmat'][None],Ks=camera['K'][None],
        width=camera['width'],height=camera['height'],sh_degree=degree,
        eps2d=.3,radius_clip=0.,tile_size=16,packed=False,absgrad=True,
        render_mode='RGB',rasterize_mode='classic')
    if covars is not None: kw['covars']=covars
    image,alpha,meta=rasterization(**kw)
    return image[0].clamp(0,1),meta


def colors_for(params,camera,degree,mask,means=None):
    from gsplat.cuda._wrapper import spherical_harmonics
    means=params['means'] if means is None else means
    sh=torch.cat((params['sh0'],params['shN']),1)
    if 'means' in inspect.signature(spherical_harmonics).parameters:
        out=spherical_harmonics(degree,means,camera['viewmat'][None],sh,masks=mask)
    else:
        view=camera['viewmat'];center=-(view[:3,:3].T@view[:3,3])
        out=spherical_harmonics(degree,(means-center)[None],sh[None],masks=mask)
    # P2: strict activation, matching the kernel's `> 0` gradient mask at exactly 0.
    return (out+.5).relu()


def state(params,camera,degree,covars=None,means=None):
    image,meta=baseline(params,camera,degree,covars,means)
    if meta['radii'].ndim!=3 or meta['radii'].shape[-1]!=2:
        raise RuntimeError('Unsupported radii layout; requires [1,N,2]. Do not silently adapt.')
    mask=(meta['radii']>0).any(-1)
    colors=colors_for(params,camera,degree,mask,means)
    return image,dict(meta=meta,colors=colors,camera=camera,mask=mask)


def features(st,ids):
    meta=st['meta']
    # P1: the renderer's native conic is carried through unchanged, so an
    # override of these values re-renders bitwise (zero-displacement gate).
    return torch.cat((meta['means2d'][0,ids],meta['conics'][0,ids]),-1)


def draw(st,ids=None,z=None,rebuild=False):
    from gsplat.cuda._wrapper import isect_tiles,isect_offset_encode,rasterize_to_pixels
    m=st['meta'];c=st['camera'];tile=16
    uv=m['means2d'];co=m['conics'];radii=m['radii'];depth=m['depths']
    opacity=m['opacities']
    if opacity.ndim==1:opacity=opacity[None]
    if ids is not None:
        uv=uv.clone();co=co.clone();radii=radii.clone()
        new_uv,new_conic=screen_unpack(z)
        uv[0,ids]=new_uv;co[0,ids]=new_conic
        # Pinned EWA bounds: opacity-aware 3.33 sigma (Config.h must match).
        new_cov=conic_to_cov(new_conic)
        extend=(2*torch.log((opacity[0,ids]*255).clamp_min(1))).sqrt().clamp_max(3.33)
        rr=torch.ceil(extend[:,None]*new_cov.diagonal(dim1=-2,dim2=-1).sqrt()).to(radii.dtype)
        inside=(new_uv[:,0]+rr[:,0]>0)&(new_uv[:,0]-rr[:,0]<c['width'])&\
               (new_uv[:,1]+rr[:,1]>0)&(new_uv[:,1]-rr[:,1]<c['height'])
        rr=torch.where(inside[:,None],rr,torch.zeros_like(rr))
        radii[0,ids]=rr
        rebuild=True
    if rebuild:
        tw=math.ceil(c['width']/tile);th=math.ceil(c['height']/tile)
        # Use ordinary bbox intersections. M0 additionally checks reconstructed
        # zero displacement; neither outdated indices nor unchanged K are used.
        _,isect,flat=isect_tiles(uv,radii,depth,tile,tw,th,packed=False)
        offsets=isect_offset_encode(isect,1,tw,th)
    else:
        flat=m['flatten_ids'];offsets=m['isect_offsets']
    out,_=rasterize_to_pixels(uv,co,st['colors'],opacity,c['width'],c['height'],
        tile,offsets,flat,packed=False,absgrad=True)
    return out[0].clamp(0,1)


def detach_state(st):
    return dict(st,meta={k:(v.detach() if torch.is_tensor(v) else v) for k,v in st['meta'].items()},
                colors=st['colors'].detach())
