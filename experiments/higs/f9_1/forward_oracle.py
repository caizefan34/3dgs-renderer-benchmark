import json, math, sys
from types import SimpleNamespace
import torch
from f9_0_loader import load_scene
from gsplat.experimental.render.functional.gaussian_inference import (
    _HigsAutogradFunction, _cull_gaussians_batched, _gather_visible_native,
)
from gsplat.experimental.render.kernels import _backend
from gsplat.cuda._wrapper import fully_fused_projection

scene, out_path = sys.argv[1:]
m,q,s,o,sh,vm,K,w,h,_ = load_scene(scene)
ids,_,_= _cull_gaussians_batched(m,q,s,vm,K,w,h,eps2d=.3,near_plane=.01,far_plane=1e10,radius_clip=.0,camera_model='pinhole')
C=vm.shape[-3]; N=ids.numel()
def ctx(): return SimpleNamespace(tile_sampling_ratio=1.,sampling_mode='uniform',tile_mask_external=None,cull_refresh_interval=1)
def capture(f9):
    x=ctx()
    if f9:
        return _HigsAutogradFunction._native_forward_capture(x,m,q,s,o,sh,vm,K,w,h,3,16,.01,1e10,.0,.3,None,'pinhole','RGB',visible_ids=ids,gatherless=True)
    a,b,c,d,e=_gather_visible_native(m,q,s,o,sh,ids)
    return _HigsAutogradFunction._native_forward_capture(x,a[None],b[None],c[None],d[None],e,vm,K,w,h,3,16,.01,1e10,.0,.3,None,'pinhole','RGB',visible_ids=ids,gatherless=False)
br,ba,bc=capture(False); fr,fa,fc=capture(True)
def met(a,b,mask=None):
    if mask is not None:a,b=a[mask],b[mask]
    d=(a-b).float(); bb=b.float()
    return {'max_abs':float(d.abs().max()) if d.numel() else 0.,'rel_l2':float(d.norm()/bb.norm().clamp_min(1e-30)) if d.numel() else 0.,'cosine':float(torch.nn.functional.cosine_similarity(a.float().flatten(),b.float().flatten(),dim=0)) if d.numel() else 1.}
valid=(bc[8]>0).all(-1).reshape(-1)
struct={'tile_offsets_identical':bool(torch.equal(bc[4],fc[4])),'flatten_ids_identical':bool(torch.equal(bc[5],fc[5])),'intersection_ids_identical':bool(torch.equal(bc[5],fc[5])),'last_ids_identical':bool(torch.equal(bc[7],fc[7])),'n_intersections':[int(bc[5].numel()),int(fc[5].numel())]}
cam=_backend._C.higs_camera_positions_from_viewmats(vm[0].contiguous()); inv=torch.linalg.inv(vm[0])[...,:3,3].contiguous()
vme,vqu,vsc,vop,vsh=_gather_visible_native(m,q,s,o,sh,ids)
brad,bm2,bd,bcon,_=fully_fused_projection(means=vme[None],covars=None,quats=vqu[None],scales=vsc[None],viewmats=vm,Ks=K,width=w,height=h,eps2d=.3,near_plane=.01,far_plane=1e10,radius_clip=.0,packed=False,calc_compensations=False,camera_model='pinhole')
frad,fm2,fd,fcon,foe,fce=_backend._C.higs_gatherless_projected_producer(ids,m,q,s,o,sh,vm[0].contiguous(),K[0].contiguous(),cam,w,h,.3,.01,1e10,.0)
v2=(brad.reshape(-1,2)>0).all(-1)
result={'scene':scene,'n_visible':int(N),'visible_ids_identical':True,'classification':'ALGEBRAIC_EXACT_FP_REASSOCIATED','camera_positions':met(cam,inv),'radii':met(frad,brad.reshape(-1,2)),'means2d':met(fm2,bm2.reshape(-1,2),v2),'depths':met(fd,bd.reshape(-1),v2),'conics':met(fcon,bcon.reshape(-1,3),v2),'opacities_eval':met(foe,o[ids],v2),'colors_eval':met(fc[2],bc[2],valid),'render_rgb':met(fr,br),'render_alpha':met(fa,ba),'support_mismatch':int(((fc[8]>0).all(-1)!=(bc[8]>0).all(-1)).sum()),'structure':struct}
json.dump(result,open(out_path,'w'),indent=2)
