import csv, math, sys, numpy as np, torch
from f9_0_loader import load_scene
from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched
from gsplat.experimental.render.kernels import _backend
from gsplat.cuda._wrapper import _make_lazy_cuda_func, isect_offset_encode

scene,out=sys.argv[1:]
m,q,s,o,sh,vm,K,w,h,_=load_scene(scene); C=vm.shape[-3]
ids,_,_=_cull_gaussians_batched(m,q,s,vm,K,w,h,eps2d=.3,near_plane=.01,far_plane=1e10,radius_clip=.0,camera_model='pinhole')
tw,th=math.ceil(w/16),math.ceil(h/16); ext=_backend._C
def f0():return _cull_gaussians_batched(m,q,s,vm,K,w,h,eps2d=.3,near_plane=.01,far_plane=1e10,radius_clip=.0,camera_model='pinhole')[0]
def cam():return ext.higs_camera_positions_from_viewmats(vm[0].contiguous())
pos=cam()
def f9():return ext.higs_gatherless_projected_producer(ids,m,q,s,o,sh,vm[0].contiguous(),K[0].contiguous(),pos,w,h,.3,.01,1e10,.0)
r,mm,d,co,oe,ce=f9()
def f4():
 return torch.ops.gsplat.intersect_tile(mm[None,None],r[None,None],d[None,None],co[None,None],oe[None,None],None,None,C,16,tw,th,True,False,None)
_,ii,flat=f4(); offs=isect_offset_encode(ii,C,tw,th).reshape(1,C,th,tw)
def f5():return _make_lazy_cuda_func('rasterize_to_pixels_3dgs')(mm[None,None].contiguous(),co[None,None].contiguous(),ce[None,None].contiguous(),oe[None,None].contiguous(),None,None,w,h,16,offs.contiguous(),flat.contiguous(),False,False)
def ev(fn):
 a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record();z=fn();b.record();b.synchronize();return a.elapsed_time(b)
ops={'F0':f0,'camera_position':cam,'F9':f9,'F4':f4,'F5':f5}
for _ in range(20):
 for fn in ops.values():fn()
rows=[]
for name,fn in ops.items():
 x=[ev(fn) for _ in range(500)]; rows.append({'scene':scene,'stage':name,'median_ms':float(np.median(x)),'mean_ms':float(np.mean(x)),'p10_ms':float(np.percentile(x,10)),'p90_ms':float(np.percentile(x,90)),'std_ms':float(np.std(x)),'n':len(x)})
with open(out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
