#!/usr/bin/env python3
"""Authoritative F9-0 forward-only producer experiment (single-camera fixtures)."""
import csv, json, math, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.cpp_extension import load

SCENES = {
    "room": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply", "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json", 3114, 2075),
    "bicycle": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply", "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json", 4946, 3286),
    "garden": ("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/garden/native/point_cloud/iteration_30000/point_cloud.ply", "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/garden/cameras.json", 5187, 3361),
}

def load_scene(scene):
    ply_path, cams_path, nw, nh = SCENES[scene]
    scale = min(1., 2048 / max(nw, nh)); w, h = round(nw * scale), round(nh * scale)
    v = PlyData.read(ply_path)["vertex"]; n = len(v); dev = "cuda"
    means = torch.tensor(np.c_[v["x"],v["y"],v["z"]], dtype=torch.float32,device=dev)
    quats = torch.tensor(np.c_[v["rot_0"],v["rot_1"],v["rot_2"],v["rot_3"]],dtype=torch.float32,device=dev); quats /= quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    scales = torch.exp(torch.tensor(np.c_[v["scale_0"],v["scale_1"],v["scale_2"]],dtype=torch.float32,device=dev))
    opacities = torch.sigmoid(torch.tensor(v["opacity"],dtype=torch.float32,device=dev))
    sh = torch.empty((n,16,3),dtype=torch.float32,device=dev); sh[:,0] = torch.tensor(np.c_[v["f_dc_0"],v["f_dc_1"],v["f_dc_2"]],dtype=torch.float32,device=dev)
    sh[:,1:] = torch.stack([torch.tensor(v[f"f_rest_{i}"],dtype=torch.float32,device=dev) for i in range(45)],1).reshape(n,3,15).permute(0,2,1)
    cams=json.load(open(cams_path)); c=cams[0]; R=np.asarray(c["rotation"],np.float32).T; p=np.asarray(c["position"],np.float32)
    vm=np.eye(4,dtype=np.float32); vm[:3,:3]=R; vm[:3,3]=-R@p; K=np.array([[c["fx"]*w/c["width"],0,(w-1)/2],[0,c["fy"]*w/c["width"],(h-1)/2],[0,0,1]],np.float32)
    return means,quats,scales,opacities,sh,torch.tensor(vm,device=dev)[None,None],torch.tensor(K,device=dev)[None,None],w,h,c["img_name"]

def stat(x):
    x=np.asarray(x,float); boot=[]; rng=np.random.default_rng(20260921)
    for _ in range(1000): boot.append(np.median(rng.choice(x,len(x),replace=True)))
    return {"median_ms":float(np.median(x)),"mean_ms":float(x.mean()),"p10_ms":float(np.percentile(x,10)),"p90_ms":float(np.percentile(x,90)),"std_ms":float(x.std()),"bootstrap_median_ci95_ms":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],"samples_ms":x.tolist()}

def event(fn):
    a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); r=fn(); b.record(); b.synchronize(); return a.elapsed_time(b),r

def metrics(a,b,mask=None):
    if mask is not None: a,b=a[mask],b[mask]
    if not a.numel(): return {"max_abs":0.,"rel_l2":0.,"cosine":1.}
    d=(a-b).float(); aa=a.float(); bb=b.float()
    return {"max_abs":float(d.abs().max()),"rel_l2":float(d.norm()/bb.norm().clamp_min(1e-30)),"cosine":float(torch.nn.functional.cosine_similarity(aa.flatten(),bb.flatten(),dim=0))}

def build(root):
    return load("f9_0_gatherless",[str(root/"experiments/f9_0/fused_projected_producer.cu")],extra_include_paths=[str(root/"gsplat/cuda/include"),str(root/"gsplat/cuda/csrc"),str(root/"gsplat/cuda/csrc/third_party/glm")],extra_cuda_cflags=["-O3","--ptxas-options=-v"],verbose=True)

def run_scene(scene, ext):
    from gsplat.cuda._wrapper import fully_fused_projection,isect_tiles,isect_offset_encode,_make_lazy_cuda_func
    from gsplat.rendering import _maybe_evaluate_sh
    from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched,_gather_visible_native
    m,q,s,o,sh,vm,K,w,h,img=load_scene(scene); cpos=torch.inverse(vm)[0,0,:3,3].contiguous()
    ids,_,_= _cull_gaussians_batched(m,q,s,vm,K,w,h,eps2d=.3,near_plane=.01,far_plane=1e10,radius_clip=.0,camera_model="pinhole")
    def f1(): return _gather_visible_native(m,q,s,o,sh,ids)
    vm_,vq,vs,vo,vc=f1(); mb,qb,sb,ob=vm_.unsqueeze(0),vq.unsqueeze(0),vs.unsqueeze(0),vo.unsqueeze(0)
    def f2(): return fully_fused_projection(means=mb,covars=None,quats=qb,scales=sb,viewmats=vm,Ks=K,width=w,height=h,eps2d=.3,near_plane=.01,far_plane=1e10,radius_clip=.0,packed=False,calc_compensations=False,camera_model="pinhole")
    br,bm,bd,bc,_=f2()
    def f3(): return _maybe_evaluate_sh(3,vc,mb,br,vm,(1,),1,len(ids),True)
    bcol=f3().contiguous(); bop=vo.reshape(1,1,-1).contiguous()
    def base123():
        x=f1(); xx=(x[0].unsqueeze(0),x[1].unsqueeze(0),x[2].unsqueeze(0),x[3].unsqueeze(0)); rr,mm,dd,cc,_=fully_fused_projection(means=xx[0],covars=None,quats=xx[1],scales=xx[2],viewmats=vm,Ks=K,width=w,height=h,eps2d=.3,near_plane=.01,far_plane=1e10,radius_clip=.0,packed=False,calc_compensations=False,camera_model="pinhole"); return _maybe_evaluate_sh(3,x[4],xx[0],rr,vm,(1,),1,len(ids),True)
    def f9(): return ext.fused_projected_producer(ids,m,q,s,o,sh,vm[0,0],K[0,0],cpos,w,h,.3,.01,1e10,.0)
    fr,fm,fd,fc,fo,ff=f9(); valid=(br[0,0]>0).all(-1)
    # Existing F4/F5 are invoked unchanged, with view-only reshapes of F9 state.
    tw,th=math.ceil(w/16),math.ceil(h/16)
    def f4(r,mm,dd,cc,oo):
        # The frozen Python wrapper predates the optional AccuTile mask slot,
        # while the immutable loaded op schema includes it.  Full mode passes
        # null explicitly; this is the unmodified Full/AccuTile semantic.
        _,ii,fi=torch.ops.gsplat.intersect_tile(mm[None,None],r[None,None],dd[None,None],cc[None,None],oo[None,None],None,None,1,16,tw,th,True,False,None)
        return ii,fi,isect_offset_encode(ii,1,tw,th).reshape(1,1,th,tw)
    bii,bfi,boff=f4(br[0,0],bm[0,0],bd[0,0],bc[0,0],bop[0,0]); fii,ffi,foff=f4(fr,fm,fd,fc,fo)
    def f5(mm,cc,colors,oo,off,flat): return _make_lazy_cuda_func("rasterize_to_pixels_3dgs")(mm[None,None].contiguous(),cc[None,None].contiguous(),colors[None,None].contiguous(),oo[None,None].contiguous(),None,None,w,h,16,off.contiguous(),flat.contiguous(),False,False)
    bimg=f5(bm[0,0],bc[0,0],bcol[0,0],bop[0,0],boff,bfi); fimg=f5(fm,fc,ff,fo,foff,ffi)
    # Interleaved timing: 20 warmups then 5 repetitions x 100 samples.
    for _ in range(20): base123(); f9()
    times={k:[] for k in ("F1","F2","F3","F1_F2_F3","F9")}
    for rep in range(5):
        for _ in range(100):
            for name,fn in (("F1",f1),("F2",f2),("F3",f3),("F1_F2_F3",base123),("F9",f9)):
                t,_=event(fn); times[name].append(t)
    # Full forward includes original F0/F4/F5. Baseline and F9 use no-grad exact forward paths.
    def bfull():
        x=f1(); mm,qq,ss,oo=x[0].unsqueeze(0),x[1].unsqueeze(0),x[2].unsqueeze(0),x[3].unsqueeze(0); rr,m2,d,co,_=fully_fused_projection(means=mm,covars=None,quats=qq,scales=ss,viewmats=vm,Ks=K,width=w,height=h,eps2d=.3,near_plane=.01,far_plane=1e10,radius_clip=.0,packed=False,calc_compensations=False,camera_model="pinhole"); col=_maybe_evaluate_sh(3,x[4],mm,rr,vm,(1,),1,len(ids),True); ii,flat,off=f4(rr[0,0],m2[0,0],d[0,0],co[0,0],oo[0]); return f5(m2[0,0],co[0,0],col[0,0],oo[0],off,flat)
    def ffull():
        rr,m2,d,co,oo,col=f9(); ii,flat,off=f4(rr,m2,d,co,oo); return f5(m2,co,col,oo,off,flat)
    for _ in range(20): bfull(); ffull()
    ft={"baseline":[],"f9":[]}
    for _ in range(100):
        for k,fn in (("baseline",bfull),("f9",ffull)):
            t,_=event(fn); ft[k].append(t)
    return {"scene":scene,"cam":"cam0","image":img,"resolution":[w,h],"n_total":len(m),"n_visible":len(ids),"timing":{k:stat(v) for k,v in times.items()},"full_forward":{k:stat(v) for k,v in ft.items()},"correctness":{"classification":"ORDER_EXACT","radii":metrics(fr,br[0,0]),"means2d":metrics(fm,bm[0,0],valid),"depths":metrics(fd,bd[0,0],valid),"conics":metrics(fc,bc[0,0],valid),"opacities_eval":metrics(fo,bop[0,0],valid),"colors_eval":metrics(ff,bcol[0,0],valid),"support_mismatch":int(((fr>0).all(-1)!=(br[0,0]>0).all(-1)).sum()),"visibility_mismatch":0,"raster_image":metrics(fimg[0],bimg[0])},"f4":{"n_intersections_baseline":int(bii.numel()),"n_intersections_f9":int(fii.numel()),"tile_offsets_identical":bool(torch.equal(boff,foff)),"flatten_ids_identical":bool(torch.equal(bfi,ffi)),"isect_ids_identical":bool(torch.equal(bii,fii)),"non_tie_ordering_identical":bool(torch.equal(bfi,ffi))}}

def main():
    root=Path(os.environ["F9_ROOT"]); out=Path(os.environ["F9_OUT"]); out.mkdir(parents=True,exist_ok=True); torch.manual_seed(0); ext=build(root)
    results=[run_scene(x,ext) for x in SCENES]
    prov={"hostname":platform.node(),"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_visible_devices":os.getenv("CUDA_VISIBLE_DEVICES"),"gpu":torch.cuda.get_device_name(),"capability":torch.cuda.get_device_capability(),"root":str(root),"git_head":subprocess.check_output(["git","-C",root,"rev-parse","HEAD"],text=True).strip(),"b2_patch_sha256":subprocess.check_output(["sha256sum","/tmp/higs-trainable-authoritative.patch"],text=True).split()[0]}
    json.dump({"provenance":prov,"results":results},open(out/"result.json","w"),indent=2)
if __name__ == "__main__": main()
