import sys
sys.path.insert(0, "/root/epic05_n1"); sys.path.insert(0, "/root/epic05_frozen/higs_c0_final30k_worktree"); sys.path.insert(0, "/root/epic05_build_cache/scene/gsplat_scene_cuda"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch, numpy as np
import n1_core as C
import cg0_prereg as P
import cg0_compiler as COM
import cg0_appearance as APP
from cg0_run import render_depth
C.bootstrap_c0()
scene="room"; it=15000
ds=C.load_scene_dataset(scene); n=len(ds)
split=P.build_split(n)
ck=C.load_checkpoint(scene,it); Ns=int(ck["xyz"].shape[0]); ss=float(ck["spatial_lr_scale"])
src_shd=int(ck["active_sh_degree"]); src_model=C.build_model(ck)
compile_idx=split["compile64"]
dm={}
for ci in compile_idx:
    cam=ds.get_camera(ci); d,_=render_depth(src_model,cam,src_shd); dm[ci]=d
C.scene_reset()
def gt_get(i): return ds.get_gt_image(i)
cams_c=[ds.get_camera(i) for i in compile_idx]
cand=COM.generate_candidates(dm,cams_c,compile_idx,gt_get,"cuda")
h,cnt,labels,order=COM.fuse_count(cand,Ns,ss)
ncl=int(labels.max().item())+1
cam=cams_c[0]; ci0=compile_idx[0]; D=dm[ci0]
cidx=cand["cidx"][order]; pix=cand["pix"][order]; Xall=cand["X"][order]
# find single-camera clusters in frame
u0,v0,z0,inf0=APP._project(cam, cand["mu"] if False else None) if False else (None,None,None,None)
# use cluster mu
mu=torch.zeros(ncl,3,device="cuda")
cntc=torch.bincount(labels,minlength=ncl).unsqueeze(1)
mu.index_add_(0,labels,Xall); mu=mu/cntc
u,v,z,inf=APP._project(cam, mu)
cams_per=torch.bincount((labels*100000+cidx)//100000, minlength=ncl)
single=torch.where((cams_per==1) & inf)[0][:6]
fx,fy,cx,cy=cam.K[0,0].item(),cam.K[1,1].item(),cam.K[0,2].item(),cam.K[1,2].item()
R=cam.viewmatrix[:3,:3]; t=cam.viewmatrix[:3,3]
for cl in single.tolist():
    mask=(labels==cl)
    px=pix[mask]; dd=D[px[:,1],px[:,0]]
    # camera-space depth of candidates
    Xc=Xall[mask]
    campt=Xc@R.t()+t.unsqueeze(0)
    zc=(mu[cl]@R.t()+t)
    zc=zc[2].item()
    ui=int(round(u[cl].item())); vi=int(round(v[cl].item()))
    Dpix=D[vi,ui].item()
    print(f"cl{cl} npix={int(mask.sum())} pix_u[{px[:,0].min().item()},{px[:,0].max().item()}] cand_z[{campt[:,2].min().item():.3f},{campt[:,2].max().item():.3f}] center_z={zc:.3f} Dpix={Dpix:.3f} reldiff={abs(Dpix-zc)/zc:.4f}")


