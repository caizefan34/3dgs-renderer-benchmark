#!/usr/bin/env python
"""diag_align.py - independent recompute of teacher(render) vs saved GT patch PSNR.
Reads workload.npz ref_teacher_f64 and g5a_gt_patches.npz gt; recomputes MSE/PSNR
per tile from scratch (no reliance on extraction self-report). Pure numpy."""
import json, os, sys
import numpy as np
WORKLOAD="/home/liaoyuanjun/cgt_f0_fused_cuda/results/workload.npz"
GTNPZ="/home/liaoyuanjun/cgt_g5_native/results/g5a_gt_patches.npz"
d=np.load(WORKLOAD)
T=d["ref_teacher_f64"]            # (150,256,4)
scene=d["scene"]; view=d["view"]
g=np.load(GTNPZ)
gt=g["gt"]; gscene=g["scene"]; gview=g["view"]; ti=g["ti"]; tj=g["tj"]
assert gt.shape==T.shape[:2]+(3,), (gt.shape,T.shape)
N=gt.shape[0]
psnr=np.zeros(N); mse=np.zeros(N); gmax=np.zeros(N); gstd=np.zeros(N)
for t in range(N):
    Tr=T[t][:,:3].reshape(-1); Gr=gt[t].reshape(-1)
    m=float(np.mean((Tr-Gr)**2)); mse[t]=m; psnr[t]=-10*np.log10(max(m,1e-30))
    gmax[t]=float(gt[t].max()); gstd[t]=float(gt[t].std())
order=np.argsort(psnr)
print("N=",N)
print("PSNR min=%.3f median=%.3f mean=%.3f max=%.3f"%(psnr.min(),np.median(psnr),psnr.mean(),psnr.max()))
print("below20=%d below25=%d below30=%d"%((psnr<20).sum(),(psnr<25).sum(),(psnr<30).sum()))
# check scene/view consistency between the two npz
sc_match=bool(np.all(scene==gscene)); vw_match=bool(np.all(view==gview))
print("scene_match=%s view_match=%s"%(sc_match,vw_match))
print("ti range",int(ti.min()),int(ti.max()),"tj range",int(tj.min()),int(tj.max()))
print("--- 25 worst tiles ---")
for t in order[:25]:
    print("t=%3d %s v=%3d ti=%d tj=%d psnr=%.2f mse=%.5f gmax=%.4f gstd=%.4f"%(
        t,scene[t],view[t],ti[t],tj[t],psnr[t],mse[t],gmax[t],gstd[t]))
# per-scene
for sc in ["room","garden","bicycle"]:
    m=scene==sc
    print("[%s] n=%d psnr min=%.2f mean=%.2f below25=%d"%(sc,m.sum(),psnr[m].min(),psnr[m].mean(),(psnr[m]<25).sum()))
