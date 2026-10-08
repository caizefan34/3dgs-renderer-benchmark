#!/usr/bin/env python
"""diag_g5a.py - explain the A vs B/C PSNR gap. CPU-only, few tiles per scene."""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/liaoyuanjun/cgt_realscene/final/SCRIPTS")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_compressibility import initialize, predict_and_vjp
from g5_cgt_renderer import cgt_student_render

WORKLOAD="/home/liaoyuanjun/cgt_f0_fused_cuda/results/workload.npz"
GTPATCHES="/home/liaoyuanjun/cgt_g5_native/results/g5a_gt_patches.npz"
RANK=4
d=np.load(WORKLOAD); coeffs0=d["coeffs"].astype(np.float64); px=d["px"].astype(np.float64)
scene=d["scene"]; ref_tch=d["ref_teacher_f64"]
g=np.load(GTPATCHES); gt_all=g["gt"].astype(np.float64); gscene=g["scene"]
P=px.shape[0]; tr=np.arange(0,P,2); te=np.arange(1,P,2)

def psnr(a,b):
    m=np.mean((a-b)**2); return float(-10*np.log10(max(m,1e-30)))

def native(px_tr,gt_tr,init,iters,lr,seed=0):
    torch.manual_seed(seed)
    c=torch.tensor(init,dtype=torch.float32,requires_grad=True)
    opt=torch.optim.Adam([c],lr=lr)
    pxt=torch.tensor(px_tr,dtype=torch.float32); yt=torch.tensor(gt_tr,dtype=torch.float32)
    for _ in range(iters):
        opt.zero_grad(); pred=cgt_student_render(c,pxt)
        loss=torch.mean((pred[:,:3]-yt)**2); loss.backward(); opt.step()
    return c.detach().numpy()

print(f"{'sc':7s} {'t':3s} {'gtstd':7s} {'gtmax':7s} {'TchGT':7s} {'A_Tch':7s} {'A_GT':7s} {'B_GT':7s} {'C_GT':7s} {'Btr':7s} {'Ctr':7s}")
for sc in ["room","garden","bicycle"]:
    idx=np.where(scene==sc)[0][:5]   # first 5 tiles per scene
    for t in idx:
        gt_rgb=gt_all[t]
        # A render (frozen coeffs) via numpy predict_and_vjp
        rA=predict_and_vjp(coeffs0[t],px)         # (256,4)
        A_E=np.clip(rA[:,:3],0,1)
        Tch_E=np.clip(ref_tch[t][:,:3],0,1)
        pTchGT=psnr(Tch_E,gt_rgb)
        pA_Tch=psnr(A_E,Tch_E)
        pA_GT=psnr(A_E,gt_rgb)
        # B native from teacher
        pB=native(px[tr],gt_rgb[tr],coeffs0[t],200,0.05,seed=int(t))
        rB=cgt_student_render(torch.tensor(pB,dtype=torch.float32),torch.tensor(px,dtype=torch.float32)).detach().numpy()
        B_E=np.clip(rB[:,:3],0,1); pB_GT=psnr(B_E[te],gt_rgb[te]); pB_tr=psnr(B_E[tr],gt_rgb[tr])
        # C native from scratch
        y4=np.concatenate([gt_rgb,np.zeros((P,1))],axis=1)
        pC0=initialize(RANK,px[tr],y4[tr],seed=int(t))
        pC=native(px[tr],gt_rgb[tr],pC0,200,0.05,seed=int(t))
        rC=cgt_student_render(torch.tensor(pC,dtype=torch.float32),torch.tensor(px,dtype=torch.float32)).detach().numpy()
        C_E=np.clip(rC[:,:3],0,1); pC_GT=psnr(C_E[te],gt_rgb[te]); pC_tr=psnr(C_E[tr],gt_rgb[tr])
        print(f"{sc:7s} {t:3d} {gt_rgb.std():7.4f} {gt_rgb.max():7.4f} {pTchGT:7.2f} {pA_Tch:7.2f} {pA_GT:7.2f} {pB_GT:7.2f} {pC_GT:7.2f} {pB_tr:7.2f} {pC_tr:7.2f}")
