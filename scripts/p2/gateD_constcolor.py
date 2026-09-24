#!/usr/bin/env python3
"""Gate D isolation: constant source color -> does native composition arithmetic match base?"""
import json, os, math, importlib.util, torch
CORE="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
R2="/mnt/storage_pool/liaoyuanjun/higs_p2_1a_r2_final_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CKPT_BASE="/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7"; CAM_BASE="/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360"
MAX_LONG=2048; TS=16
def ls(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def mvm(cam,dev):
    R=torch.tensor(cam["rotation"],dtype=torch.float32,device=dev); po=torch.tensor(cam["position"],dtype=torch.float32,device=dev)
    w,h=float(cam["width"]),float(cam["height"]); sf=MAX_LONG/max(w,h); W,H=int(round(w*sf)),int(round(h*sf))
    vm=torch.eye(4,dtype=torch.float32,device=dev); vm[:3,:3]=R; vm[:3,3]=-R@po
    K=torch.tensor([[cam["fx"]*sf,0,W/2-0.5],[0,cam["fy"]*sf,H/2-0.5],[0,0,1]],dtype=torch.float32,device=dev)[None]
    return vm[None],K,W,H
dev=torch.device('cuda'); torch.set_grad_enabled(False)
core=ls(CORE,"gsplat_cuda"); r2=ls(R2,"experimental_gaussian_render_inference_scene_cuda")
for sc in ["room"]:
    cam=json.load(open(f"{CAM_BASE}/{sc}/cameras.json"))[0]
    ck=torch.load(f"{CKPT_BASE}/a100_30k_{sc}_t16_16/a100_30k_{sc}_t16_16_latest.pt",map_location="cpu",weights_only=False)
    st=ck.get("model_state",ck)
    means=st["xyz"].to(dev).contiguous(); quats=st["rotations"].to(dev).contiguous()
    scales=torch.exp(st["scales"].to(dev)).contiguous(); opac=torch.sigmoid(st["opacity"].to(dev).flatten()).contiguous()
    sh=st["shs"].to(dev).contiguous(); vm,K,W,H=mvm(cam,dev)
    N=means.numel()//3; vis=torch.arange(N,device=dev,dtype=torch.int64)
    cp=r2.higs_camera_positions_from_viewmats(vm.contiguous())
    radii,means2d,depths,conics,opac_bc,colors0=r2.higs_gatherless_projected_producer(
        vis,means,quats,scales,opac,sh,vm.contiguous(),K.contiguous(),cp,W,H,0.3,0.01,1e10,0.0)
    TW,TH=math.ceil(W/TS),math.ceil(H/TS)
    mm2d=means2d.reshape(1,1,N,2); rr=radii.reshape(1,1,N,2); dd=depths.reshape(1,1,N)
    cc=conics.reshape(1,1,N,3); oo=opac_bc.reshape(1,1,N)
    for COL,label in [(0.35,"gray035"),(0.9,"bright090")]:
        colors=torch.full_like(colors0,COL)
        col=colors.reshape(1,1,N,3)
        tp,isect_ids,flatten=torch.ops.gsplat.intersect_tile(mm2d,rr,dd,cc,oo,None,None,1,TS,TW,TH,True,False)
        offs=torch.ops.gsplat.intersect_offset(isect_ids.clone(),1,TW,TH).reshape(1,TH,TW)
        bgv=torch.tensor([0.02,0.03,0.04],device=dev)
        out=torch.ops.gsplat.rasterize_to_pixels_3dgs(mm2d,cc,col,oo,bgv.reshape(1,1,3),None,W,H,TS,offs.contiguous(),flatten.contiguous(),False,False)
        b_rgb,b_al,b_last=out[0],out[1],out[2]
        rgb,al,diag=torch.ops.experimental.higs_native_hierarchy_from_projected(vis,radii,means2d,depths,conics,opac_bc,colors,W,H,TS,bgv,False)
        d=(b_rgb-rgb).abs()
        conv_al = al.squeeze(-1)  # [H,W]
        # restrict analysis to opaque pixels (alpha>0.999)
        m = conv_al>0.999
        bm=b_rgb.permute(0,3,1,2)  # [1,3,H,W]
        rm=rgb.permute(2,0,1)      # [3,H,W]
        db=(bm[0,:,m]).mean()      # mean over opaque px, all channels
        dr=(rm[:,m]).mean()
        rel_ratio = (dr.item() )

        print(sc,label,"constant-src color")
        print("   diff max",format(d.max().item(),'.4f'),"mean",format(d.mean().item(),'.4f'),"rel_l2",format(((b_rgb-rgb).norm()/b_rgb.norm()).item(),'.4f'))
        print("   opaque-px mean base",format(db.item(),'.4f'),"native",format(dr.item(),'.4f'),"ratio(nat/base)",format((dr/db).item(),'.3f'))
        print("   base absmax",format(b_rgb.abs().max().item(),'.4f'),"native absmax",format(rgb.abs().max().item(),'.4f'))