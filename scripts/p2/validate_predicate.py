#!/usr/bin/env python3
"""Validate native fine-tile reconstruction against ground-truth active masks."""
import json, os, math, importlib.util, torch
CORE="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
R2="/mnt/storage_pool/liaoyuanjun/higs_p2_1a_r2_final_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CKPT_BASE="/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7"; CAM_BASE="/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360"
MAX_LONG=2048; TS=16; MTW,MTH=8,4
CHOL_SCALE=0.84932180028*TS; MAX_EXTEND=3.33; LOG2_INV_AT=math.log2(255.0)
MAX_EXTEND_CHOL_SQ=(math.log2(math.e)*0.5)*MAX_EXTEND*MAX_EXTEND
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
bg=torch.tensor([0.,0.,0.],device=dev).float()
for sc in ["room"]:
    cam=json.load(open(f"{CAM_BASE}/{sc}/cameras.json"))[0]
    ck=torch.load(f"{CKPT_BASE}/a100_30k_{sc}_t16_16/a100_30k_{sc}_t16_16_latest.pt",map_location="cpu",weights_only=False)
    st=ck.get("model_state",ck)
    means=st["xyz"].to(dev).contiguous(); quats=st["rotations"].to(dev).contiguous()
    scales=torch.exp(st["scales"].to(dev)).contiguous(); opac=torch.sigmoid(st["opacity"].to(dev).flatten()).contiguous()
    sh=st["shs"].to(dev).contiguous(); vm,K,W,H=mvm(cam,dev)
    N=means.numel()//3; vis=torch.arange(N,device=dev,dtype=torch.int64)
    cp=r2.higs_camera_positions_from_viewmats(vm.contiguous())
    radii,means2d,depths,conics,opac_bc,colors=r2.higs_gatherless_projected_producer(
        vis,means,quats,scales,opac,sh,vm.contiguous(),K.contiguous(),cp,W,H,0.3,0.01,1e10,0.0)
    TW,TH=math.ceil(W/TS),math.ceil(H/TS); mc=math.ceil(TW/MTW); mr=math.ceil(TH/MTH)
    rgb,al,diag=torch.ops.experimental.higs_native_hierarchy_from_projected(vis,radii,means2d,depths,conics,opac_bc,colors,W,H,TS,bg,True)
    mo,sr,bo,am=diag[0].cpu().numpy(),diag[1].cpu().numpy(),diag[2].cpu().numpy(),diag[3].cpu().numpy()
    print("n_macro",mo.shape[0]-1,"macroG",int(mo[-1]),"n_batch",int(bo[-1]),"n_am",am.shape[0],"batch_offsets[-1]",bo[-1])
    # ground truth active fine tiles: from active_masks per batch -> but batch->macro mapping via find via batch_offsets scanning
    # reconstruct native per-fine-tile from macro bins with predicate
    l0=torch.sqrt(torch.clamp_min(conics[...,0],0.0)); l1=torch.where(l0>1e-12,conics[...,1]/l0,torch.zeros_like(l0)); l2=torch.sqrt(torch.clamp_min(conics[...,2]-l1*l1,0.0))
    lg=torch.log2(torch.clamp_min(opac_bc,1e-30)); tr=torch.min(torch.full_like(lg,MAX_EXTEND_CHOL_SQ),lg+LOG2_INV_AT)
    a0=l0*CHOL_SCALE; a1=l1*CHOL_SCALE; a2=l2*CHOL_SCALE; C=a1*a1+a2*a2; nr=torch.where(C>0,-a1/C,torch.zeros_like(C))
    my_tiles={}; n_tiles_total=TW*TH
    for mt in range(mo.shape[0]-1):
        s,e=int(mo[mt]),int(mo[mt+1])
        if s==e: continue
        gids=sr[s:e]; mtc=mt%mc; mtr=mt//mc; mcx=mtc*MTW+MTW*0.5; mcy=mtr*MTH+MTH*0.5
        g=torch.tensor(gids,device='cuda'); xt=means2d[g,0]/TS-mcx; yt=means2d[g,1]/TS-mcy
        L0=a0[g];L1=a1[g];L2=a2[g];TR=tr[g];NR=nr[g]
        for r in range(MTH):
            ny=yt-(r+0.5-MTH*0.5); dy0=-0.5-ny; dy1=dy0+1.0; dyh=torch.where(ny>=0,dy1,dy0); l1dy=L1*dyh; vh=L2*dyh; vhs=vh*vh
            for c in range(MTW):
                nx=xt-(c+0.5-MTW*0.5); dx0=-0.5-nx; l0x0=L0*dx0; l0x1=l0x0+L0
                uh=torch.minimum(torch.maximum(torch.zeros_like(l0x0),l0x0+l1dy),l0x1+l1dy); qh=uh*uh+vhs
                l0dx=torch.where(nx>=0,l0x1,l0x0); dyv=torch.clamp(l0dx*NR,dy0,dy1); uv=L1*dyv+l0dx; vv=L2*dyv; qv=vv*vv+uv*uv
                hit=((torch.abs(nx)<0.5)&(torch.abs(ny)<0.5))|(torch.minimum(qh,qv)<=TR)
                tid=(mtr*MTH+r)*TW+(mtc*MTW+c)
                if tid<n_tiles_total and bool(hit.any().item()): my_tiles[tid]=my_tiles.get(tid,0)+1
    # my active tile count
    print("my_native_active_fine_tiles",len(my_tiles), "total_tiles", n_tiles_total)
    # ground truth: any active mask bit per batch, mapped to macro+local tile
    gt_active=set()
    n_tiles=TW*TH; tbits=int(math.floor(math.log2(n_tiles)))+1
    # map batch -> macro by scanning offsets
    for bidx in range(am.shape[0]):
        # find mt: smallest mt such that bo[mt+1] > bidx
        import bisect
        mt=bisect.bisect_right(bo[bidx:], bidx)  # not used; do linear
        mt=0
        for mm in range(bo.shape[0]-1):
            if bo[mm]<=bidx<bo[mm+1]: mt=mm; break
        msk=int(am[bidx]); mtc=mt%mc; mtr=mt//mc
        for t in range(MTW*MTH):
            if msk&(1<<t):
                r=t//MTW; c=t%MTW; tid=(mtr*MTH+r)*TW+(mtc*MTW+c); gt_active.add(tid)
    print("gt_native_active_fine_tiles",len(gt_active))
    # agreement
    my=set(my_tiles)
    fp=list(my-gt_active)[:10]; fn=list(gt_active-my)[:10]
    print("my but not gt (first 10):",fp)
    print("gt but not my (first 10):",fn)
    print("intersection",len(my&gt_active),"|my|",len(my),"|gt|",len(gt_active))