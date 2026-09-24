#!/usr/bin/env python3
"""Gate D root cause: is RGB divergence caused by differing per-fine-tile gid coverage
between BASE (isect_tile radius coverage) and TEST (native macro predicate)?"""
import json, os, math, importlib.util, torch
CORE="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
R2="/mnt/storage_pool/liaoyuanjun/higs_p2_1a_r2_final_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CKPT_BASE="/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7"; CAM_BASE="/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360"
MAX_LONG=2048; TS=16; MTW,MTH=8,4
CHOL_SCALE=0.84932180028*TS; MAX_EXTEND=3.33
LOG2_INV_AT=math.log2(255.0); MAX_EXTEND_CHOL_SQ=(math.log2(math.e)*0.5)*MAX_EXTEND*MAX_EXTEND
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
    radii,means2d,depths,conics,opac_bc,colors=r2.higs_gatherless_projected_producer(
        vis,means,quats,scales,opac,sh,vm.contiguous(),K.contiguous(),cp,W,H,0.3,0.01,1e10,0.0)
    TW,TH=math.ceil(W/TS),math.ceil(H/TS)
    mm2d=means2d.reshape(1,1,N,2); rr=radii.reshape(1,1,N,2); dd=depths.reshape(1,1,N); cc=conics.reshape(1,1,N,3); oo=opac_bc.reshape(1,1,N)
    col=colors.reshape(1,1,N,3)
    # native diagnostics (debug)
    bgv=torch.tensor([0.0,0.0,0.0],device=dev)
    rgb,al,diag=torch.ops.experimental.higs_native_hierarchy_from_projected(vis,radii,means2d,depths,conics,opac_bc,colors,W,H,TS,bgv,True)
    macro_offsets, sorted_rows, batch_offsets, act_mask = diag[0],diag[1],diag[2],diag[3]
    # BASE tiles from isect
    tp,isect_ids,flatten=torch.ops.gsplat.intersect_tile(mm2d,rr,dd,cc,oo,None,None,1,TS,TW,TH,True,False)
    ids=isect_ids.cpu().numpy(); flat=flatten.cpu().numpy()
    n_tiles=TW*TH; tile_n_bits=int(math.floor(math.log2(n_tiles)))+1
    mask=(1<<tile_n_bits)-1; tids=(ids>>32)&mask
    base_tiles={}
    for i in range(ids.shape[0]):
        base_tiles.setdefault(int(tids[i]),[]).append(int(flat[i]))
    # native tiles via predicate replay
    l0=torch.sqrt(torch.clamp_min(conics[...,0],0.0)); l1=torch.where(l0>1e-12,conics[...,1]/l0,torch.zeros_like(l0))
    l2=torch.sqrt(torch.clamp_min(conics[...,2]-l1*l1,0.0)); log2_opac=torch.log2(torch.clamp_min(opac_bc,1e-30))
    t_rast=torch.min(torch.full_like(log2_opac,MAX_EXTEND_CHOL_SQ),log2_opac+LOG2_INV_AT)
    rl0=l0*CHOL_SCALE; rl1=l1*CHOL_SCALE; rl2=l2*CHOL_SCALE; C=rl1*rl1+rl2*rl2
    nlr=torch.where(C>0,-rl1/C,torch.zeros_like(C))
    native_tiles={}
    n_macro=macro_offsets.shape[0]-1; mc_cols=(TW+MTW-1)//MTW
    for mt in range(n_macro):
        s,e=int(macro_offsets[mt]),int(macro_offsets[mt+1])
        if s==e: continue
        gids=torch.tensor(sorted_rows[s:e].cpu(),device=dev)
        mt_col=mt%mc_cols; mt_row=mt//mc_cols; mt_cx=mt_col*MTW+MTW*0.5; mt_cy=mt_row*MTH+MTH*0.5
        g=torch.tensor([int(x) for x in sorted_rows[s:e].cpu().tolist()],device=dev)
        x_t=means2d[g,0]/TS-mt_cx; y_t=means2d[g,1]/TS-mt_cy
        a0=rl0[g]; a1=rl1[g]; a2=rl2[g]; tr=t_rast[g]; nr=nlr[g]
        seen=set()
        for r in range(MTH):
            ny=y_t-(r+0.5-MTH*0.5); dy0=-0.5-ny; dy1=dy0+1.0; dy_h=torch.where(ny>=0,dy1,dy0)
            l1_dy=a1*dy_h; v_h=a2*dy_h; v_h_sq=v_h*v_h
            for c in range(MTW):
                nx=x_t-(c+0.5-MTW*0.5); dx0=-0.5-nx; l0dx0=a0*dx0; l0dx1=l0dx0+a0
                u_h=torch.minimum(torch.maximum(torch.zeros_like(l0dx0),l0dx0+l1_dy),l0dx1+l1_dy); q_h=u_h*u_h+v_h_sq
                l0dx=torch.where(nx>=0,l0dx1,l0dx0); dy_v=torch.clamp(l0dx*nr,dy0,dy1)
                u_v=a1*dy_v+l0dx; v_v=a2*dy_v; q_v=v_v*v_v+u_v*u_v
                c_hit=(torch.abs(nx)<0.5)&(torch.abs(ny)<0.5); e_hit=torch.minimum(q_h,q_v)<=tr
                hit=(c_hit|e_hit)
                if bool(hit.any()):
                    tid=(mt_row*MTH+r)*TW+(mt_col*MTW+c)
                    for idx in hit.nonzero().flatten().tolist():
                        key=int(g[idx])
                        if key not in seen:
                            seen.add(key); native_tiles.setdefault(tid,[]).append(key)
    tid0=0
    b0=set(base_tiles.get(tid0,[])); n0=set(native_tiles.get(tid0,[]))
    print(sc,"corner fine-tile 0 coverage")
    print("   base n",len(b0),"native n",len(n0),"base-only",len(b0-n0),"native-only",len(n0-b0),"same",len(b0&n0))
    # front-most gids & colors in each
    depg=depths.cpu().numpy()
    def fronts(st):
        return sorted(st,key=lambda g:float(depg[g]))[:5]
    fb=fronts(b0); fn=fronts(n0)
    for tag,st in [("base",b0),("native",n0)]:
        lst=sorted(st,key=lambda g:float(depg[g]))[:6]
        print(f"   {tag} front gids",lst,"colors",[ [round(float(c),3) for c in colors[int(g)].tolist()] for g in lst])
    # global fine-tile coverage equality stats (only tiles present in both authors)
    allt=set(base_tiles)|set(native_tiles)
    eq=0; diff=0; nbase_only=0; nnat_only=0
    for t in allt:
        if set(base_tiles.get(t,[]))==set(native_tiles.get(t,[])): eq+=1
        else:
            diff+=1
            if t in base_tiles and t not in native_tiles: nbase_only+=1
            elif t in native_tiles and t not in base_tiles: nnat_only+=1
    print(f"   global fine-tiles: equal-coverage {eq}, differing {diff}, base-only-tiles {nbase_only}, native-only-tiles {nnat_only}")
    print("   total fine tiles",TW*TH)