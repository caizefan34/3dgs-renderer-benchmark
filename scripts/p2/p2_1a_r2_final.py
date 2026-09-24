#!/usr/bin/env python3
"""P2-1A-R2 final correctness run: F9 identity, Gate A, Gate D metrics, basic hierarchy counts,
across room / bicycle / garden. Correctness only -- no timing (Gate D governs whether timing runs)."""
import json, os, math, hashlib, importlib.util, torch
OUT="/mnt/storage_pool/liaoyuanjun/higs_p2_1a_r2_runtime"
os.makedirs(OUT, exist_ok=True)
CORE="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
R2="/mnt/storage_pool/liaoyuanjun/higs_p2_1a_r2_final_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
CKPT_BASE="/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7"
CAM_BASE="/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360"
MAX_LONG=2048; TS=16
SCENES=["room","bicycle","garden"]
def ls(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def mvm(cam,dev):
    R=torch.tensor(cam["rotation"],dtype=torch.float32,device=dev); po=torch.tensor(cam["position"],dtype=torch.float32,device=dev)
    w,h=float(cam["width"]),float(cam["height"]); sf=MAX_LONG/max(w,h); W,H=int(round(w*sf)),int(round(h*sf))
    vm=torch.eye(4,dtype=torch.float32,device=dev); vm[:3,:3]=R; vm[:3,3]=-R@po
    K=torch.tensor([[cam["fx"]*sf,0,W/2-0.5],[0,cam["fy"]*sf,H/2-0.5],[0,0,1]],dtype=torch.float32,device=dev)[None]
    return vm[None],K,W,H
def b32(t):  # sha256 of float32 bytes
    tb=t.detach().cpu().to(torch.float32).contiguous()
    return hashlib.sha256(tb.numpy().tobytes()).hexdigest()
dev=torch.device('cuda'); torch.set_grad_enabled(False)
core=ls(CORE,"gsplat_cuda"); r2=ls(R2,"experimental_gaussian_render_inference_scene_cuda")
info={"gpu":torch.cuda.get_device_name(0), "uuid":str(torch.cuda.get_device_properties(0).uuid),
      "so_sha":hashlib.sha256(open(R2,"rb").read()).hexdigest(), "core_so":CORE,
      "core_sha":hashlib.sha256(open(CORE,"rb").read()).hexdigest()}
gateA={}; gateD={}; counts={}; f9={}
for sc in SCENES:
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
    nproj=int((radii[...,0]>0).sum().item())
    TW,TH=math.ceil(W/TS),math.ceil(H/TS)
    f9[sc]={"N_GS":N,"N_projected_radii_gt0":nproj,"W":W,"H":H,"TW":TW,"TH":TH,
            "r_radii":b32(radii),"r_means2d":b32(means2d),"r_depths":b32(depths),
            "r_conics":b32(conics),"r_opac":b32(opac_bc),"r_colors":b32(colors)}
    # ---- Gate A: adapter cholesky reproduces conic quadratic form (SPD-scoped) ----
    l0=torch.sqrt(torch.clamp_min(conics[...,0],0.0)); l1=torch.where(l0>1e-12,conics[...,1]/l0,torch.zeros_like(l0))
    l2=torch.sqrt(torch.clamp_min(conics[...,2]-l1*l1,0.0))
    spd = (conics[...,2]-l1*l1)>=0.0   # conic is positive-definite when c2-l1^2>=0
    nspd_total=int(spd.numel()); nspd=int(spd.sum().item()); nnon= nspd_total-nspd
    dn=12; gs=torch.randperm(min(N,40000),device=dev)
    dX=torch.linspace(-5,5,dn,device=dev); dY=torch.linspace(-5,5,dn,device=dev)
    DX,DY=torch.meshgrid(dX,dY,indexing="xy"); dx=DX.reshape(1,-1); dy=DY.reshape(1,-1)
    c0=conics[gs,0,None]; c1=conics[gs,1,None]; c2=conics[gs,2,None]
    q_base=c0*dx*dx+(2*c1)*dx*dy+c2*dy*dy  # [n, npts]
    la0=l0[gs,None]*dx; la1=l1[gs,None]*dy; la2=l2[gs,None]*dy
    q_ad=(la0+la1)**2+la2*la2
    err=(q_base-q_ad).abs()
    s_mask=spd[gs]
    if bool(s_mask.any()):
        mx=err[s_mask].max().item(); mn=err[s_mask].mean().item()
        rel=err[s_mask].norm().item()/(q_base[s_mask].norm().item()+1e-12)
    else:
        mx=mn=rel=float('nan')
    out=float((err[s_mask] > (q_base[s_mask].abs()+1e-9)*1.5e-3).count_nonzero()) if bool(s_mask.any()) else 0.0
    gateA[sc]={"max_abs":mx,"mean_abs":mn,"rel_l2":rel,"outside_tol":out,
               "n_projected_sample":int(s_mask.sum().item()),
               "n_nonspd_clamped":nnon,"frac_nonspd":round(nnon/max(1,nspd_total),5),
               "pass":(rel if bool(s_mask.any()) else float('nan'))<1e-3}
    # ---- shared F9 -> BASE flat + TEST native ----
    mm2d=means2d.reshape(1,1,N,2); rr=radii.reshape(1,1,N,2); dd=depths.reshape(1,1,N)
    cc=conics.reshape(1,1,N,3); oo=opac_bc.reshape(1,1,N); col=colors.reshape(1,1,N,3)
    tp,isect_ids,flatten=torch.ops.gsplat.intersect_tile(mm2d,rr,dd,cc,oo,None,None,1,TS,TW,TH,True,False)
    base_fine=int(flatten.numel())
    offs=torch.ops.gsplat.intersect_offset(isect_ids.clone(),1,TW,TH).reshape(1,TH,TW)
    bgv=torch.tensor([0.0,0.0,0.0],device=dev)
    bout=torch.ops.gsplat.rasterize_to_pixels_3dgs(mm2d,cc,col,oo,bgv.reshape(1,1,3),None,W,H,TS,offs.contiguous(),flatten.contiguous(),False,False)
    b_rgb,b_al,b_last=bout[0],bout[1],bout[2]
    rgb,al,diag=torch.ops.experimental.higs_native_hierarchy_from_projected(vis,radii,means2d,depths,conics,opac_bc,colors,W,H,TS,bgv,False)
    rgb2,al2,diag2=torch.ops.experimental.higs_native_hierarchy_from_projected(vis,radii,means2d,depths,conics,opac_bc,colors,W,H,TS,bgv,True)
    nobj=len(diag2) if diag2 else 0
    macro_offsets=diag2[0] if nobj>0 else None
    # ---- Gate D metrics ----
    br=b_rgb.squeeze(0)                 # [H,W,3]
    ar=torch.tensor(0) if False else rgb  # [H,W,3]
    dra=(br-ar).float(); aldn=br.norm().item()+1e-12
    ndn=float(dra.norm().item()/aldn)
    cosv=float(torch.nn.functional.cosine_similarity(br.flatten()[None],ar.flatten()[None]).item())
    da=(b_al.squeeze(0)-al).abs()        # [H,W,1]
    support_diff=float(da.mean().item())
    gateD[sc]={"rgb_max_abs":float(dra.abs().max().item()),"rgb_mean_abs":float(dra.abs().mean().item()),
               "rgb_rel_l2":ndn,"rgb_cosine":cosv,
               "alpha_max_abs":float(da.abs().max().item()),"alpha_mean_abs":float(da.abs().mean().item()),
               "alpha_cosine":float(torch.nn.functional.cosine_similarity((b_al.flatten())[None],(al.flatten())[None]).item()),
               "support_mismatch_mean_alpha":support_diff,
               "rgb_nan":int(torch.isnan(ar).sum().item())+int(torch.isnan(br).sum().item()),
               "rgb_inf":int(torch.isinf(ar).sum().item())+int(torch.isinf(br).sum().item())}
    # ---- hierarchy counters ----
    macro_entries=int((macro_offsets[-1]).item()) if macro_offsets is not None else 0
    n_macro=int(macro_offsets.numel()-1) if macro_offsets is not None else 0
    n_macro_batches=int(diag2[2][-1].item()) if (nobj>2 and diag2[2] is not None) else 0
    n_act_words=int(diag2[3].numel()) if (nobj>3 and diag2[3] is not None) else 0
    n_sorted_rows=int((diag2[1]).numel()) if (nobj>1 and diag2[1] is not None) else 0
    counts[sc]={"native_macro_tile_G_entries":macro_entries,"baseline_fine_pair_count":base_fine,
                "compression_ratio_btch":round(base_fine/max(1,macro_entries),3),
                "n_macro_tiles":n_macro,"n_macro_batches":n_macro_batches,"active_mask_words":n_act_words,
                "mini_batch_capacity_32":macro_entries,  # each macro entry ~ one 32-lane mini-batch lane group
                "n_gs":N,"n_projected":nproj,"res":"%dx%d"%(W,H),"n_sorted_rows":n_sorted_rows}
    print(sc,"gateA",json.dumps(gateA[sc]),"gateD.rgb_rel_l2",round(ndn,4),"alpha_cos",round(gateD[sc]["alpha_cosine"],7),
          "macro_entries",macro_entries,"base_fine",base_fine)
res={"info":info,"gateA":gateA,"gateD":gateD,"counts":counts,"f9":f9,
     "gateA_pass":all(gateA[s]["pass"] for s in SCENES),
     "gateD_classification":"OFF (outside FP32 envelope) <P2_1A_R2_CORRECTNESS_FAIL>" }
with open(f"{OUT}/final_correctness.json","w") as f: json.dump(res,f,indent=2)
print("WROTE",f"{OUT}/final_correctness.json")