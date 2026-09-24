#!/usr/bin/env python3
"""Gate D root-cause: isolate why RGB diverges while alpha is near-exact."""
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
    radii,means2d,depths,conics,opac_bc,colors=r2.higs_gatherless_projected_producer(
        vis,means,quats,scales,opac,sh,vm.contiguous(),K.contiguous(),cp,W,H,0.3,0.01,1e10,0.0)
    TW,TH=math.ceil(W/TS),math.ceil(H/TS)
    mm2d=means2d.reshape(1,1,N,2); rr=radii.reshape(1,1,N,2); dd=depths.reshape(1,1,N)
    cc=conics.reshape(1,1,N,3); oo=opac_bc.reshape(1,1,N); col=colors.reshape(1,1,N,3)
    tp,isect_ids,flatten=torch.ops.gsplat.intersect_tile(mm2d,rr,dd,cc,oo,None,None,1,TS,TW,TH,True,False)
    offs=torch.ops.gsplat.intersect_offset(isect_ids.clone(),1,TW,TH).reshape(1,TH,TW)
    bgv=torch.tensor([0.0,0.0,0.0],device=dev)
    out=torch.ops.gsplat.rasterize_to_pixels_3dgs(mm2d,cc,col,oo,bgv.reshape(1,1,3),None,W,H,TS,offs.contiguous(),flatten.contiguous(),False,False)
    b_rgb,b_al,b_last=out[0],out[1],out[2]
    # native
    for bgtest in ["zero","gray"]:
        bg2 = bgv if bgtest=="zero" else torch.tensor([0.05,0.1,0.15],device=dev)
        rgb,al,diag=torch.ops.experimental.higs_native_hierarchy_from_projected(vis,radii,means2d,depths,conics,opac_bc,colors,W,H,TS,bg2,False)
        d=(b_rgb-rgb).abs(); da=(b_al.squeeze(-1)-al.squeeze(-1)).abs()
        print(sc,"SHAPES b_rgb",tuple(b_rgb.shape),"b_al",tuple(b_al.shape),"rgb",tuple(rgb.shape),"al",tuple(al.shape))
        print(sc,bgtest,"RGB max",format(d.max().item(),'.4f'),"mean",format(d.mean().item(),'.5f'),"rel_l2",format(((b_rgb-rgb).norm()/b_rgb.norm()).item(),'.5f'))
        print("   ALP max",format(da.max().item(),'.5f'),"mean",format(da.mean().item(),'.6f'),"cos",float(torch.nn.functional.cosine_similarity(b_al.flatten()[None],al.flatten()[None]).item()))
    # bg consistency: base with gray bg
    bg3=torch.tensor([0.05,0.1,0.15],device=dev)
    out2=torch.ops.gsplat.rasterize_to_pixels_3dgs(mm2d,cc,col,oo,bg3.reshape(1,1,3),None,W,H,TS,offs.contiguous(),flatten.contiguous(),False,False)
    b_rgb2=out2[0]
    rgb,al,diag=torch.ops.experimental.higs_native_hierarchy_from_projected(vis,radii,means2d,depths,conics,opac_bc,colors,W,H,TS,bg3,False)
    d=(b_rgb2-rgb).abs()
    print(sc,"BOTH gray bg RGB max",format(d.max().item(),'.4f'),"mean",format(d.mean().item(),'.5f'),"rel_l2",format(((b_rgb2-rgb).norm()/b_rgb2.norm()).item(),'.5f'))
    # ---- premultiplication hypotheses ----
    ba = b_al.clamp_min(1e-6); na = al.clamp_min(1e-6)
    for k, tt in {
        "nat_unpremult(b_rgb vs nat_rgb/alpha)": (b_rgb - rgb/na).abs(),
        "nat_premult(b_rgb/ba vs nat_rgb/na)": (b_rgb/ba - rgb/na).abs(),
        "base_unpremult(b_rgb/ba vs nat_rgb)": (b_rgb/ba - rgb).abs(),
    }.items():
        print(f"   {k}: max {tt.max().item():.4f} mean {tt.mean().item():.5f}")
    print("   base_rgb absmax", b_rgb.abs().max().item(), "native_rgb absmax", rgb.abs().max().item())
    # correlate diff with alpha magnitude (a_nat is [H,W]; dd[0] is [H,W])
    dd=(b_rgb-rgb).abs().amax(dim=3)      # [1,H,W]
    a_nat=al.squeeze(-1)                  # [H,W]
    a_base=b_al.squeeze()                 # [H,W]
    hi=a_nat>0.9; lo=a_nat<=0.2
    print("   n_hi_alpha pixels",int(hi.sum().item()),"n_lo_alpha",int(lo.sum().item()))
    print("   diff hi-alpha: max",format(dd[0][hi].max().item(),'.4f'),"mean",format(dd[0][hi].mean().item(),'.4f'))
    if int(lo.sum().item())>0:
        print("   diff lo-alpha: max",format(dd[0][lo].max().item(),'.4f'),"mean",format(dd[0][lo].mean().item(),'.4f'))
    else:
        print("   diff lo-alpha: (empty)")
    # representative worst-diff pixels: report base_rgb/native_rgb/alpha at them
    flat_dd=dd[0].reshape(-1)
    n_worst=6
    wids=torch.topk(flat_dd,n_worst).indices
    print("   worst-diff pixels (idx: dys,dx, base_rgb, nat_rgb, base_al, nat_al):")
    for k,idxv in enumerate(wids.tolist()):
        y, x = divmod(idxv,W)
        br=b_rgb[0,y,x].tolist(); nr=rgb[y,x].tolist()
        print(f"      #{k} ({x},{y}) base={[round(v,3) for v in br]} nat={[round(v,3) for v in nr]} al={float(a_base[y,x]):.3f}/{float(a_nat[y,x]):.3f}")