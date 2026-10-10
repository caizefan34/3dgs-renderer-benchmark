#!/usr/bin/env python3
"""S1A-R1 Phase 2: three-way render comparison.

A. Native FasterGSFused renderer (_C.forward, the ORIGINAL training rasterizer)
B. Standard gsplat low-level path (fully_fused_projection + isect + rasterize_to_pixels)
C. HiGS full-tile path (rasterize_gaussian_higs_dynamic, use_higs_culling=False, no mask)

Same ckpt/camera/res/background/GT. Reports PSNR/MAE/SSIM pairwise + provenance.
Render sanity only — NO sparse mask, NO default-pixel fill.
"""
import os, sys, json, math, time, hashlib
import torch, numpy as np
from PIL import Image
BUILD_DIR = "/home/liaoyuanjun/higs-13scene/artifacts/cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda"
sys.path.insert(0, BUILD_DIR)
import torch as _t
_C_so_dir = "/mnt/storage_pool/liaoyuanjun/fused_profile0/faster-gaussian-splatting/FasterGSFusedCudaBackend/FasterGSFusedCudaBackend"
sys.path.insert(0, _C_so_dir)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/xdg")
dev = torch.device("cuda:6"); torch.manual_seed(0)
import _C  # native FasterGS backend (torch preloaded)

OUT = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/s1a_r1_deliverables"
IMGDIR = f"{OUT}/renders"; os.makedirs(IMGDIR, exist_ok=True)
CKDIR = "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints"
CKPTS = {"5K":"0005000.pt","10K":"0010000.pt","15K":"final.pt"}

cams = json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536), b""): h.update(b)
    return h.hexdigest()

def load_ckpt(ckpt):
    blob=torch.load(ckpt,map_location="cpu",weights_only=False)
    g={k:v for k,v in blob["model_state_dict"].items() if k.startswith("gaussians.")}
    means=g["gaussians._means"].to(dev).float()
    quats=g["gaussians._rotations"].to(dev).float(); quats=quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    scales=g["gaussians._scales"].to(dev).float().exp()
    opac=g["gaussians._opacities"].to(dev).float().sigmoid().squeeze(-1)
    sh0=g["gaussians._sh_coefficients_0"].to(dev).float()       # [N,1,3]
    shrest=g["gaussians._sh_coefficients_rest"].to(dev).float() # [N,15,3]
    colors=torch.cat([sh0,shrest],dim=1).contiguous()           # [N,16,3]
    return means,quats,scales,opac,sh0,shrest,colors,blob.get("num_iterations_trained")

def get_gt(c,w2,h2):
    for sub in ("eval_images","images"):
        for ext in (".JPG",".jpg",".png"):
            pp=f"/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/{sub}/{c['img_name']}{ext}"
            if os.path.exists(pp) and os.path.getsize(pp)>1000:
                return Image.open(pp).convert("RGB").resize((w2,h2),Image.LANCZOS), pp, sha256(pp)
    return None,None,None

def psnr(a,b):
    mse=(a.astype(np.float64)-b.astype(np.float64)).mean()
    return float(-10*np.log10(max(mse,1e-12)))
def mae(a,b):
    return float(np.abs(a.astype(np.float64)-b.astype(np.float64)).mean())
def ssim_torch(a,b):
    # a,b [3,H,W] in [0,1]
    a=torch.tensor(a,dtype=torch.float32,device=dev).unsqueeze(0)
    b=torch.tensor(b,dtype=torch.float32,device=dev).unsqueeze(0)
    win=11; sigma=1.5
    coords=torch.arange(win,device=dev,dtype=torch.float32)-win//2
    g=torch.exp(-(coords**2)/(2*sigma*sigma)); g=g/g.sum()
    win2d=g.unsqueeze(0)*g.unsqueeze(1)
    win3d=win2d.expand(3,1,win,win).contiguous()
    def filt(x): return torch.nn.functional.conv2d(x,win3d,padding=win//2,groups=3)
    mu_a=filt(a); mu_b=filt(b)
    sa=((a-mu_a)**2); sb=((b-mu_b)**2); sab=(a-mu_a)*(b-mu_b)
    va=filt(sa); vb=filt(sb); vab=filt(sab)
    C1=0.01**2; C2=0.03**2
    s=((2*mu_a*mu_b+C1)*(2*vab+C2))/((mu_a**2+mu_b**2+C1)*(va+vb+C2))
    return float(s.mean())

# gsplat low-level path (B)
from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode
def gsplat_lowlevel(means,quats,scales,opac,colors,viewmats,Ks,width,height,sh_degree=3):
    C=viewmats.shape[0]; N=means.shape[0]
    from gsplat.rendering import _maybe_evaluate_sh
    radii,means2d,depths,conics,_=fully_fused_projection(
        means=means.contiguous(),covars=None,quats=quats.contiguous(),scales=scales.contiguous(),
        viewmats=viewmats,Ks=Ks,width=width,height=height,eps2d=0.3,near_plane=0.01,far_plane=1e10,
        radius_clip=0.0,packed=False,calc_compensations=False,camera_model="pinhole")
    opac_bc=torch.broadcast_to(opac[...,None,:],(1,C,N)).contiguous()
    tw=math.ceil(width/16); th=math.ceil(height/16)
    _,isect_ids,flatten_ids=isect_tiles(means2d,radii,depths,16,tw,th,packed=False,n_images=C,
        image_ids=None,gaussian_ids=None,conics=conics,opacities=opac_bc)
    isect_offsets=isect_offset_encode(isect_ids,C,tw,th).reshape((1,C,th,tw))
    colors_eval=_maybe_evaluate_sh(sh_degree,colors,means,radii,viewmats,(1,),C,N,True).contiguous()
    bg=torch.zeros((1,C,3),device=means.device)
    fn=_r._make_lazy_cuda_func("rasterize_to_pixels_3dgs")
    rc,ra,_,_=fn(means2d.contiguous(),conics.contiguous(),colors_eval.contiguous(),opac_bc.contiguous(),
                 bg,None,width,height,16,isect_offsets.contiguous(),flatten_ids.contiguous(),False,False)
    return rc,ra

# HiGS full-tile path (C)
from gsplat.experimental.render.functional.gaussian_inference import rasterize_gaussian_higs_dynamic as higs_render
def higs_fulltile(means,quats,scales,opac,colors,viewmats,Ks,width,height):
    out=higs_render(means,quats,scales,opac,colors,sh_degree=3,viewmats=viewmats,Ks=Ks,
        width=width,height=height,tile_size=16,near_plane=0.2,far_plane=10000.0,
        background=torch.zeros(3,device=dev),enable_culling=False,use_higs_culling=False)
    return out["frame"], out.get("metadata")

# Native FasterGS path (A): _C.forward(means,scales,rotations,opacities,sh0,shrest,bg_color, w2c,cam_position, active_sh_bases,width,height,focal_x,focal_y,center_x,center_y,near,far, current_mean_lr, adam_step_count)
def native_fastergs(means,scales,rotations,opacities,sh0,shrest,bg_color,w2c,cam_position,
                    active_sh_bases,width,height,focal_x,focal_y,center_x,center_y,near,far):
    N=means.shape[0]
    autograd_dummy=torch.zeros(1,device=means.device)
    densification_info=torch.empty(0,device=means.device)
    res=_C.forward(means,scales,rotations,opacities,sh0,shrest,bg_color,
                   w2c,cam_position,
                   int(active_sh_bases),int(width),int(height),
                   float(focal_x),float(focal_y),float(center_x),float(center_y),
                   float(near),float(far),0.0,0)
    image=res[0]  # [3,H,W]
    return image, res

# ---- run on camera 0, native 0.25 scale, 5K first ----
W,H=396,264
sf=W/4946.0
results=[]
for ckpt_tag, ckpt_file in CKPTS.items():
    ckpt=f"{CKDIR}/{ckpt_file}"
    means,quats,scales,opac,sh0,shrest,colors,num_iters=load_ckpt(ckpt)
    c=cams[0]
    R=np.asarray(c["rotation"],np.float64); p=np.asarray(c["position"],np.float64)
    Rw2c=R.T; vm=np.eye(4); vm[:3,:3]=Rw2c; vm[:3,3]=-Rw2c@p
    w2c=torch.tensor(vm,dtype=torch.float32,device=dev).unsqueeze(0)  # [1,4,4]
    cam_position=torch.tensor(p,dtype=torch.float32,device=dev).reshape(1,3)  # [1,3]
    Ks=torch.tensor([[c["fx"]*sf,0,(W-1)/2.0],[0,c["fy"]*sf,(H-1)/2.0],[0,0,1.0]],dtype=torch.float32,device=dev).unsqueeze(0)
    gt_img, gt_path, gt_hash = get_gt(c,W,H)
    gt_np=np.asarray(gt_img,np.float32)/255.0
    gt_t=torch.tensor(gt_np,dtype=torch.float32,device=dev).permute(2,0,1).unsqueeze(0)
    bg=torch.zeros(3,device=dev)
    rec={"ckpt":ckpt_tag,"num_iters":int(num_iters),"N":int(means.shape[0]),
         "camera":c["img_name"],"cam_id":c["id"],"W":W,"H":H,"scale":float(sf),
         "fx":float(c["fx"]*sf),"fy":float(c["fy"]*sf),"cx":float((W-1)/2.0),"cy":float((H-1)/2.0),
         "gt_path":gt_path,"gt_sha256":gt_hash,
         "w2c":vm.tolist(),"cam_position":p.tolist()}
    renders={}
    # A native
    try:
        imgA,resA=native_fastergs(means,scales,quats,opac,sh0,shrest,bg,w2c,cam_position,
                                  3,W,H,c["fx"]*sf,c["fy"]*sf,(W-1)/2.0,(H-1)/2.0,0.2,10000.0)
        a=imgA.detach()
        if a.dim()==4: a=a[0]  # [1,H,W,3]->[H,W,3]
        elif a.dim()==3 and a.shape[0]==3: a=a.permute(1,2,0)  # [3,H,W]->[H,W,3]
        a=a.clamp(0,1).cpu().numpy()
        renders["A_native_fastergs"]=a
        rec["A_native_meta"]={"image_shape":list(imgA.shape),"res_counts":[int(resA[5]),int(resA[6]),int(resA[7])] if len(resA)>=8 else None}
    except Exception as e:
        import traceback; traceback.print_exc(); rec["A_native_error"]=f"{type(e).__name__}: {e}"
    # B gsplat low-level
    try:
        vmB=w2c.unsqueeze(0)  # [1,1,4,4]
        KsB=Ks.unsqueeze(0)   # [1,1,3,3]
        rcB,raB=gsplat_lowlevel(means.unsqueeze(0),quats.unsqueeze(0),scales.unsqueeze(0),opac.view(1,1,-1),colors.unsqueeze(0),vmB,KsB,W,H,3)
        b=rcB.reshape(1,H,W,3).permute(0,3,1,2).clamp(0,1)
        renders["B_gsplat_lowlevel"]=b[0].cpu().numpy().transpose(1,2,0)
        rec["B_gsplat_meta"]={"alpha_mean":float(raB.mean())}
    except Exception as e:
        import traceback; traceback.print_exc(); rec["B_gsplat_error"]=f"{type(e).__name__}: {e}"
    # C HiGS full-tile
    try:
        fc,metaC=higs_fulltile(means,quats,scales,opac,colors,vmB,KsB,W,H)
        cimg=fc.permute(0,3,1,2).clamp(0,1)
        renders["C_higs_fulltile"]=cimg[0].cpu().numpy().transpose(1,2,0)
        rec["C_higs_meta"]={k:(int(v) if isinstance(v,(int,np.integer)) else float(v)) for k,v in metaC.items()} if metaC else {}
    except Exception as e:
        import traceback; traceback.print_exc(); rec["C_higs_error"]=f"{type(e).__name__}: {e}"
    # save render images + pairwise metrics vs GT
    for name,arr in renders.items():
        Image.fromarray((arr*255).astype(np.uint8)).save(f"{IMGDIR}/{ckpt_tag}_{name}.png")
    names=list(renders.keys())
    rec["pairwise_vs_gt"]={}
    for name in names:
        arr=renders[name]
        rec["pairwise_vs_gt"][name]={"psnr":psnr(arr,gt_np),"mae":mae(arr,gt_np),"ssim":ssim_torch(arr.transpose(2,0,1),gt_np.transpose(2,0,1))}
    rec["pairwise_renders"]={}
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            rec["pairwise_renders"][f"{names[i]}_vs_{names[j]}"]={"psnr":psnr(renders[names[i]],renders[names[j]]),"mae":mae(renders[names[i]],renders[names[j]])}
    results.append(rec)
    print(f"[{ckpt_tag}] " + " | ".join(f"{n}: PSNR={rec['pairwise_vs_gt'][n]['psnr']:.2f} SSIM={rec['pairwise_vs_gt'][n]['ssim']:.3f}" for n in names))

json.dump(results, open(f"{OUT}/S1A_R1_RENDER_THREEWAY.json","w"), indent=2, default=str)
print("saved S1A_R1_RENDER_THREEWAY.json")
