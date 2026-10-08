import os, sys, json, importlib.util, math, random
import numpy as np
import torch
import torch.nn.functional as F

REPO="/root/3dgs-renderer-benchmark"; CKROOT="/root/epic05_p16rb/runs/P12_C0"
TEST_STEP=8  # match Faster-GS MipNeRF360 TEST_STEP
os.environ["HIGS_PX_RUNTIME"]="2"; os.environ["HIGS_BWD_SCALAR_ADJOINT"]="scalar_adjoint"
os.environ["HIGS_BWD_H8_MR"]="1"; os.environ["HIGS_BWD_ABSGRAD"]="1"; os.environ.pop("HIGS_DISABLE_F9",None)
C0_WT="/root/epic05_frozen/higs_c0_final30k_worktree"; SCENE_CUDA_DIR="/root/epic05_build_cache/scene/gsplat_scene_cuda"
CORE_SO="/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
C0_SO="/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
PHASE7_DIR=REPO+"/scripts/epic05/phase7"; SRC_DIR=REPO+"/src"; REFERENCE_V1_DIR=REPO+"/baseline/reference_v1"
def load_ext(name,p):
    s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); sys.modules[name]=m; return m
for p in (C0_WT, SCENE_CUDA_DIR):
    if p not in sys.path: sys.path.insert(0,p)
load_ext("gsplat_cuda",CORE_SO); sys.modules["gsplat.csrc"]=sys.modules["gsplat_cuda"]
load_ext("experimental_gaussian_render_inference_scene_cuda",C0_SO)
sys.modules["gsplat.experimental.render.kernels.csrc"]=sys.modules["experimental_gaussian_render_inference_scene_cuda"]
for p in (PHASE7_DIR,SRC_DIR,REFERENCE_V1_DIR):
    if p not in sys.path: sys.path.insert(0,p)
from gsplat.experimental import rasterize_gaussian_higs_dynamic
from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE
from gaussian_model import GaussianModel
from config import ReferenceV1Config
from dataset import GTDataset
dev="cuda"
config=ReferenceV1Config(); config.repo_root=REPO; config.iterations=30000

class SepSSIM:
    def __init__(self, device="cuda", window_size=11, sigma=1.5):
        coords=torch.arange(window_size,dtype=torch.float32,device=device); coords-=window_size//2
        k1d=torch.exp(-(coords**2)/(2*sigma**2)); k1d=k1d/k1d.sum()
        self.k_h=k1d.unsqueeze(0).unsqueeze(0).repeat(15,1,1,1).contiguous()
        self.k_v=self.k_h.permute(0,1,3,2).contiguous(); self.padding=window_size//2
    def __call__(self,pred,target):
        if pred.ndim==3: pred=pred.unsqueeze(0).permute(0,3,1,2); target=target.unsqueeze(0).permute(0,3,1,2)
        st=torch.cat([pred,target,pred**2,target**2,pred*target],dim=1)
        b=F.conv2d(st,self.k_h,padding=(0,self.padding),groups=15); b=F.conv2d(b,self.k_v,padding=(self.padding,0),groups=15)
        mu_p,mu_t=b[:,0:3],b[:,3:6]; bp2,bt2,bpt=b[:,6:9],b[:,9:12],b[:,12:15]
        mu_p2,mu_t2,mu_pt=mu_p**2,mu_t**2,mu_p*mu_t; sp2,st2,spt=bp2-mu_p2,bt2-mu_t2,bpt-mu_pt
        ssim=(2*mu_pt+0.01**2)*(2*spt+0.03**2)/((mu_p2+mu_t2+0.01**2)*(sp2+st2+0.03**2))
        return 1.0-ssim.mean()
def compute_psnr(pred,gt):
    mse=F.mse_loss(pred,gt); return float(20*math.log10(1.0/math.sqrt(mse.item()))) if mse>1e-10 else 100.0
ssim_fn=SepSSIM(dev)

out={}
SCENES = [sys.argv[1]] if len(sys.argv)>1 else ["room","garden","bicycle"]
for SCENE in SCENES:
    dataset=GTDataset(scene=SCENE, repo_root=REPO, resolution="1080p", device="cuda", background="black")
    n=len(dataset)
    ckpt=torch.load(os.path.join(CKROOT,SCENE,"ckpt_30000.pt"),map_location="cpu",weights_only=False)
    model=GaussianModel(max_sh_degree=config.sh_degree)
    model._xyz=torch.nn.Parameter(ckpt["xyz"].to(dev).clone(),requires_grad=True)
    model._shs=torch.nn.Parameter(ckpt["shs"].to(dev).clone(),requires_grad=True)
    model._scaling=torch.nn.Parameter(ckpt["scaling"].to(dev).clone(),requires_grad=True)
    model._rotation=torch.nn.Parameter(ckpt["rotation"].to(dev).clone(),requires_grad=True)
    model._opacity=torch.nn.Parameter(ckpt["opacity"].to(dev).clone(),requires_grad=True)
    model.active_sh_degree=int(ckpt["active_sh_degree"])
    _HIGS_DYNAMIC_SCENE.mark_dirty()
    def render(ci):
        cam,gt=dataset.get_item(ci)
        with torch.no_grad():
            o=rasterize_gaussian_higs_dynamic(model.get_xyz,model.get_rotation,model.get_scaling,model.get_opacity,model.get_features,sh_degree=model.active_sh_degree,viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),Ks=cam.K.unsqueeze(0).unsqueeze(0),width=cam.image_width,height=cam.image_height,backward_mode="higs_native",enable_culling=True,camera_model="pinhole",render_mode="RGB")
        return o["frame"].reshape(cam.image_height,cam.image_width,3).clamp(0,1), gt
    heldout=list(range(0,n,TEST_STEP))
    allidx=list(range(n))
    def ev(idxs):
        ps=[];ss=[]
        for ci in idxs:
            img,gt=render(ci); ps.append(compute_psnr(img,gt)); ss.append(1.0-ssim_fn(img,gt).item())
        return float(np.mean(ps)),float(np.mean(ss)),len(idxs)
    p_all,s_all,n_all=ev(allidx)
    p_ho,s_ho,n_ho=ev(heldout)
    out[SCENE]={"n_cameras":n,"heldout_step":TEST_STEP,"n_heldout":n_ho,
                "all_cams_trainview_psnr":round(p_all,3),"all_cams_trainview_ssim":round(s_all,4),
                "heldout_pseudo_unseen_psnr":round(p_ho,3),"heldout_pseudo_unseen_ssim":round(s_ho,4)}
    print(f"[{SCENE}] n={n} heldout={n_ho} | ALL(train-view) PSNR={p_all:.3f} SSIM={s_all:.4f} | heldout(pseudo-unseen) PSNR={p_ho:.3f} SSIM={s_ho:.4f}",flush=True)
print("RESULT "+json.dumps(out),flush=True)
json.dump(out,open("/root/sb0_fgs/c0_unseen_eval_"+SCENE+".json","w"),indent=2)