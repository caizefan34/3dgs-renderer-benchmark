import os, sys, json, importlib.util, time, random
import numpy as np
import torch
import torch.nn.functional as F

REPO="/root/3dgs-renderer-benchmark"; CKROOT="/root/epic05_p16rb/runs/P12_C0"
SCENE=sys.argv[1] if len(sys.argv)>1 else "bicycle"
NCAM=int(sys.argv[2]) if len(sys.argv)>2 else 8
NITER=int(sys.argv[3]) if len(sys.argv)>3 else 50
NROUND=int(sys.argv[4]) if len(sys.argv)>4 else 5
WARM=5
PROFILE={"room":29.441,"bicycle":41.058,"garden":37.867}

os.environ["HIGS_PX_RUNTIME"]="2"; os.environ["HIGS_BWD_SCALAR_ADJOINT"]="scalar_adjoint"
os.environ["HIGS_BWD_H8_MR"]="1"; os.environ["HIGS_BWD_ABSGRAD"]="1"; os.environ.pop("HIGS_DISABLE_F9",None)
C0_WT="/root/epic05_frozen/higs_c0_final30k_worktree"; SCENE_CUDA_DIR="/root/epic05_build_cache/scene/gsplat_scene_cuda"
CORE_SO="/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
C0_SO="/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"
PHASE7_DIR=REPO+"/scripts/epic05/phase7"; SRC_DIR=REPO+"/src"; REFERENCE_V1_DIR=REPO+"/baseline/reference_v1"
def load_ext(name,p):
    s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); sys.modules[name]=m; return m
for p in (C0_WT,SCENE_CUDA_DIR):
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
config=ReferenceV1Config(); config.scene=SCENE; config.repo_root=REPO; config.iterations=30000
dataset=GTDataset(scene=SCENE, repo_root=REPO, resolution="1080p", device="cuda", background="black")
n_cam=len(dataset)
rng=random.Random(42); stack=list(range(n_cam)); cams=[]
for i in range(NCAM):
    if not stack: stack=list(range(n_cam))
    cams.append(stack.pop(rng.randint(0,len(stack)-1)))
ckpt=torch.load(os.path.join(CKROOT,SCENE,"ckpt_30000.pt"),map_location="cpu",weights_only=False)
model=GaussianModel(max_sh_degree=config.sh_degree)
model._xyz=torch.nn.Parameter(ckpt["xyz"].to(dev).clone(),requires_grad=True)
model._shs=torch.nn.Parameter(ckpt["shs"].to(dev).clone(),requires_grad=True)
model._scaling=torch.nn.Parameter(ckpt["scaling"].to(dev).clone(),requires_grad=True)
model._rotation=torch.nn.Parameter(ckpt["rotation"].to(dev).clone(),requires_grad=True)
model._opacity=torch.nn.Parameter(ckpt["opacity"].to(dev).clone(),requires_grad=True)
model.active_sh_degree=int(ckpt["active_sh_degree"])
model.max_radii2D=ckpt["max_radii2D"].to(dev).clone(); model.xyz_gradient_accum=ckpt["xyz_gradient_accum"].to(dev).clone(); model.denom=ckpt["denom"].to(dev).clone()
model.training_setup({"position_lr_init":config.position_lr_init,"position_lr_final":config.position_lr_final,"position_lr_delay_mult":config.position_lr_delay_mult,"position_lr_max_steps":config.position_lr_max_steps,"feature_lr":config.feature_lr,"opacity_lr":config.opacity_lr,"scaling_lr":config.scaling_lr,"rotation_lr":config.rotation_lr,"percent_dense":config.percent_dense})
N=model._xyz.shape[0]
print("[%s] N=%d n_cam=%d"%(SCENE,N,n_cam),flush=True)
def render_one(ci):
    cam,gt=dataset.get_item(ci)
    out=rasterize_gaussian_higs_dynamic(model.get_xyz,model.get_rotation,model.get_scaling,model.get_opacity,model.get_features,sh_degree=model.active_sh_degree,viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),Ks=cam.K.unsqueeze(0).unsqueeze(0),width=cam.image_width,height=cam.image_height,backward_mode="higs_native",enable_culling=True,camera_model="pinhole",render_mode="RGB")
    image=out["frame"].reshape(cam.image_height,cam.image_width,3)
    return F.l1_loss(image.clamp(0,1),gt)
def measure(tag):
    for i in range(WARM):
        model.optimizer.zero_grad(set_to_none=True); loss=render_one(cams[i%len(cams)]); loss.backward(); model.optimizer.step()
    torch.cuda.synchronize(); fb=[]; op=[]
    for i in range(NITER):
        model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); s0=torch.cuda.Event(True); e0=torch.cuda.Event(True); s0.record()
        loss=render_one(cams[i%len(cams)]); loss.backward(); e0.record(); torch.cuda.synchronize(); fb.append(s0.elapsed_time(e0))
        s1=torch.cuda.Event(True); e1=torch.cuda.Event(True); s1.record(); model.optimizer.step(); e1.record(); torch.cuda.synchronize(); op.append(s1.elapsed_time(e1))
    return float(np.median(fb)), float(np.median(op))
def morton_perm():
    pts=model.get_xyz.detach().cpu().numpy().astype(np.float64); mins=pts.min(0); maxs=pts.max(0)
    scale=(2**20-1)/np.maximum(maxs-mins,1e-9); q=np.clip(((pts-mins)*scale).astype(np.int64),0,2**20-1)
    def p1b2(n):
        n=n&0x1FFFFF; n=(n|(n<<32))&0x1F00000000FFFF; n=(n|(n<<16))&0x1F0000FF0000FF; n=(n|(n<<8))&0x100F00F00F00F00F; n=(n|(n<<4))&0x10C30C30C30C30C3; n=(n|(n<<2))&0x1249249249249249; return n
    code=p1b2(q[:,0])|(p1b2(q[:,1])<<1)|(p1b2(q[:,2])<<2); order=np.argsort(code,kind="stable")
    return torch.from_numpy(order.astype(np.int64)).to(dev)
def apply_perm(perm):
    with torch.no_grad():
        for name in ["_xyz","_shs","_scaling","_rotation","_opacity"]:
            p=getattr(model,name); p.data.copy_(p.data[perm])
        for group in model.optimizer.param_groups:
            for p in group["params"]:
                st=model.optimizer.state.get(p,None)
                if st is not None:
                    for k in ("exp_avg","exp_avg_sq"):
                        if k in st: st[k].copy_(st[k][perm])
perm=morton_perm(); inv=torch.argsort(perm)
# warmup both layouts once
measure("warmA"); apply_perm(perm); measure("warmB"); apply_perm(inv)
As=[]; Bs=[]
for r in range(NROUND):
    a,oa=measure("A%d"%r); apply_perm(perm); b,ob=measure("B%d"%r); apply_perm(inv)
    As.append(a); Bs.append(b)
    print("[%s] round%d A=%.3f B=%.3f  speedup=%.2f%%"%(SCENE,r,a,b,(a-b)/a*100),flush=True)
Amed=float(np.median(As)); Bmed=float(np.median(Bs))
saved=Amed-Bmed; e2e=saved/PROFILE[SCENE]*100
res={"scene":SCENE,"N":int(N),"A_rounds":As,"B_rounds":Bs,"A_median_ms":Amed,"B_median_ms":Bmed,"raster_fwd_bwd_speedup_pct":float((Amed-Bmed)/Amed*100),"e2e_speedup_pct":float(e2e),"profile_iter_ms":PROFILE[SCENE],"saved_ms_per_iter":float(saved)}
print("RESULT "+json.dumps(res),flush=True)