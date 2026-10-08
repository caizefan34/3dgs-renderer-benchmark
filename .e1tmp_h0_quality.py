import os, sys, time, json, importlib.util, glob
os.environ.setdefault("COMAX0_POWER_CLAMP","1")
os.environ["TMPDIR"]="/mnt/storage_pool/liaoyuanjun/comax0_mx/.e1tmp"
import torch, numpy as np
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
from pathlib import Path
ROOT=Path("/mnt/storage_pool/liaoyuanjun/comax0_mx/package/COMAX0_MX_DSH")
sys.path.insert(0,str(ROOT/"scripts")); sys.path.insert(0,str(ROOT/"scripts"/"comax_gs"))
import common
cands=[os.path.expanduser("~/.cache/torch_extensions/py311_cu118/gsplat_cuda/gsplat_cuda.so")]
cands+=sorted(glob.glob(os.path.expanduser("~/.cache/torch_extensions/*/gsplat_cuda/gsplat_cuda.so")),reverse=True)
so=next(p for p in cands if os.path.isfile(p))
spec=importlib.util.spec_from_file_location("gsplat_cuda",so); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.modules["gsplat.csrc"]=m
from gsplat.cuda import _backend
if getattr(_backend,"_C",None) is None: _backend._C=m
from gsplat.rendering import fully_fused_projection as _ffp
print("SHIM_OK",flush=True)
dev="cuda"
CELL=float(os.environ.get("CELL","0.5"))
CAMS=[0,8]
CK="/mnt/storage_pool/liaoyuanjun/retiregs/RETIRE0_room_source/checkpoints/iter_15000.pt"
ck=common.load_checkpoint(CK,device=dev)
gm=common.params_from_ckpt(ck,device=dev)
for _p in (gm._xyz,gm._shs,gm._opacity,gm._scaling,gm._rotation): _p.requires_grad_(False)
# frozen grouping
frozen=common.load_groups("room",17,device=dev)
# voxel grouping (view-independent, cell CELL)
xyz=gm.get_xyz.detach().cpu().numpy().astype(np.float64)
lo=xyz.min(0)
vids=np.floor((xyz-lo)/CELL).astype(np.int64)
key=(vids[:,0]*1000003+vids[:,1]*1000003+vids[:,2]).astype(np.int64)
_,inv=np.unique(key,return_inverse=True)
voxel=torch.from_numpy(inv.astype(np.int32)).to(dev)
print("GROUPS frozen",int(frozen.max())+1,"voxel",int(voxel.max())+1,flush=True)
sys.path.insert(0,common.REPO)
from scripts.epic05.phase7.dataset import GTDataset
ds=GTDataset(scene="room",repo_root=common.REPO,resolution="720p",device=dev,background="black")
met=common.Metrics(device=dev)
bg=torch.zeros(3,device=dev)
out={"scene":"room","resolution":"720p","cell":CELL,"cams":CAMS,"note":"QUALITY-SPEED TRADEOFF probe: PSNR/SSIM of COMAX render (frozen kNN grouping vs coarse VOXEL grouping) vs ordinary-GS baseline, same checkpoint, no training. Quantifies representation change (quality cost) of coarse grouping that raises rho. FUTURE_HYPOTHESIS; NOT frozen COMAX; master verdict unchanged. NUMERICS/PROFILING ONLY."}
for ci in CAMS:
    cam,_=ds.get_item(ci)
    torch.cuda.synchronize(); t0=time.time()
    base=common.render_baseline(gm,cam,background=bg).clamp(0,1)
    torch.cuda.synchronize(); t_base=time.time()-t0
    torch.cuda.synchronize(); t1=time.time()
    cf=common.render_comax(gm,cam,frozen,eta=1.0,background=bg).clamp(0,1)
    torch.cuda.synchronize(); t_froz=time.time()-t1
    torch.cuda.synchronize(); t2=time.time()
    cv=common.render_comax(gm,cam,voxel,eta=1.0,background=bg).clamp(0,1)
    torch.cuda.synchronize(); t_vox=time.time()-t2
    def stats(a,b):
        return {"psnr":round(float(met.psnr(a,b)),4),"ssim":round(float(met.ssim(a,b)),6),
                "max_abs":round(float((a-b).abs().max()),6),"mean_abs":round(float((a-b).abs().mean()),6)}
    out["cam_%d"%ci]={"t_base_s":round(t_base,2),"t_frozen_s":round(t_froz,2),"t_voxel_s":round(t_vox,2),
       "frozen_vs_base":stats(cf,base),"voxel_vs_base":stats(cv,base)}
    print("cam",ci,"frozen psnr",out["cam_%d"%ci]["frozen_vs_base"]["psnr"],"voxel psnr",out["cam_%d"%ci]["voxel_vs_base"]["psnr"],"| frozen ssim",out["cam_%d"%ci]["frozen_vs_base"]["ssim"],"voxel ssim",out["cam_%d"%ci]["voxel_vs_base"]["ssim"],flush=True)
    del base,cf,cv; torch.cuda.empty_cache()
p=Path("/mnt/storage_pool/liaoyuanjun/comax0_mx/COMAX0_H0/H0_QUALITY_TRADEOFF_cell%s.json"%str(CELL).replace(".","p"))
p.write_text(json.dumps(out,indent=2,allow_nan=False))
print("WROTE",p.name,flush=True)
