import os, sys, time, json, importlib.util, glob, math
os.environ.setdefault("COMAX0_POWER_CLAMP","1")
os.environ["TMPDIR"]="/mnt/storage_pool/liaoyuanjun/comax0_mx/.e1tmp"
import numpy as np
import torch
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
from gsplat.rendering import fully_fused_projection as _ffp, isect_tiles, isect_offset_encode
print("SHIM_OK",flush=True)
dev="cuda"
CELL=float(os.environ["CELL"])
CK="/mnt/storage_pool/liaoyuanjun/retiregs/RETIRE0_room_source/checkpoints/iter_15000.pt"
ck=common.load_checkpoint(CK,device=dev)
gm=common.params_from_ckpt(ck,device=dev)
for _p in (gm._xyz,gm._shs,gm._opacity,gm._scaling,gm._rotation): _p.requires_grad_(False)
op_all=gm.get_opacity
# ---- build voxel grouping (view-independent, coarse spatial) ----
xyz=gm.get_xyz.detach().cpu().numpy().astype(np.float64)
lo=xyz.min(0); 
vids=np.floor((xyz-lo)/CELL).astype(np.int64)
# pack (ix,iy,iz) into a single int key
key=(vids[:,0]*1000003 + vids[:,1]*1000003 + vids[:,2]).astype(np.int64)
uniq,inv=np.unique(key,return_inverse=True)
gid=inv.astype(np.int32)
G=int(uniq.size)
print("VOXEL cell",CELL,"N",gid.size,"G",G,"mean",round(gid.size/G,3),"max",round((gid.size/G),1),flush=True)
groups=torch.from_numpy(gid).to(dev)
sys.path.insert(0,common.REPO)
from scripts.epic05.phase7.dataset import GTDataset
ds=GTDataset(scene="room",repo_root=common.REPO,resolution="1080p",device=dev,background="black")
cam_i=0
cam,_=ds.get_item(cam_i)
torch.cuda.synchronize()
radii,means2d,depths,conics,_=_ffp(means=gm.get_xyz,covars=None,quats=gm.get_rotation,scales=gm.get_scaling,viewmats=cam.viewmatrix.unsqueeze(0),Ks=cam.K.unsqueeze(0),width=cam.image_width,height=cam.image_height,eps2d=0.3,near_plane=0.01,far_plane=1e10,radius_clip=0.0,packed=False,opacities=op_all)
torch.cuda.synchronize()
r0=radii[0]; visible=(r0>0).any(dim=1)
c0=conics[0]; ca=c0[:,0]; cb=c0[:,1]; cq=c0[:,2]
gidv=groups[visible]; m2v=means2d[0][visible]; opv=op_all[visible]
Gv=int(gidv.max().item())+1
lo2=torch.zeros(Gv,2,device=dev); hi2=torch.zeros(Gv,2,device=dev)
lo2.index_reduce_(0,gidv,m2v,"amin",include_self=False); hi2.index_reduce_(0,gidv,m2v,"amax",include_self=False)
torch.cuda.synchronize()
m3=means2d.unsqueeze(0) if means2d.dim()==2 else means2d
r3=radii.unsqueeze(0) if radii.dim()==2 else radii
d3=depths.unsqueeze(0) if depths.dim()==1 else depths
tw=(cam.image_width+15)//16; th=(cam.image_height+15)//16; n_tiles=tw*th
_tpg,_ids,flatten=isect_tiles(means2d=m3,radii=r3.clamp(min=0),depths=d3,tile_size=16,tile_width=tw,tile_height=th,sort=True,packed=False,n_images=1)
Fp=int(flatten.numel())
offs=torch.cat([isect_offset_encode(_ids,1,tw,th).reshape(-1).to(torch.long),torch.tensor([Fp],device=dev)])
torch.manual_seed(1234+cam_i)
NSAMP=16
sampled=torch.randperm(n_tiles,device=dev)[:NSAMP]
s_idx=offs[sampled]; e_idx=offs[sampled+1]; counts=e_idx-s_idx
cumcounts=torch.cat([torch.zeros(1,dtype=torch.long,device=dev),counts.cumsum(0)[:-1]])
local_ar=torch.arange(int(counts.sum()),device=dev)-torch.repeat_interleave(cumcounts,counts)
fp_idx=torch.repeat_interleave(s_idx,counts)+local_ar
ids=flatten[fp_idx]; g_ids=groups[ids]
m2=means2d[0][ids]; opi=op_all[ids]; con=conics[0][ids]
ca_i=con[:,0]; cb_i=con[:,1]; cq_i=con[:,2]
rx=r0[ids][:,0]; ry=r0[ids][:,1]
gid_all=g_ids.tolist(); opi_all=opi.tolist(); ca_l=ca_i.tolist(); cb_l=cb_i.tolist(); cq_l=cq_i.tolist()
mx_l=m2[:,0].tolist(); my_l=m2[:,1].tolist(); rx_l=rx.tolist(); ry_l=ry.tolist()
cnt=counts.tolist(); cum=cumcounts.tolist()
px_canon=0; px_rej=0; ghit_total=0; npix=0
for bi in range(NSAMP):
    s=cum[bi]; n=cnt[bi]
    if n<=0: continue
    tile_id=int(sampled[bi].item()); tox=(tile_id%tw)*16; toy=(tile_id//tw)*16
    for py in range(16):
        for pxx in range(16):
            cxp=tox+pxx; cyp=toy+py
            runmax={}; gcount={}; pcan=0; prej=0
            for j in range(s,s+n):
                if abs(cxp-mx_l[j])>rx_l[j] or abs(cyp-my_l[j])>ry_l[j]: continue
                a=opi_all[j]*math.exp(-0.5*(ca_l[j]*(cxp-mx_l[j])**2+2*cb_l[j]*(cxp-mx_l[j])*(cyp-my_l[j])+cq_l[j]*(cyp-my_l[j])**2))
                if a<0.0: a=0.0
                elif a>1.0: a=1.0
                g=gid_all[j]; gcount[g]=gcount.get(g,0)+1
                pcan+=1
                rm=runmax.get(g)
                if rm is None: runmax[g]=a
                elif a>rm: runmax[g]=a
                else: prej+=1
            if pcan>0:
                px_canon+=pcan; px_rej+=prej; ghit_total+=len(gcount); npix+=1
rho=px_rej/max(px_canon,1); eff_g=ghit_total/max(npix,1); kbar=px_canon/max(1,sum(1 for _ in range(0)) or 1)
out={"cell":CELL,"G":Gv,"per_pixel_canon":px_canon,"per_pixel_rej":px_rej,"rho_per_pixel":round(rho,4),
  "eff_groups_per_pixel":round(eff_g,3),"mean_k_per_group_per_pixel":round(px_canon/max(1.0,eff_g*npix/max(npix,1)),4),
  "arith_ceiling":round(1/(1-rho),4),"nsamp":NSAMP,
  "note":"FUTURE_HYPOTHESIS method-version exploration: view-independent coarse VOXEL grouping (cell=CELL m) replaces the frozen kNN grouping. Tests whether per-ray overlap k (and rho) can exceed the frozen k~2 / rho~0.36. NOT the frozen COMAX; master verdict unchanged. NUMERICS/PROFILING ONLY."}
p=Path("/mnt/storage_pool/liaoyuanjun/comax0_mx/COMAX0_H0/H0_VOXEL_RHO_cell%s.json"%str(CELL).replace(".","p"))
p.write_text(json.dumps(out,indent=2,allow_nan=False))
print("WROTE",p.name,"rho",round(rho,4),"ceil",round(1/(1-rho),4),"eff_g",round(eff_g,1),flush=True)
