import sys
sys.path.insert(0, "/root/epic05_n1"); sys.path.insert(0, "/root/epic05_frozen/higs_c0_final30k_worktree"); sys.path.insert(0, "/root/epic05_build_cache/scene/gsplat_scene_cuda"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch, numpy as np
import n1_core as C
import cg0_prereg as P
import cg0_compiler as COM
import cg0_appearance as APP
from cg0_run import render_depth
C.bootstrap_c0()
scene="room"; it=15000
ds=C.load_scene_dataset(scene); n=len(ds)
split=P.build_split(n)
ck=C.load_checkpoint(scene,it); Ns=int(ck["xyz"].shape[0]); ss=float(ck["spatial_lr_scale"])
src_shd=int(ck["active_sh_degree"]); src_model=C.build_model(ck)
compile_idx=split["compile64"][:4]
dm={}
for ci in compile_idx:
    cam=ds.get_camera(ci); d,_=render_depth(src_model,cam,src_shd); dm[ci]=d
    print("cam%d depth: shape=%s min=%.3f max=%.3f mean=%.3f finite_frac=%.3f"%(ci,tuple(d.shape),float(d[d>0].min()),float(d.max()),float(d[d>0].mean()),float(torch.isfinite(d).float().mean())))
C.scene_reset()
def gt_get(i): return ds.get_gt_image(i)
cams_c=[ds.get_camera(i) for i in compile_idx]
cand=COM.generate_candidates(dm,cams_c,compile_idx,gt_get,"cuda")
X=cand["X"].cpu().numpy().astype(np.float32)
src_xyz=src_model.get_xyz.detach().cpu().numpy().astype(np.float32)
S=src_xyz[::8]  # ~105K
print("M=%d src_sub=%d scene_scale=%.4f"%(X.shape[0],S.shape[0],ss))
# candidate -> nearest src (chunk)
Xs=X[:20000]
d_c=[]
for i in range(0,Xs.shape[0],2000):
    d=torch.cdist(torch.from_numpy(Xs[i:i+2000]),torch.from_numpy(S))
    d_c.append(d.min(dim=1).values.numpy())
d_c=np.concatenate(d_c)
# src -> nearest src (calibration), subset of S
Ss=S[:5000]
d_s=[]
for i in range(0,Ss.shape[0],500):
    d=torch.cdist(torch.from_numpy(Ss[i:i+500]),torch.from_numpy(S))
    d_s.append(d.min(dim=1).values.numpy())
d_s=np.concatenate(d_s)
print("cand->src NN: median=%.4f p90=%.4f"%(np.median(d_c),np.percentile(d_c,90)))
print("src->src NN (subsample): median=%.4f p90=%.4f"%(np.median(d_s),np.percentile(d_s,90)))
print("ratio cand/src = %.2f"%(np.median(d_c)/np.median(d_s)))
