import sys
sys.path.insert(0, "/root/epic05_n1"); sys.path.insert(0, "/root/epic05_frozen/higs_c0_final30k_worktree"); sys.path.insert(0, "/root/epic05_build_cache/scene/gsplat_scene_cuda"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch, numpy as np
import n1_core as C
import cg0_prereg as P
C.bootstrap_c0()
scene="room"; it=15000
ds=C.load_scene_dataset(scene)
ck=C.load_checkpoint(scene,it); ss=float(ck["spatial_lr_scale"])
src_model=C.build_model(ck)
src_xyz=src_model.get_xyz.detach().cpu().numpy().astype(np.float32)
src_scale=src_model.get_scaling.detach().cpu().numpy().astype(np.float32)
print("scene_scale=%.4f"%ss)
print("src scaling: median=%.5f mean=%.5f p90=%.5f max=%.5f"%(np.median(src_scale),np.mean(src_scale),np.percentile(src_scale,90),src_scale.max()))
# true NN distance with DISJOINT subsamples
A=src_xyz[::16]; B=src_xyz[1::16]   # disjoint, ~52K each
print("A=%d B=%d"%(A.shape[0],B.shape[0]))
d=[]
for i in range(0,A.shape[0],2000):
    dist=torch.cdist(torch.from_numpy(A[i:i+2000]),torch.from_numpy(B))
    d.append(dist.min(dim=1).values.numpy())
d=np.concatenate(d)
print("src A->B NN (disjoint, 16x sub): median=%.5f p90=%.5f"%(np.median(d),np.percentile(d,90)))
print("  -> approx full-density NN ~ median/16^(1/3)=%.5f"%(np.median(d)/16**(1/3)))
