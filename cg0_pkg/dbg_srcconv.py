import sys
sys.path.insert(0, "/root/epic05_n1"); sys.path.insert(0, "/root/epic05_frozen/higs_c0_final30k_worktree"); sys.path.insert(0, "/root/epic05_build_cache/scene/gsplat_scene_cuda"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/scripts/epic05/phase7"); sys.path.insert(0, "/root/3dgs-renderer-benchmark/baseline/reference_v1")
import torch, n1_core as C
import cg0_prereg as P
C.bootstrap_c0()
ck=C.load_checkpoint("room",15000)
shs=ck["shs"]
print("src shs shape", tuple(shs.shape))
print("src shs[:,0,:] mean", [round(x,4) for x in shs[:,0,:].mean(0).tolist()], "std", [round(x,4) for x in shs[:,0,:].std(0).tolist()])
C0=P.C0
print("src DC color (0.5+C0*sh0) mean", [round(x,4) for x in (0.5+C0*shs[:,0,:]).clamp(0,1).mean(0).tolist()])
print("src active_sh_degree", int(ck["active_sh_degree"]))
from gaussian_model import GaussianModel
m=C.build_model(ck)
print("get_features shape", tuple(m.get_features.shape), "get_features[:,0,:] mean", [round(x,4) for x in m.get_features[:,0,:].mean(0).tolist()])
