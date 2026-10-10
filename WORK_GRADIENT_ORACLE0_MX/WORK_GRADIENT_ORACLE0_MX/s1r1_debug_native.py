#!/usr/bin/env python3
"""Debug native FasterGS _C.forward signature in isolation."""
import os, sys, math
import torch
_C_so_dir = "/mnt/storage_pool/liaoyuanjun/fused_profile0/faster-gaussian-splatting/FasterGSFusedCudaBackend/FasterGSFusedCudaBackend"
sys.path.insert(0, _C_so_dir)
os.environ["TORCH_EXTENSIONS_DIR"] = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/torch_ext"
import _C
dev = torch.device("cuda:6")

# tiny synthetic test first
N=100
means=torch.randn(N,3,device=dev)*0.5
means[:,2]+=3.0
quats=torch.zeros(N,4,device=dev); quats[:,0]=1.0
scales=torch.ones(N,3,device=dev)*0.1
opac=torch.full((N,),0.9,device=dev)
sh0=torch.zeros(N,1,3,device=dev)  # grayish
shrest=torch.zeros(N,15,3,device=dev)
bg=torch.zeros(3,device=dev)
w2c=torch.eye(4,device=dev).unsqueeze(0)  # [1,4,4]
cam_position=torch.tensor([[0.0,0.0,0.0]],device=dev)  # [1,3]
W,H=64,64; fx=fy=50.0; cx=cy=31.5
print("shapes:", means.shape, scales.shape, quats.shape, opac.shape, sh0.shape, shrest.shape, bg.shape, w2c.shape, cam_position.shape)
try:
    res=_C.forward(means,scales,quats,opac,sh0,shrest,bg,
                   w2c,cam_position,
                   0,W,H,fx,fy,cx,cy,0.2,100.0,0.0,0)
    print("OK native forward. result tuple len:", len(res))
    for i,r in enumerate(res):
        print(f"  [{i}] type={type(r).__name__} shape={getattr(r,'shape',None)} dtype={getattr(r,'dtype',None)}")
except Exception as e:
    import traceback; traceback.print_exc()
    print("FAILED:", e)
