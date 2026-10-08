import sys, os
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH0_MX")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/pylibs")
import torch
import mx_probe as MP, mx_perf as PERF
trainer, dataset, train_ds, sampler, ad, g = MP.build("/mnt/storage_pool/liaoyuanjun/fused_profile0/profile/mat_room.yaml", "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_room_2026-10-06-01-01-33/checkpoints/final.pt")
views = list(train_ds); view = views[0]
W=int(view.camera.width); H=int(view.camera.height)
st = PERF.h_settings(view, g, 0.0)
from FasterGSFusedCudaBackend import _C
(image,pb,tb,ib,bb,n_instances,n_buckets,sel) = _C.forward(g.means,g.raw_scales,g.raw_rotations,g.raw_opacities,g.sh_coefficients_0,g.sh_coefficients_rest,*st.as_tuple())
torch.cuda.synchronize()
n_tiles=(W+15)//16*(H+11)//12
print("W,H,n_tiles,tb_bytes,n_instances,n_buckets",W,H,n_tiles,tb.numel(),n_instances,n_buckets)
print("tb dtype",tb.dtype,"shape",tb.shape)
base=tb.data_ptr(); off=0
def ob(c,e):
    global off
    cur=base+off; al=(cur+127)&~127; off=al-base; s=off; off+=c*e; return s
o1=ob(n_tiles,8); o2=ob(n_tiles*192,4); o3=ob(n_tiles,4); o4=ob(n_tiles,4); o5=ob(n_tiles,4); o6=ob(n_tiles*192,4)
print("offsets ir,ft,nb,bo,mp,np:",o1,o2,o3,o4,o5,o6)
print("np bytes=",n_tiles*192*4,"off_np%4=",o6%4)
print("blob end off=",off,"tb=",tb.numel())
