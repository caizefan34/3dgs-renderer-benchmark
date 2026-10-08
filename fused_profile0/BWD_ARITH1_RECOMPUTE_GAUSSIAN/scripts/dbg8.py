import sys, os
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH0_MX")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/pylibs")
import torch
import mx_probe as MP, mx_perf as PERF
from FasterGSFusedCudaBackend import _C

def main():
    trainer, dataset, train_ds, sampler, ad, g = MP.build("/mnt/storage_pool/liaoyuanjun/fused_profile0/profile/mat_room.yaml", "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_room_2026-10-06-01-01-33/checkpoints/final.pt")
    views = list(train_ds); view = views[0]
    st = PERF.h_settings(view, g, 0.0)
    (image,pb,tb,ib,bb,n_instances,n_buckets,sel) = _C.forward(g.means,g.raw_scales,g.raw_rotations,g.raw_opacities,g.sh_coefficients_0,g.sh_coefficients_rest,*st.as_tuple())
    torch.cuda.synchronize()
    print("image.shape",tuple(image.shape),"st.wh",st.width,st.height,"tb",tb.numel(),"n_buckets",n_buckets,"n_instances",n_instances)
    # try both tile assumptions
    for tw,th in [(16,16),(16,12)]:
        _,H,W = image.shape
        nt = (W+tw-1)//tw * (H+th-1)//th
        bs = tw*th
        # replay off_np
        base=tb.data_ptr(); off=0
        def ob(c,e):
            nonlocal off
            cur=base+off; al=(cur+127)&~127; off=al-base; s=off; off+=c*e; return s
        ob(nt,8); ob(nt*bs,4); ob(nt,4); ob(nt,4); ob(nt,4); onp=ob(nt*bs,4)
        print(f"tile {tw}x{th}: n_tiles={nt} off_np={onp} np_bytes={nt*bs*4} end={onp+nt*bs*4} tb={tb.numel()} fits={onp+nt*bs*4<=tb.numel()}")

if __name__=="__main__":
    main()
