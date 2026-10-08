import sys
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH0_MX")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/pylibs")
import torch
import mx_probe as MP, mx_perf as PERF

def main():
    trainer, dataset, train_ds, sampler, ad, g = MP.build("/mnt/storage_pool/liaoyuanjun/fused_profile0/profile/mat_room.yaml", "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_room_2026-10-06-01-01-33/checkpoints/final.pt")
    views = list(train_ds); view = views[0]
    st = PERF.h_settings(view, g, 0.0)
    from FasterGSFusedCudaBackend import _C
    (image,pb,tb,ib,bb,n_instances,n_buckets,sel) = _C.forward(g.means,g.raw_scales,g.raw_rotations,g.raw_opacities,g.sh_coefficients_0,g.sh_coefficients_rest,*st.as_tuple())
    torch.cuda.synchronize()
    _,H,W = image.shape
    n_tiles = (W+15)//16 * (H+11)//12
    print("image_shape",image.shape,"H,W",H,W,"n_tiles",n_tiles,"tb",tb.numel(),"n_buckets",n_buckets)
    base=tb.data_ptr(); off=0
    def ob(c,e):
        nonlocal off
        cur=base+off; al=(cur+127)&~127; off=al-base; s=off; off+=c*e; return s
    o1=ob(n_tiles,8); o2=ob(n_tiles*192,4); o3=ob(n_tiles,4); o4=ob(n_tiles,4); o5=ob(n_tiles,4); o6=ob(n_tiles*192,4)
    print("offsets",o1,o2,o3,o4,o5,o6)
    end_fixed = off
    print("end_fixed",end_fixed,"tb",tb.numel(),"cub_ws=",tb.numel()-end_fixed)
    np_bytes = tb[o6:o6+n_tiles*192*4]
    print("slice len",np_bytes.numel(),"div4",np_bytes.numel()%4==0)
    n_processed = np_bytes.cpu().view(torch.int32)
    N_valid_G = int(n_processed.sum().item())
    print("N_valid_G",N_valid_G,"n_buckets*32*192",n_buckets*32*192)

if __name__ == "__main__":
    main()
