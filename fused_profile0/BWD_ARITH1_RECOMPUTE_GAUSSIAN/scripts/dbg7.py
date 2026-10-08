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
    print("n_tiles",n_tiles,"tb",tb.numel(),"n_buckets",n_buckets)
    # n_processed is last fixed field before cub_workspace. off_np = tb - n_tiles*192*4
    np_bytes = n_tiles*192*4
    off_np = tb.numel() - np_bytes
    print("off_np",off_np,"off_np%4",off_np%4)
    n_processed = tb[off_np:off_np+np_bytes].cpu().view(torch.int32)
    N_valid_G = int(n_processed.sum().item())
    print("N_valid_G",N_valid_G,"dense_ub",n_buckets*32*192)
    # also report max n_processed per tile
    np_r = n_processed.view(n_tiles,192)
    print("max_n_processed_per_pixel",int(np_r.max().item()),"mean",float(n_processed.float().mean()))

if __name__ == "__main__":
    main()
