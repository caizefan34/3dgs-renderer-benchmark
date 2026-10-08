import sys
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH0_MX")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/pylibs")
import torch
import mx_probe as MP, mx_perf as PERF

def main():
    trainer, dataset, train_ds, sampler, ad, g = MP.build("/mnt/storage_pool/liaoyuanjun/fused_profile0/profile/mat_room.yaml", "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_room_2026-10-06-01-01-33/checkpoints/final.pt")
    views = list(train_ds); view = views[0]
    print("view.camera WH", int(view.camera.width), int(view.camera.height))
    st = PERF.h_settings(view, g, 0.0)
    print("settings WH", st.width, st.height)
    from FasterGSFusedCudaBackend import _C
    (image,pb,tb,ib,bb,n_instances,n_buckets,sel) = _C.forward(g.means,g.raw_scales,g.raw_rotations,g.raw_opacities,g.sh_coefficients_0,g.sh_coefficients_rest,*st.as_tuple())
    torch.cuda.synchronize()
    print("image.shape", tuple(image.shape), "tb", tb.numel(), "n_buckets", n_buckets, "n_instances", n_instances)
    _,H,W = image.shape
    for tw,th in [(16,12),(16,16)]:
        nt = (W+tw-1)//tw * (H+th-1)//th
        print(f"tile {tw}x{th}: n_tiles={nt}, nt*192*4={nt*192*4}")

if __name__ == "__main__":
    main()
