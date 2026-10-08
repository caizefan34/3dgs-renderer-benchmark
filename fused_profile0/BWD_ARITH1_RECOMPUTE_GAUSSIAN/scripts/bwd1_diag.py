import sys, os, json
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH0_MX")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/pylibs")
import numpy as np
import torch
import mx_probe as MP, mx_perf as PERF
from FasterGSFusedCudaBackend import _C

TILE_W, TILE_H = 16, 16
BS = TILE_W * TILE_H
ALIGN = 128

def obtain_off(off, count, elemsize):
    al = (off + ALIGN - 1) & ~(ALIGN - 1)
    return al, al + count * elemsize

def parse_offsets(n_tiles):
    off = 0
    o_ir, off = obtain_off(off, n_tiles, 8)
    o_ft, off = obtain_off(off, n_tiles * BS, 4)
    o_nb, off = obtain_off(off, n_tiles, 4)
    o_bo, off = obtain_off(off, n_tiles, 4)
    o_mp, off = obtain_off(off, n_tiles, 4)
    o_np, off = obtain_off(off, n_tiles * BS, 4)
    return o_ir, o_ft, o_nb, o_bo, o_mp, o_np, off

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--scene', required=True)
    a = ap.parse_args()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    trainer, dataset, train_ds, sampler, ad, g = MP.build(a.config, a.checkpoint)
    views = list(train_ds); view = views[0]
    st = PERF.h_settings(view, g, 0.0)
    (image, pb, tb, ib, bb, n_instances, n_buckets, sel) = _C.forward(
        g.means, g.raw_scales, g.raw_rotations, g.raw_opacities,
        g.sh_coefficients_0, g.sh_coefficients_rest, *st.as_tuple())
    torch.cuda.synchronize()
    _, H, W = image.shape
    n_pixels = W * H
    n_tiles = ((W + TILE_W - 1)//TILE_W) * ((H + TILE_H - 1)//TILE_H)
    tb_bytes = tb.cpu().numpy().tobytes()
    o_ir, o_ft, o_nb, o_bo, o_mp, o_np, end_off = parse_offsets(n_tiles)
    print("SCENE", a.scene, "W,H", W, H, "n_pixels", n_pixels, "n_tiles", n_tiles)
    print("tb.numel", tb.numel(), "len(tb_bytes)", len(tb_bytes), "end_off", end_off)
    print("offsets ir,ft,nb,bo,mp,np:", o_ir, o_ft, o_nb, o_bo, o_mp, o_np)
    # parse n_buckets per tile
    nb_arr = np.frombuffer(tb_bytes, dtype=np.uint32, count=n_tiles, offset=o_nb)
    print("n_buckets: returned", n_buckets, "sum(blob)", int(nb_arr.sum()), "max(blob)", int(nb_arr.max()), "min", int(nb_arr.min()))
    # parse max_n_processed
    mp_arr = np.frombuffer(tb_bytes, dtype=np.uint32, count=n_tiles, offset=o_mp)
    print("max_n_processed: max", int(mp_arr.max()), "mean", float(mp_arr.mean()))
    # parse n_processed
    full = np.frombuffer(tb_bytes, dtype=np.uint32, count=n_tiles*BS, offset=o_np)
    valid = full[:n_pixels].astype(np.float64)
    print("n_processed[:W*H]: sum", int(valid.sum()), "max", int(valid.max()), "mean", float(valid.mean()))
    print("n_processed tail [W*H:]: sum", int(full[n_pixels:].astype(np.float64).sum()), "max", int(full[n_pixels:].max()))
    # consistency: max(mp_arr) should be <= max tile_n_primitives; sum(nb)*32*BS is dense bound
    dense_bound = int(nb_arr.sum()) * 32 * BS
    print("dense_bound(sum_nb*32*BS)", dense_bound, "N_valid_G", int(valid.sum()), "violates", int(valid.sum()) > dense_bound)
    # also check instance_ranges to get tile_n_primitives
    ir = np.frombuffer(tb_bytes, dtype=np.uint32, count=n_tiles*2, offset=o_ir).reshape(n_tiles, 2)
    tile_nprim = (ir[:,1].astype(np.int64) - ir[:,0].astype(np.int64))
    print("tile_n_primitives: max", int(tile_nprim.max()), "sum", int(tile_nprim.sum()), "n_instances_returned", int(n_instances))
    per_pixel_bound = int((tile_nprim.astype(np.float64) * 1).sum())  # sum of tile_nprim * (pixels per tile=256)? no
    # correct bound: sum over pixels of last_contributor <= sum over tiles of 256*tile_nprim
    bound2 = int(tile_nprim.sum()) * BS
    print("bound sum(tile_nprim)*256 =", bound2, "N_valid_G", int(valid.sum()), "violates", int(valid.sum()) > bound2)

if __name__ == "__main__":
    main()
