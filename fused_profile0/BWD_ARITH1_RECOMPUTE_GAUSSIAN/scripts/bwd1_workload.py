import sys, os, json, argparse
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH0_MX")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/pylibs")
import numpy as np
import torch
import mx_probe as MP, mx_perf as PERF
from FasterGSFusedCudaBackend import _C

TILE_W, TILE_H = 16, 16
BS = TILE_W * TILE_H  # 256
ALIGN = 128

def obtain_off(off, count, elemsize):
    cur = off
    al = (cur + ALIGN - 1) & ~(ALIGN - 1)
    return al, al + count * elemsize

def parse_n_processed_off(n_tiles):
    off = 0
    o_ir, off = obtain_off(off, n_tiles, 8)            # instance_ranges uint2
    o_ft, off = obtain_off(off, n_tiles * BS, 4)       # final_transmittances float
    o_nb, off = obtain_off(off, n_tiles, 4)            # n_buckets uint
    o_bo, off = obtain_off(off, n_tiles, 4)            # buckets_offset uint
    o_mp, off = obtain_off(off, n_tiles, 4)            # max_n_processed uint
    o_np, off = obtain_off(off, n_tiles * BS, 4)       # n_processed uint
    return o_np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--view', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    trainer, dataset, train_ds, sampler, ad, g = MP.build(a.config, a.checkpoint)
    views = list(train_ds); view = views[a.view]
    st = PERF.h_settings(view, g, 0.0)
    (image, pb, tb, ib, bb, n_instances, n_buckets, sel) = _C.forward(
        g.means, g.raw_scales, g.raw_rotations, g.raw_opacities,
        g.sh_coefficients_0, g.sh_coefficients_rest, *st.as_tuple())
    torch.cuda.synchronize()
    _, H, W = image.shape
    n_pixels = W * H
    n_tiles = ((W + TILE_W - 1)//TILE_W) * ((H + TILE_H - 1)//TILE_H)
    tb_bytes = tb.cpu().numpy().tobytes()
    o_np = parse_n_processed_off(n_tiles)
    # n_processed is indexed by GLOBAL pixel index (width*py+px); only first W*H entries valid
    full = np.frombuffer(tb_bytes, dtype=np.uint32, count=n_tiles*BS, offset=o_np)
    valid = full[:n_pixels].astype(np.float64)
    N_valid_G = int(valid.sum())
    max_n_proc = int(valid.max())
    mean_n_proc = float(valid.mean())
    n_zero_pixels = int((valid == 0).sum())
    res = {
        "schema": "BWD_ARITH1_WORKLOAD",
        "scene": a.scene, "view": a.view,
        "n_gaussians": int(g.means.shape[0]),
        "image_wh": [W, H], "n_pixels": int(n_pixels), "n_tiles": int(n_tiles),
        "tile": [TILE_W, TILE_H], "block_size_blend": BS,
        "n_instances": int(n_instances), "n_buckets": int(n_buckets),
        "N_valid_G": N_valid_G,
        "N_processed": N_valid_G,
        "max_n_processed_per_pixel": max_n_proc,
        "mean_n_processed_per_pixel": mean_n_proc,
        "n_zero_processed_pixels": n_zero_pixels,
        "n_valid_pixel_slots": int(n_pixels),
        "n_total_buffer_slots": int(n_tiles * BS),
        "bytes_per_valid_G_FP32": 4,
        "M_G_bytes_exact_FP32": N_valid_G * 4,
        "N_candidate_dense_upper_bound": int(n_buckets) * 32 * BS,
        "M_G_bytes_dense_upper_bound": int(n_buckets) * 32 * BS * 4,
        "method": "parse tile_buffers.n_processed[:W*H] (global-pixel-indexed per-pixel last_contributor); backward evaluates expf for exactly last_contributor primitives per pixel => exact backward recompute count",
    }
    print("DBG_WORKLOAD " + json.dumps(res) + " WL_END", flush=True)
    json.dump(res, open(a.out, 'w'), indent=2)

if __name__ == "__main__":
    main()
