# MSB0-P0 economics: exact full-image saved-state bytes + per-candidate reduction.
# Runs on EPIC-05 (A100). Reuses project_gaussians. NO training, NO CUDA kernels.
import argparse, json, os, sys, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msb0_oracle import project_gaussians, TILE_W, TILE_H, BUCKET
from msb0_run import read_colmap_images, read_cam_intrinsics, camera_to_w2c

FLOAT4 = 16; FLOAT3 = 12; FLOAT = 4; UINT = 4
BLOCK = TILE_W * TILE_H  # 256 pixels per tile

def per_tile_counts(bounds, gw, gh):
    """bounds = (xmin,xmax,ymin,ymax) tile-coord arrays (half-open [min,max))."""
    xmin, xmax, ymin, ymax = [np.asarray(b) for b in bounds]
    diff = np.zeros((gh + 1, gw + 1), dtype=np.int64)
    np.add.at(diff, (ymin, xmin), 1)
    np.add.at(diff, (ymax, xmin), -1)
    np.add.at(diff, (ymin, xmax), -1)
    np.add.at(diff, (ymax, xmax), 1)
    counts = np.cumsum(np.cumsum(diff, axis=0), axis=1)[:gh, :gw]
    return counts

def run(scene, ckpt_path, sparse_dir, device):
    _sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt = _sd["model_state_dict"] if "model_state_dict" in _sd else _sd
    cams = read_cam_intrinsics(sparse_dir)
    imgs = read_colmap_images(sparse_dir)
    iid = sorted(imgs.keys())[0]
    qvec, tvec, cid, name = imgs[iid]
    model, W, H, params = cams[cid]
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    w2c_np = camera_to_w2c(qvec, tvec)
    w2c = torch.tensor(w2c_np, dtype=torch.float64, device=device)
    cam_pos = torch.tensor(-w2c_np[:3, :3].T @ w2c_np[:3, 3], dtype=torch.float64, device=device)
    proj = project_gaussians(ckpt, w2c, cam_pos, 16, W, H, fx, fy, cx, cy, 0.01, 1000.0, device=device)
    gw, gh = proj["gw"], proj["gh"]
    xmin, xmax, ymin, ymax = proj["bounds"]
    counts = per_tile_counts((xmin.cpu().numpy(), xmax.cpu().numpy(),
                              ymin.cpu().numpy(), ymax.cpu().numpy()), gw, gh)
    n_prims_per_tile = counts.astype(np.int64)
    total_prims = int(n_prims_per_tile.sum())
    buckets_per_tile = (n_prims_per_tile + BUCKET - 1) // BUCKET
    total_buckets = int(buckets_per_tile.sum())
    n_pixels = W * H
    n_tiles = gw * gh
    # saved-state bytes (full image, frozen camera)
    bucket_ct_bytes = total_buckets * BLOCK * FLOAT4          # float4 per (bucket, tile-pixel)
    tile_ft_bytes = n_pixels * FLOAT                          # final transmittance per pixel
    tile_np_bytes = n_pixels * UINT                           # n_processed per pixel
    m0_total = bucket_ct_bytes + tile_ft_bytes + tile_np_bytes
    # M1: drop C_b -> keep T_b only (float per bucket-pixel)
    m1_total = total_buckets * BLOCK * FLOAT + tile_ft_bytes + tile_np_bytes
    # M2: drop T_b -> keep C_b only (float3 per bucket-pixel)
    m2_total = total_buckets * BLOCK * FLOAT3 + tile_ft_bytes + tile_np_bytes
    # M3: forward-carry eliminates bucket_ct entirely
    m3_total = tile_ft_bytes + tile_np_bytes
    def red(m): return (m0_total - m) / m0_total
    return {
        "scene": scene, "image": name, "image_wh": [W, H], "n_tiles": n_tiles,
        "gw": gw, "gh": gh, "n_visible": int(proj["idx"].numel()),
        "primitive_tile_overlaps_total": total_prims, "total_buckets": total_buckets,
        "saved_state_bytes": {
            "bucket_color_transmittance": bucket_ct_bytes,
            "tile_final_transmittances": tile_ft_bytes,
            "tile_n_processed": tile_np_bytes,
            "M0_total": m0_total,
        },
        "candidate_reduction": {
            "M1_drop_Cb": {"bytes": m1_total, "reduction_frac": red(m1_total),
                           "reduction_pct": round(100 * red(m1_total), 3)},
            "M2_drop_Tb": {"bytes": m2_total, "reduction_frac": red(m2_total),
                           "reduction_pct": round(100 * red(m2_total), 3)},
            "M3_forward_carry": {"bytes": m3_total, "reduction_frac": red(m3_total),
                                  "reduction_pct": round(100 * red(m3_total), 3)},
        },
        "note": "bucket_color_transmittance dominates saved state; tile_final_transmittances "
                "and tile_n_processed are per-pixel and ~256x smaller. M1 keeps T_b (float), "
                "M2 keeps C_b (float3), M3 eliminates bucket_ct via sequential forward carry.",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = run(args.scene, args.ckpt, args.sparse, device)
    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, f"msb0_economics_{args.scene}.json")
    with open(outp, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("WROTE", outp)

if __name__ == "__main__":
    main()
