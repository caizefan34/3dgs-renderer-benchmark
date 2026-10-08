
# MSB0-P0 runner: projection -> tile selection -> exact-replay backward fidelity
# -> byte/op accounting -> JSON.  Runs on EPIC-05 (A100). NO CUDA, NO training.
import argparse, json, os, struct, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msb0_oracle import (project_gaussians, composite_tile, backward_tile,
                         autograd_ground_truth, fidelity_metrics, passes_gate4,
                         BUCKET, TILE_W, TILE_H, BLOCK_SIZE_BLEND, MIN_ALPHA_THRESHOLD)

CAM_MODEL_NUM_PARAMS = {0:3,1:4,2:4,3:5,4:8,5:8,6:12,7:5,8:4,9:5,10:12}

def read_colmap_images(path):
    imgs = {}
    with open(os.path.join(path, 'images.bin'), 'rb') as f:
        (n,) = struct.unpack('<Q', f.read(8))
        for _ in range(n):
            iid = struct.unpack('<i', f.read(4))[0]
            qvec = struct.unpack('<4d', f.read(32))
            tvec = struct.unpack('<3d', f.read(24))
            cid = struct.unpack('<i', f.read(4))[0]
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00': break
                name += c
            (n2d,) = struct.unpack('<Q', f.read(8))
            f.read(24*n2d)
            imgs[iid] = (qvec, tvec, cid, name.decode('utf-8', 'replace'))
    return imgs

def qvec_to_rot(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]], dtype=np.float64)

def camera_to_w2c(qvec, tvec):
    R = qvec_to_rot(qvec)
    t = np.array(tvec, dtype=np.float64)
    w2c = np.eye(4, dtype=np.float64)
    w2c[0,:3] = R[0]; w2c[1,:3] = R[1]; w2c[2,:3] = R[2]
    w2c[0,3] = t[0]; w2c[1,3] = t[1]; w2c[2,3] = t[2]
    return w2c

def read_cam_intrinsics(path):
    cams = {}
    with open(os.path.join(path, 'cameras.bin'), 'rb') as f:
        (n,) = struct.unpack('<Q', f.read(8))
        for _ in range(n):
            cid = struct.unpack('<i', f.read(4))[0]
            model = struct.unpack('<i', f.read(4))[0]
            w = struct.unpack('<q', f.read(8))[0]
            h = struct.unpack('<q', f.read(8))[0]
            np_ = CAM_MODEL_NUM_PARAMS[model]
            params = struct.unpack('<' + 'd'*np_, f.read(8*np_))
            cams[cid] = (model, w, h, params)
    return cams


def build_tile_problem(proj, tile_x, tile_y, gw, gh, width, height, bg, device,
                       max_positions=None):
    """Select gaussians whose screen bounds overlap the tile; build per-pixel
    alpha/color for the tile's 16x16 pixels, sorted front-to-back per pixel."""
    x0, x1 = tile_x * TILE_W, min((tile_x + 1) * TILE_W, width)
    y0, y1 = tile_y * TILE_H, min((tile_y + 1) * TILE_H, height)
    P = (x1 - x0) * (y1 - y0)
    # pixel centers
    px = np.repeat(np.arange(x0, x1), (y1 - y0)) + 0.5
    py = np.tile(np.arange(y0, y1), (x1 - x0)) + 0.5
    pixel_xy = torch.tensor(np.stack([px, py], axis=1), dtype=torch.float64, device=device)

    # proj arrays are already filtered to visible gaussians; 'ov' is a mask
    # over that filtered set.  positions index into the filtered arrays.
    bxmin, bxmax, bymin, bymax = proj['bounds']
    ov = (bxmax > tile_x) & (bxmin <= tile_x) & (bymax > tile_y) & (bymin <= tile_y)
    positions = torch.nonzero(ov, as_tuple=False).squeeze(1)
    if positions.numel() == 0:
        return None
    depth_all = proj['depth']
    if max_positions is not None and positions.numel() > max_positions:
        # keep the max_positions nearest to camera (smallest depth), deterministic
        order = torch.argsort(depth_all[positions])
        positions = positions[order[:max_positions]]
    M = positions.numel()
    mean2d = proj['mean2d'][positions].to(torch.float64)     # (M,2)
    conic = proj['conic'][positions].to(torch.float64)       # (M,3)
    opacity = proj['opacity'][positions].to(torch.float64)   # (M,)
    color = proj['color'][positions].to(torch.float64)       # (M,3)
    depth = depth_all[positions].to(torch.float64)           # (M,)

    # per-pixel alpha: (P, M)
    delta = mean2d.unsqueeze(0) - pixel_xy.unsqueeze(1)   # (P,M,2)
    exponent = (-0.5 * (conic[:,0].unsqueeze(0) * delta[:,:,0]**2
                        + conic[:,2].unsqueeze(0) * delta[:,:,1]**2)
                - conic[:,1].unsqueeze(0) * delta[:,:,0] * delta[:,:,1])
    gaussian = torch.exp(torch.clamp(exponent, max=0.0))
    alpha = opacity.unsqueeze(0) * gaussian               # (P,M)
    colors = color.unsqueeze(0).expand(P, M, 3).clone()   # (P,M,3)

    gaussians = {'conic': conic, 'mean2d': mean2d, 'opacity': opacity,
                 'pixel_xy': pixel_xy, 'depth': depth}
    return {'P': P, 'M': M, 'alpha': alpha, 'colors': colors,
            'gaussians': gaussians, 'bg': torch.tensor(bg, dtype=torch.float64, device=device)}


def run_scene(scene, ckpt_path, sparse_dir, device, max_positions=2048):
    _sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt = _sd['model_state_dict'] if 'model_state_dict' in _sd else _sd
    cams = read_cam_intrinsics(sparse_dir)
    imgs = read_colmap_images(sparse_dir)
    # pick image id 0 (deterministic frozen camera)
    iid = sorted(imgs.keys())[0]
    qvec, tvec, cid, name = imgs[iid]
    model, W, H, params = cams[cid]
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    w2c_np = camera_to_w2c(qvec, tvec)
    w2c = torch.tensor(w2c_np, dtype=torch.float64, device=device)
    cam_pos = torch.tensor(-w2c_np[:3,:3].T @ w2c_np[:3,3], dtype=torch.float64, device=device)
    active_bases = 16  # SH_DEGREE 3 -> total_sh_bases_rest = 15, active = 15+1

    proj = project_gaussians(ckpt, w2c, cam_pos, active_bases, W, H,
                             fx, fy, cx, cy, 0.01, 1000.0, device=device)
    gw, gh = proj['gw'], proj['gh']
    bg = [0.0, 0.0, 0.0]

    # choose the tile with the most overlapping gaussians (deterministic, deep compositing)
    idx = proj['idx']; bxmin, bxmax, bymin, bymax = proj['bounds']
    best_tile, best_cnt = None, -1
    for ty in range(gh):
        for tx in range(gw):
            ov = (bxmax > tx) & (bxmin <= tx) & (bymax > ty) & (bymin <= ty)
            c = int(ov.sum())
            if c > best_cnt:
                best_cnt, best_tile = c, (tx, ty)
    tx, ty = best_tile
    prob = build_tile_problem(proj, tx, ty, gw, gh, W, H, bg, device, max_positions)
    if prob is None:
        return {'scene': scene, 'error': 'no tile'}

    saved = composite_tile(prob['alpha'], prob['colors'], prob['gaussians']['opacity'],
                           prob['gaussians'], prob['bg'], prob['M'])
    # upstream gradient: random-but-fixed dL/dpixel
    g = torch.Generator(device=device).manual_seed(1234)
    grad_image = torch.randn(prob['P'], 3, dtype=torch.float64, device=device, generator=g)

    ref = backward_tile(prob['alpha'], prob['colors'], prob['gaussians'], prob['bg'],
                        saved, grad_image, mode='M0')
    # Independent autograd ground truth (validates the hand-derived backward M0)
    ag = autograd_ground_truth(prob['alpha'], prob['colors'],
                               prob['gaussians']['opacity'], prob['gaussians']['conic'],
                               prob['gaussians']['mean2d'], prob['gaussians']['pixel_xy'],
                               prob['bg'], grad_image)
    results = {}
    for mode in ['M0', 'M1', 'M2', 'M3']:
        cand = backward_tile(prob['alpha'], prob['colors'], prob['gaussians'], prob['bg'],
                             saved, grad_image, mode=mode)
        results[mode] = fidelity_metrics(ref, cand)
    # Compare the four TRUE per-Gaussian parameter gradients.  Autograd sums
    # shared-parameter grads over pixels; backward_tile keeps them per-pixel,
    # so sum ref over the pixel dim.  blend opacity grad = gaussian * dL_dalpha.
    opacity = prob['gaussians']['opacity']
    gaussian = prob['alpha'] / opacity.clamp_min(1e-30)        # (P,M)
    ref_sum = {
        'dL_dcolor': ref['dL_dcolor'].sum(0),                  # (M,3)
        'dL_dconic': ref['dL_dconic'].sum(0),                  # (M,3)
        'dL_dmean2d': ref['dL_dmean2d'].sum(0),                # (M,2)
        'dL_dopacity': (gaussian * ref['dL_dalpha']).sum(0),   # (M,)
    }
    ag_sum = {
        'dL_dcolor': ag['dL_dcolor'].sum(0),   # (P,M,3) -> (M,3)
        'dL_dconic': ag['dL_dconic'],
        'dL_dmean2d': ag['dL_dmean2d'],
        'dL_dopacity': ag['dL_dopacity'],
    }
    results['M0_vs_autograd'] = fidelity_metrics(ag_sum, ref_sum)

    # byte accounting (saved blending state, per render)
    n_buckets = saved['n_buckets']
    n_pixels = W * H
    n_tiles = gw * gh
    # bucket_color_transmittance: float4 per (bucket, pixel-in-tile) -> 16 B * BLOCK * n_buckets_total
    # tile_final_transmittances: float per pixel -> 4 B * n_pixels
    # We report per-scene structural bytes for the strongest backend's saved state.
    bytes = {
        'bucket_color_transmittance_bytes_per_bucket_pixel': 16,
        'tile_final_transmittances_bytes_per_pixel': 4,
        'note': 'bucket_color_transmittance = float4(C.rgb,T) per (bucket,tile-pixel); '
                'tile_final_transmittances = float T per pixel',
    }
    return {
        'scene': scene, 'image': name, 'n_gaussians_total': int(ckpt['gaussians._means'].shape[0]),
        'n_visible': int(idx.numel()), 'image_wh': [W, H], 'n_tiles': n_tiles,
        'chosen_tile': [tx, ty], 'tile_overlap_gaussians': best_cnt,
        'tile_positions_replayed': prob['M'], 'n_buckets': n_buckets,
        'fidelity': results, 'gate4_pass_M0': passes_gate4(results['M0']),
        'saved_state_bytes': bytes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--sparse', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-positions', type=int, default=2048)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    res = run_scene(args.scene, args.ckpt, args.sparse, device, args.max_positions)
    res['runtime_s'] = time.time() - t0
    res['device'] = device.type
    if torch.cuda.is_available():
        res['gpu'] = torch.cuda.get_device_name(0)
    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, f"msb0_replay_{args.scene}.json")
    with open(outp, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2)[:1500])
    print('WROTE', outp)

if __name__ == '__main__':
    main()
