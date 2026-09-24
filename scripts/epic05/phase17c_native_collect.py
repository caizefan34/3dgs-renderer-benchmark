"""C17-2 Data Integrity: Native-resolution membership collection + cross-tile audit.
Uses raw PLY but at native camera resolution (matching corrected baseline camera parameter).
"""
import json, math, gc, sys, time
from pathlib import Path
import numpy as np
import torch
torch.manual_seed(0)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from benchmark_framework.scene import load_ply
from benchmark_framework.cameras import load_cameras_from_json, resize_cameras
from gsplat import rasterization

# NATIVE camera resolution from corrected baseline
SCENES_NATIVE = {
    "room":    {"w": 3114, "h": 2075},
    "bicycle": {"w": 4946, "h": 3286},
    "garden":  {"w": 5187, "h": 3361},
}
OUT_DIR = REPO / "results" / "phase-a100"
OUT_DIR.mkdir(parents=True, exist_ok=True)
Q = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]
THRESHOLDS = [1, 2, 4, 8, 16, 32, 64]

def pct(v, total): return float(v / total) if total else 0.0

def stat(arr):
    if not isinstance(arr, np.ndarray): arr = np.array(arr)
    if not arr.size: return {"count": 0}
    p = np.quantile(arr.astype(np.float64), Q)
    return {"count": int(arr.size), "min": float(p[0]), "p1": float(p[1]), "p5": float(p[2]),
            "p25": float(p[3]), "p50": float(p[4]), "p75": float(p[5]), "p95": float(p[6]),
            "p99": float(p[7]), "p99_9": float(p[8]), "max": float(p[9]),
            "mean": float(arr.mean()), "std": float(arr.std())}

def collect_native(name, w, h):
    base = REPO / "data" / "official" / "mipnerf360" / name
    scene = load_ply(str(base / "point_cloud.ply"), device="cuda")
    # Load all cameras, use camera 0 (same as corrected baseline)
    cams = load_cameras_from_json(str(base / "cameras.json"), device="cuda")
    cam = cams[0]
    # Camera at native resolution (cam has native width/height from JSON)
    # Build viewmat/K directly from camera 0 at native resolution
    # Note: we need viewmatrix at native res, not resized
    vm = cam.viewmatrix.unsqueeze(0)
    # Build native K from camera intrinsic
    K = cam.K.unsqueeze(0)  # This is the K at native resolution
    # But load_cameras_from_json may already store native parameters
    # Let's verify by checking shapes
    
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = torch.exp(scene["scales"]).contiguous()
    opacities = torch.sigmoid(scene["opacity"]).contiguous()
    
    with torch.no_grad():
        _, _, meta = rasterization(means=scene["xyz"], quats=quats, scales=scales,
            opacities=opacities, colors=scene["shs"], viewmats=vm, Ks=K,
            width=w, height=h, tile_size=16, packed=True,
            sh_degree=scene.get("sh_degree", 3), render_mode="RGB")
    torch.cuda.synchronize()
    
    flat = meta["flatten_ids"].cpu().numpy().astype(np.int64)
    offsets = meta["isect_offsets"].cpu().numpy()
    gids = meta["gaussian_ids"].cpu().numpy().astype(np.int64)
    depths = meta["depths"].cpu().numpy().astype(np.float64)
    isect_ids = meta["isect_ids"].cpu().numpy().astype(np.int64)
    assert offsets.shape[0] == 1
    th, tw = offsets.shape[-2:]
    ntiles, nisect = th * tw, len(flat)
    nmaster = int(scene["num_points"])
    off = offsets[0].reshape(-1).astype(np.int64)
    ends = np.empty_like(off); ends[:-1] = off[1:]; ends[-1] = nisect
    members = gids[flat]
    
    # --- Within-tile delta ---
    deltas, run_lengths, ranges, width_bits = [], [], [], []
    zero = 0; compared = 0; small = {str(t): 0 for t in THRESHOLDS}
    depth_nonmonotonic = 0; depth_pairs = 0
    exact_dups = 0
    
    # --- Cross-tile overlap ---
    neighbor = {"h": [], "v": [], "ddr": [], "ddl": []}
    shared_dir = {k: [] for k in neighbor}
    # Order consistency: for shared pairs in adjacent tiles
    order_agreement = {"h": [], "v": [], "ddr": [], "ddl": []}
    
    t_freq = np.bincount(members, minlength=nmaster)
    
    for t in range(ntiles):
        s, e = int(off[t]), int(ends[t])
        seq = members[s:e]
        dseq = depths[flat[s:e]]
        n = e - s
        if n:
            r = int(seq.max() - seq.min())
            ranges.append(r); width_bits.append(max(math.ceil(math.log2(r + 1)), 0) if r else 0)
        if n > 1:
            ds = np.diff(seq); ad = np.abs(ds)
            deltas.append(ds); compared += len(ds)
            zero += int((ds == 0).sum())
            for x in THRESHOLDS: small[str(x)] += int((ad <= x).sum())
            dd = np.diff(dseq)
            depth_pairs += len(dd)
            depth_nonmonotonic += int((dd < 0).sum())
            good = ad == 1
            if good.any():
                cuts = np.flatnonzero(np.diff(np.r_[False, good, False]))
                run_lengths.extend((cuts[1::2] - cuts[::2] + 1).tolist())
        
        y, x = divmod(t, tw)
        a = set(seq.tolist())
        # Map gid → position index for order consistency
        # (lower index = farther from camera in depth sort = rendered earlier)
        a_pos = {gid: i for i, gid in enumerate(seq.tolist())}
        exact_dups += n - len(a)
        
        for lbl, yy, xx in [("h", y, x + 1), ("v", y + 1, x),
                            ("ddr", y + 1, x + 1), ("ddl", y + 1, x - 1)]:
            if yy >= th or xx < 0 or xx >= tw:
                continue
            ss, ee = int(off[yy * tw + xx]), int(ends[yy * tw + xx])
            bseq = members[ss:ee].tolist()
            b = set(bseq)
            inter = len(a & b); union = len(a | b)
            j = inter / union if union else 1.0
            neighbor[lbl].append(j)
            shared_dir[lbl].append({
                "a_share": inter / len(a) if a else 1.0,
                "b_share": inter / len(b) if b else 1.0
            })
            
            # Order consistency for shared Gaussians
            if inter > 1 and a and b:
                b_pos = {gid: i for i, gid in enumerate(bseq)}
                shared = list(a & b)
                agreements = 0
                pairs_checked = 0
                for i in range(len(shared)):
                    for j in range(i + 1, len(shared)):
                        gi, gj = shared[i], shared[j]
                        if gi in a_pos and gj in a_pos and gi in b_pos and gj in b_pos:
                            pairs_checked += 1
                            # Both are depth-sorted, so depth order = position order
                            # In both tiles, lower index = further from camera
                            if (a_pos[gi] < a_pos[gj]) == (b_pos[gi] < b_pos[gj]):
                                agreements += 1
                if pairs_checked > 0:
                    order_agreement[lbl].append(agreements / pairs_checked)
    
    d = np.concatenate(deltas) if deltas else np.empty(0, dtype=np.int64)
    nf = t_freq[t_freq > 0]
    zigzag = (d << 1) ^ (d >> 63)
    vb = np.where(zigzag < (1 << 7), 1, np.where(zigzag < (1 << 14), 2,
        np.where(zigzag < (1 << 21), 3, np.where(zigzag < (1 << 28), 4, 5))))
    
    return {
        "scene": name, "tile_size": 16, "width": w, "height": h,
        "n_gaussians": nmaster, "n_isects": nisect, "n_tiles": ntiles,
        "tile_grid": f"{tw}x{th}",
        "n_visible_gaussians": int(nf.sum() / (nisect / nf.sum() if nf.sum() > 0 else 1)),
        "flatten_ids": {"dtype": "int32 (CUDA)", "length": nisect, "bytes": nisect * 4,
            "valid_range": [int(members.min()), int(members.max())],
            "per_tile_duplicates": int(exact_dups)},
        "tile_offsets": {"dtype": "int32 (CUDA)", "shape": [1, th, tw],
            "entries": ntiles, "bytes": ntiles * 4},
        "gaussian_tile_membership": stat(nf),
        "within_tile_delta": {"pairs": int(compared), "zero_delta": int(zero),
            "zero_ratio": pct(zero, compared),
            "small_delta_ratio": {str(t): pct(small[str(t)], compared) for t in THRESHOLDS},
            "stat": stat(d),
            "leb128_bytes_per_delta": {"mean": float(vb.mean()) if len(vb) else None}},
        "run_length": stat(run_lengths),
        "id_range_per_tile": {"global_range": stat(ranges),
            "bits_for_range": stat(width_bits),
            "global_id_bits_needed": max(math.ceil(math.log2(nmaster)), 1) if nmaster else 0,
            "unused_vs_int32": max(32 - max(math.ceil(math.log2(nmaster)), 1), 0) if nmaster else 0},
        "depth_order": {"pairs": int(depth_pairs),
            "strict_decrease": int(depth_nonmonotonic),
            "strict_decrease_ratio": pct(depth_nonmonotonic, depth_pairs)},
        "cross_tile_overlap": {k: {"jaccard": stat(v),
            "a_shared_pct": stat([x["a_share"] for x in shared_dir[k]]),
            "b_shared_pct": stat([x["b_share"] for x in shared_dir[k]]),
            "pairs": len(v)} for k, v in neighbor.items()},
        "relative_order_consistency": {k: stat(v) for k, v in order_agreement.items()},
    }

def main():
    results = {"audit": "C17-2 Data Integrity + Cross-Tile Overlap",
        "status": "OPTIMIZATION: NOT STARTED",
        "n_isect_reconciliation": {},
        "cross_tile_overlap": {},
        "representation_properties": {},
        "scenes": {}}
    
    for name in ["room", "bicycle", "garden"]:
        wh = SCENES_NATIVE[name]
        print(f"\n{'='*60}", flush=True)
        print(f"  COLLECT: {name} at native {wh['w']}x{wh['h']}", flush=True)
        print(f"  Tile grid: {math.ceil(wh['w']/16)}x{math.ceil(wh['h']/16)} = {math.ceil(wh['w']/16)*math.ceil(wh['h']/16)}", flush=True)
        t0 = time.time()
        try:
            result = collect_native(name, wh["w"], wh["h"])
            results["scenes"][name] = result
            elapsed = time.time() - t0
            print(f"  DONE in {elapsed:.1f}s: isects={result['n_isects']:,} tiles={result['n_tiles']:,} visible_G={result.get('n_visible_gaussians', '?'):,}", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results["scenes"][name] = {"error": str(e)}
        gc.collect(); torch.cuda.empty_cache()
    
    out_path = OUT_DIR / "c17_2_membership_native.pt"
    torch.save(results, str(out_path))
    print(f"\nSAVED {out_path}", flush=True)

if __name__ == "__main__":
    main()
