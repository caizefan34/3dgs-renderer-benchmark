"""C17-2 collect flatten_ids + tile_offsets from real scenes (gsplat env)."""
import json, math, gc, sys
from pathlib import Path
import numpy as np
import torch
torch.manual_seed(0)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from benchmark_framework.scene import load_ply
from benchmark_framework.cameras import load_cameras_from_json, resize_cameras
from gsplat import rasterization

SCENES = {"room": (1080, 1080), "bicycle": (1920, 1080), "garden": (1920, 1080)}
OUT_DIR = REPO / "results" / "phase-a100"
OUT_DIR.mkdir(parents=True, exist_ok=True)
Q = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]
THRESHOLDS = [1, 2, 4, 8, 16, 32, 64]

def pct(v, total): return float(v / total) if total else 0.0

def collect(name, wh):
    w, h = wh
    base = REPO / "data" / "official" / "mipnerf360" / name
    scene = load_ply(str(base / "point_cloud.ply"), device="cuda")
    cams = resize_cameras(load_cameras_from_json(str(base / "cameras.json"), device="cuda"), w, h)
    cam = cams[0]
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = torch.exp(scene["scales"]).contiguous()
    opacities = torch.sigmoid(scene["opacity"]).contiguous()
    vm = cam.viewmatrix.unsqueeze(0); Ks = cam.K.unsqueeze(0)
    with torch.no_grad():
        _, _, meta = rasterization(means=scene["xyz"], quats=quats, scales=scales,
            opacities=opacities, colors=scene["shs"], viewmats=vm, Ks=Ks,
            width=w, height=h, tile_size=16, packed=True,
            sh_degree=scene.get("sh_degree", 3), render_mode="RGB")
    torch.cuda.synchronize()
    flat = meta["flatten_ids"].cpu().numpy().astype(np.int64)
    offsets = meta["isect_offsets"].cpu().numpy()
    gids = meta["gaussian_ids"].cpu().numpy().astype(np.int64)
    depths = meta["depths"].cpu().numpy().astype(np.float64)
    assert offsets.shape[0] == 1
    th, tw = offsets.shape[-2:]
    ntiles, nisect = th * tw, len(flat)
    nmaster = int(scene["num_points"])
    off = offsets[0].reshape(-1).astype(np.int64)
    ends = np.empty_like(off); ends[:-1] = off[1:]; ends[-1] = nisect
    members = gids[flat]
    deltas, run_lengths, ranges, width_bits = [], [], [], []
    zero = 0; compared = 0; small = {str(t): 0 for t in THRESHOLDS}
    depth_nonmonotonic = 0; depth_pairs = 0
    neighbor = {"h": [], "v": [], "ddr": [], "ddl": []}
    shared_dir = {k: [] for k in neighbor}
    t_freq = np.bincount(members, minlength=nmaster)
    exact_dups = 0
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
        exact_dups += n - len(a)
        for lbl, yy, xx in [("h", y, x + 1), ("v", y + 1, x), ("ddr", y + 1, x + 1), ("ddl", y + 1, x - 1)]:
            if yy >= th or xx < 0 or xx >= tw: continue
            ss, ee = int(off[yy * tw + xx]), int(ends[yy * tw + xx])
            b = set(members[ss:ee].tolist())
            inter = len(a & b); union = len(a | b)
            j = inter / union if union else 1.0
            neighbor[lbl].append(j)
            shared_dir[lbl].append({"a_share": inter / len(a) if a else 1.0, "b_share": inter / len(b) if b else 1.0})
    d = np.concatenate(deltas) if deltas else np.empty(0, dtype=np.int64)
    nf = t_freq[t_freq > 0]
    zigzag = (d << 1) ^ (d >> 63)
    vb = np.where(zigzag < (1 << 7), 1, np.where(zigzag < (1 << 14), 2, np.where(zigzag < (1 << 21), 3, np.where(zigzag < (1 << 28), 4, 5))))
    def stat(arr):
        if not isinstance(arr, np.ndarray): arr = np.array(arr)
        if not arr.size: return {"count": 0}
        p = np.quantile(arr.astype(np.float64), Q)
        return {"count": int(arr.size), "min": float(p[0]), "p1": float(p[1]), "p5": float(p[2]),
                "p25": float(p[3]), "p50": float(p[4]), "p75": float(p[5]), "p95": float(p[6]),
                "p99": float(p[7]), "p99_9": float(p[8]), "max": float(p[9]),
                "mean": float(arr.mean()), "std": float(arr.std())}
    return {
        "scene": name, "tile_size": 16, "width": w, "height": h,
        "n_gaussians": nmaster, "n_isects": nisect, "n_tiles": ntiles,
        "flatten_ids": {"dtype": "int32 (CUDA)", "length": nisect, "bytes": nisect * 4,
            "valid_range": [int(members.min()), int(members.max())],
            "global_id_via": "gaussian_ids[flatten_ids]", "per_tile_duplicates": int(exact_dups)},
        "tile_offsets": {"dtype": "int32 (CUDA)", "shape": [1, th, tw], "entries": ntiles, "bytes": ntiles * 4,
            "sentinel": "tile_offsets[t+1] for interior, n_isects for final", "all_ranges_valid": True},
        "within_tile_delta": {"pairs": int(compared), "zero_delta": int(zero), "zero_ratio": pct(zero, compared),
            "small_delta_ratio": {str(t): pct(small[str(t)], compared) for t in THRESHOLDS},
            "stat": stat(d), "leb128_bytes_per_delta": {"mean": float(vb.mean()) if len(vb) else None, "note": "coding-model estimate only"}},
        "cross_tile_overlap": {k: {"jaccard": stat(v), "a_shared_pct": stat([x["a_share"] for x in shared_dir[k]]), "b_shared_pct": stat([x["b_share"] for x in shared_dir[k]]), "pairs": len(v)} for k, v in neighbor.items()},
        "gaussian_tile_membership": stat(nf),
        "run_length": stat(run_lengths),
        "id_range_per_tile": {"global_range": stat(ranges), "bits_for_range": stat(width_bits),
            "global_id_bits_needed": max(math.ceil(math.log2(nmaster)), 1) if nmaster else 0,
            "unused_vs_int32": 32 - max(math.ceil(math.log2(nmaster)), 1) if nmaster else 0},
        "depth_order": {"pairs": int(depth_pairs), "strict_decrease": int(depth_nonmonotonic), "strict_decrease_ratio": pct(depth_nonmonotonic, depth_pairs)},
    }

def main():
    out = {"collection": "deterministic real-scene baseline via gsplat rasterization, no kernel change", "scenes": {}}
    for name, wh in SCENES.items():
        print(f"  {name}...", flush=True)
        out["scenes"][name] = collect(name, wh)
        gc.collect(); torch.cuda.empty_cache()
    torch.save(out, str(OUT_DIR / "c17_2_membership_collected.pt"))
    print(f"SAVED {OUT_DIR / 'c17_2_membership_collected.pt'}", flush=True)

if __name__ == "__main__":
    main()
