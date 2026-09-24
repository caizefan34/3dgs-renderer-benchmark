"""C17-2 Cross-tile differential feasibility — targeted CPU analysis.
Phase 1: Capture raw tensors from native-resolution gsplat.
Phase 2: Stratified-sampled pair analysis (set, order, insertion, LCS).
Phase 3: Tile-path sequential scan.
"""
import gc, json, math, sys, time, random
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from benchmark_framework.scene import load_ply
from benchmark_framework.cameras import load_cameras_from_json
from gsplat import rasterization

OUT_DIR = REPO / "results" / "phase-a100"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENES = {
    "room":    {"w": 3114, "h": 2075},
    "bicycle": {"w": 4946, "h": 3286},
}
PS = [25, 50, 75, 90, 95, 99]

def pctile(arr, percentiles=PS):
    if not isinstance(arr, np.ndarray): arr = np.array(arr)
    if not arr.size: return {}
    p = np.percentile(arr.astype(np.float64), percentiles)
    return {f"p{pp}": float(v) for pp, v in zip(percentiles, p)} | {"mean": float(arr.mean()), "max": float(arr.max()), "min": float(arr.min())}

def capture(name, w, h):
    base = REPO / "data" / "official" / "mipnerf360" / name
    scene = load_ply(str(base / "point_cloud.ply"), device="cuda")
    cams = load_cameras_from_json(str(base / "cameras.json"), device="cuda")
    cam = cams[0]
    vm = cam.viewmatrix.unsqueeze(0); K = cam.K.unsqueeze(0)
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
    del scene, cams, meta, quats, scales, opacities, vm, K
    gc.collect(); torch.cuda.empty_cache()
    return {"flat": flat, "offsets": offsets, "gids": gids}

def extract_membership(flat, offsets, gids):
    """Extract per-tile ordered membership arrays."""
    assert offsets.shape[0] == 1
    th, tw = offsets.shape[-2:]
    off = offsets[0].reshape(-1).astype(np.int64)
    ends = np.empty_like(off); ends[:-1] = off[1:]; ends[-1] = len(flat)
    members = gids[flat]
    tiles = []
    for t in range(len(off)):
        s, e = int(off[t]), int(ends[t])
        tiles.append(members[s:e].copy())
    return tiles, th, tw

def find_neighbors(t, tw, th):
    y, x = divmod(t, tw)
    n = {}
    if x + 1 < tw: n["h"] = y * tw + (x + 1)
    if y + 1 < th: n["v"] = (y + 1) * tw + x
    if y + 1 < th and x + 1 < tw: n["ddr"] = (y + 1) * tw + (x + 1)
    if y + 1 < th and x - 1 >= 0: n["ddl"] = (y + 1) * tw + (x - 1)
    return n

def lcs_length(seq_a, seq_b, max_size=5000):
    """Fast LCS length using unique ID intersection + shared subsequence."""
    # For Gaussian IDs in depth order, the shared subsequence IS the LCS
    # because order preservation holds. So LCS = |shared ∩ ordered subsequence|
    a_set = set(seq_a.tolist())
    b_set = set(seq_b.tolist())
    shared = a_set & b_set
    sub_a = [x for x in seq_a.tolist() if x in shared]
    sub_b = [x for x in seq_b.tolist() if x in shared]
    if sub_a == sub_b:
        return len(shared), True
    else:
        # If not identical subsequence, do actual LCS DP (expensive)
        # Only on sampled/small cases
        if len(seq_a) * len(seq_b) < max_size:
            return _lcs_dp(seq_a, seq_b), False
        else:
            return len(shared), False  # fallback: intersection size

def _lcs_dp(a, b):
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(1, m + 1):
        ai = a[i - 1]
        dp_row = dp[i]
        dp_prev = dp[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                dp_row[j] = dp_prev[j - 1] + 1
            else:
                dp_row[j] = max(dp_prev[j], dp_row[j - 1])
    return int(dp[m, n])

def compute_block_structure(insert_only, b_ids):
    """Compute insertion block lengths from the set of new IDs and full target list."""
    if len(insert_only) <= 1:
        return [len(insert_only)]
    b_pos = {gid: i for i, gid in enumerate(b_ids.tolist())}
    positions = sorted([b_pos[g] for g in insert_only])
    blocks = []
    cur = [positions[0]]
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] == 1:
            cur.append(positions[i])
        else:
            blocks.append(len(cur))
            cur = [positions[i]]
    blocks.append(len(cur))
    return blocks

def analyze_scene(name, scene_data, sample_count=2000):
    tiles, th, tw = extract_membership(scene_data["flat"], scene_data["offsets"], scene_data["gids"])
    ntiles = len(tiles)
    random.seed(42)
    
    results = {
        "scene": name, "ntiles": ntiles, "th": th, "tw": tw,
        "membership_size": {"n_isects": int(scene_data["flat"].shape[0])},
        "tile_stats": {},
    }
    
    # Phase 1: Full scan — count empty tiles, size distribution
    tile_sizes = np.array([len(t) for t in tiles], dtype=np.int64)
    results["tile_stats"] = {
        "size": pctile(tile_sizes),
        "empty_tiles": int((tile_sizes == 0).sum()),
        "nonempty_tiles": int((tile_sizes > 0).sum()),
    }
    
    # Phase 2: Sample random tile pairs for each direction
    dir_counts = {"h": 0, "v": 0, "ddr": 0, "ddl": 0}
    dir_data = {k: [] for k in dir_counts}
    
    # Collect all neighbor pairs, weighted by direction
    all_pairs = []
    for t in range(ntiles):
        if len(tiles[t]) == 0:
            continue
        neigh = find_neighbors(t, tw, th)
        for label, nt in neigh.items():
            if len(tiles[nt]) == 0:
                continue
            all_pairs.append((label, t, nt))
    
    random.shuffle(all_pairs)
    
    # Detailed analysis on sampled pairs
    sample_pairs = all_pairs[:min(sample_count, len(all_pairs))]
    
    for label, t_a, t_b in sample_pairs:
        a_ids, b_ids = tiles[t_a], tiles[t_b]
        len_a, len_b = len(a_ids), len(b_ids)
        a_set = set(a_ids.tolist())
        b_set = set(b_ids.tolist())
        shared = a_set & b_set
        insert_only = b_set - a_set
        remove_only = a_set - b_set
        
        len_s = len(shared)
        len_i = len(insert_only)
        
        pair = {
            "len_A": len_a, "len_B": len_b, "len_S": len_s, "len_I": len_i,
            "rho_I": len_i / len_b if len_b else 0,
        }
        
        # Order-preserving subsequence
        if len_s > 0:
            sub_a = [x for x in a_ids.tolist() if x in shared]
            sub_b = [x for x in b_ids.tolist() if x in shared]
            pair["order_preserving_subseq"] = (sub_a == sub_b)
        else:
            pair["order_preserving_subseq"] = True
        
        # Insertion positions (normalized)
        if len_i > 0:
            b_pos = {gid: i for i, gid in enumerate(b_ids.tolist())}
            pos_norm = [b_pos[g] / len_b for g in insert_only]
            pair["insert_positions_norm"] = pos_norm
            
            # Block structure
            blocks = compute_block_structure(insert_only, b_ids)
            pair["insert_blocks"] = blocks
        
        dir_data[label].append(pair)
    
    # Aggregate per direction
    for label in dir_counts:
        pairs = dir_data[label]
        if not pairs:
            continue
        results[f"dir_{label}"] = {
            "pairs": len(pairs),
            "len_B": pctile([p["len_B"] for p in pairs]),
            "len_S": pctile([p["len_S"] for p in pairs]),
            "len_I": pctile([p["len_I"] for p in pairs]),
            "rho_I": pctile([p["rho_I"] for p in pairs]),
            "order_preserving_subseq": {
                "count": sum(1 for p in pairs if p["order_preserving_subseq"]),
                "total": len(pairs),
                "ratio": sum(1 for p in pairs if p["order_preserving_subseq"]) / len(pairs),
            },
        }
        
        # Insertion position statistics
        all_pos = []
        for p in pairs:
            all_pos.extend(p.get("insert_positions_norm", []))
        if all_pos:
            results[f"dir_{label}"]["insert_pos_norm"] = pctile(all_pos)
        
        # Insertion block statistics
        all_blocks = []
        for p in pairs:
            all_blocks.extend(p.get("insert_blocks", []))
        if all_blocks:
            results[f"dir_{label}"]["insert_blocks"] = pctile(all_blocks)
    
    # Phase 3: Row-major path analysis
    path_deltas = []
    path_I = []
    path_B_sizes = []
    for t in range(ntiles - 1):
        if (t + 1) % tw == 0:  # row boundary
            continue
        if len(tiles[t]) == 0 or len(tiles[t + 1]) == 0:
            continue
        a_ids, b_ids = tiles[t], tiles[t + 1]
        len_a, len_b = len(a_ids), len(b_ids)
        a_set = set(a_ids.tolist()); b_set = set(b_ids.tolist())
        len_i = len(b_set - a_set)
        path_B_sizes.append(len_b)
        path_I.append(len_i)
        path_deltas.append(len_i / len_b if len_b else 0)
    
    results["tile_path_consecutive"] = {
        "pairs": len(path_deltas),
        "rho_I": pctile(path_I),  # "insert only" count (not normalized)
        "insert_count": pctile(path_I),
    }
    
    # Phase 4: Worst-case analysis — full scan for extremes
    max_I, max_R, max_delta, min_overlap_t, min_overlap_nt = 0, 0, 0, -1, -1
    for label, t_a, t_b in all_pairs:
        a_ids, b_ids = tiles[t_a], tiles[t_b]
        a_set = set(a_ids.tolist()); b_set = set(b_ids.tolist())
        shared = a_set & b_set
        len_i = len(b_set - a_set)
        len_r = len(a_set - b_set)
        union = len(a_set | b_set)
        jaccard = len(shared) / union if union else 1.0
        
        if len_i > max_I:
            max_I = len_i
        if len_r > max_R:
            max_R = len_r
        if len_i + len_r > max_delta:
            max_delta = len_i + len_r
        if jaccard < min_overlap_t:
            min_overlap_t = jaccard
    
    results["worst_case"] = {
        "scan_pairs": len(all_pairs),
        "max_len_I": int(max_I),
        "max_len_R": int(max_R),
        "max_delta_entries": int(max_delta),
    }
    
    return results

def main():
    all_results = {
        "audit": "C17-2 Cross-Tile Differential Feasibility",
        "status": "OPTIMIZATION: NOT STARTED",
        "workload_identity": {},
        "scenes": {},
    }
    
    for name in ["room", "bicycle"]:
        wh = SCENES[name]
        print(f"\n  CAPTURE + ANALYZE: {name} native {wh['w']}x{wh['h']}",
              f"grid={math.ceil(wh['w']/16)}x{math.ceil(wh['h']/16)}", flush=True)
        t0 = time.time()
        sd = capture(name, wh["w"], wh["h"])
        ct = time.time() - t0
        print(f"  Captured {len(sd['flat']):,} isects in {ct:.0f}s", flush=True)
        
        all_results["workload_identity"][name] = {
            "gaussian_source": "raw SfM PLY",
            "resolution": f"{wh['w']}x{wh['h']}",
            "tile_grid": f"{math.ceil(wh['w']/16)}x{math.ceil(wh['h']/16)}",
            "n_isects": int(len(sd["flat"])), "tile_size": 16, "packed": True,
        }
        
        t1 = time.time()
        results = analyze_scene(name, sd, sample_count=2000)
        at = time.time() - t1
        print(f"  Analyzed in {at:.0f}s", flush=True)
        all_results["scenes"][name] = results
        del sd; gc.collect(); torch.cuda.empty_cache()
    
    out_json = OUT_DIR / "c17_2_differential_feasibility.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_json}", flush=True)

if __name__ == "__main__":
    main()
