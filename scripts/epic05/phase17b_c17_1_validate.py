"""
Phase 17B — C17-1: Python Prototype Validation (TRUE BASELINE)
===============================================
Proves intersection-set equivalence and depth-ordering equivalence
by reconstructing per-tile groups from baseline data and applying
per-tile local sort.

Baseline key format (true v1.5.3):
  bits [63:32+tn] = image_id
  bits [32+tn-1:32] = tile_id  
  bits [31:0] = full float32 depth (bitcast to uint32)

C17-1: per-tile local sort on full float32 depth → must match baseline sorting
because within a tile, the sort key is just depth, and CUB sorts full 47-bit key.
"""

import json, math, os, sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_framework.scene import load_ply
from benchmark_framework.cameras import load_cameras_from_json, resize_cameras

DEVICE = "cuda"
OUTPUT_DIR = REPO_ROOT / "results" / "epic05" / "phase17b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTIONS = {"room": (1080, 1080), "bicycle": (1920, 1080), "garden": (1920, 1080)}
TILE_SIZES = [16, 20, 32]

def get_first_camera(scene_name, target_w, target_h):
    cam_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "cameras.json"
    cameras = load_cameras_from_json(str(cam_path), device=DEVICE)
    cameras = resize_cameras(cameras, target_w, target_h)
    return cameras[0]

@torch.no_grad()
def run_validation(scene_name, tile_size):
    print(f"\n{'='*70}")
    print(f"VALIDATION — {scene_name}, tile_size={tile_size}")
    print(f"{'='*70}")
    
    torch.cuda.reset_peak_memory_stats()
    
    ply_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "point_cloud.ply"
    scene = load_ply(str(ply_path), device=DEVICE)
    sh_degree = scene.get("sh_degree", 3)
    
    target_w, target_h = RESOLUTIONS[scene_name]
    cam = get_first_camera(scene_name, target_w, target_h)
    W, H = cam.image_width, cam.image_height
    viewmat = cam.viewmatrix.unsqueeze(0).to(DEVICE)
    K = cam.K.unsqueeze(0).to(DEVICE)
    
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = torch.exp(scene["scales"]).contiguous()
    opacities = torch.sigmoid(scene["opacity"]).contiguous()
    
    from gsplat import rasterization
    
    # Warmup + capture meta
    _, _, meta = rasterization(
        means=scene["xyz"], quats=quats, scales=scales,
        opacities=opacities, colors=scene["shs"],
        viewmats=viewmat, Ks=K, width=W, height=H,
        tile_size=tile_size, packed=True,
        sh_degree=sh_degree, render_mode="RGB")
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    renders, alphas, meta = rasterization(
        means=scene["xyz"], quats=quats, scales=scales,
        opacities=opacities, colors=scene["shs"],
        viewmats=viewmat, Ks=K, width=W, height=H,
        tile_size=tile_size, packed=True,
        sh_degree=sh_degree, render_mode="RGB")
    end.record()
    end.synchronize()
    fwd_ms = start.elapsed_time(end)
    
    # Extract
    isect_ids = meta["isect_ids"]
    flatten_ids = meta["flatten_ids"]
    tile_offsets = meta["isect_offsets"]  # [I, H, W]
    depths_proj = meta["depths"]  # [nnz]
    gaussian_ids = meta["gaussian_ids"]
    tile_height, tile_width = tile_offsets.shape[-2], tile_offsets.shape[-1]
    n_tiles = tile_width * tile_height
    n_isects = isect_ids.shape[0]
    nnz = depths_proj.shape[0]
    
    print(f"  n_isects={n_isects:,}, nnz={nnz:,}, tiles={tile_width}x{tile_height}={n_tiles}")
    
    # Per-intersection depth and Gaussian ID
    fl_np = flatten_ids.cpu().numpy().astype(np.int64)
    depths_np = depths_proj.cpu().numpy().astype(np.float64)
    per_isect_depths = depths_np[fl_np]
    
    gid_np = gaussian_ids.cpu().numpy().astype(np.int64)
    per_isect_gaussians = gid_np[fl_np]
    
    # Decode tile_id (BASELINE format: >> 32)
    tile_n_bits = max(int(math.ceil(math.log2(n_tiles))), 1)
    tile_mask = (1 << tile_n_bits) - 1
    tile_ids = ((isect_ids >> 32) & tile_mask).cpu().numpy().astype(np.int64)
    
    # ── Baseline per-tile groups (from tile_offsets) ──────────────────
    off_np = tile_offsets[0].cpu().numpy()  # [TH, TW]
    
    bl_by_tile = {}
    for tid in range(n_tiles):
        h = tid // tile_width
        w = tid % tile_width
        start = int(off_np[h, w])
        
        if tid == n_tiles - 1:
            end = n_isects
        else:
            nh = (tid + 1) // tile_width
            nw = (tid + 1) % tile_width
            end = int(off_np[nh, nw])
        
        if start >= n_isects:
            bl_by_tile[tid] = np.zeros((0, 3), dtype=np.float64)
            continue
        
        entries = []
        for pos in range(start, min(end, n_isects)):
            gidx = int(per_isect_gaussians[pos])
            d = float(per_isect_depths[pos])
            entries.append([gidx, d, pos])
        
        bl_by_tile[tid] = np.array(entries, dtype=np.float64) if entries else np.zeros((0, 3), dtype=np.float64)
    
    # ── C17-1: reconstruct from decoded tile_ids ─────────────────────
    c17_raw = {}
    for pos in range(n_isects):
        tid = int(tile_ids[pos])
        gidx = int(per_isect_gaussians[pos])
        d = float(per_isect_depths[pos])
        if tid not in c17_raw:
            c17_raw[tid] = []
        c17_raw[tid].append([gidx, d, pos])
    
    # Per-tile local sort by depth, then gaussian_idx as tiebreaker
    c17_sorted = {}
    for tid in range(n_tiles):
        entries = c17_raw.get(tid, [])
        if len(entries) > 0:
            arr = np.array(entries, dtype=np.float64)
            idx = np.lexsort((arr[:, 0], arr[:, 1]))
            c17_sorted[tid] = arr[idx]
        else:
            c17_sorted[tid] = np.zeros((0, 3), dtype=np.float64)
    
    # ── 1. Intersection set equivalence ──────────────────────────────
    total_missing = 0
    total_extra = 0
    tile_reports = {}
    
    for tid in range(n_tiles):
        bl_set = set(bl_by_tile[tid][:, 0].astype(int).tolist()) if len(bl_by_tile[tid]) > 0 else set()
        c17_set = set(c17_sorted[tid][:, 0].astype(int).tolist()) if len(c17_sorted[tid]) > 0 else set()
        
        missing = bl_set - c17_set
        extra = c17_set - bl_set
        
        if len(missing) > 0 or len(extra) > 0:
            tile_reports[str(tid)] = {"bl": len(bl_set), "c17": len(c17_set), "missing": len(missing), "extra": len(extra)}
        
        total_missing += len(missing)
        total_extra += len(extra)
    
    equiv = {
        "total_missing": int(total_missing),
        "total_extra": int(total_extra),
        "tile_reports": tile_reports,
        "PASS": bool(total_missing == 0 and total_extra == 0),
    }
    print(f"  Set equiv: missing={total_missing}, extra={total_extra}, PASS={equiv['PASS']}")
    
    # ── 2. Depth ordering equivalence ────────────────────────────────
    exact_match = 0
    total_active = 0
    mismatches = []
    
    for tid in range(n_tiles):
        bl_arr = bl_by_tile[tid]
        c17_arr = c17_sorted[tid]
        
        if len(bl_arr) == 0:
            continue
        total_active += 1
        
        bl_seq = bl_arr[:, 0].astype(int).tolist()
        c17_seq = c17_arr[:, 0].astype(int).tolist()
        
        if bl_seq == c17_seq:
            exact_match += 1
        else:
            min_len = min(len(bl_seq), len(c17_seq))
            diff_count = sum(1 for i in range(min_len) if bl_seq[i] != c17_seq[i])
            mismatches.append({
                "tile_id": int(tid),
                "n_bl": len(bl_seq), "n_c17": len(c17_seq),
                "diff_positions": diff_count,
                "same_length": len(bl_seq) == len(c17_seq),
            })
    
    order = {
        "total_active_tiles": total_active,
        "exact_order_match": exact_match,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:30],
        "PASS": bool(exact_match == total_active),
    }
    print(f"  Order match: {exact_match}/{total_active} tiles, PASS={order['PASS']}")
    if not order['PASS']:
        for m in mismatches[:5]:
            print(f"    Tile {m['tile_id']}: {m['n_bl']} items, diff={m['diff_positions']} pos")
    
    # ── 3. Depth monotonic ───────────────────────────────────────────
    bl_viol = 0
    c17_viol = 0
    for tid in range(n_tiles):
        for arr, viol in [(bl_by_tile[tid], 'bl'), (c17_sorted[tid], 'c17')]:
            if len(arr) > 1:
                v = int(np.sum(np.diff(arr[:, 1]) < -1e-8))
                if viol == 'bl':
                    bl_viol += v
                else:
                    c17_viol += v
    
    mono = {"bl_violations": bl_viol, "c17_violations": c17_viol, "PASS": bool(c17_viol == 0)}
    print(f"  Depth mono: BL={bl_viol}, C17={c17_viol}, PASS={mono['PASS']}")
    
    # ── 4. Duplicate check ─────────────────────────────────────────────
    bl_dups = sum(len(bl_by_tile[tid]) - len(np.unique(bl_by_tile[tid][:,0])) for tid in range(n_tiles) if len(bl_by_tile[tid]) > 0)
    c17_dups = sum(len(c17_sorted[tid]) - len(np.unique(c17_sorted[tid][:,0])) for tid in range(n_tiles) if len(c17_sorted[tid]) > 0)
    print(f"  Duplicates: BL={int(bl_dups)}, C17={int(c17_dups)}")
    
    # ── 5. Verify tile_id decode matches offset-based grouping ──────
    decode_ok = True
    for pos in range(min(50000, n_isects)):
        tid = int(tile_ids[pos])
        h = tid // tile_width
        w = tid % tile_width
        start = int(off_np[h, w])
        if tid == n_tiles - 1:
            end = n_isects
        else:
            nh = (tid + 1) // tile_width
            nw = (tid + 1) % tile_width
            end = int(off_np[nh, nw])
        if pos < start or pos >= end:
            decode_ok = False
            break
    
    print(f"  Tile ID decode match: {decode_ok}")
    
    feasible = equiv["PASS"] and order["PASS"] and mono["PASS"]
    
    return {
        "scene": scene_name,
        "tile_size": tile_size,
        "n_isects": int(n_isects),
        "n_tiles": int(n_tiles),
        "n_gaussians": scene["num_points"],
        "forward_time_ms": fwd_ms,
        "intersection_equivalence": equiv,
        "depth_ordering": order,
        "depth_monotonic": mono,
        "bl_duplicates": int(bl_dups),
        "c17_duplicates": int(c17_dups),
        "tile_decode_match": decode_ok,
        "feasible": bool(feasible),
    }


if __name__ == "__main__":
    results = {}
    for scene in ["room", "bicycle", "garden"]:
        for ts in TILE_SIZES:
            key = f"{scene}_t{ts}"
            try:
                r = run_validation(scene, ts)
                results[key] = r
            except Exception as e:
                print(f"ERROR {key}: {e}")
                import traceback; traceback.print_exc()
                results[key] = {"error": str(e)}
    
    out_path = OUTPUT_DIR / "c17_1_equivalence.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # Summary
    print("\n" + "="*100)
    print("C17-1 VALIDATION SUMMARY (TRUE BASELINE v1.5.3)")
    print("="*100)
    h = f"{'Scene':>10} | {'tile':>4} | {'n_isects':>10} | {'missing':>7} | {'extra':>5} | {'order':>7} | {'mono':>5} | {'dups':>4} | {'decode':>6} | {'status':>12}"
    print(h)
    print("-"*100)
    all_pass = True
    for k, r in results.items():
        if "error" in r:
            print(f"{k:>15}: ERROR: {r['error']}")
            all_pass = False
            continue
        eq = r["intersection_equivalence"]
        od = r["depth_ordering"]
        mo = r["depth_monotonic"]
        fe = r["feasible"]
        if not fe:
            all_pass = False
        print(f"{r['scene']:>10} | {r['tile_size']:>4} | {r['n_isects']:>10,} | {eq['total_missing']:>7} | {eq['total_extra']:>5} | {od['exact_order_match']}/{od['total_active_tiles']:<4} | {'OK' if mo['PASS'] else 'FAIL':>5} | {r['c17_duplicates']:>4} | {str(r['tile_decode_match']):>6} | {'PASS' if fe else 'FAIL':>12}")
    
    print(f"\nOVERALL: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
