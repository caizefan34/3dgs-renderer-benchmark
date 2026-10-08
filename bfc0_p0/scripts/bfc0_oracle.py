#!/usr/bin/env python3
"""BFC0 DSH-4/5: O1 and O2 oracle with fragmentation analysis.

Implements:
- O1 frontier-only oracle (upper bound)
- O2 exact-validity oracle (primary gate)
- Warp metrics from BFC0 reference
- Tail region analysis (preregistered: <=25% live pixels)
- Stratification by traversal percentile, active fraction, tile density
"""
import os, sys, json, math
import numpy as np
from typing import Dict, Any, List, Tuple

WARP_SIZE = 32

def warp_metrics(active: np.ndarray, warp_size: int = WARP_SIZE):
    """Compute static-warp vs ideal-compacted-warp counts for one traversal position."""
    active = np.asarray(active, dtype=np.bool_).reshape(-1)
    n = active.size
    pad = (-n) % warp_size
    if pad:
        active = np.pad(active, (0, pad), constant_values=False)
    groups = active.reshape(-1, warp_size)
    w_any = int(np.any(groups, axis=1).sum())
    a = int(active.sum())
    w_comp = (a + warp_size - 1) // warp_size
    return a, w_any, w_comp

def aggregate_oracle(sequence_of_active_masks: List[np.ndarray], warp_size: int = WARP_SIZE):
    """Aggregate oracle metrics over all traversal positions."""
    A = W_any = W_comp = 0
    rows = []
    for d, active in enumerate(sequence_of_active_masks):
        a, wa, wc = warp_metrics(active, warp_size)
        A += a
        W_any += wa
        W_comp += wc
        rows.append((d, a, wa, wc))
    
    occ = A / (warp_size * W_any) if W_any else 1.0
    reduction = 1.0 - W_comp / W_any if W_any else 0.0
    
    return {
        "sum_active_lanes": A,
        "sum_w_any": W_any,
        "sum_w_comp": W_comp,
        "lane_occupancy": occ,
        "warp_reduction_bound": reduction,
        "rows": rows,
    }

def compute_tail_metrics(rows, tile_pixels, threshold=0.25):
    """Compute tail region metrics where <=25% of pixels are live."""
    tail_rows = []
    for d, a, wa, wc in rows:
        live_fraction = a / tile_pixels if tile_pixels > 0 else 0
        if live_fraction <= threshold:
            tail_rows.append((d, a, wa, wc))
    
    if not tail_rows:
        return {
            "tail_lane_occupancy": 0.0,
            "tail_warp_reduction": 0.0,
            "tail_positions": 0,
            "tail_active_lanes": 0,
        }
    
    tail_A = sum(r[1] for r in tail_rows)
    tail_W_any = sum(r[2] for r in tail_rows)
    tail_W_comp = sum(r[3] for r in tail_rows)
    
    tail_occ = tail_A / (WARP_SIZE * tail_W_any) if tail_W_any else 1.0
    tail_reduction = 1.0 - tail_W_comp / tail_W_any if tail_W_any else 0.0
    
    return {
        "tail_lane_occupancy": tail_occ,
        "tail_warp_reduction": tail_reduction,
        "tail_positions": len(tail_rows),
        "tail_active_lanes": tail_A,
        "tail_w_any": tail_W_any,
        "tail_w_comp": tail_W_comp,
    }

def run_oracle_from_trace(trace_path: str, output_dir: str):
    """Run O1 and O2 oracle on collected trace data."""
    
    data = np.load(trace_path)
    # Handle both old and new API names
    isect_offsets = data["isect_offsets"] if "isect_offsets" in data else data["tile_offsets"]
    flatten_ids = data["flatten_ids"]
    last_ids = data["last_ids"]
    means2d = data["means2d"]
    conics = data["conics"]
    opacities = data["opacities"]
    width = int(data["width"])
    height = int(data["height"])
    tile_size = int(data["tile_size"])
    
    C = isect_offsets.shape[0]
    tile_h = isect_offsets.shape[1]
    tile_w = isect_offsets.shape[2]
    n_isects = len(flatten_ids)
    tile_pixels = tile_size * tile_size
    
    print(f"Trace: {C} cameras, {tile_h}x{tile_w} tiles, {n_isects} intersections")
    print(f"Image: {width}x{height}, tile_size: {tile_size}")
    
    results = {
        "O1_FRONTIER": [],
        "O2_EXACT_VALIDITY": [],
    }
    
    # Process first camera only for now (C=1 typically)
    cam_id = 0
    
    for tile_idx in range(tile_h * tile_w):
        tile_row = tile_idx // tile_w
        tile_col = tile_idx % tile_w
        
        range_start = int(isect_offsets[cam_id, tile_row, tile_col])
        is_last_tile = (tile_idx == tile_h * tile_w - 1)
        if tile_col < tile_w - 1:
            range_end = int(isect_offsets[cam_id, tile_row, tile_col + 1])
        elif tile_row < tile_h - 1:
            range_end = int(isect_offsets[cam_id, tile_row + 1, 0])
        else:
            range_end = n_isects
        
        n_gaussians_in_tile = range_end - range_start
        if n_gaussians_in_tile == 0:
            continue
        
        # Get Gaussian indices for this tile
        gauss_ids = flatten_ids[range_start:range_end].astype(np.int64)
        
        # Get pixel coordinates for this tile
        tile_y_start = tile_row * tile_size
        tile_x_start = tile_col * tile_size
        
        # Build per-pixel data
        pixel_last_ids = np.full((tile_size, tile_size), -1, dtype=np.int32)
        
        for ty in range(tile_size):
            for tx in range(tile_size):
                py = tile_y_start + ty
                px = tile_x_start + tx
                if py < height and px < width:
                    pixel_last_ids[ty, tx] = last_ids[cam_id, py, px]
        
        # Build sorted list of unique gaussian indices in this tile
        sorted_gauss_ids = gauss_ids  # Already depth-sorted
        
        n_gauss = len(sorted_gauss_ids)
        
        o1_active_per_pos = []
        o2_active_per_pos = []
        
        for d in range(n_gauss):
            # d=0: last gaussian, d=n-1: first gaussian
            # Reverse: current gaussian index in original order
            orig_idx = n_gauss - 1 - d
            
            # O1: pixel is active if its last_id >= orig_idx
            o1_mask = (pixel_last_ids >= orig_idx).flatten()
            o1_active_per_pos.append(o1_mask)
            
            # O2: pixel is active if alpha > 1/255 AND within tile AND sigma < 0
            o2_mask = np.zeros(tile_pixels, dtype=np.bool_)
            for ty in range(tile_size):
                for tx in range(tile_size):
                    py = tile_y_start + ty
                    px = tile_x_start + tx
                    if py >= height or px >= width:
                        continue
                    
                    pix_id = ty * tile_size + tx
                    g_id = sorted_gauss_ids[orig_idx]
                    
                    if g_id >= len(means2d[cam_id]):
                        continue
                    
                    mx = means2d[cam_id, g_id, 0]
                    my = means2d[cam_id, g_id, 1]
                    cx, cy, cz = conics[cam_id, g_id]
                    opac = opacities[g_id]
                    
                    dx = px + 0.5 - mx
                    dy = py + 0.5 - my
                    sigma = 0.5 * (cx * dx * dx + cz * dy * dy) + cy * dx * dy
                    
                    if sigma < 0:
                        vis = math.exp(-sigma)
                        alpha = min(0.999, float(opac) * vis)
                        if alpha > 1.0 / 255.0:
                            o2_mask[pix_id] = True
            o2_active_per_pos.append(o2_mask)
        
        # Run O1 oracle
        o1_result = aggregate_oracle(o1_active_per_pos)
        o1_result["tile_idx"] = tile_idx
        o1_result["tile_row"] = tile_row
        o1_result["tile_col"] = tile_col
        o1_result["n_gaussians"] = n_gauss
        o1_result["n_pixels"] = tile_pixels
        
        # Compute tail metrics
        o1_tail = compute_tail_metrics(o1_result["rows"], tile_pixels)
        o1_result.update(o1_tail)
        
        results["O1_FRONTIER"].append(o1_result)
        
        # Run O2 oracle
        o2_result = aggregate_oracle(o2_active_per_pos)
        o2_result["tile_idx"] = tile_idx
        o2_result["tile_row"] = tile_row
        o2_result["tile_col"] = tile_col
        o2_result["n_gaussians"] = n_gauss
        o2_result["n_pixels"] = tile_pixels
        
        o2_tail = compute_tail_metrics(o2_result["rows"], tile_pixels)
        o2_result.update(o2_tail)
        
        results["O2_EXACT_VALIDITY"].append(o2_result)
    
    # Aggregate global metrics
    for oracle_name in ["O1_FRONTIER", "O2_EXACT_VALIDITY"]:
        oracle_results = results[oracle_name]
        if not oracle_results:
            continue
        
        total_A = sum(r["sum_active_lanes"] for r in oracle_results)
        total_W_any = sum(r["sum_w_any"] for r in oracle_results)
        total_W_comp = sum(r["sum_w_comp"] for r in oracle_results)
        
        global_occ = total_A / (WARP_SIZE * total_W_any) if total_W_any else 1.0
        global_reduction = 1.0 - total_W_comp / total_W_any if total_W_any else 0.0
        
        # Tail metrics
        tail_positions = sum(r["tail_positions"] for r in oracle_results)
        tail_A = sum(r["tail_active_lanes"] for r in oracle_results)
        tail_W_any = sum(r["tail_w_any"] for r in oracle_results)
        tail_W_comp = sum(r["tail_w_comp"] for r in oracle_results)
        
        tail_occ = tail_A / (WARP_SIZE * tail_W_any) if tail_W_any else 1.0
        tail_reduction = 1.0 - tail_W_comp / tail_W_any if tail_W_any else 0.0
        
        # Count tiles meeting gates
        tiles_tail_ok = sum(1 for r in oracle_results if r["tail_lane_occupancy"] <= 0.50)
        tiles_warp_ok = sum(1 for r in oracle_results if r["warp_reduction_bound"] >= 0.20)
        
        results[oracle_name + "_GLOBAL"] = {
            "total_active_lanes": total_A,
            "total_w_any": total_W_any,
            "total_w_comp": total_W_comp,
            "lane_occupancy": global_occ,
            "warp_reduction_bound": global_reduction,
            "tail_lane_occupancy": tail_occ,
            "tail_warp_reduction": tail_reduction,
            "tail_positions": tail_positions,
            "n_tiles_processed": len(oracle_results),
            "tiles_tail_ok": tiles_tail_ok,
            "tiles_warp_ok": tiles_warp_ok,
            "tiles_warp_ok_fraction": tiles_warp_ok / len(oracle_results) if oracle_results else 0,
        }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    detail_path = os.path.join(output_dir, "BFC0_ORACLE_METRICS.json")
    
    # Convert to serializable format
    serializable = {}
    for k, v in results.items():
        if k.endswith("_GLOBAL"):
            serializable[k] = v
        else:
            serializable[k] = []
            for r in v:
                r_copy = {kk: vv for kk, vv in r.items() if kk != "rows"}
                serializable[k].append(r_copy)
    
    with open(detail_path, "w") as f:
        json.dump(serializable, f, indent=2)
    
    # Print summary
    print("\n=== BFC0 Oracle Results ===")
    for oracle_name in ["O1_FRONTIER", "O2_EXACT_VALIDITY"]:
        global_key = oracle_name + "_GLOBAL"
        if global_key in results:
            g = results[global_key]
            print(f"\n{oracle_name}:")
            print(f"  Lane occupancy: {g['lane_occupancy']:.4f}")
            print(f"  Warp reduction bound: {g['warp_reduction_bound']:.4f}")
            print(f"  Tail lane occupancy: {g['tail_lane_occupancy']:.4f}")
            print(f"  Tail warp reduction: {g['tail_warp_reduction']:.4f}")
            print(f"  Tiles processed: {g['n_tiles_processed']}")
            print(f"  Tiles with tail occ <= 0.50: {g['tiles_tail_ok']}/{g['n_tiles_processed']}")
            print(f"  Tiles with warp reduction >= 0.20: {g['tiles_warp_ok']}/{g['n_tiles_processed']}")
    
    return results

if __name__ == "__main__":
    out_dir = os.environ.get("BFC0_OUT_DIR", "C:\\Users\\36570\\3dgs-renderer-benchmark\\bfc0_p0\\output")
    trace_dir = os.path.join(out_dir, "trace")
    
    # Find trace files
    trace_files = [f for f in os.listdir(trace_dir) if f.endswith("_trace.npz")]
    
    if not trace_files:
        print("No trace files found. Run bfc0_trace_collector.py first.")
        sys.exit(1)
    
    for trace_file in trace_files:
        scene_name = trace_file.replace("_trace.npz", "")
        trace_path = os.path.join(trace_dir, trace_file)
        print(f"\nProcessing {scene_name}...")
        run_oracle_from_trace(trace_path, out_dir)
