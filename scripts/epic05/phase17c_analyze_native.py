"""Post-process native collection: fix visible_G bug, compute cross-tile stats."""
import torch, json, math, numpy as np

d = torch.load('results/phase-a100/c17_2_membership_native.pt', map_location='cpu', weights_only=True)
Q = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]

def stat(arr):
    if not isinstance(arr, np.ndarray): arr = np.array(arr)
    if not arr.size: return {"count": 0}
    p = np.quantile(arr.astype(np.float64), Q)
    return {"count": int(arr.size), "min": float(p[0]), "p1": float(p[1]), "p5": float(p[2]),
            "p25": float(p[3]), "p50": float(p[4]), "p75": float(p[5]), "p95": float(p[6]),
            "p99": float(p[7]), "p99_9": float(p[8]), "max": float(p[9]),
            "mean": float(arr.mean()), "std": float(arr.std())}

# Corrected baseline values
BASELINE = {
    "room": {"n_isect": 25475247, "n_visible": 16076, "n_tiles": 25350, "w": 3114, "h": 2075},
    "bicycle": {"n_isect": 18550853, "n_visible": 13654, "n_tiles": 63860, "w": 4946, "h": 3286},
    "garden": {"n_isect": 11257252, "n_visible": 100582, "n_tiles": 68575, "w": 5187, "h": 3361},
}

for name in ["room", "bicycle", "garden"]:
    v = d["scenes"][name]
    bl = BASELINE[name]
    
    nisect = v['n_isects']
    nmaster = v['n_gaussians']
    gm = v['gaussian_tile_membership']
    n_visible = gm['count']
    mean_tpg = gm['mean']
    
    print("=" * 70)
    print("  %s (native %dx%d, tile grid %s)" % (name, v['width'], v['height'], v['tile_grid']))
    print("=" * 70)
    
    print("  MEMBERSHIP AUDIT (native PLY):")
    print("    master=%d visible=%d(%d%%) isects=%d mean=%.4f" % (
        nmaster, n_visible, int(n_visible*100/nmaster), nisect, mean_tpg))
    print("    n_visible*mean = %d*%.4f = %.1f (diff=%.2f)" % (
        n_visible, mean_tpg, n_visible*mean_tpg, nisect - n_visible*mean_tpg))
    
    print("  CORRECTED BASELINE (trained ckpt):")
    print("    n_visible=%d isects=%d" % (bl['n_visible'], bl['n_isect']))
    
    ratio_v = nisect / bl['n_isect'] if bl['n_isect'] else 0
    print("  RATIO (PLY/ckpt): n_isect=%.4f" % ratio_v)
    
    # --- Cross-tile overlap ---
    for nd in ['h', 'v', 'ddr', 'ddl']:
        if nd in v['cross_tile_overlap']:
            j = v['cross_tile_overlap'][nd]['jaccard']
            print("  overlap[%s]: j_mean=%.4f j_p50=%.4f j_p75=%.4f j_p99=%.4f pairs=%d" % (
                nd, j['mean'], j['p50'], j['p75'], j['p99'], j['count']))
    
    # Order consistency
    if 'relative_order_consistency' in v:
        for nd in ['h', 'v', 'ddr', 'ddl']:
            if nd in v['relative_order_consistency']:
                oc = v['relative_order_consistency'][nd]
                if oc['count'] > 0:
                    print("  order_agree[%s]: mean=%.6f p50=%.6f p99=%.6f count=%d" % (
                        nd, oc['mean'], oc['p50'], oc['p99'], oc['count']))
    
    # Delta
    dl = v['within_tile_delta']
    print("  delta: pairs=%d zero_ratio=%.6f <=1=%.4f <=4=%.4f <=16=%.4f" % (
        dl['pairs'], dl['zero_ratio'],
        dl['small_delta_ratio']['1'], dl['small_delta_ratio']['4'], dl['small_delta_ratio']['16']))
    
    # Run length
    rl = v['run_length']
    print("  run_len: count=%d p50=%.0f max=%.0f" % (rl['count'], rl['p50'], rl['max']))
    
    # ID range
    ir = v['id_range_per_tile']
    print("  id_bits: need=%d unused=%d range_p50=%.0f" % (
        ir['global_id_bits_needed'], ir['unused_vs_int32'], ir['global_range']['p50']))
    
    # Depth
    dp = v['depth_order']
    print("  depth: pairs=%d dec_ratio=%.6f" % (dp['pairs'], dp['strict_decrease_ratio']))
    
    print()
