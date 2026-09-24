import torch, json, sys
d = torch.load('results/phase-a100/c17_2_membership_collected.pt', map_location='cpu', weights_only=True)
for k, v in d['scenes'].items():
    si = v['within_tile_delta']['stat']
    print(f"{k}: isects={v['n_isects']} tiles={v['n_tiles']} gauss={v['n_gaussians']}")
    print(f"  flat_bytes={v['flatten_ids']['bytes']} tile_bytes={v['tile_offsets']['bytes']} dup={v['flatten_ids']['per_tile_duplicates']}")
    print(f"  delta: pairs={si['count']} zero={v['within_tile_delta']['zero_ratio']:.4f} mean={si['mean']:.2f} p50={si['p50']} p99={si['p99']}")
    print(f"  small: <=1={v['within_tile_delta']['small_delta_ratio']['1']:.4f} <=2={v['within_tile_delta']['small_delta_ratio']['2']:.4f} <=4={v['within_tile_delta']['small_delta_ratio']['4']:.4f} <=16={v['within_tile_delta']['small_delta_ratio']['16']:.4f}")
    for nd in ['h','v','ddr','ddl']:
        j = v['cross_tile_overlap'][nd]['jaccard']
        print(f"  overlap[{nd}]: jac_p50={j['p50']:.4f} mean={j['mean']:.4f} p99={j['p99']:.4f} n={j['count']}")
    m = v['gaussian_tile_membership']
    print(f"  gauss_tiles: p50={m['p50']} p99={m['p99']} max={m['max']:.0f} mean={m['mean']:.2f}")
    rl = v['run_length']
    print(f"  run_len: p50={rl['p50']} max={rl['max']:.0f} count={rl['count']}")
    dr = v['depth_order']
    print(f"  depth: pairs={dr['pairs']} dec_ratio={dr['strict_decrease_ratio']:.6f}")
    ub = v['id_range_per_tile']
    print(f"  id_bits: needed={ub['global_id_bits_needed']} unused={ub['unused_vs_int32']} range_p50={ub['global_range']['p50']:.0f}")
    print()
