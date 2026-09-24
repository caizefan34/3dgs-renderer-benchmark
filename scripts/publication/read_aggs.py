import json

A = "/mnt/storage_pool/liaoyuanjun/pubphase/aggregates"
for name in ("p2_a1_vs_a0", "p2_a2_vs_a1", "p2_c0_vs_a2", "p2_c0_vs_a0",
             "p1_b1_vs_b1a", "p1_c0_vs_b1"):
    t = json.load(open(f"{A}/{name}.json"))
    agg = t["aggregate"]
    rows = [r for r in t["rows"] if r["status"] == "ok"]
    print(f"== {name}: pairs={agg['n_pairs']} "
          f"speedup_geomean={agg['speedup_geomean'] and round(agg['speedup_geomean'],4)} "
          f"faster={agg['n_faster']} slower={agg['n_slower']} "
          f"mean_dPSNR={agg['mean_d_psnr'] and round(agg['mean_d_psnr'],3)} "
          f"N_geo={agg['n_geo_ratio'] and round(agg['n_geo_ratio'],4)}")
    for r in rows:
        print(f"   {r['scene']:9s} speedup={r['speedup'] and round(r['speedup'],4)} "
              f"dPSNR={r['d_psnr']:+.3f} dSSIM={r['d_ssim']:+.4f} "
              f"dLPIPS={r['d_lpips'] and round(r['d_lpips'],4)} N_ratio={r['n_ratio']:.3f} "
              f"{'CLEAN' if r['timing_clean_pair'] else 'TIMING-DOWNGRADED'}")
    print()
