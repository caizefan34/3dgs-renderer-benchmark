import json, sys
for fn in sys.argv[1:]:
    d = json.load(open(fn))
    print(f"{fn}: decision={d.get('decision','?' )}")
    if "camera_records" in d:
        print(f"  cameras={len(d['camera_records'])}")
        if len(d['camera_records'])>0:
            r0=d['camera_records'][0]
            for g,gd in r0.get('density_groups',{}).items():
                print(f"  {g}: {gd['n_tiles']} tiles")
            if "kernel_rows" in r0:
                print(f"  top kernel: {r0['kernel_rows'][0]['name'] if len(r0['kernel_rows'])>0 else 'none'}")
    if "rows" in d:
        print(f"  total_tile_rows={len(d['rows'])}")
    if "by_density" in d:
        print(f"  density_groups={list(d['by_density'].keys())}")
        for g,gv in d['by_density'].items():
            if gv:
                print(f"    {g}: n_tiles={gv.get('n_tiles','?')}")
                if gv.get("tail_low_util_warp_fraction_at_90pct"):
                    w=gv["tail_low_util_warp_fraction_at_90pct"]
                    print(f"      tail_low_util_warp@90pct: mean={w.get('mean',''):.4f} p50={w.get('p50',''):.4f}")
                if gv.get("baseline_avoided_suffix_fraction"):
                    b=gv["baseline_avoided_suffix_fraction"]
                    print(f"      baseline_avoided_suffix: mean={b.get('mean',''):.4f}")
                if gv.get("executed_over_logical"):
                    eo=gv["executed_over_logical"]
                    print(f"      executed_over_logical: mean={eo.get('mean',''):.4f}")
    if "high_density_aggregate" in d and d["high_density_aggregate"]:
        h=d["high_density_aggregate"]
        for k,v in h.items():
            if isinstance(v,dict):
                print(f"  high.{k}: mean={v.get('mean',''):.4f} p50={v.get('p50',''):.4f}")
    if "kernel_rows" in d:
        print("  top profiles:")
        for r in d["kernel_rows"][:5]:
            name=r["name"]
            if len(name)>70: name=name[:67]+"..."
            print(f"    {r['device_total_us']:10.1f}us  {name}")
        fwd = [r for r in d["kernel_rows"] if "fwd_kernel" in r["name"]]
        bwd = [r for r in d["kernel_rows"] if "bwd_kernel" in r["name"]]
        if fwd: print(f"  forward kernel: {fwd[0]['device_total_us']:.1f}us")
        if bwd: print(f"  backward kernel: {bwd[0]['device_total_us']:.1f}us, ratio={bwd[0]['device_total_us']/fwd[0]['device_total_us']:.1f}x" if fwd and bwd else "")
