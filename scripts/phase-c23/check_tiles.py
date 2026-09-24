import json, sys, numpy as np
d=json.load(open(sys.argv[1]))
for r in d.get("camera_records",[])[:2]:
    print(f"camera {r['camera']}:")
    for g,gd in r.get("density_groups",{}).items():
        tiles=gd.get("tile_records",[])
        if not tiles: 
            print(f"  {g}: 0 tiles"); continue
        its=[t["total_intersections"] for t in tiles]
        acts=[t["mean_activity"] for t in tiles]
        tails=[t["active_lane_fraction_at_90pct"] for t in tiles]
        lowwp=[t["warps_le_4_active_lanes_at_90pct"] for t in tiles]
        print(f"  {g}: {len(tiles)} tiles, intersect p50={np.percentile(its,50):.0f}, p90={np.percentile(its,90):.0f}")
        print(f"       mean_activity: mean={np.mean(acts):.3f}, p50={np.percentile(acts,50):.3f}, p90={np.percentile(acts,90):.3f}")
        print(f"       tail_active@90pct: mean={np.mean(tails):.3f}, p50={np.percentile(tails,50):.3f}, p90={np.percentile(tails,90):.3f}")
        print(f"       warps_le_4_lanes@90pct: mean={np.mean(lowwp):.3f}, p50={np.percentile(lowwp,50):.3f}, p90={np.percentile(lowwp,90):.3f}")
