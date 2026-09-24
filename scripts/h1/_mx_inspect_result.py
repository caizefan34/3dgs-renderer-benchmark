import json, glob

for f in sorted(glob.glob("/tmp/h1_smoke/train_cam*.json")):
    d = json.load(open(f))
    print(f"\n=== {f.split('/')[-1]} ===")
    print("keys:", sorted(d.keys()))
    for k in ["B1_forward_stages_single", "B1_workload", "B2_workload", "B2_metadata",
              "forward_correctness", "backward_correctness"]:
        if k in d:
            v = d[k]
            if isinstance(v, dict):
                print(f"  {k}: {sorted(v.keys()) if len(v) > 5 else v}")
            else:
                print(f"  {k}: {v}")
    # print timing summaries
    for k in ["B1_forward_decomposed_timing", "B2_forward_timing", "B1_fwd_bwd_timing",
              "B2_fwd_bwd_timing", "B1_forward_autograd_timing"]:
        if k in d:
            t = d[k]
            print(f"  {k}: median={t.get('median_ms', '?'):.3f} mean={t.get('mean_ms', '?'):.3f}")