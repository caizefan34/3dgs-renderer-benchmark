#!/bin/bash
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
import numpy as np
for s in ("room", "bicycle"):
    d = json.load(open(f"/root/epic05_n1/artifacts/N1_GRADIENT_CONFLICT_{s}.json"))["rows"]
    g = np.array([r["global_cosine"] for r in d])
    j = np.array([r["vis_jaccard"] for r in d])
    print(f"=== {s}: n={len(d)}")
    print(f"  global_cosine: median={np.median(g):.4f} mean={g.mean():.4f} "
          f"p10={np.percentile(g,10):.4f} p90={np.percentile(g,90):.4f} "
          f"neg_share={np.mean(g<0):.3f}")
    for tier in ("high","medium","low"):
        idx = [i for i,r in enumerate(d) if r["tier"]==tier]
        gt = g[idx]; jt = j[idx]
        print(f"  {tier}: cos median={np.median(gt):.4f} neg={np.mean(gt<0):.3f} "
              f"vis_jaccard median={np.median(jt):.4f}")
    # per checkpoint
    for ck in (15000,20000,25000):
        idx = [i for i,r in enumerate(d) if r["checkpoint"]==ck]
        print(f"  @{ck}: cos median={np.median(g[idx]):.4f}")
# action loss sanity: A0 loss_v1 distribution
for s in ("room", "bicycle"):
    d = json.load(open(f"/root/epic05_n1/artifacts/N1_ACTION_RESULTS_{s}.json"))["rows"]
    a0 = [r["loss_v1"] for r in d if r["action"]=="A0"]
    hb = {}
    for r in d:
        if r["action"]=="A0":
            hb.setdefault(r["checkpoint"], []).append(r["heldout_fused_before"])
    print(f"{s} A0 loss_v1: median={np.median(a0):.5f} p10={np.percentile(a0,10):.5f} p90={np.percentile(a0,90):.5f}")
EOF
