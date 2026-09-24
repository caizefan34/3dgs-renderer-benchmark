import json

for b in ("faster-gs", "fastgs", "speedy-splat"):
    p = f"/mnt/storage_pool/liaoyuanjun/strong_baseline_results/{b}/training_summary.json"
    d = json.load(open(p))
    print(f"== {b}: {json.dumps({k: v for k, v in d.items() if k != 'jobs'})[:400]}")
    jobs = d.get("jobs", [])
    if jobs:
        j0 = jobs[0]
        print("   job keys:", sorted(j0.keys())[:20])
        for k in ("gpu", "gpu_id", "env", "python", "config", "iterations", "iters",
                  "command", "started", "timestamp", "date"):
            if k in j0:
                print(f"   {k}: {str(j0[k])[:120]}")
