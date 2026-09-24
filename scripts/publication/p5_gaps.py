import json

ALL13 = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
         "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]

for b in ("faster-gs", "fastgs", "speedy-splat"):
    d = json.load(open(f"/mnt/storage_pool/liaoyuanjun/strong_baseline_results/{b}/all_metrics.json"))
    have = set()
    rc_bad = []
    for k, v in d.items():
        scene = v.get("scene")
        method = v.get("method")
        have.add((scene, method))
        if v.get("exit_code", 0) != 0:
            rc_bad.append(k)
    missing = [(s, m) for s in ALL13 for m in ("native", "c42") if (s, m) not in have]
    print(f"{b}: {len(d)} entries, rc!=0: {rc_bad}, missing: {missing}")
