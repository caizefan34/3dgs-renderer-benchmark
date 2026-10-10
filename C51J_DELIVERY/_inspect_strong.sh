#!/bin/bash
set +e
echo "=== strong_baselines tree (train.py files) ==="
find /mnt/storage_pool/liaoyuanjun/strong_baselines -maxdepth 3 -name "train.py" -o -maxdepth 3 -name "full_eval.py" 2>/dev/null | head -30
echo
echo "=== fastgs train.py: prune + radius_clip + opacity logic ==="
grep -nE "radius_clip|opacity.*prune|prune.*opacity|densif.*interval|opac.*reset|0.05|every.*10|fused" /mnt/storage_pool/liaoyuanjun/strong_baselines/fastgs/train.py 2>/dev/null | head -30
echo
echo "=== paper-protocol: method algorithm blocks (keys) ==="
/home/liaoyuanjun/miniforge3/envs/nerficg/bin/python - <<'PYEOF' 2>&1 | head -60
import json
try:
    d=json.load(open("/mnt/storage_pool/liaoyuanjun/tco0_work/benchmark/higs-paper-protocol.json"))
    print("top keys:", list(d.keys()))
    m=d.get("methods",{})
    for k in m:
        alg=m[k].get("algorithm", m[k])
        s=json.dumps(alg)[:240]
        print(f"- {k}: {s}")
except Exception as e:
    print("err", repr(e)[:200])
PYEOF
echo
echo "=== git commit of strong_baselines/fastgs ==="
git -C /mnt/storage_pool/liaoyuanjun/strong_baselines/fastgs rev-parse HEAD 2>/dev/null || echo "not a git repo"
git -C /mnt/storage_pool/liaoyuanjun/strong_baselines/speedy-splat rev-parse HEAD 2>/dev/null || echo "speedy not git"
echo "DONE"
