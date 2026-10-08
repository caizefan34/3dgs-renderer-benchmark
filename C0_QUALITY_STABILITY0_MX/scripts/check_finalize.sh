#!/usr/bin/env bash
Q=/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX
echo "=== seed42 reuse checkpoints in campaign checkpoints/ ==="
ls "$Q/checkpoints"/*_s42_*.ckpt_lite_30000.pt 2>/dev/null | wc -l
echo "=== seed42 ckpt names ==="
ls "$Q/checkpoints"/*_s42_*.ckpt_lite_30000.pt 2>/dev/null | xargs -n1 basename
echo "=== watcher calls run_pipeline? ==="
grep -n "run_pipeline" "$Q/scripts/watch_and_aggregate.sh"
echo "=== scheduler records contamination? (index keys) ==="
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
$PY -c "import json; d=json.load(open('$Q/RUN_INDEX_SEED4344.json')); print('keys:', sorted(d.keys())); print('n_runs:', len(d.get('runs',[]))); r=d['runs'][0] if d.get('runs') else {}; print('run0 keys:', sorted(r.keys()))"
