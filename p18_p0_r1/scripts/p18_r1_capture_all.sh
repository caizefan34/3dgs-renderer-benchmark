#!/bin/bash
# P18-P0-R1 full trace capture: frozen / off / on for room+bicycle (8 probe cams each),
# then bit-equality check of render hashes across the three modes.
set -uo pipefail
cd /root/epic05_p18_r1
PY=/root/miniforge3/envs/gsplat/bin/python
export CUDA_VISIBLE_DEVICES=2

for mode in frozen off; do
    echo "===== MODE $mode ====="
    $PY scripts/p18_r1_capture.py --mode $mode || { echo "MODE_$mode FAILED"; exit 1; }
done

echo "===== MODE on ====="
$PY scripts/p18_r1_capture.py --mode on || { echo "MODE_on FAILED"; exit 1; }

echo "===== RENDER EQUIVALENCE CHECK ====="
$PY - <<'PYEOF'
import json
base = "/root/epic05_p18_r1/artifacts/"
f = json.load(open(base + "P18_R1_RENDER_HASH_frozen.json"))
o = json.load(open(base + "P18_R1_RENDER_HASH_off.json"))
g = json.load(open(base + "P18_R1_RENDER_HASH_on.json"))
bad, n = [], 0
for scene in f["hashes"]:
    for cam in f["hashes"][scene]:
        a = f["hashes"][scene][cam]["renders"]
        b = o["hashes"][scene][cam]["renders"]
        c = g["hashes"][scene][cam]["renders"]
        n += 1
        if not (a == b == c):
            bad.append((scene, cam, a[:12], b[:12], c[:12]))
print(f"[equiv] compared {n} (scene,cam) render hashes")
if bad:
    print("[equiv] MISMATCH:", bad)
    raise SystemExit(1)
print("[equiv] frozen == off == on : PASS (bit-identical renders)")

m = json.load(open(base + "P18_R1_TRACE_MANIFEST.json"))
for scene, s in m["scenes"].items():
    print(f"[manifest] {scene}: n_primary_records={s['n_primary_records']} "
          f"calls={s['n_primary_calls']} pass={s['n_primary_records_pass']} "
          f"mean_pop={s['live_pop_mean']:.3f} median={s['live_pop_median']} "
          f"max_cta={s['max_cta']} cams={len(s['cams'])}")
print("[manifest] n_primary_records_min =", m["n_primary_records_min"])
PYEOF
echo "CAPTURE_ALL_DONE"
