#!/bin/bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo ---
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open('/root/epic05_p14/artifacts/P14_CAMERA_CLUSTERS.json'))
for s in ('room', 'bicycle'):
    sc = d['scenes'][s]
    print(f"{s}: n_cams={sc['n_cameras']} N={sc['N_gaussians']}")
    for c in sc['cells']:
        js = list(c['cluster_jaccard_to_seed'].values())
        print(f"  seed {c['seed_camera']}: cluster {c['cluster_cameras']}")
        print(f"    jaccards {[round(x,3) for x in js]}  audit {c['audit_cameras']}")
EOF
