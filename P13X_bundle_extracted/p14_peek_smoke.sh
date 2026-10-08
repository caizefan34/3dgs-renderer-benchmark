#!/bin/bash
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
u = json.load(open('/root/epic05_p14/P14_O2_UNITS_SMOKE/room_cell0_q20_H32.json'))
print("pre  totals:", json.dumps(u["pre"], indent=1))
print("post totals:", json.dumps(u["post"], indent=1))
print("R_stream pre/post/max:", u["R_stream_pre"], u["R_stream_post"], u["R_stream_max"])
print("n_seg_max:", u["n_frozen_segments_max"], "cache GB:", u["cache_gb_fp32_max"])
print("rebuild:", json.dumps({k: v for k, v in u["rebuild"].items() if k != "per_camera"}, indent=1))
d1 = json.load(open('/root/epic05_p14/P14_O1_UNITS_SMOKE/room_cell0_d1_h32.json'))
f = json.load(open('/root/epic05_p14/P14_O1_UNITS_SMOKE/room_cell0_f_q20_h32.json'))
print("R_train:", f["G_train"]/d1["G_train"])
print("P_audit:", (f["L_audit_after"]-d1["L_audit_after"])/f["L_audit_before"])
print("timings d1:", {k: round(v,3) for k,v in d1["timings_ms"].items()})
print("timings f :", {k: round(v,3) for k,v in f["timings_ms"].items()})
EOF
