#!/bin/bash
echo "===== ROOM LOG (key lines) ====="
grep -E "done in|COMPLETE|heldout_before|future-anchor|phaseB anchors 64|FAIL|Error|Traceback" /root/epic05_n1/logs/stage23_room.log | head -40
echo "===== BICYCLE LOG (key lines) ====="
grep -E "done in|COMPLETE|heldout_before|future-anchor|phaseB anchors 64|FAIL|Error|Traceback" /root/epic05_n1/logs/stage23_bicycle.log | head -40
echo "===== ARTIFACTS ====="
ls -la /root/epic05_n1/artifacts/
echo "===== ROW COUNTS ====="
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
for s in ("room", "bicycle"):
    d = json.load(open(f"/root/epic05_n1/artifacts/N1_ACTION_RESULTS_{s}.json"))
    acts = {}
    for r in d["rows"]:
        acts[r["action"]] = acts.get(r["action"], 0) + 1
    print(s, "rows:", len(d["rows"]), acts)
    g = json.load(open(f"/root/epic05_n1/artifacts/N1_GRADIENT_CONFLICT_{s}.json"))
    print(s, "conflict rows:", len(g["rows"]))
    import numpy as np
    z = np.load(f"/root/epic05_n1/artifacts/N1_SKETCHES_{s}.npz")
    print(s, "sketches:", len(z.files), "dim:", z[z.files[0]].shape)
    v = json.load(open(f"/root/epic05_n1/artifacts/N1_VIEW_LOSSES_{s}.json"))
    for ck, losses in v["per_view_fused_loss_by_ckpt"].items():
        print(s, "view losses @", ck, ":", len(losses))
EOF
