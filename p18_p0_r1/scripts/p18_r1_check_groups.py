#!/usr/bin/env python3
"""R1-B decisive record-integrity test.

For each dumped call group (warp, slot) the kernel wrote 16 records = the 16
rows of one (gaussian, tile). If the per-pp overwrite is not guarded, only the
last pp iteration's content survives and each group would show exactly 4
distinct mask16 values (4x replication). If records are correct, the 16 masks
are 16 real per-row masks (mostly distinct).
"""
import sys

import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "/root/epic05_p18_r1/artifacts"

for scene in ["room", "bicycle"]:
    d = np.load(f"{out}/P18_R1_TRACE_{scene}.npz")
    warp, slot, m16 = d["warp"], d["slot"], d["mask16"]
    g = d["g"]
    key = (d["mt_batch"].astype(np.int64) << 24) ^ (warp.astype(np.int64) << 16) ^ \
          (slot.astype(np.int64) << 8) ^ (g.astype(np.int64) & 0xFF)
    # group by (cta, warp, slot, g, x0) to identify one call's 16 records
    grp = np.stack([d["cta"].astype(np.int64), warp.astype(np.int64),
                    slot.astype(np.int64), g.astype(np.int64),
                    d["x0"].astype(np.int64)], axis=1)
    uniq, inv, cnt = np.unique(grp, axis=0, return_inverse=True, return_counts=True)
    print(f"== {scene}: records={len(m16)} groups={len(uniq)} sizedist="
          f"{np.bincount(cnt).tolist()[:20]}")
    ndist = np.zeros(20, dtype=np.int64)
    for gi in range(len(uniq)):
        sel = inv == gi
        nd = len(np.unique(m16[sel]))
        ndist[min(nd, 19)] += 1
    print(f"   distinct-mask-count per group (idx=count): {ndist[:18].tolist()}")
    # row-label coverage within groups
    rows = (d["y0"] & 15).astype(np.int64)
    rowcov = np.zeros(17, dtype=np.int64)
    for gi in range(min(len(uniq), 200000)):
        sel = inv == gi
        rowcov[len(np.unique(rows[sel]))] += 1
    print(f"   distinct-row-count per group: {rowcov[:18].tolist()}")
