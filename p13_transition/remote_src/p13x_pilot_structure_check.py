"""P13-X pilot structural validation of saved NPZ snapshots (instrumentation check only).

Independently re-derives the integration-doc checks from the persisted files:
1. each anchor emits 3 NPZ files;
2. flatten_ids (gaussian_ids) range is [0,N);
3. isect_offsets monotone and last tile end equals len(flatten_ids);
plus tile_ids/tile_starts/tile_ends/visible_ids internal consistency.
"""
import glob
import json
import re
from pathlib import Path

import numpy as np

SNAP = "/root/epic05_p13x/artifacts/x12_snapshots"
OUT = "/root/epic05_p13x/artifacts/P13X_PILOT_STRUCTURE_CHECK.json"
EXPECTED_ITERS = {15000, 15250, 15500}

PAT = re.compile(r"(?P<scene>.+)_a(?P<anchor>\d+)_i(?P<iter>\d+)\.npz$")

per_anchor = {}
problems = []
n_checked = 0
for p in sorted(Path(SNAP).glob("*.npz")):
    m = PAT.match(p.name)
    if not m:
        problems.append(f"{p.name}: name does not match pattern")
        continue
    key = (m["scene"], int(m["anchor"]))
    per_anchor.setdefault(key, []).append(int(m["iter"]))
    z = np.load(p)
    n_checked += 1

    N = int(z["N"][0])
    M = int(z["gaussian_ids"].size)
    g = z["gaussian_ids"]
    ts = z["tile_starts"]
    te = z["tile_ends"]
    ti = z["tile_ids"]

    if M and (g.min() < 0 or g.max() >= N):
        problems.append(f"{p.name}: gaussian_ids range [{g.min()},{g.max()}] vs N={N}")
    if ts.size and int(ts[0]) != 0:
        problems.append(f"{p.name}: tile_starts[0]={int(ts[0])} != 0")
    if np.any(np.diff(ts) < 0) or np.any(te < ts):
        problems.append(f"{p.name}: tile_starts not monotone / ends<starts")
    if ts.size and int(te[-1]) != M:
        problems.append(f"{p.name}: last tile_end {int(te[-1])} != M {M}")
    if ti.size != M:
        problems.append(f"{p.name}: tile_ids size {ti.size} != M {M}")
    else:
        expect_ti = np.repeat(np.arange(ts.size, dtype=np.int64), te - ts)
        if not np.array_equal(ti, expect_ti):
            problems.append(f"{p.name}: tile_ids != repeat(arange, ends-starts)")
    radii = z["radii"]
    if not np.array_equal(z["visible_ids"], np.flatnonzero(radii > 0).astype(np.int64)):
        problems.append(f"{p.name}: visible_ids != flatnonzero(radii>0)")

anchor_ok = {f"{k[0]}:a{k[1]:02d}": sorted(v) == sorted(EXPECTED_ITERS)
             for k, v in sorted(per_anchor.items())}

out = {
    "scope": "pilot structural validation only; not X1/X2 scientific evidence",
    "n_npz_checked": n_checked,
    "n_anchors": len(per_anchor),
    "per_anchor_iters": {f"{k[0]}:a{k[1]:02d}": sorted(v) for k, v in sorted(per_anchor.items())},
    "each_anchor_has_expected_3_iters": all(anchor_ok.values()),
    "gaussian_ids_within_0_N": not any("gaussian_ids range" in x for x in problems),
    "tile_offsets_monotone_and_last_end_eq_M": not any("tile_end" in x or "monotone" in x or "tile_starts[0]" in x for x in problems),
    "tile_ids_consistent": not any("tile_ids" in x for x in problems),
    "visible_ids_eq_radii_gt_0": not any("visible_ids" in x for x in problems),
    "problems": problems,
    "status": "PILOT_STRUCTURE_VALID" if not problems else "PILOT_STRUCTURE_PROBLEMS",
}
Path(OUT).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
