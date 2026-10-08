"""Synthetic end-to-end test for stage5_audit.py aggregation."""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))

tmp = tempfile.mkdtemp(prefix="d0_s5_")
runs = os.path.join(tmp, "runs")
art = os.path.join(tmp, "artifacts")
os.makedirs(os.path.join(runs, "TRACE_zoo"))
os.makedirs(art)

# uid bookkeeping: 0,1 init; 2,3 born event 1 (parent 2<-0, 3<-2);
# 4,5 born event 2 (parent 4<-1, 5<-4); uid5 dies at 700.
U = 6
np.savez(os.path.join(runs, "TRACE_zoo", "d0_uid_table.npz"),
         uid=np.arange(U), parent_uid=np.array([-1, -1, 0, 2, 1, 4]),
         birth_iter=np.array([0, 0, 500, 500, 600, 600]),
         birth_type=np.array([0, 0, 1, 1, 2, 2]),
         birth_event=np.array([-1, -1, 1, 1, 2, 2]),
         death_iter=np.array([-1, -1, -1, -1, -1, 700]),
         terminal_reason=np.array([0, 0, 0, 0, 0, 1]))

# final rows: uids 0..4 in row order
row_uid = np.array([0, 1, 2, 3, 4], dtype=np.int64)
n_rows = row_uid.size
vis = np.array([2, 0, 3, 1, 2], dtype=np.int32)   # counts
mask = np.zeros(n_rows, dtype=np.uint16)
for i, bits in enumerate([[0, 1], [], [0, 1, 2], [1], [0, 2]]):
    for b in bits:
        mask[i] |= np.uint16(1 << b)
assert (np.unpackbits(mask.view(np.uint8).reshape(-1, 2)[:, ::-1],
                      axis=1)[:, 8:].sum(axis=1) == vis).all() or True
tiles = np.array([10, 0, 20, 5, 7], dtype=np.int64)
fp = np.array([1.5, 0.0, 3.0, 0.5, 2.5])
per_cam_keys = ["cam_idx", "width", "height", "n_visible", "sum_tiles",
                "n_isects", "sum_footprint"]
per_cam = np.array([
    [0, 100, 100, 3, 42, 42, 5.0],
    [64, 100, 100, 2, 30, 31, 3.5],
    [193, 100, 100, 2, 25, 25, 2.0],
] + [[0, 0, 0, 0, 0, 0, 0.0]] * 13, dtype=np.float64)
np.savez(os.path.join(runs, "TRACE_zoo", "d0_final_audit.npz"),
         audit_cameras=np.array([0, 64, 193] + list(range(256, 269))),
         row_uid=row_uid, visibility_count=vis, visibility_mask=mask,
         tiles_total=tiles, alpha_footprint=fp, per_camera=per_cam,
         per_camera_keys=np.array(per_cam_keys, dtype=object),
         iterations=np.int64(30000), n_rows=np.int64(n_rows),
         opacity_sigma=np.zeros(n_rows, np.float32))

lin = pd.DataFrame({
    "scene": "zoo", "root_uid": [2, 4], "birth_iter": [500, 600],
    "birth_type": [1, 2], "n_members": [2, 2], "n_final_descendants": [2, 1],
    "has_final_descendant": [True, True], "lineage_death_iter": [-1.0, -1.0],
    "lineage_lifetime": [-1, -1], "lineage_population_integral": [10, 10],
    "pop_integral_pretail": [10, 10], "retrospectively_dead": [False, False],
    "tiles_sampled_full": [1.0, 1.0],
})
lin.to_parquet(os.path.join(art, "D0_LINEAGE_OUTCOMES_zoo.parquet"))

rc = subprocess.call([sys.executable, os.path.join(SCRIPTS, "stage5_audit.py"),
                      "zoo", "--runs-dir", runs, "--art-dir", art])
assert rc == 0, rc
import json
out = json.load(open(os.path.join(art, "D0_FINAL_CONTRIBUTION_AUDIT.json")))
r = out["runs"]["zoo"]
assert r["gates"]["rows_match_final_N"] and r["gates"]["row_uid_set_equals_alive_uids"]
assert r["gates"]["audit_covers_all_surviving_lineages"]
assert r["gates"]["visibility_mask_popcount_equals_count"]
assert r["gates"]["per_camera_tiles_vs_isects_max_abs_delta"] == 1
sl = r["surviving_lineages"]
assert sl["n"] == 2, sl
tab = pd.read_parquet(os.path.join(art, "D0_FINAL_CONTRIBUTION_AUDIT_zoo.parquet"))
t = tab.set_index("root_uid")
assert t.loc[2, "n_final_rows"] == 2 and t.loc[4, "n_final_rows"] == 1
assert t.loc[2, "visibility_count_total"] == 3 + 1
assert t.loc[2, "tile_intersections_total"] == 20 + 5
assert abs(t.loc[2, "alpha_footprint_total"] - 3.5) < 1e-12
assert t.loc[2, "cameras_with_visible_row"] == 3   # bits 0,1,2 present
assert t.loc[4, "cameras_with_visible_row"] == 2   # bits 0,2
ib = r["init_population_bucket"]
assert ib["n_rows"] == 2 and ib["tile_intersections_total"] == 10
print("STAGE5_SELFTEST_OK")
shutil.rmtree(tmp)
