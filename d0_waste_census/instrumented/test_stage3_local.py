#!/usr/bin/env python3
"""Synthetic local test for stage3_accounting.py (no GPU, no real data).

Fixture (7 uids):
  0,1  init, alive
  2    clone(0)  b100 d250  |  3  clone(0)  b100 d250
  4    split(2)  b200 d250  |  5  split(1)  b150 d250
  6    clone(0)  b100 alive
main roots: 2={2,4} dead | 3={3} dead | 5={5} dead | 6={6} alive
init-root sensitivity: tree1={1,5} dead | tree0={0,2,3,4,6} alive
(uid 1 itself dies at 250, so its init tree is fully extinct)
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import stage3_accounting as s3  # noqa: E402

tmp = tempfile.mkdtemp(prefix="d0_stage3_")
runs = os.path.join(tmp, "runs")
os.makedirs(os.path.join(runs, "TRACE_test", "d0_cost_samples"))
s3.RUNS = runs
s3.ART = tmp

d = os.path.join(runs, "TRACE_test")

np.savez(os.path.join(d, "d0_uid_table.npz"),
         uid=np.arange(7, dtype=np.int64),
         parent_uid=np.array([-1, -1, 0, 0, 2, 1, 0], dtype=np.int64),
         birth_iter=np.array([0, 0, 100, 100, 200, 150, 100], dtype=np.int64),
         birth_type=np.array([0, 0, 1, 1, 2, 2, 1], dtype=np.int64),
         birth_event=np.array([0, 0, 1, 1, 2, 2, 1], dtype=np.int64),
         death_iter=np.array([-1, 250, 250, 250, 250, 250, -1], dtype=np.int64),
         terminal_reason=np.array([0, 1, 1, 1, 4, 4, 0], dtype=np.int64),
         tiles_cum=np.zeros(7, dtype=np.int64),
         cam_count=np.ones(7, dtype=np.int64),
         hist_n=np.ones(7, dtype=np.int64))

iters = np.arange(1, 301, dtype=np.int64)
I_t = np.full(300, 1000, dtype=np.int64)
np.savez(os.path.join(d, "d0_population_trajectory.npz"), iter=iters,
         I_t=I_t, I_t_sumchecked=I_t.copy(), loss=np.zeros(300))

sam_iters = np.array([1, 21, 41], dtype=np.int64)
np.savez(os.path.join(d, "d0_sample_summary.npz"), iter=sam_iters,
         I_t=np.array([1000, 1000, 1000]),
         sum_tiles=np.array([990, 980, 970]),
         N_t=np.array([7, 7, 7]), n_visible=np.array([7, 7, 7]),
         n_nonzero=np.array([7, 7, 7]),
         tiles_uid_sum=np.array([990.0, 980.0, 970.0]))

# tiles per uid: dead uids {2,3,4,5} hold 400/1000
TILES = np.array([200, 200, 100, 100, 100, 100, 200], dtype=np.uint16)
for it in sam_iters:
    np.savez(os.path.join(d, "d0_cost_samples", "sample_%05d.npz" % it),
             uid=np.arange(7, dtype=np.int32), tiles=TILES)

eval_rows = [
    {"step": 5000, "tag": "d0_ckpt", "psnr": 29.95, "ssim": 0.90, "lpips": 0.30},
    {"step": 10000, "tag": "d0_ckpt", "psnr": 30.0, "ssim": 0.90, "lpips": 0.30},
    {"step": 15000, "tag": "d0_ckpt", "psnr": 30.0, "ssim": 0.90, "lpips": 0.30},
    {"step": 20000, "tag": "d0_ckpt", "psnr": 30.0, "ssim": 0.90, "lpips": 0.30},
    {"step": 25000, "tag": "d0_ckpt", "psnr": 30.0, "ssim": 0.90, "lpips": 0.30},
    {"step": 30000, "tag": "final", "psnr": 30.0, "ssim": 0.90, "lpips": 0.30},
]
with open(os.path.join(d, "results.json"), "w") as f:
    json.dump({"eval_rows": eval_rows, "final_N": 7}, f)
with open(os.path.join(d, "d0_trace_meta.json"), "w") as f:
    json.dump({"n_features": 30}, f)

sys.argv = ["stage3_accounting.py", "test"]
rc = s3.main()

acc = json.load(open(os.path.join(tmp, "D0_ACCOUNTING.json")))
s = acc["scenes"]["test"]
assert s["t_star"] == 5000, s["t_star"]
assert s["lineage"]["n_post_init_roots"] == 4, s["lineage"]
assert s["lineage"]["n_dead_roots"] == 3, s["lineage"]
# pop_full: r2=150+50=200, r3=150, r5=100, r6=29901
assert s["lineage"]["pop_integral_full_total"] == 200 + 150 + 100 + 29901, s["lineage"]
exp_rN = 450 / (450 + 4901)
assert abs(s["pre_tail"]["r_N_dead"] - exp_rN) < 1e-12, s["pre_tail"]
assert abs(s["pre_tail"]["r_I_dead"] - 0.4) < 1e-12, s["pre_tail"]
assert s["pre_tail"]["n_samples_used"] == 3
exp_R = ((s3.F_FB * 0.4 + s3.F_OPT * exp_rN) * 5000
         + (s3.F_FB + s3.F_OPT) * 25000) / ((s3.F_FB + s3.F_OPT) * 30000)
assert abs(s["headroom"]["R_upper"] - exp_R) < 1e-12, s["headroom"]
assert s["gates"]["accounting_valid"] is True
assert s["recon_sampled"]["pass"] is True
# sensitivity: dead tree = {1,5}; pop 30100 full, 5101 at T; r_N_i = 5101/(5101+10252)
si = s["init_root_sensitivity"]
assert si["n_roots"] == 2 and si["n_dead_roots"] == 1, si
exp_rN_i = 350 / (350 + 10252)
assert abs(si["r_N_dead"] - exp_rN_i) < 1e-12, si
assert abs(si["r_I_dead"] - 0.3) < 1e-12, si
print("STAGE3 SYNTHETIC OK rc=%d R_upper=%.6f rN=%.6f rI=%.3f" % (rc, exp_R, exp_rN, 0.4))
shutil.rmtree(tmp, ignore_errors=True)
