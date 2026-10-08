import sys, time
sys.path.insert(0, "/root/mr0_pkg")
import mr0_cell as CC
t0 = time.perf_counter()
CC.fit_shard("room@10000", "/root/mr0_out_test2", 0, 8, n_steps=8, n_cam_fit=8)
print("FIT_DONE", round(time.perf_counter()-t0, 1), "s")
