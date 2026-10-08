import sys, time
sys.path.insert(0, "/root/mr0_pkg")
import mr0_cell as CC
t0 = time.perf_counter()
CC.build_groups("room@10000", "/root/mr0_out_test2")
print("BUILD_GROUPS_DONE", round(time.perf_counter()-t0, 1), "s")
