import sys, os
os.environ["MAX_JOBS"] = "8"
os.environ["VERBOSE"] = "1"
sys.path.insert(0, "/root/epic05_dpgr0/src_dpgr0/gsplat/experimental/render/kernels/cuda")
import build_dpgr0
try:
    mod = build_dpgr0.build_and_load_dpgr0()
    print("BUILD_OK")
    print("file:", getattr(mod, "__file__", "?"))
    for name in ["higs_rasterize_backward_dpgr0", "dpgr0_get_stage_times", "dpgr0_get_stats"]:
        print("has", name, hasattr(mod, name))
except Exception as e:
    import traceback; traceback.print_exc()
    print("BUILD_FAIL")
