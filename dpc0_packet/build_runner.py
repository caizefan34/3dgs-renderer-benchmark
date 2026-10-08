import sys, os
os.environ["MAX_JOBS"] = "8"
os.environ["VERBOSE"] = "1"
sys.path.insert(0, "/root/epic05_dpc0/src_full/gsplat/experimental/render/kernels/cuda")
import build_dpc
try:
    mod = build_dpc.build_and_load_dpc0()
    print("BUILD_OK")
    print("has higs_rasterize_backward:", hasattr(mod, "higs_rasterize_backward"))
    print("file:", getattr(mod, "__file__", "?"))
except Exception as e:
    import traceback; traceback.print_exc()
    print("BUILD_FAIL")