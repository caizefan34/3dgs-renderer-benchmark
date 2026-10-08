import sys, os
os.environ["MAX_JOBS"] = "8"
os.environ["VERBOSE"] = "1"
os.environ.setdefault("FAST_MATH", "1")
sys.path.insert(0, "/root/epic05_dpc0/src_timed/gsplat/experimental/render/kernels/cuda")
import build_dpc
try:
    mod = build_dpc.build_and_load_dpc0()
    print("BUILD_OK")
    print("file:", getattr(mod, "__file__", "?"))
    for name in ["higs_rasterize_backward","dpc0_get_stage_times","dpc0_get_n_isects","dpc0_get_pstride","dpc0_get_N","dpc0_get_partial_key"]:
        print("has", name, hasattr(mod, name))
except Exception as e:
    import traceback; traceback.print_exc()
    print("BUILD_FAIL")
