import sys, importlib.util, torch
torch.cuda.init()
CORE = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
SRC = "/tmp/higs_h3_fwd_1a_source"
for nm in ("gsplat.csrc", "gsplat_csrc"):
    try:
        spec = importlib.util.spec_from_file_location(nm, CORE)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        sys.modules[nm] = m
        print("loaded as", nm)
    except Exception as e:
        print("load %s fail:" % nm, repr(e)[:200])
sys.path.insert(0, SRC)
try:
    import gsplat
    from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode
    print("wrapper OK", getattr(gsplat, "__version__", "?"))
except Exception as e:
    import traceback; traceback.print_exc()
    print("wrapper FAIL:", repr(e)[:300])