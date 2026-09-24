import sys, importlib.util
import ctypes, torch  # import torch first so libc10/libtorch are loaded
torch.cuda.init()
sys.path.insert(0, "/tmp/higs_h3_fwd_1a_source")
CORE = "/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so"
spec = importlib.util.spec_from_file_location("gsplat_csrc", CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
sys.modules["gsplat.csrc"] = core
print("core loaded, torch ops:", hasattr(__import__("torch").ops, "gsplat") or "n/a")
import gsplat
print("gsplat:", getattr(gsplat, "__version__", "?"))
try:
    from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched, _gather_visible_native
    print("experimental OK")
except Exception as e:
    print("experimental FAIL:", repr(e)[:300])
try:
    from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode
    print("wrapper OK")
except Exception as e:
    print("wrapper FAIL:", repr(e)[:300])