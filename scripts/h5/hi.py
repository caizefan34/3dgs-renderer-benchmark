import sys, torch
sys.path = [p for p in sys.path if '/.local/' not in p]
PY = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python"
print("python", sys.executable, "torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())
try:
    import gsplat
    print("gsplat", gsplat.__file__)
    from gsplat.cuda import _backend
    print("backend _C:", type(_backend._C))
except Exception as e:
    import traceback; traceback.print_exc()
    print("BASE-IMPORT FAIL", repr(e)[:250])
try:
    from gsplat.experimental.render.functional.gaussian_inference import _cull_gaussians_batched, _gather_visible_native
    print("experimental OK", _cull_gaussians_batched)
except Exception as e:
    print("EXP FAIL", repr(e)[:250])