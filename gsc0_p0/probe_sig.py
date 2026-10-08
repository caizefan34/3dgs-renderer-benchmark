import sys, inspect, traceback
trees = {
 "true_accutile": "/root/epic05_frozen/gsplat-true-accutile-v153",
 "b1_clean": "/root/epic05_frozen/gsplat-b1-clean-v153",
}
for name, parent in trees.items():
    for m in list(sys.modules):
        if m=="gsplat" or m.startswith("gsplat."): del sys.modules[m]
    sys.path.insert(0, parent)
    try:
        import gsplat
        from gsplat import rasterization
        sig = inspect.signature(rasterization)
        print(f"[{name}] {gsplat.__file__}")
        print(f"   ver={getattr(gsplat,'__version__','?')} has_accutile={'accutile' in sig.parameters} has_absgrad={'absgrad' in sig.parameters}")
        print(f"   params={list(sig.parameters.keys())}")
    except Exception as e:
        print(f"[{name}] FAIL {repr(e)[:160]}")
        for ln in traceback.format_exc().splitlines()[-3:]: print("   ",ln)
    finally:
        sys.path.remove(parent)