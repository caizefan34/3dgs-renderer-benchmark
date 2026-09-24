"""Resolve which gsplat provides gsplat.experimental (the HiGS B2 renderer)."""
import os, sys, site, inspect
print("ENABLE_USER_SITE", getattr(site, "ENABLE_USER_SITE", None))
print("usersite", site.getusersitepackages())
print("sys.path[:6]")
for p in sys.path[:6]:
    print("  ", p)
try:
    import gsplat
    print("gsplat", os.path.dirname(gsplat.__file__))
    print("gsplat_ver", getattr(gsplat, "__version__", "?"))
except Exception as e:
    print("gsplat_ERR", repr(e))
try:
    import gsplat.experimental as ex
    print("experimental", os.path.dirname(ex.__file__))
    print("experimental_files", sorted(os.listdir(os.path.dirname(ex.__file__)))[:60])
except Exception as e:
    print("experimental_ERR", repr(e))
for base in [
    "/home/liaoyuanjun/miniforge3/envs/anysplat/lib/python3.10/site-packages/gsplat",
    "/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat",
]:
    print("DIR", base, "exists", os.path.isdir(base))
    if os.path.isdir(base):
        print("  contents", sorted(os.listdir(base))[:30])
        exp = os.path.join(base, "experimental")
        print("  experimental_exists", os.path.isdir(exp))
        if os.path.isdir(exp):
            print("  experimental_files", sorted(os.listdir(exp))[:60])
try:
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    print("frozen_file", inspect.getsourcefile(rasterize_gaussian_higs_frozen))
    print("frozen_sig", str(inspect.signature(rasterize_gaussian_higs_frozen))[:800])
except Exception as e:
    print("frozen_ERR", repr(e))
try:
    from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer
    print("create_higs_file", inspect.getsourcefile(create_higs_renderer))
    print("create_higs_sig", str(inspect.signature(create_higs_renderer))[:600])
except Exception as e:
    print("create_higs_ERR", repr(e))
print("=== PROBE2_DONE ===")
