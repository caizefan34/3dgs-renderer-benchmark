"""Verify the HiGS (B2) build imports and report its API + provenance.
Run with: PYTHONNOUSERSITE=1 PYTHONPATH=<higs-build-root> CUDA_VISIBLE_DEVICES=0 python this.py
"""
import os, sys, inspect, site
print("argv0", sys.argv[0])
print("ENABLE_USER_SITE", getattr(site, "ENABLE_USER_SITE", None))
print("PYTHONPATH", os.environ.get("PYTHONPATH", "UNSET"))
print("sys.path[:5]")
for p in sys.path[:5]:
    print("  ", p)
try:
    import gsplat
    print("gsplat", os.path.dirname(gsplat.__file__))
    print("gsplat_ver", getattr(gsplat, "__version__", "?"))
    print("has_experimental_dir", os.path.isdir(os.path.join(os.path.dirname(gsplat.__file__), "experimental")))
except Exception as e:
    print("gsplat_ERR", repr(e))
try:
    import gsplat.experimental as ex
    print("experimental", os.path.dirname(ex.__file__))
    print("experimental_files", sorted(os.listdir(os.path.dirname(ex.__file__)))[:60])
except Exception as e:
    print("experimental_ERR", repr(e))
try:
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    print("frozen_file", inspect.getsourcefile(rasterize_gaussian_higs_frozen))
    print("frozen_sig", str(inspect.signature(rasterize_gaussian_higs_frozen))[:1200])
except Exception as e:
    print("frozen_ERR", repr(e))
try:
    from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER
    print("create_higs_file", inspect.getsourcefile(create_higs_renderer))
    print("create_higs_sig", str(inspect.signature(create_higs_renderer))[:800])
except Exception as e:
    print("create_higs_ERR", repr(e))
# also confirm the clean rasterization still importable in this tree
try:
    from gsplat.rendering import rasterization
    print("rendering_file", inspect.getsourcefile(rasterization))
    print("rasterization_ok", rasterization is not None)
except Exception as e:
    print("rendering_ERR", repr(e))
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("=== PROBE3_DONE ===")
