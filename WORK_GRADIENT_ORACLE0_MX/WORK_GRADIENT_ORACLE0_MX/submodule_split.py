#!/usr/bin/env python3
"""Check the full split: which gsplat submodules resolve to source vs ~/.local,
and whether the source is self-consistent for a HiGS training run."""
import gsplat, os
src = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"
local = "/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat"

print("gsplat base:", gsplat.__file__)
subs = ["cuda._wrapper", "cuda._torch_impl", "rendering", "scene",
        "experimental", "experimental.render.functional.gaussian_inference",
        "strategy.default"]
for s in subs:
    try:
        m = __import__(f"gsplat.{s}", fromlist=["x"])
        f = m.__file__
        origin = "SOURCE" if src in f else ("LOCAL" if local in f else "OTHER")
        print(f"  {s:55s} {origin:7s} {f}")
    except Exception as e:
        print(f"  {s:55s} FAIL {e}")

print("\nsource gsplat/version.py:", os.path.exists(f"{src}/gsplat/version.py"))
try:
    import importlib
    v = importlib.import_module("gsplat.version")
    print("source version attr:", getattr(v, "__version__", "?"))
except Exception as e:
    print("version import:", e)
