"""Directly load the higs-tree .so files to capture the REAL error (not JIT fallback)."""
import os, sys, importlib, importlib.util, traceback
H = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"
print("H exists", os.path.isdir(H))
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cxx11_abi", torch._C._GLIBCXX_USE_CXX11_ABI)

for label, name, path in [
    ("MAIN", "csrc", os.path.join(H, "gsplat/csrc.so")),
    ("EXP", "csrc", os.path.join(H, "gsplat/experimental/render/kernels/csrc.so")),
]:
    print(f"\n=== {label} direct load {path} exists={os.path.exists(path)} ===")
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        pub = [a for a in dir(m) if not a.startswith("_")]
        print("  OK loaded; n_public_attrs", len(pub), "sample", pub[:12])
    except Exception as e:
        print("  FAIL", repr(e))
        traceback.print_exc(limit=4)
print("\n=== PROBE4_DONE ===")
