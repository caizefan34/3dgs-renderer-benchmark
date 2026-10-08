#!/usr/bin/env python3
import sys, os, hashlib
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
ident = C.bootstrap_c0()
print("c0_ext_so:", ident["so_sha256"])
print("core_so:", ident["core_sha256"])
print("c0_so_match_prereg:", ident["so_sha256"] == C.C0_SO)
print("core_so_match_prereg:", ident["core_sha256"] == C.CORE_SO)
# renderer signature default tile_size
from gsplat.experimental import rasterize_gaussian_higs_dynamic
import inspect
sig = inspect.signature(rasterize_gaussian_higs_dynamic)
print("renderer params:", list(sig.parameters.keys()))
print("tile_size default:", sig.parameters["tile_size"].default if "tile_size" in sig.parameters else "ABSENT")
# fused loss lock
fused = C.make_fused_loss()
print("fused loss so:", fused.so_path)
print("fused so sha:", fused.so_sha256)
print("P11 lock match:", fused.so_sha256 == C.P11_FUSED_SO_SHA256)
