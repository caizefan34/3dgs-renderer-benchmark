#!/usr/bin/env python3
"""BWD-ARITH1 common: monkey-patch _C_arith0 (R1 build) into production diff_rasterize.
R0: BWD1_CACHE_ENABLE unset -> production recompute (inside _C_arith0).
R1: BWD1_CACHE_ENABLE=1 -> exact FP32 forward Gaussian cache reuse.
"""
import sys, os

BWD1_DIR = "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN"
sys.path.insert(0, os.path.join(BWD1_DIR, "rasterization"))
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH0_MX")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/fused_profile0/pylibs")


def patch_C():
    """Import _C_arith0 (R1 build) and monkey-patch into diff_rasterize's module globals."""
    import mx_probe as MP
    if BWD1_DIR not in sys.path:
        sys.path.insert(0, BWD1_DIR)
    import _C_arith0 as C
    g = MP.diff_rasterize.__globals__
    g['_C'] = C
    return C


def set_mode(mode):
    if mode == "r1":
        os.environ["BWD1_CACHE_ENABLE"] = "1"
    else:
        os.environ.pop("BWD1_CACHE_ENABLE", None)
