#!/usr/bin/env python3
import os
SRC = "/mnt/storage_pool/liaoyuanjun/comax0_mx/COMAX0_CUDA_PROD/src"
p = os.path.join(SRC, "comax_envelope_ext.py")
s = open(p, encoding="utf-8").read()
old = """        bundle["node_base"], bundle["node_pow2"],
        bundle["tile_ids"], bundle["tile_offsets"],
        bundle["n_tiles"], bundle["tw"], bundle["th"], bundle["width"], bundle["height"],
        bundle["background"].to(bundle["means2d"].dtype))"""
new = """        bundle["node_base"], bundle["node_pow2"],
        bundle["n_tiles"], bundle["tw"], bundle["th"], bundle["width"], bundle["height"],
        bundle["tile_ids"], bundle["tile_offsets"],
        bundle["background"].to(bundle["means2d"].dtype))"""
assert old in s, "render call order not found"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("FIX_OK")
